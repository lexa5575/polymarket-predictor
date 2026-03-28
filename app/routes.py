"""
Custom API Routes
-----------------

Additional endpoints beyond AgentOS defaults:
- POST /api/scan-and-fanout — batch scan + fan-out workflow runs
- POST /api/settle — check resolved markets, update paper trades
- GET  /api/dashboard — current bankroll snapshot
- POST /api/price-prediction — predict coin price direction
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter

from agents.settings import get_paper_trade_store
from schemas.price_prediction import PricePrediction, PricePredictionRequest
from schemas.workflow_input import PredictionRequest
from tools.polymarket import PolymarketTools

logger = logging.getLogger(__name__)
router = APIRouter()

_polymarket_tools = PolymarketTools()


def _extract_record(wf_result) -> dict:
    """Extract the Record step result from workflow output.

    The Record step (conditional_logging) returns StepOutput(content=dict).
    The workflow's final result carries this as wf_result.content.
    """
    if not wf_result or wf_result.content is None:
        return {"action": "unknown", "trade_id": None, "reason": "No workflow result"}
    content = wf_result.content
    if isinstance(content, dict):
        return content
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json")
    # Fallback: try JSON parse
    try:
        return json.loads(str(content))
    except (json.JSONDecodeError, TypeError):
        pass
    return {"action": "unknown", "trade_id": None, "reason": "Could not parse workflow result"}


@router.post("/scan-and-fanout")
async def scan_and_fanout(max_candidates: int = 20):
    """Job 1: Scan crypto markets → fan-out prediction workflow runs.

    Deterministic candidate selection (no LLM scanner).
    Only supported assets with sufficient liquidity are processed.
    """
    from app.scanner_service import scan_candidates
    from workflows import prediction_workflow

    # 1. Deterministic scan — no LLM
    candidates = scan_candidates(max_candidates=max_candidates)

    # 2. Fan-out: run workflow for each candidate
    results = []
    for candidate in candidates:
        try:
            wf_result = prediction_workflow.run(
                input=PredictionRequest(
                    mode="single_market",
                    condition_id=candidate["condition_id"],
                    gamma_market_id=candidate.get("gamma_market_id"),
                )
            )
            record = _extract_record(wf_result)

            results.append({
                "condition_id": candidate["condition_id"],
                "question": candidate["question"],
                "status": "completed",
                "action": record.get("action", "unknown"),
                "trade_id": record.get("trade_id"),
                "side": record.get("side"),
                "stake": record.get("stake"),
                "reason": record.get("reason"),
            })
        except Exception as e:
            results.append({
                "condition_id": candidate["condition_id"],
                "question": candidate["question"],
                "status": "error",
                "error": str(e),
            })
    return {
        "scan_completed": True,
        "candidates_processed": len(results),
        "results": results,
    }


@router.post("/settle")
async def settle_trades():
    """Job 2: Check resolved markets, update paper trades, snapshot bankroll."""
    import json

    store = get_paper_trade_store()
    open_trades = store.get_open_trades()

    if not open_trades:
        return {"message": "No open trades to settle", "settled": 0}

    settled = []
    errors = []

    for trade in open_trades:
        try:
            resolution_json = _polymarket_tools.get_market_resolution(
                trade.condition_id
            )
            resolution = json.loads(resolution_json)

            if resolution.get("status") == "resolved":
                outcome = resolution.get("final_outcome")
                if outcome in ("YES", "NO"):
                    resolved = store.resolve_trade(trade.id, outcome)
                    settled.append({
                        "trade_id": trade.id,
                        "question": trade.question,
                        "side": trade.side,
                        "outcome": outcome,
                        "won": resolved.status == "won",
                        "pnl": resolved.pnl,
                        "brier_score": resolved.brier_score,
                    })
        except Exception as e:
            errors.append({"trade_id": trade.id, "error": str(e)})

    # Create bankroll snapshot after settlement
    snapshot = store.create_bankroll_snapshot()

    return {
        "settled": len(settled),
        "errors": len(errors),
        "trades": settled,
        "bankroll": snapshot.model_dump(),
    }


@router.get("/dashboard")
async def dashboard():
    """Current bankroll snapshot + open positions with unrealized PnL."""
    import json as _json
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    from storage.math_utils import calculate_mtm_pnl, get_exit_price_from_orderbook

    store = get_paper_trade_store()
    snapshot = store.get_bankroll_snapshot()
    open_trades = store.get_open_trades()
    now = _dt.now(_tz.utc)

    positions = []
    for t in open_trades:
        pos = {
            "id": t.id,
            "question": t.question,
            "side": t.side,
            "stake": t.stake,
            "edge": t.edge,
            "underlier_group": t.underlier_group,
            "created_at": t.created_at.isoformat(),
            "age_minutes": round(
                (now - (t.created_at.replace(tzinfo=_tz.utc) if t.created_at.tzinfo is None else t.created_at)).total_seconds() / 60, 1,
            ),
            "exit_policy": {
                "take_profit_pct": t.take_profit_pct,
                "stop_loss_pct": t.stop_loss_pct,
                "max_hold_seconds": t.max_hold_seconds,
            },
        }
        # Try to get current price for unrealized PnL
        try:
            book = _json.loads(_polymarket_tools.get_orderbook(t.token_id))
            price = get_exit_price_from_orderbook(book)
            if price is not None:
                pnl = calculate_mtm_pnl(t.entry_fill_price, price, t.stake)
                pos["current_price"] = price
                pos["unrealized_pnl"] = round(pnl, 2)
                pos["unrealized_pnl_pct"] = round(pnl / t.stake, 4) if t.stake else 0.0
        except Exception:
            pass  # graceful degradation
        positions.append(pos)

    return {
        "bankroll": snapshot.model_dump(),
        "open_positions": positions,
    }


@router.post("/monitor")
async def monitor_positions():
    """Check all open trades, close by exit conditions (TP/SL/max_hold/resolution)."""
    from app.monitor import run_monitor
    return run_monitor()


@router.get("/analytics")
async def analytics():
    """Detailed trading performance analytics."""
    import statistics as _stats

    store = get_paper_trade_store()
    all_trades = store.get_all_trades()

    finished = [t for t in all_trades if t.status in ("won", "lost", "closed")]
    resolved = [t for t in all_trades if t.status in ("won", "lost")]
    closed_early = [t for t in all_trades if t.status == "closed"]
    won = [t for t in resolved if t.status == "won"]
    lost = [t for t in resolved if t.status == "lost"]

    total_pnl = sum(t.pnl or 0.0 for t in finished)
    profitable = [t for t in finished if (t.pnl or 0) > 0]
    unprofitable = [t for t in finished if (t.pnl or 0) < 0]

    profits = [t.pnl for t in profitable if t.pnl is not None]
    losses = [t.pnl for t in unprofitable if t.pnl is not None]
    avg_profit = _stats.mean(profits) if profits else 0.0
    avg_loss = _stats.mean(losses) if losses else 0.0
    profit_factor = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else 0.0

    # Two win rates
    n_finished = len(finished)
    n_resolved = len(resolved)
    trading_win_rate = len(profitable) / n_finished if n_finished > 0 else 0.0
    resolution_win_rate = len(won) / n_resolved if n_resolved > 0 else 0.0

    # Sharpe
    pnl_list = [t.pnl for t in finished if t.pnl is not None]
    sharpe = None
    if len(pnl_list) >= 2:
        std = _stats.stdev(pnl_list)
        if std > 0:
            sharpe = _stats.mean(pnl_list) / std

    # Avg hold time
    hold_times = []
    for t in finished:
        end = t.exit_time or t.resolution_time
        if end and t.created_at:
            hold_times.append((end - t.created_at).total_seconds() / 60)
    avg_hold = _stats.mean(hold_times) if hold_times else 0.0

    # By exit reason (closed trades only)
    by_exit_reason = {}
    for t in closed_early:
        reason = t.exit_reason or "unknown"
        bucket = by_exit_reason.setdefault(reason, {"count": 0, "total_pnl": 0.0})
        bucket["count"] += 1
        bucket["total_pnl"] += t.pnl or 0.0
    for bucket in by_exit_reason.values():
        bucket["avg_pnl"] = round(bucket["total_pnl"] / bucket["count"], 2) if bucket["count"] else 0.0

    # By resolution
    by_resolution = {
        "won": {"count": len(won), "avg_pnl": round(_stats.mean([t.pnl for t in won if t.pnl]) if won else 0.0, 2)},
        "lost": {"count": len(lost), "avg_pnl": round(_stats.mean([t.pnl for t in lost if t.pnl]) if lost else 0.0, 2)},
    }

    # By underlier group
    by_group = {}
    for t in finished:
        g = t.underlier_group
        bucket = by_group.setdefault(g, {"count": 0, "total_pnl": 0.0})
        bucket["count"] += 1
        bucket["total_pnl"] += t.pnl or 0.0
    for bucket in by_group.values():
        bucket["total_pnl"] = round(bucket["total_pnl"], 2)

    # Edge vs result
    edges = [t.edge for t in finished if t.edge is not None]
    returns = [(t.pnl or 0) / t.stake if t.stake else 0.0 for t in finished]
    avg_edge = _stats.mean(edges) if edges else 0.0
    avg_return = _stats.mean(returns) if returns else 0.0

    return {
        "total_trades": len(all_trades),
        "finished_trades": n_finished,
        "trading_win_rate": round(trading_win_rate, 4),
        "resolution_win_rate": round(resolution_win_rate, 4),
        "avg_profit": round(avg_profit, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe_ratio": round(sharpe, 4) if sharpe else None,
        "avg_hold_minutes": round(avg_hold, 1),
        "by_exit_reason": by_exit_reason,
        "by_resolution": by_resolution,
        "by_underlier_group": by_group,
        "edge_vs_result": {
            "avg_predicted_edge": round(avg_edge, 4),
            "avg_realized_return": round(avg_return, 4),
        },
    }


# ---------------------------------------------------------------------------
# Price Prediction
# ---------------------------------------------------------------------------


def _extract_dict_from_response(response) -> dict:
    """Extract dict from agent response content (Pydantic model, dict, or JSON string)."""
    if not response or response.content is None:
        return {}
    content = response.content
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json", exclude_none=True)
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        # Try full JSON parse
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: find last JSON object in text
        for match in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content))):
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                continue
    return {}


@router.post("/price-prediction")
async def price_prediction(request: PricePredictionRequest):
    """Predict whether a coin's price will be above/below a target within timeframe.

    Calls Market Data Agent, News Agent, and Risk Agent directly — no Polymarket needed.
    """
    from agents import market_data_agent, risk_agent

    # 1. Get market data
    try:
        market_response = await market_data_agent.arun(
            f"Get current {request.coin} price, funding rate, open interest, and Fear & Greed index. "
            f"Context: predicting whether {request.coin} will be {request.direction} "
            f"${request.price_target:,.0f} in {request.timeframe}."
        )
        market_data = _extract_dict_from_response(market_response)
    except Exception as e:
        logger.error("Market data agent failed: %s", e)
        market_data = {}

    # 2. Get sentiment (deterministic news service — same as workflow)
    try:
        from app.news_service import fetch_sentiment

        query = f"{request.coin} price {request.direction} ${request.price_target:,.0f} {request.timeframe}"
        sentiment_report = fetch_sentiment(query)
        sentiment = sentiment_report.model_dump(mode="json")
    except Exception as e:
        logger.error("News sentiment service failed: %s", e)
        sentiment = {"sentiment_score": 0.0, "key_narratives": ["Sentiment service unavailable"], "confidence": 0.1}

    # 3. Get prediction from Risk Agent
    current_price = market_data.get("price_usd", 0)
    fear_greed = market_data.get("fear_greed_index", 50)
    signal = market_data.get("signal", "Unknown")
    sent_score = sentiment.get("sentiment_score", 0)
    narratives = sentiment.get("key_narratives", [])

    try:
        risk_response = await risk_agent.arun(
            f"Predict: Will {request.coin} be {request.direction} ${request.price_target:,.0f} "
            f"in the next {request.timeframe}?\n\n"
            f"Current price: ${current_price:,.0f}\n"
            f"24h change: {market_data.get('change_24h_pct', 0):.1f}%\n"
            f"Fear & Greed Index: {fear_greed} ({market_data.get('fear_greed_label', 'Unknown')})\n"
            f"Market signal: {signal}\n"
            f"Funding rate: {market_data.get('funding_rate', 'N/A')}\n"
            f"Open interest: {market_data.get('open_interest', 'N/A')}\n"
            f"News sentiment score: {sent_score}\n"
            f"Key narratives: {json.dumps(narratives)}\n\n"
            f"Respond with your estimated probability (0-1) that {request.coin} "
            f"WILL be {request.direction} ${request.price_target:,.0f} in {request.timeframe}, "
            f"plus your confidence (High/Medium/Low) and brief reasoning."
        )
        prediction = _extract_dict_from_response(risk_response)
    except Exception as e:
        logger.error("Risk agent failed: %s", e)
        prediction = {"estimated_prob_of_side": 0.5, "risk_rating": "Unknown"}

    # 4. Parse prediction
    # estimated_prob_of_side = P(recommended_side wins), NOT P(YES).
    # Normalize to P(event will happen) = P(YES) for the response.
    raw_prob = prediction.get("estimated_prob_of_side")
    if raw_prob is None:
        raw_prob = prediction.get("estimated_probability")
    if raw_prob is None:
        raw_prob = 0.5
    recommended_side = prediction.get("recommended_side", "YES")
    prob_yes = raw_prob if recommended_side == "YES" else 1.0 - raw_prob

    confidence = prediction.get("confidence") or prediction.get("risk_rating") or "Medium"
    if confidence not in ("High", "Medium", "Low"):
        confidence = "Medium"
    rationale = (
        prediction.get("reasoning")
        or prediction.get("rationale")
        or " | ".join(prediction.get("warnings", []))
        or f"Signal: {signal}, Sentiment: {sent_score}"
    )

    return PricePrediction(
        coin=request.coin,
        current_price=current_price,
        price_target=request.price_target,
        direction=request.direction,
        timeframe=request.timeframe,
        prediction="YES" if prob_yes > 0.5 else "NO",
        estimated_probability=round(prob_yes, 3),
        confidence=confidence,
        signal=signal,
        fear_greed_index=fear_greed,
        sentiment_score=round(sent_score, 2),
        rationale=rationale,
        market_data=market_data,
        sentiment=sentiment,
    )
