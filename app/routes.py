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
async def scan_and_fanout(max_candidates: int = 5):
    """Job 1: Scan crypto markets → fan-out prediction workflow runs.

    Calls polymarket_scanner programmatically, then runs the prediction
    workflow for each candidate. No self-POST — all in-process.
    """
    from agents import polymarket_scanner
    from workflows import prediction_workflow

    # 1. Scan for candidates
    scan_result = polymarket_scanner.run(
        f"Scan active crypto prediction markets. Return top {max_candidates} candidates "
        "ranked by liquidity and potential edge."
    )

    # 2. Fan-out: run workflow for each candidate
    results = []
    if scan_result and scan_result.content:
        content = scan_result.content
        # If structured output, iterate candidates
        candidates = getattr(content, "candidates", None)
        if candidates:
            for candidate in candidates[:max_candidates]:
                try:
                    wf_result = prediction_workflow.run(
                        input=PredictionRequest(
                            mode="single_market",
                            condition_id=candidate.condition_id,
                        )
                    )
                    record = _extract_record(wf_result)

                    results.append({
                        "condition_id": candidate.condition_id,
                        "question": candidate.question,
                        "status": "completed",
                        "action": record.get("action", "unknown"),
                        "trade_id": record.get("trade_id"),
                        "side": record.get("side"),
                        "stake": record.get("stake"),
                        "reason": record.get("reason"),
                    })
                except Exception as e:
                    results.append({
                        "condition_id": candidate.condition_id,
                        "status": "error",
                        "error": str(e),
                    })
        else:
            results.append({"status": "no_candidates_found"})

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
    """Current bankroll snapshot + open positions summary."""
    store = get_paper_trade_store()
    snapshot = store.get_bankroll_snapshot()
    open_trades = store.get_open_trades()

    return {
        "bankroll": snapshot.model_dump(),
        "open_positions": [
            {
                "id": t.id,
                "question": t.question,
                "side": t.side,
                "stake": t.stake,
                "edge": t.edge,
                "underlier_group": t.underlier_group,
                "created_at": t.created_at.isoformat(),
            }
            for t in open_trades
        ],
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
    from agents import market_data_agent, news_agent, risk_agent

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

    # 2. Get sentiment
    try:
        sentiment_response = await news_agent.arun(
            f"What is the current sentiment for {request.coin} price "
            f"in the next {request.timeframe}? Any catalysts or risks that could move the price?"
        )
        sentiment = _extract_dict_from_response(sentiment_response)
    except Exception as e:
        logger.error("News agent failed: %s", e)
        sentiment = {"sentiment_score": 0.0, "key_narratives": ["News agent unavailable"], "confidence": 0.1}

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
