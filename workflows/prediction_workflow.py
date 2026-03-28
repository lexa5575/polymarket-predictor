"""
Prediction Workflow — Structured Step-to-Step Handoff
-----------------------------------------------------

Pipeline: Event Scan → Data+News (parallel) → Data Quality → Risk (LLM) → Edge & Gate (code) → Sizing (code) → Decision (code) → Record

All steps after Data Collection are function-steps (executor=) that:
1. Read typed outputs from prior steps via get_step_output()
2. Call agents with complete prompts when needed
3. Return typed StepOutput (Pydantic model or plain dict)

No tagged string blocks. No JSON parsing from text.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from agno.workflow import Parallel, Step, Workflow
from agno.workflow.types import StepInput, StepOutput

from agents import (
    logger_agent,
    risk_agent,
)
from agents.settings import get_paper_trade_store
from tools.polymarket import PolymarketTools
from schemas.market import (
    BetDecision,
    EventCandidate,
    MarketSnapshot,
    RiskAssessment,
    RiskEstimate,
    SentimentReport,
)
from schemas.workflow_input import PredictionRequest
from storage.math_utils import (
    calculate_entry_price,
    check_liquidity,
    check_portfolio_limits,
    compute_edge,
    confidence_to_score,
    determine_risk_rating,
    estimate_slippage,
    fractional_kelly,
    kelly_criterion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_content_to_dict(step_output: StepOutput | None) -> dict | None:
    """Extract dict from step content. For function-step plain dict outputs."""
    if not step_output or step_output.content is None:
        return None
    content = step_output.content
    if hasattr(content, "model_dump"):
        return content.model_dump(mode="json", exclude_none=True)
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _step_content_to_model(
    step_output: StepOutput | None,
    model_class: type[BaseModel],
) -> BaseModel | None:
    """Extract and validate a Pydantic model from step content. Returns None if invalid."""
    if not step_output or step_output.content is None:
        return None
    content = step_output.content
    if isinstance(content, model_class):
        return content
    try:
        # Handle Pydantic model of different type
        if hasattr(content, "model_dump"):
            d = content.model_dump(mode="json")
            return model_class.model_validate(d)
        # Handle dict
        if isinstance(content, dict):
            return model_class.model_validate(content)
        # Handle JSON string
        if isinstance(content, str):
            # Try full string as JSON first
            try:
                d = json.loads(content)
                return model_class.model_validate(d)
            except (json.JSONDecodeError, Exception):
                pass
            # Fallback: find last JSON object in text (agent may embed JSON in prose)
            import re
            for match in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content))):
                try:
                    d = json.loads(match.group())
                    return model_class.model_validate(d)
                except (json.JSONDecodeError, Exception):
                    continue
    except Exception:
        pass
    return None


def _safe_risk_assessment(condition_id: str = "unknown", warnings: list[str] | None = None) -> RiskAssessment:
    """Create a safe Unacceptable RiskAssessment fallback."""
    return RiskAssessment(
        condition_id=condition_id,
        risk_rating="Unacceptable",
        recommended_side="YES",
        estimated_prob_of_side=0.5,
        market_prob_of_side=0.5,
        edge=0.0,
        underlier_group="other",
        warnings=warnings or ["Safe fallback"],
        liquidity_ok=False,
        correlated_positions=0,
    )


def _safe_bet_decision(
    condition_id: str = "unknown",
    market_slug: str = "unknown",
    rationale: str = "Safe fallback",
) -> BetDecision:
    """Create a safe SKIP BetDecision fallback."""
    return BetDecision(
        condition_id=condition_id,
        market_slug=market_slug,
        token_id="",
        side="YES",
        action="SKIP",
        estimated_prob_of_side=0.5,
        market_prob_of_side_at_entry=0.5,
        edge=0.0,
        entry_price=0.0,
        slippage_estimate=0.0,
        stake=0.0,
        underlier_group="other",
        rationale=rationale,
        exit_conditions=[],
        confidence="Low",
    )


def _safe_risk_estimate(
    condition_id: str = "unknown",
    warnings: list[str] | None = None,
) -> RiskEstimate:
    """Create a safe Low-confidence RiskEstimate fallback."""
    return RiskEstimate(
        condition_id=condition_id,
        recommended_side="YES",
        estimated_prob_of_side=0.5,
        confidence="Low",
        underlier_group="other",
        reasoning="Safe fallback — insufficient data",
        warnings=warnings or ["Safe fallback"],
    )


def _extract_cid_slug(step_input: StepInput) -> tuple[str, str]:
    """Safely extract condition_id and market_slug from Event Scan step."""
    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    if event:
        return event.condition_id, event.market_slug
    return "unknown", "unknown"


def _should_trade(
    risk: RiskAssessment,
    confidence_score: float,
    snapshot,  # BankrollSnapshot
    new_stake: float,
) -> tuple[bool, list[str]]:
    """Single source of truth for BET/SKIP decision.

    Returns (should_bet, skip_reasons).
    Aggregates risk.warnings so specific causes appear in rationale.
    """
    reasons: list[str] = []

    # From compute_edge_and_gate (encoded in risk_rating + warnings)
    if risk.risk_rating == "Unacceptable":
        reasons.append(f"Risk rating: {risk.risk_rating}")
        if risk.warnings:
            reasons.extend(risk.warnings)

    # Confidence gate
    if confidence_score < 0.5:
        reasons.append(f"Confidence score {confidence_score} below 0.5 minimum")

    # Portfolio gates (circuit breaker, max positions, capital at risk, reserve)
    portfolio_ok, portfolio_warnings = check_portfolio_limits(
        snapshot=snapshot,
        new_stake=new_stake,
    )
    if not portfolio_ok:
        reasons.extend(portfolio_warnings)

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Step: Event Scan (deterministic — no LLM)
# ---------------------------------------------------------------------------

_polymarket = PolymarketTools()


def _build_token_book(book_data: dict, token_id: str) -> dict:
    """Build TokenBook dict from CLOB orderbook response.

    IMPORTANT: Polymarket CLOB API does NOT guarantee sort order.
    bids may come lowest-first, asks may come highest-first.
    We must sort explicitly: bids descending, asks ascending.
    """
    raw_bids = book_data.get("bids", [])
    raw_asks = book_data.get("asks", [])

    # Sort: bids descending (highest first), asks ascending (lowest first)
    bids = sorted(raw_bids, key=lambda x: float(x["price"]), reverse=True)
    asks = sorted(raw_asks, key=lambda x: float(x["price"]))

    best_bid = float(bids[0]["price"]) if bids else 0.0
    best_ask = float(asks[0]["price"]) if asks else 0.0
    spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0.0

    # depth_10pct: sum of $ liquidity within 10% of midpoint
    midpoint = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
    depth = 0.0
    if midpoint > 0:
        lo = midpoint * 0.9
        hi = midpoint * 1.1
        for bid in bids:
            p = float(bid["price"])
            if p >= lo:
                depth += float(bid["size"]) * p
        for ask in asks:
            p = float(ask["price"])
            if p <= hi:
                depth += float(ask["size"]) * p

    return {
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": round(spread, 4),
        "depth_10pct": round(depth, 2),
    }


def run_event_scan(step_input: StepInput) -> StepOutput:
    """Deterministic event scan: fetch market data + orderbooks from Polymarket API.

    No LLM involved — all data comes directly from API.
    Reads condition_id from workflow input.
    """
    # Get condition_id from workflow input (step_input.input = PredictionRequest)
    wf_input = step_input.input
    condition_id = None
    if hasattr(wf_input, "condition_id"):
        condition_id = wf_input.condition_id
    elif isinstance(wf_input, dict):
        condition_id = wf_input.get("condition_id")

    if not condition_id:
        logger.error("Event Scan: no condition_id in workflow input")
        return StepOutput(content=None)

    try:
        # Find market in active markets
        markets = json.loads(_polymarket.get_active_crypto_markets(limit=100))
        market = None
        for m in markets:
            if m.get("condition_id") == condition_id:
                market = m
                break

        if not market:
            logger.error("Event Scan: market %s not found", condition_id)
            return StepOutput(content=None)

        token_ids = market.get("clob_token_ids", [])
        if len(token_ids) < 2:
            logger.error("Event Scan: market %s has < 2 token IDs", condition_id)
            return StepOutput(content=None)

        yes_token_id = token_ids[0]
        no_token_id = token_ids[1]

        # Fetch orderbooks
        yes_book_raw = json.loads(_polymarket.get_orderbook(yes_token_id))
        no_book_raw = json.loads(_polymarket.get_orderbook(no_token_id))

        yes_book = _build_token_book(yes_book_raw, yes_token_id)
        no_book = _build_token_book(no_book_raw, no_token_id)

        # Outcome prices
        prices = market.get("outcome_prices", [])
        market_prob_yes = float(prices[0]) if prices else 0.5

        event = EventCandidate(
            gamma_market_id=str(market.get("gamma_market_id", "")),
            condition_id=condition_id,
            market_slug=market.get("slug", ""),
            question=market.get("question", ""),
            category="crypto",
            end_date=market.get("end_date", ""),
            yes_book=yes_book,
            no_book=no_book,
            market_prob_yes=market_prob_yes,
            volume_24h=float(market.get("volume_24h", 0)),
            total_liquidity=yes_book["depth_10pct"] + no_book["depth_10pct"],
        )
        return StepOutput(content=event)

    except Exception as e:
        logger.error("Event Scan failed: %s", e)
        return StepOutput(content=None)


# ---------------------------------------------------------------------------
# Step: Market Data (deterministic — no LLM)
# ---------------------------------------------------------------------------


def run_market_data(step_input: StepInput) -> StepOutput:
    """Deterministic market data fetch. Thin wrapper over market_data_service."""
    from app.market_data_service import fetch_market_snapshot
    from storage.supported_assets import match_asset

    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    if not event:
        return StepOutput(content=None)

    asset = match_asset(event.question)
    if not asset:
        logger.warning("Unsupported asset for question: %s", event.question)
        return StepOutput(content=None)

    snapshot = fetch_market_snapshot(asset["coin_id"], asset["symbol"])
    return StepOutput(content=snapshot)


# ---------------------------------------------------------------------------
# Step: News & Sentiment (uses shared deterministic service)
# ---------------------------------------------------------------------------


def run_news_sentiment(step_input: StepInput) -> StepOutput:
    """Thin wrapper: build query from EventCandidate, call shared news service."""
    from app.news_service import fetch_sentiment

    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    if not event:
        return StepOutput(content=SentimentReport(
            query="fallback", sentiment_score=0.0,
            key_narratives=["No event data"], sources_count=0, confidence=0.1,
        ))

    query = f"{event.question} crypto prediction market"
    return StepOutput(content=fetch_sentiment(query))


# ---------------------------------------------------------------------------
# Step: Data Quality Check
# ---------------------------------------------------------------------------


def ensure_data_quality(step_input: StepInput) -> StepOutput:
    """Validate that required data from parallel agents is present and schema-valid."""
    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    market = _step_content_to_model(step_input.get_step_output("Market Data"), MarketSnapshot)
    sentiment = _step_content_to_model(step_input.get_step_output("News & Sentiment"), SentimentReport)

    result: dict = {
        "event_missing": event is None,
        "market_data_missing": market is None,
        "sentiment_missing": sentiment is None,
        "force_skip": False,
    }

    if event is None:
        result["force_skip"] = True
        result["skip_reason"] = "EventCandidate missing or invalid"
    elif market is None:
        result["force_skip"] = True
        result["skip_reason"] = "MarketSnapshot missing or invalid"

    if sentiment is None:
        result["sentiment_fallback"] = {
            "query": "fallback_no_search",
            "sentiment_score": 0.0,
            "key_narratives": ["No sentiment data — search timed out or failed"],
            "sources_count": 0,
            "confidence": 0.1,
        }

    return StepOutput(content=result)


# ---------------------------------------------------------------------------
# Step: Risk Assessment (fn-step wrapping agent)
# ---------------------------------------------------------------------------


def run_risk_assessment(step_input: StepInput) -> StepOutput:
    """Gather all data, call risk_agent, validate response as RiskEstimate."""
    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    dq = _step_content_to_dict(step_input.get_step_output("Data Quality"))
    market = _step_content_to_model(step_input.get_step_output("Market Data"), MarketSnapshot)
    sentiment = _step_content_to_model(step_input.get_step_output("News & Sentiment"), SentimentReport)

    # Sentiment fallback from Data Quality
    if not sentiment and dq and dq.get("sentiment_fallback"):
        sentiment = dq["sentiment_fallback"]

    event_d = event.model_dump(mode="json") if event else None
    market_d = market.model_dump(mode="json") if market else None
    sentiment_d = sentiment.model_dump(mode="json") if hasattr(sentiment, "model_dump") else sentiment

    # Force skip from data quality
    if dq and dq.get("force_skip"):
        return StepOutput(content=_safe_risk_estimate(
            condition_id=event_d.get("condition_id", "unknown") if event_d else "unknown",
            warnings=[dq.get("skip_reason", "Data quality failure")],
        ))

    # Missing event (double-check even if DQ didn't catch it)
    if not event_d:
        return StepOutput(content=_safe_risk_estimate(warnings=["EventCandidate not available"]))

    prompt = (
        f"Estimate the probability for this prediction market:\n\n"
        f"Event:\n{json.dumps(event_d, indent=2)}\n\n"
        f"Market Data:\n{json.dumps(market_d, indent=2) if market_d else 'Not available'}\n\n"
        f"Sentiment:\n{json.dumps(sentiment_d, indent=2) if sentiment_d else 'Not available'}"
    )

    try:
        response = risk_agent.run(prompt)
    except Exception as e:
        logger.error("Risk agent failed: %s", e)
        return StepOutput(content=_safe_risk_estimate(
            condition_id=event_d.get("condition_id", "unknown"),
            warnings=[f"Risk agent exception: {e}"],
        ))

    # Validate response type
    if isinstance(response.content, RiskEstimate):
        return StepOutput(content=response.content)
    try:
        d = response.content.model_dump(mode="json") if hasattr(response.content, "model_dump") else response.content
        return StepOutput(content=RiskEstimate.model_validate(d))
    except Exception:
        pass

    return StepOutput(content=_safe_risk_estimate(
        condition_id=event_d.get("condition_id", "unknown"),
        warnings=["Agent returned invalid response"],
    ))


# ---------------------------------------------------------------------------
# Step: Edge & Gate (deterministic — builds RiskAssessment from RiskEstimate)
# ---------------------------------------------------------------------------


def compute_edge_and_gate(step_input: StepInput) -> StepOutput:
    """Deterministic: compute edge, risk rating, liquidity, correlation.

    Reads RiskEstimate (LLM output) + EventCandidate (market data).
    Returns a full RiskAssessment for downstream steps.

    Codifies rules from mandate.md and risk_policy.md:
    - Edge >= 5%
    - Depth >= $10K, volume >= $50K, spread <= 5%
    - Correlated positions < 3
    """
    dq = _step_content_to_dict(step_input.get_step_output("Data Quality"))
    if dq and dq.get("force_skip"):
        return StepOutput(content=_safe_risk_assessment(
            condition_id="unknown",
            warnings=[dq.get("skip_reason", "Data quality failure")],
        ))

    estimate = _step_content_to_model(
        step_input.get_step_output("Risk Assessment"), RiskEstimate)
    event = _step_content_to_model(
        step_input.get_step_output("Event Scan"), EventCandidate)

    if not estimate or not event:
        return StepOutput(content=_safe_risk_assessment(
            warnings=["RiskEstimate or EventCandidate not available"],
        ))

    # Guard: condition_id mismatch between LLM and EventCandidate
    if estimate.condition_id != event.condition_id:
        return StepOutput(content=_safe_risk_assessment(
            condition_id=event.condition_id,
            warnings=[
                f"condition_id mismatch: estimate={estimate.condition_id}, "
                f"event={event.condition_id}",
            ],
        ))

    # 1. Market prob — midpoint implied probability from Polymarket, NOT LLM.
    #    Edge is analytical (midpoint). Execution price (best_ask + slippage)
    #    is computed separately in compute_position_sizing.
    if estimate.recommended_side == "YES":
        market_prob = event.market_prob_yes
        book = event.yes_book
    else:
        market_prob = 1.0 - event.market_prob_yes
        book = event.no_book

    # 2. Edge (code)
    edge = compute_edge(estimate.estimated_prob_of_side, market_prob)

    # 3. Composite liquidity check (code)
    liquidity_ok, liquidity_warnings = check_liquidity(
        depth_10pct=book.depth_10pct,
        volume_24h=event.volume_24h,
        spread=book.spread,
    )

    # 4. Correlated positions (code, from DB) — FAIL-CLOSED on error
    try:
        store = get_paper_trade_store()
        correlated = store.get_correlated_count(estimate.underlier_group)
    except Exception as e:
        logger.error("Cannot check correlated positions: %s", e)
        return StepOutput(content=_safe_risk_assessment(
            condition_id=event.condition_id,
            warnings=[f"DB error checking correlation: {e}"],
        ))

    # 5. Risk rating (code)
    risk_rating = determine_risk_rating(edge, liquidity_ok, correlated)

    # Collect all warnings: LLM warnings + code-generated warnings
    all_warnings = list(estimate.warnings) + liquidity_warnings
    if correlated >= 2:
        all_warnings.append(
            f"Correlated positions: {correlated}/3 in {estimate.underlier_group}")
    if edge < 0.05:
        all_warnings.append(f"Edge {edge:.1%} below 5% minimum")

    return StepOutput(content=RiskAssessment(
        condition_id=event.condition_id,
        risk_rating=risk_rating,
        recommended_side=estimate.recommended_side,
        estimated_prob_of_side=estimate.estimated_prob_of_side,
        market_prob_of_side=market_prob,
        edge=edge,
        underlier_group=estimate.underlier_group,
        warnings=all_warnings,
        liquidity_ok=liquidity_ok,
        correlated_positions=correlated,
    ))


# ---------------------------------------------------------------------------
# Step: Position Sizing (deterministic)
# ---------------------------------------------------------------------------


def compute_position_sizing(step_input: StepInput) -> StepOutput:
    """Deterministic sizing: Kelly, slippage, entry price."""
    dq = _step_content_to_dict(step_input.get_step_output("Data Quality"))
    if dq and dq.get("force_skip"):
        return StepOutput(content={"force_skip": True, "sizing_note": dq.get("skip_reason")})

    risk_model = _step_content_to_model(step_input.get_step_output("Edge & Gate"), RiskAssessment)
    event_model = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    risk = risk_model.model_dump(mode="json") if risk_model else None
    event = event_model.model_dump(mode="json") if event_model else None

    # --- Required field guards ---
    if risk is None:
        return StepOutput(content={"force_skip": True, "sizing_note": "Risk data missing or invalid"})
    if risk.get("estimated_prob_of_side") is None:
        return StepOutput(content={"force_skip": True, "sizing_note": "estimated_prob_of_side missing"})
    if risk.get("market_prob_of_side") is None:
        return StepOutput(content={"force_skip": True, "sizing_note": "market_prob_of_side missing"})
    recommended_side = risk.get("recommended_side")
    if recommended_side not in {"YES", "NO"}:
        return StepOutput(content={"force_skip": True, "sizing_note": f"Invalid recommended_side: {recommended_side}"})

    if event is None:
        return StepOutput(content={"force_skip": True, "sizing_note": "Event data missing or invalid"})
    side_key = "yes_book" if recommended_side == "YES" else "no_book"
    book = event.get(side_key) or {}
    best_ask = book.get("best_ask")
    depth = book.get("depth_10pct")
    if best_ask is None or best_ask <= 0:
        return StepOutput(content={"force_skip": True, "sizing_note": f"No valid best_ask in {side_key}"})
    if depth is None or depth <= 0:
        return StepOutput(content={"force_skip": True, "sizing_note": f"No depth in {side_key}"})

    # --- Bankroll --- FAIL-CLOSED on error
    try:
        store = get_paper_trade_store()
        snapshot = store.get_bankroll_snapshot()
        bankroll = snapshot.current_bankroll
    except Exception as e:
        logger.error("Cannot load bankroll for sizing: %s", e)
        return StepOutput(content={"force_skip": True, "sizing_note": f"Cannot load bankroll: {e}"})

    max_stake = bankroll * 0.20
    estimated_prob = risk["estimated_prob_of_side"]
    market_prob = risk["market_prob_of_side"]

    if estimated_prob <= market_prob:
        return StepOutput(content={"force_skip": True, "sizing_note": "No positive edge"})

    # --- Kelly ---
    raw_kelly = kelly_criterion(estimated_prob, market_prob)
    quarter_kelly = fractional_kelly(raw_kelly, 0.25)
    raw_stake = quarter_kelly * bankroll
    capped_stake = min(raw_stake, max_stake)

    min_stake = bankroll * 0.02
    if capped_stake < min_stake:
        return StepOutput(content={"force_skip": True, "sizing_note": f"Stake ${capped_stake:.2f} below minimum ${min_stake:.2f}"})

    # --- Slippage from real orderbook ---
    third = depth / 3.0
    asks = [(best_ask, third), (best_ask + 0.01, third), (best_ask + 0.02, third)]
    slippage = estimate_slippage(capped_stake, asks)
    entry_price = calculate_entry_price(best_ask, slippage)

    # Hard gate: slippage > 2%
    slippage_pct = slippage / best_ask if best_ask > 0 else 0.0
    if slippage_pct > 0.02:
        return StepOutput(content={"force_skip": True, "sizing_note": f"Slippage {slippage_pct:.1%} exceeds 2% budget"})

    return StepOutput(content={
        "force_skip": False,
        "recommended_side": recommended_side,
        "kelly_fraction_raw": round(raw_kelly, 4),
        "kelly_fraction_quarter": round(quarter_kelly, 4),
        "recommended_stake": round(capped_stake, 2),
        "entry_price": round(entry_price, 4),
        "slippage_estimate": round(slippage, 4),
        "bankroll": round(bankroll, 2),
    })


# ---------------------------------------------------------------------------
# Step: Decision (deterministic — replaces Decision Agent)
# ---------------------------------------------------------------------------


def build_decision(step_input: StepInput) -> StepOutput:
    """Deterministic: build BetDecision from risk + sizing + portfolio state.

    No LLM call. Uses _should_trade() as single decision point.
    """
    sizing = _step_content_to_dict(step_input.get_step_output("Position Sizing"))
    if not sizing or sizing.get("force_skip"):
        note = sizing.get("sizing_note", "Forced skip") if sizing else "No sizing data"
        cid, slug = _extract_cid_slug(step_input)
        return StepOutput(content=_safe_bet_decision(cid, slug, note))

    risk = _step_content_to_model(
        step_input.get_step_output("Edge & Gate"), RiskAssessment)
    event = _step_content_to_model(
        step_input.get_step_output("Event Scan"), EventCandidate)
    estimate = _step_content_to_model(
        step_input.get_step_output("Risk Assessment"), RiskEstimate)

    if not risk or not event:
        cid, slug = _extract_cid_slug(step_input)
        return StepOutput(content=_safe_bet_decision(cid, slug, "Missing risk or event data"))

    confidence_val = estimate.confidence if estimate else "Low"
    confidence_score = confidence_to_score(confidence_val)
    new_stake = sizing.get("recommended_stake", 0.0)

    # Get portfolio state — FAIL-CLOSED on error
    try:
        store = get_paper_trade_store()
        snapshot = store.get_bankroll_snapshot()
    except Exception as e:
        logger.error("Cannot load portfolio state: %s", e)
        cid, slug = _extract_cid_slug(step_input)
        return StepOutput(content=_safe_bet_decision(
            cid, slug, f"Cannot load portfolio state: {e}"))

    # Single decision point
    should_bet, skip_reasons = _should_trade(
        risk=risk,
        confidence_score=confidence_score,
        snapshot=snapshot,
        new_stake=new_stake,
    )
    action = "BET" if should_bet else "SKIP"

    side = risk.recommended_side
    book = event.yes_book if side == "YES" else event.no_book

    # Rationale (template, not LLM)
    reasoning_text = estimate.reasoning if estimate else ""
    if action == "BET":
        rationale = (
            f"Edge {risk.edge:.1%} on {side}. "
            f"Risk: {risk.risk_rating}. "
            f"Confidence: {confidence_val}. "
            f"{reasoning_text}"
        )
    else:
        rationale = (
            f"SKIP: {'; '.join(skip_reasons)}. "
            f"Edge {risk.edge:.1%} on {side}. "
            f"{reasoning_text}"
        )

    # Exit conditions (template, not LLM)
    exit_conditions = [
        "Exit if market probability moves against us by more than 15%",
        "Exit if new information fundamentally changes the thesis",
        "Hold until resolution if conditions remain stable",
    ] if action == "BET" else []

    return StepOutput(content=BetDecision(
        condition_id=risk.condition_id,
        market_slug=event.market_slug,
        token_id=book.token_id if action == "BET" else "",
        side=side,
        action=action,
        estimated_prob_of_side=risk.estimated_prob_of_side,
        market_prob_of_side_at_entry=risk.market_prob_of_side,
        edge=risk.edge,
        entry_price=sizing.get("entry_price", 0.0) if action == "BET" else 0.0,
        slippage_estimate=sizing.get("slippage_estimate", 0.0) if action == "BET" else 0.0,
        stake=new_stake if action == "BET" else 0.0,
        underlier_group=risk.underlier_group,
        rationale=rationale,
        exit_conditions=exit_conditions,
        confidence=confidence_val,
    ))


# ---------------------------------------------------------------------------
# Step: Record (conditional logging — sole DB writer)
# ---------------------------------------------------------------------------


def conditional_logging(step_input: StepInput) -> StepOutput:
    """Record paper trade in DB + write audit memo. Only for BET decisions."""
    # Hard backstop: force_skip from sizing overrides everything
    sizing = _step_content_to_dict(step_input.get_step_output("Position Sizing"))
    if sizing and sizing.get("force_skip"):
        return StepOutput(content={"action": "SKIP", "trade_id": None, "reason": sizing.get("sizing_note")})

    # Read typed BetDecision
    decision = _step_content_to_model(step_input.get_step_output("Decision"), BetDecision)
    if not decision:
        return StepOutput(content={"action": "SKIP", "trade_id": None, "reason": "No valid BetDecision"})

    if decision.action != "BET" or decision.stake <= 0:
        return StepOutput(content={"action": "SKIP", "trade_id": None, "reason": decision.rationale})

    # Get question from EventCandidate
    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    question = event.question if event else decision.market_slug

    # Record trade in DB (source of truth)
    try:
        store = get_paper_trade_store()
        trade = store.open_trade(decision, question)
    except Exception as e:
        logger.error("Failed to record paper trade: %s", e)
        return StepOutput(content={"action": "ERROR", "trade_id": None, "reason": f"DB write failed: {e}"})

    # Logger agent writes audit memo (best-effort, non-blocking)
    try:
        logger_agent.run(
            f"Write audit memo: Trade {trade.id}, {decision.side} ${decision.stake:.2f}, "
            f"edge {decision.edge:.2%}, {question}"
        )
    except Exception as e:
        logger.warning("Memo generation failed (trade recorded): %s", e)

    return StepOutput(content={
        "action": "BET",
        "trade_id": trade.id,
        "side": decision.side,
        "stake": decision.stake,
        "entry_price": decision.entry_price,
    })


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------

prediction_workflow = Workflow(
    id="prediction-workflow",
    name="Crypto Prediction Pipeline",
    steps=[
        Step(name="Event Scan", executor=run_event_scan),
        Parallel(
            Step(name="Market Data", executor=run_market_data),
            Step(name="News & Sentiment", executor=run_news_sentiment),
            name="Data Collection",
        ),
        Step(name="Data Quality", executor=ensure_data_quality),
        Step(name="Risk Assessment", executor=run_risk_assessment),   # LLM → RiskEstimate
        Step(name="Edge & Gate", executor=compute_edge_and_gate),     # code → RiskAssessment
        Step(name="Position Sizing", executor=compute_position_sizing),
        Step(name="Decision", executor=build_decision),               # code → BetDecision
        Step(name="Record", executor=conditional_logging),
    ],
)
