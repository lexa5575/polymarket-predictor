"""
Prediction Workflow — Structured Step-to-Step Handoff
-----------------------------------------------------

Pipeline: Event Scan → Data+News (parallel) → Data Quality → Risk → Sizing → Decision → Record

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
    decision_agent,
    logger_agent,
    market_data_agent,
    news_agent,
    polymarket_agent,
    risk_agent,
)
from agents.settings import get_paper_trade_store
from schemas.market import (
    BetDecision,
    EventCandidate,
    MarketSnapshot,
    RiskAssessment,
    SentimentReport,
)
from schemas.workflow_input import PredictionRequest
from storage.math_utils import (
    calculate_entry_price,
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
            d = json.loads(content)
            return model_class.model_validate(d)
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
    """Gather all data, call risk_agent with complete context, validate response."""
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
        return StepOutput(content=_safe_risk_assessment(
            condition_id=event_d.get("condition_id", "unknown") if event_d else "unknown",
            warnings=[dq.get("skip_reason", "Data quality failure")],
        ))

    # Missing event (double-check even if DQ didn't catch it)
    if not event_d:
        return StepOutput(content=_safe_risk_assessment(warnings=["EventCandidate not available"]))

    prompt = (
        f"Analyze this prediction market for risk:\n\n"
        f"Event:\n{json.dumps(event_d, indent=2)}\n\n"
        f"Market Data:\n{json.dumps(market_d, indent=2) if market_d else 'Not available'}\n\n"
        f"Sentiment:\n{json.dumps(sentiment_d, indent=2) if sentiment_d else 'Not available'}"
    )

    try:
        response = risk_agent.run(prompt)
    except Exception as e:
        logger.error("Risk agent failed: %s", e)
        return StepOutput(content=_safe_risk_assessment(
            condition_id=event_d.get("condition_id", "unknown"),
            warnings=[f"Risk agent exception: {e}"],
        ))

    # Validate response type
    if isinstance(response.content, RiskAssessment):
        return StepOutput(content=response.content)
    try:
        d = response.content.model_dump(mode="json") if hasattr(response.content, "model_dump") else response.content
        return StepOutput(content=RiskAssessment.model_validate(d))
    except Exception:
        pass

    return StepOutput(content=_safe_risk_assessment(
        condition_id=event_d.get("condition_id", "unknown"),
        warnings=["Agent returned invalid response"],
    ))


# ---------------------------------------------------------------------------
# Step: Position Sizing (deterministic)
# ---------------------------------------------------------------------------


def compute_position_sizing(step_input: StepInput) -> StepOutput:
    """Deterministic sizing: Kelly, slippage, entry price."""
    dq = _step_content_to_dict(step_input.get_step_output("Data Quality"))
    if dq and dq.get("force_skip"):
        return StepOutput(content={"force_skip": True, "sizing_note": dq.get("skip_reason")})

    risk_model = _step_content_to_model(step_input.get_step_output("Risk Assessment"), RiskAssessment)
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

    # --- Bankroll ---
    try:
        store = get_paper_trade_store()
        snapshot = store.get_bankroll_snapshot()
        bankroll = snapshot.current_bankroll
    except Exception:
        bankroll = 10_000.0

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
# Step: Decision (fn-step wrapping agent)
# ---------------------------------------------------------------------------


def run_decision(step_input: StepInput) -> StepOutput:
    """Gather ALL data, call decision_agent with complete context, validate response."""
    event = _step_content_to_model(step_input.get_step_output("Event Scan"), EventCandidate)
    risk = _step_content_to_model(step_input.get_step_output("Risk Assessment"), RiskAssessment)
    sizing = _step_content_to_dict(step_input.get_step_output("Position Sizing"))
    market = _step_content_to_model(step_input.get_step_output("Market Data"), MarketSnapshot)
    sentiment = _step_content_to_model(step_input.get_step_output("News & Sentiment"), SentimentReport)

    dq = _step_content_to_dict(step_input.get_step_output("Data Quality"))
    if not sentiment and dq and dq.get("sentiment_fallback"):
        sentiment = dq["sentiment_fallback"]

    event_d = event.model_dump(mode="json") if event else None
    risk_d = risk.model_dump(mode="json") if risk else None
    market_d = market.model_dump(mode="json") if market else None
    sentiment_d = sentiment.model_dump(mode="json") if hasattr(sentiment, "model_dump") else sentiment

    cid = event_d.get("condition_id", "unknown") if event_d else "unknown"
    slug = event_d.get("market_slug", "unknown") if event_d else "unknown"

    # Force skip from sizing
    if sizing and sizing.get("force_skip"):
        return StepOutput(content=_safe_bet_decision(cid, slug, sizing.get("sizing_note", "Forced skip")))

    # Missing event
    if not event_d:
        return StepOutput(content=_safe_bet_decision(rationale="EventCandidate not available"))

    prompt = (
        f"Make a final BET/SKIP decision:\n\n"
        f"Event:\n{json.dumps(event_d, indent=2)}\n\n"
        f"Market Data:\n{json.dumps(market_d, indent=2) if market_d else 'N/A'}\n\n"
        f"Sentiment:\n{json.dumps(sentiment_d, indent=2) if sentiment_d else 'N/A'}\n\n"
        f"Risk Assessment:\n{json.dumps(risk_d, indent=2) if risk_d else 'N/A'}\n\n"
        f"Position Sizing:\n{json.dumps(sizing, indent=2) if sizing else 'N/A'}"
    )

    try:
        response = decision_agent.run(prompt)
    except Exception as e:
        logger.error("Decision agent failed: %s", e)
        return StepOutput(content=_safe_bet_decision(cid, slug, f"Decision agent exception: {e}"))

    # Validate response type
    if isinstance(response.content, BetDecision):
        return StepOutput(content=response.content)
    try:
        d = response.content.model_dump(mode="json") if hasattr(response.content, "model_dump") else response.content
        return StepOutput(content=BetDecision.model_validate(d))
    except Exception:
        pass

    return StepOutput(content=_safe_bet_decision(cid, slug, "Agent returned invalid response"))


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
        Step(name="Event Scan", agent=polymarket_agent),
        Parallel(
            Step(name="Market Data", agent=market_data_agent),  # type: ignore[arg-type]
            Step(name="News & Sentiment", agent=news_agent),  # type: ignore[arg-type]
            name="Data Collection",
        ),
        Step(name="Data Quality", executor=ensure_data_quality),
        Step(name="Risk Assessment", executor=run_risk_assessment),
        Step(name="Position Sizing", executor=compute_position_sizing),
        Step(name="Decision", executor=run_decision),
        Step(name="Record", executor=conditional_logging),
    ],
)
