"""
Prediction Workflow
-------------------

Deterministic pipeline for evaluating a single prediction market:
Event Scan → Market Data + News (parallel) → Risk → Sizing → Decision → Record

Each function step emits a tagged JSON block (<!-- STEP_NAME_DATA --> ... <!-- /STEP_NAME_DATA -->)
so downstream steps parse only their intended input, not the full accumulated text.
This avoids regex collisions like "side" matching inside "recommended_side".

Function steps:
- compute_position_sizing: deterministic Kelly/slippage/stake from real data
- conditional_logging: gates DB write + memo on BET vs SKIP (sole DB writer)
"""

from __future__ import annotations

import json
import logging
import re

from agno.workflow import Parallel, Step, Workflow

from agents import (
    decision_agent,
    logger_agent,
    market_data_agent,
    news_agent,
    polymarket_agent,
    risk_agent,
)
from agents.settings import get_paper_trade_store
from schemas.market import BetDecision
from schemas.workflow_input import PredictionRequest
from storage.math_utils import (
    calculate_entry_price,
    estimate_slippage,
    fractional_kelly,
    kelly_criterion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tagged JSON block protocol
# ---------------------------------------------------------------------------
# Each function step writes:
#   <!-- TAG_DATA -->{"key": "value", ...}<!-- /TAG_DATA -->
# Downstream steps extract only the block they need via _get_tagged_block().

_SIZING_TAG = "SIZING_DATA"
_RECORD_TAG = "RECORD_RESULT"


def _emit_tagged_block(tag: str, data: dict) -> str:
    """Wrap a dict as a tagged JSON block in the workflow context."""
    return f"<!-- {tag} -->{json.dumps(data)}<!-- /{tag} -->"


def _get_tagged_block(context: str, tag: str) -> dict | None:
    """Extract a tagged JSON block from workflow context. Returns None if not found."""
    pattern = rf"<!-- {tag} -->(\{{.*?\}})<!-- /{tag} -->"
    match = re.search(pattern, context, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _find_json_block(context: str) -> dict | None:
    """Find the last well-formed JSON object in the context (fallback for agent output).

    Agents with output_schema produce JSON. We search backwards for the last
    complete JSON object, which is typically the most recent agent's output.
    """
    # Find all JSON-like blocks (starting with { and ending with })
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", context)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_agent_json(context: str, key_marker: str) -> dict | None:
    """Extract a JSON block from context that contains a specific key.

    This is more reliable than regex on individual fields because it finds
    the complete JSON object that owns the key.
    """
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", context):
        try:
            obj = json.loads(match.group())
            if key_marker in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Function step: Position Sizing
# ---------------------------------------------------------------------------


def compute_position_sizing(context: str, **kwargs) -> str:  # noqa: ARG001
    """Deterministic sizing — computes Kelly, slippage, stake, entry price.

    Reads structured JSON from prior agent outputs:
    - RiskAssessment JSON (contains estimated_prob_of_side, market_prob_of_side, recommended_side)
    - EventCandidate JSON (contains yes_book, no_book with best_ask, depth_10pct)
    """
    # --- Extract RiskAssessment from agent output ---
    risk_data = _extract_agent_json(context, "estimated_prob_of_side")
    event_data = _extract_agent_json(context, "yes_book") or _extract_agent_json(context, "gamma_market_id")

    estimated_prob = risk_data.get("estimated_prob_of_side") if risk_data else None
    market_prob = risk_data.get("market_prob_of_side") if risk_data else None
    recommended_side = risk_data.get("recommended_side", "YES") if risk_data else "YES"

    # --- Get real bankroll from store ---
    try:
        store = get_paper_trade_store()
        snapshot = store.get_bankroll_snapshot()
        bankroll = snapshot.current_bankroll
    except Exception as e:
        logger.warning("Could not get bankroll snapshot, using default: %s", e)
        bankroll = 10_000.0

    max_stake = bankroll * 0.20  # 20% cap from mandate

    # --- Default sizing (safe fallback) ---
    sizing = {
        "bankroll": round(bankroll, 2),
        "max_allowed_stake": round(max_stake, 2),
        "recommended_side": recommended_side,
        "kelly_fraction_raw": 0.0,
        "kelly_fraction_quarter": 0.0,
        "recommended_stake": 0.0,
        "entry_price": 0.0,
        "slippage_estimate": 0.0,
        "force_skip": False,
        "sizing_method": "fallback",
        "sizing_note": "",
    }

    if not (estimated_prob and market_prob and estimated_prob > market_prob):
        sizing["force_skip"] = True
        sizing["sizing_note"] = "No positive edge detected or data missing"
        return _emit_tagged_block(_SIZING_TAG, sizing)

    # --- Kelly Criterion ---
    raw_kelly = kelly_criterion(estimated_prob, market_prob)
    quarter_kelly = fractional_kelly(raw_kelly, 0.25)
    raw_stake = quarter_kelly * bankroll
    capped_stake = min(raw_stake, max_stake)

    # Enforce minimum bet of 2% bankroll
    min_stake = bankroll * 0.02
    if capped_stake < min_stake:
        sizing["force_skip"] = True
        sizing["sizing_note"] = f"Stake ${capped_stake:.2f} below minimum ${min_stake:.2f}"
        return _emit_tagged_block(_SIZING_TAG, sizing)

    # --- Extract orderbook for the recommended side ---
    best_ask = None
    depth = None
    if event_data:
        side_key = f"{'yes' if recommended_side == 'YES' else 'no'}_book"
        book = event_data.get(side_key, {})
        if isinstance(book, dict):
            best_ask = book.get("best_ask")
            depth = book.get("depth_10pct")

    # Build approximate orderbook for slippage estimation
    # HARD RULE: if no valid orderbook for the recommended side, force skip.
    # Never synthesize a fake orderbook — that hides broken handoffs.
    if best_ask and depth and depth > 0:
        third = depth / 3.0
        asks = [
            (best_ask, third),
            (best_ask + 0.01, third),
            (best_ask + 0.02, third),
        ]
    else:
        sizing["force_skip"] = True
        sizing["sizing_note"] = (
            f"No valid orderbook for {recommended_side} side — "
            "cannot compute entry price or slippage safely"
        )
        return _emit_tagged_block(_SIZING_TAG, sizing)

    slippage = estimate_slippage(capped_stake, asks)
    entry_price = calculate_entry_price(best_ask, slippage)

    # --- HARD GATE: slippage budget (max 2% of best_ask) ---
    slippage_pct = (slippage / best_ask) if best_ask > 0 else 0.0
    if slippage_pct > 0.02:
        sizing["force_skip"] = True
        sizing["sizing_note"] = (
            f"Slippage {slippage_pct:.1%} exceeds 2% budget. "
            f"Stake zeroed — mandate violation."
        )
        sizing["slippage_estimate"] = round(slippage, 4)
        return _emit_tagged_block(_SIZING_TAG, sizing)

    sizing.update({
        "kelly_fraction_raw": round(raw_kelly, 4),
        "kelly_fraction_quarter": round(quarter_kelly, 4),
        "recommended_stake": round(capped_stake, 2),
        "entry_price": round(entry_price, 4),
        "slippage_estimate": round(slippage, 4),
        "sizing_method": "kelly_0.25x",
    })

    return _emit_tagged_block(_SIZING_TAG, sizing)


# ---------------------------------------------------------------------------
# Function step: Conditional Logging (sole DB writer)
# ---------------------------------------------------------------------------


def conditional_logging(context: str, **kwargs) -> str:  # noqa: ARG001
    """Conditional gate: records paper trade in DB + writes audit memo for BET.

    This is the SOLE owner of DB writes for paper trades.
    For SKIP decisions, only a trace event is recorded.

    Returns a tagged JSON block with {action, trade_id, status} for route-level validation.
    """
    # --- Check if sizing forced a skip ---
    sizing = _get_tagged_block(context, _SIZING_TAG)
    if sizing and sizing.get("force_skip"):
        result = {"action": "SKIP", "trade_id": None, "reason": sizing.get("sizing_note", "Sizing forced skip")}
        logger.info("Sizing forced SKIP: %s", result["reason"])
        return _emit_tagged_block(_RECORD_TAG, result)

    # --- Extract Decision Agent output (JSON with "action" key) ---
    decision_data = _extract_agent_json(context, "action")
    if not decision_data:
        result = {"action": "SKIP", "trade_id": None, "reason": "Could not parse Decision Agent output"}
        logger.warning(result["reason"])
        return _emit_tagged_block(_RECORD_TAG, result)

    action = (decision_data.get("action") or "").upper()

    if action == "SKIP":
        result = {"action": "SKIP", "trade_id": None, "reason": decision_data.get("rationale", "Agent chose SKIP")}
        logger.info("Decision Agent chose SKIP.")
        return _emit_tagged_block(_RECORD_TAG, result)

    if action != "BET":
        result = {"action": "SKIP", "trade_id": None, "reason": f"Unknown action: {action}"}
        logger.warning("Unknown action '%s' — treating as SKIP.", action)
        return _emit_tagged_block(_RECORD_TAG, result)

    # --- Build BetDecision from the Decision Agent's JSON output ---
    # Also pull sizing data for entry_price/slippage/stake if the agent didn't include them.
    # IMPORTANT: use `is None` checks, not `or`, to preserve explicit zeros from the agent.
    # If the agent says stake=0, that means "don't bet" — don't override with sizing.
    sizing = sizing or {}
    try:
        # Stake: agent value takes precedence (even if 0). Only fall back to sizing if key absent.
        agent_stake = decision_data.get("stake")
        stake = agent_stake if agent_stake is not None else (sizing.get("recommended_stake") or 0.0)

        # Entry price / slippage: same None-aware fallback
        agent_entry = decision_data.get("entry_price")
        entry_price = agent_entry if agent_entry is not None else (sizing.get("entry_price") or 0.5)

        agent_slippage = decision_data.get("slippage_estimate")
        slippage_est = agent_slippage if agent_slippage is not None else (sizing.get("slippage_estimate") or 0.01)

        decision = BetDecision(
            condition_id=decision_data.get("condition_id", "unknown"),
            market_slug=decision_data.get("market_slug", "unknown"),
            token_id=decision_data.get("token_id", "unknown"),
            side=decision_data.get("side", "YES"),
            action="BET",
            estimated_prob_of_side=decision_data.get("estimated_prob_of_side", 0.5),
            market_prob_of_side_at_entry=decision_data.get("market_prob_of_side_at_entry", 0.5),
            edge=decision_data.get("edge", 0.0),
            entry_price=entry_price,
            slippage_estimate=slippage_est,
            stake=stake,
            underlier_group=decision_data.get("underlier_group", "other"),
            rationale=decision_data.get("rationale", "See workflow trace."),
            exit_conditions=decision_data.get("exit_conditions", []),
            confidence=decision_data.get("confidence", "Medium"),
        )
    except Exception as e:
        result = {"action": "ERROR", "trade_id": None, "reason": f"Failed to build BetDecision: {e}"}
        logger.error(result["reason"])
        return _emit_tagged_block(_RECORD_TAG, result)

    if decision.stake <= 0:
        result = {"action": "SKIP", "trade_id": None, "reason": "BET with zero stake — treated as SKIP"}
        logger.warning(result["reason"])
        return _emit_tagged_block(_RECORD_TAG, result)

    # --- 1. Record paper trade in DB (source of truth) ---
    # Extract question from EventCandidate JSON
    event_data = _extract_agent_json(context, "question")
    question = (event_data.get("question") if event_data else None) or decision.market_slug

    try:
        store = get_paper_trade_store()
        trade = store.open_trade(decision, question)
        trade_id = trade.id
        logger.info("Paper trade recorded: %s (side=%s, stake=$%.2f)", trade_id, decision.side, decision.stake)
    except Exception as e:
        result = {"action": "ERROR", "trade_id": None, "reason": f"DB write failed: {e}"}
        logger.error(result["reason"])
        return _emit_tagged_block(_RECORD_TAG, result)

    # --- 2. Logger agent writes audit memo ---
    memo_status = "skipped"
    try:
        response = logger_agent.run(
            f"Write an audit memo for this BET decision.\n"
            f"Trade ID: {trade_id}\n"
            f"Question: {question}\n"
            f"Side: {decision.side}\n"
            f"Stake: ${decision.stake:.2f}\n"
            f"Entry price: {decision.entry_price:.4f}\n"
            f"Edge: {decision.edge:.2%}\n"
            f"Confidence: {decision.confidence}\n"
            f"Underlier group: {decision.underlier_group}\n"
        )
        memo_status = "written" if response else "failed"
    except Exception as e:
        logger.warning("Memo generation failed (trade still recorded): %s", e)
        memo_status = f"error: {e}"

    result = {
        "action": "BET",
        "trade_id": trade_id,
        "side": decision.side,
        "stake": decision.stake,
        "entry_price": decision.entry_price,
        "memo_status": memo_status,
    }
    return _emit_tagged_block(_RECORD_TAG, result)


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------

prediction_workflow = Workflow(
    id="prediction-workflow",
    name="Crypto Prediction Pipeline",
    input_schema=PredictionRequest,
    steps=[
        Step(name="Event Scan", agent=polymarket_agent),
        Parallel(
            Step(name="Market Data", agent=market_data_agent),  # type: ignore[arg-type]
            Step(name="News & Sentiment", agent=news_agent),  # type: ignore[arg-type]
            name="Data Collection",
        ),
        Step(name="Risk Assessment", agent=risk_agent),
        Step(name="Position Sizing", function=compute_position_sizing),
        Step(name="Decision", agent=decision_agent),
        Step(name="Record", function=conditional_logging),
    ],
)
