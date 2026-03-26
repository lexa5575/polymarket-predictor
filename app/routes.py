"""
Custom API Routes
-----------------

Additional endpoints beyond AgentOS defaults:
- POST /api/scan-and-fanout — batch scan + fan-out workflow runs
- POST /api/settle — check resolved markets, update paper trades
- GET  /api/dashboard — current bankroll snapshot
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter

from agents.settings import get_paper_trade_store
from schemas.workflow_input import PredictionRequest
from tools.polymarket import PolymarketTools

logger = logging.getLogger(__name__)
router = APIRouter()

_polymarket_tools = PolymarketTools()

_RECORD_TAG = "RECORD_RESULT"


def _get_record_result(wf_content: str) -> dict:
    """Extract the structured RECORD_RESULT block from workflow output."""
    pattern = rf"<!-- {_RECORD_TAG} -->(\{{.*?\}})<!-- /{_RECORD_TAG} -->"
    match = re.search(pattern, wf_content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
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
                    # Extract structured result from RECORD_RESULT tag
                    wf_content = str(wf_result.content) if wf_result and wf_result.content else ""
                    record = _get_record_result(wf_content)

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
