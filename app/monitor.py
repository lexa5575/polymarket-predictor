"""
Position Monitor Service
------------------------

Pure function `run_monitor()` that checks all open trades and closes
positions based on exit conditions. Called by:
- POST /api/monitor (manual trigger)
- Agno scheduler (cron: */1 * * * *)

Two exit lifecycles:
- Market resolution → resolve_trade() (binary PnL + brier_score)
- Early exit (TP/SL/max_hold) → close_trade() (mark-to-market PnL)

Also records resolution for previously closed trades (what-if analytics).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from agents.settings import get_paper_trade_store
from storage.exit_policy import MAX_HOLD_SECONDS, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from storage.math_utils import calculate_mtm_pnl, get_exit_price_from_orderbook
from tools.polymarket import PolymarketTools

logger = logging.getLogger(__name__)

_polymarket_tools = PolymarketTools()


def run_monitor() -> dict:
    """Check all open trades, close by exit conditions.

    Resolution is checked BEFORE orderbook (resolution doesn't need best_bid).
    Per-trade try/except ensures one API failure doesn't kill the whole pass.

    Returns {checked, closed, trades_closed, trades_open}.
    """
    store = get_paper_trade_store()
    open_trades = store.get_open_trades()
    now = datetime.now(timezone.utc)

    closed_results: list[dict] = []
    open_status: list[dict] = []

    for trade in open_trades:
        try:
            created = trade.created_at.replace(tzinfo=timezone.utc) if trade.created_at.tzinfo is None else trade.created_at
            age = (now - created).total_seconds()

            # Safe fallback for legacy trades without policy snapshot
            tp = trade.take_profit_pct if trade.take_profit_pct is not None else TAKE_PROFIT_PCT
            sl = trade.stop_loss_pct if trade.stop_loss_pct is not None else STOP_LOSS_PCT
            mh = trade.max_hold_seconds if trade.max_hold_seconds is not None else MAX_HOLD_SECONDS

            # === RESOLUTION CHECK FIRST (before orderbook) ===
            res = json.loads(_polymarket_tools.get_market_resolution(trade.condition_id))
            if res.get("status") == "resolved":
                outcome = res.get("final_outcome")
                if outcome in ("YES", "NO"):
                    result = store.resolve_trade(trade.id, outcome)
                    closed_results.append({
                        "trade_id": trade.id,
                        "reason": "resolution",
                        "pnl": result.pnl,
                        "outcome": outcome,
                    })
                    continue
                # resolved but ambiguous outcome → skip, log warning
                logger.warning(
                    "Market %s resolved but outcome=%s, skipping",
                    trade.condition_id, outcome,
                )

            # === ORDERBOOK ONLY FOR EARLY EXIT ===
            book = json.loads(_polymarket_tools.get_orderbook(trade.token_id))
            exit_price = get_exit_price_from_orderbook(book)
            if exit_price is None:
                logger.warning("No bids for %s (%s), skipping", trade.token_id, trade.question)
                open_status.append({"trade_id": trade.id, "warning": "no bids"})
                continue

            unrealized_pnl = calculate_mtm_pnl(trade.entry_fill_price, exit_price, trade.stake)
            pnl_pct = unrealized_pnl / trade.stake if trade.stake > 0 else 0.0

            # Check early exit conditions (resolution already handled above)
            should_exit = False
            reason = None
            if pnl_pct <= sl:
                should_exit, reason = True, "stop_loss"
            elif pnl_pct >= tp:
                should_exit, reason = True, "take_profit"
            elif age >= mh:
                should_exit, reason = True, "max_hold"

            if should_exit:
                result = store.close_trade(trade.id, exit_price, reason)
                closed_results.append({
                    "trade_id": trade.id,
                    "reason": reason,
                    "pnl": result.pnl,
                    "exit_price": exit_price,
                })
            else:
                open_status.append({
                    "trade_id": trade.id,
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(pnl_pct, 4),
                    "current_price": exit_price,
                    "age_minutes": round(age / 60, 1),
                })

        except Exception as e:
            logger.error("Monitor error for trade %s: %s", trade.id, e)
            open_status.append({"trade_id": trade.id, "error": str(e)})

    # === BACKGROUND: record resolution for previously closed trades ===
    for trade in store.get_closed_without_resolution():
        try:
            res = json.loads(_polymarket_tools.get_market_resolution(trade.condition_id))
            if res.get("status") == "resolved":
                outcome = res.get("final_outcome")
                if outcome in ("YES", "NO"):
                    store.record_resolution(trade.id, outcome)
        except Exception as e:
            logger.warning("Failed to check resolution for closed trade %s: %s", trade.id, e)

    return {
        "checked": len(open_trades),
        "closed": len(closed_results),
        "trades_closed": closed_results,
        "trades_open": open_status,
    }
