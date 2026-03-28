"""
Deterministic Candidate Scanner
--------------------------------

No LLM. Fetches markets from Polymarket, filters by supported assets
and liquidity, sorts by tradability score.

Single source of candidates for scan-and-fanout route.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from storage.math_utils import check_liquidity
from storage.orderbook_utils import parse_orderbook
from storage.supported_assets import match_asset
from tools.polymarket import PolymarketTools

logger = logging.getLogger(__name__)

# Strategy-fit horizon: short-term markets where price moves fast
MIN_DAYS_TO_RESOLUTION = 1    # skip already-expired markets
MAX_DAYS_TO_RESOLUTION = 14   # near-term markets with active price movement

_polymarket = PolymarketTools()


def scan_candidates(max_candidates: int = 20) -> list[dict]:
    """Deterministic candidate selection. No LLM.

    1. Fetch crypto-prices markets from Polymarket Gamma API
    2. Filter: only supported assets (match_asset with word-boundary regex)
    3. Fetch orderbook for each → compute spread, depth
    4. Prefilter: reuse check_liquidity() (shared thresholds)
    5. Sort by tradability score
    6. Return top N as minimal dicts: {condition_id, question}

    Returns list of dicts. coin_id/symbol NOT included —
    workflow re-derives them via match_asset() (single source of truth).
    """
    try:
        markets = json.loads(_polymarket.get_active_crypto_markets(limit=100))
    except Exception as e:
        logger.error("Failed to fetch markets: %s", e)
        return []

    # Funnel stats for debugging
    stats = {
        "total_fetched": len(markets),
        "passed_asset": 0,
        "passed_horizon": 0,
        "passed_tokens": 0,
        "passed_liquidity": 0,
        "final_candidates": 0,
    }
    candidates = []

    for m in markets:
        question = m.get("question", "")
        condition_id = m.get("condition_id", "")

        # 1. Supported asset filter
        asset = match_asset(question)
        if not asset:
            continue
        stats["passed_asset"] += 1

        # 2. Horizon filter — skip near-settled and ultra-long markets
        end_date_str = m.get("end_date", "")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days_left = (end_date - now).days
                if days_left < MIN_DAYS_TO_RESOLUTION:
                    continue
                if days_left > MAX_DAYS_TO_RESOLUTION:
                    continue
            except (ValueError, TypeError):
                pass  # can't parse date → don't filter
        stats["passed_horizon"] += 1

        # 3. Must have 2 token IDs
        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) < 2:
            continue
        stats["passed_tokens"] += 1

        volume_24h = float(m.get("volume_24h", 0))

        # 3. Fetch orderbooks and compute market-level liquidity
        try:
            yes_book = json.loads(_polymarket.get_orderbook(token_ids[0]))
            no_book = json.loads(_polymarket.get_orderbook(token_ids[1]))
        except Exception as e:
            logger.warning("Orderbook fetch failed for %s: %s", condition_id, e)
            continue

        yes_parsed = parse_orderbook(yes_book)
        no_parsed = parse_orderbook(no_book)
        yes_spread, yes_depth = yes_parsed["spread"], yes_parsed["depth_10pct"]
        no_spread, no_depth = no_parsed["spread"], no_parsed["depth_10pct"]

        # Market-level metrics (before side selection)
        # Spread: worst case (max) — conservative
        # Depth: best side (max) — on extreme markets (prob <5% or >95%)
        #   one side always has near-zero depth, but the tradeable side is deep.
        #   NOTE: this means a candidate can pass scanner but fail side-specific
        #   depth check in compute_position_sizing. This is acceptable because:
        #   1) Risk Agent may choose the deep side (NO) where edge exists
        #   2) If Risk Agent picks the shallow side, sizing correctly force_skips
        #   3) Without max(), ALL extreme markets would be filtered (100% of them)
        #   The alternative (min) was tested and rejected: it filtered everything.
        market_spread = max(yes_spread, no_spread)
        market_depth = max(yes_depth, no_depth)

        # 4. Shared liquidity check (same thresholds as Edge & Gate)
        liquidity_ok, warnings = check_liquidity(market_depth, volume_24h, market_spread)
        if not liquidity_ok:
            logger.debug("Filtered %s: %s", question[:40], "; ".join(warnings))
            continue
        stats["passed_liquidity"] += 1

        # 5. Tradability score
        score = volume_24h * market_depth / (1 + market_spread * 100)

        candidates.append({
            "condition_id": condition_id,
            "gamma_market_id": str(m.get("gamma_market_id", "")),
            "question": question,
            "score": score,
            "volume_24h": volume_24h,
            "market_spread": round(market_spread, 4),
            "market_depth": round(market_depth, 2),
        })

    # Sort by tradability score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    result = candidates[:max_candidates]
    stats["final_candidates"] = len(result)
    logger.info("Scanner funnel: %s", stats)

    return result
