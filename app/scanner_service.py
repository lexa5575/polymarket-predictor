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

from storage.math_utils import check_liquidity
from storage.orderbook_utils import parse_orderbook
from storage.supported_assets import match_asset
from tools.polymarket import PolymarketTools

logger = logging.getLogger(__name__)

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

    candidates = []

    for m in markets:
        question = m.get("question", "")
        condition_id = m.get("condition_id", "")

        # 1. Supported asset filter
        asset = match_asset(question)
        if not asset:
            continue

        # 2. Must have 2 token IDs
        token_ids = m.get("clob_token_ids", [])
        if len(token_ids) < 2:
            continue

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

        # Conservative market-level metrics (before side selection)
        market_spread = max(yes_spread, no_spread)
        market_depth = min(yes_depth, no_depth)

        # 4. Shared liquidity check (same thresholds as Edge & Gate)
        liquidity_ok, warnings = check_liquidity(market_depth, volume_24h, market_spread)
        if not liquidity_ok:
            logger.debug("Filtered %s: %s", question[:40], "; ".join(warnings))
            continue

        # 5. Tradability score
        score = volume_24h * market_depth / (1 + market_spread * 100)

        candidates.append({
            "condition_id": condition_id,
            "question": question,
            "score": score,
            "volume_24h": volume_24h,
            "market_spread": round(market_spread, 4),
            "market_depth": round(market_depth, 2),
        })

    # Sort by tradability score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)

    return candidates[:max_candidates]
