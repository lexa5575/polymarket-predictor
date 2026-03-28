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
from storage.supported_assets import match_asset
from tools.polymarket import PolymarketTools

logger = logging.getLogger(__name__)

_polymarket = PolymarketTools()


def _get_spread_and_depth(book_data: dict) -> tuple[float, float]:
    """Extract best spread and depth from raw orderbook.

    Sorts bids descending, asks ascending (CLOB doesn't guarantee order).
    Returns (spread, depth_10pct).
    """
    raw_bids = book_data.get("bids", [])
    raw_asks = book_data.get("asks", [])

    if not raw_bids or not raw_asks:
        return 1.0, 0.0  # worst case

    bids = sorted(raw_bids, key=lambda x: float(x["price"]), reverse=True)
    asks = sorted(raw_asks, key=lambda x: float(x["price"]))

    best_bid = float(bids[0]["price"])
    best_ask = float(asks[0]["price"])
    spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 1.0

    # depth_10pct
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

    return spread, depth


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

        yes_spread, yes_depth = _get_spread_and_depth(yes_book)
        no_spread, no_depth = _get_spread_and_depth(no_book)

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
