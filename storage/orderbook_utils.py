"""
Shared orderbook parsing utilities.

Single source of truth for extracting spread, depth, and best prices
from Polymarket CLOB API orderbook responses.

IMPORTANT: Polymarket CLOB does NOT guarantee sort order.
Bids may come lowest-first, asks may come highest-first.
All functions sort explicitly before extracting.
"""

from __future__ import annotations


def parse_orderbook(book_data: dict) -> dict:
    """Parse raw CLOB orderbook into normalized format.

    Returns dict with:
    - best_bid, best_ask, spread, depth_10pct
    - sorted_bids (descending), sorted_asks (ascending)

    Handles unsorted CLOB data safely.
    """
    raw_bids = book_data.get("bids", [])
    raw_asks = book_data.get("asks", [])

    bids = sorted(raw_bids, key=lambda x: float(x["price"]), reverse=True)
    asks = sorted(raw_asks, key=lambda x: float(x["price"]))

    best_bid = float(bids[0]["price"]) if bids else 0.0
    best_ask = float(asks[0]["price"]) if asks else 0.0
    spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else (1.0 if not bids or not asks else 0.0)

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
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": round(spread, 4),
        "depth_10pct": round(depth, 2),
    }


def build_token_book(book_data: dict, token_id: str) -> dict:
    """Build TokenBook dict from CLOB orderbook response.

    Returns dict compatible with schemas.market.TokenBook.
    """
    parsed = parse_orderbook(book_data)
    return {
        "token_id": token_id,
        "best_bid": parsed["best_bid"],
        "best_ask": parsed["best_ask"],
        "spread": parsed["spread"],
        "depth_10pct": parsed["depth_10pct"],
    }
