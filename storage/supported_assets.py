"""Supported assets for betting universe.

Maps question keywords → CoinGecko coin_id via word-boundary regex.
Only assets with full data coverage (CoinGecko + Binance + Exa news) are included.

This is the single source of truth for "which assets can we analyze."
If match_asset() returns None, the market is excluded from betting path.
"""

from __future__ import annotations

import re

SUPPORTED_ASSETS = {
    "bitcoin": {
        "coin_id": "bitcoin",
        "symbol": "BTC",
        "pattern": re.compile(r"\b(bitcoin|btc)\b", re.IGNORECASE),
    },
    "ethereum": {
        "coin_id": "ethereum",
        "symbol": "ETH",
        "pattern": re.compile(r"\b(ethereum|eth)\b", re.IGNORECASE),
    },
    "solana": {
        "coin_id": "solana",
        "symbol": "SOL",
        "pattern": re.compile(r"\b(solana|sol)\b", re.IGNORECASE),
    },
}


def match_asset(question: str) -> dict | None:
    """Match a market question to a supported asset.

    Uses word-boundary regex to avoid false positives:
    - "Netherlands" does NOT match "eth"
    - "resolution" does NOT match "sol"

    Returns {"coin_id": ..., "symbol": ...} or None if unsupported.
    """
    for asset in SUPPORTED_ASSETS.values():
        if asset["pattern"].search(question):
            return {"coin_id": asset["coin_id"], "symbol": asset["symbol"]}
    return None
