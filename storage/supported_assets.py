"""Supported assets for betting universe.

Maps question keywords → CoinGecko coin_id via word-boundary regex.
Only assets with full data coverage (CoinGecko + Binance + Exa news) are included.

This is the single source of truth for "which assets can we analyze."
If match_asset() returns None, the market is excluded from betting path.

v2: expanded from BTC/ETH/SOL to top-12 liquid crypto assets.
"""

from __future__ import annotations

import re

SUPPORTED_ASSETS = {
    # Tier 1 — highest liquidity, most data coverage
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
    # Tier 2 — large caps with good CoinGecko + Binance coverage
    "dogecoin": {
        "coin_id": "dogecoin",
        "symbol": "DOGE",
        "pattern": re.compile(r"\b(dogecoin|doge)\b", re.IGNORECASE),
    },
    "ripple": {
        "coin_id": "ripple",
        "symbol": "XRP",
        "pattern": re.compile(r"\b(xrp|ripple)\b", re.IGNORECASE),
    },
    "chainlink": {
        "coin_id": "chainlink",
        "symbol": "LINK",
        "pattern": re.compile(r"\b(chainlink|link)\b", re.IGNORECASE),
    },
    "cardano": {
        "coin_id": "cardano",
        "symbol": "ADA",
        "pattern": re.compile(r"\b(cardano|ada)\b", re.IGNORECASE),
    },
    "avalanche": {
        "coin_id": "avalanche-2",
        "symbol": "AVAX",
        "pattern": re.compile(r"\b(avalanche|avax)\b", re.IGNORECASE),
    },
    "bnb": {
        "coin_id": "binancecoin",
        "symbol": "BNB",
        "pattern": re.compile(r"\b(bnb|binance\s*coin)\b", re.IGNORECASE),
    },
    "toncoin": {
        "coin_id": "the-open-network",
        "symbol": "TON",
        "pattern": re.compile(r"\b(toncoin|ton)\b", re.IGNORECASE),
    },
    "polkadot": {
        "coin_id": "polkadot",
        "symbol": "DOT",
        "pattern": re.compile(r"\b(polkadot|dot)\b", re.IGNORECASE),
    },
    "litecoin": {
        "coin_id": "litecoin",
        "symbol": "LTC",
        "pattern": re.compile(r"\b(litecoin|ltc)\b", re.IGNORECASE),
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
