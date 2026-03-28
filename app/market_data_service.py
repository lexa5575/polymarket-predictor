"""
Deterministic Market Data Service
----------------------------------

Thin service over existing tools: CoinGecko, Binance Futures, Fear & Greed.
No LLM — all data fetched and assembled by code.

Single source of truth for MarketSnapshot in all betting paths.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from schemas.market import MarketSnapshot

import time

from tools.coingecko import CoinGeckoTools
from tools.coinglass import CoinglassTools
from tools.fear_greed import FearGreedTools

logger = logging.getLogger(__name__)

# Simple TTL cache: {coin_id: (snapshot, timestamp)}
_cache: dict[str, tuple] = {}
_CACHE_TTL = 60  # seconds — same BTC data reused across candidates in one scan


def _get_coingecko():
    return CoinGeckoTools()


def _get_coinglass():
    return CoinglassTools()


def _get_fear_greed():
    return FearGreedTools()


def derive_signal(
    change_24h_pct: float,
    fear_greed_index: int,
    funding_rate: float | None,
) -> Literal["Bullish", "Neutral", "Bearish"]:
    """Deterministic signal from price action + sentiment + derivatives."""
    bullish = 0
    bearish = 0

    if change_24h_pct > 2:
        bullish += 1
    elif change_24h_pct < -2:
        bearish += 1

    if fear_greed_index > 60:
        bullish += 1
    elif fear_greed_index < 30:
        bearish += 1

    if funding_rate is not None:
        if funding_rate < -0.0001:
            bullish += 1
        elif funding_rate > 0.0005:
            bearish += 1

    if bullish >= 2:
        return "Bullish"
    if bearish >= 2:
        return "Bearish"
    return "Neutral"


def fetch_market_snapshot(coin_id: str, symbol: str) -> MarketSnapshot | None:
    """Deterministic market data fetch. No LLM.

    Calls existing tools directly:
    - CoinGecko: price, change_24h, market_cap (REQUIRED)
    - Binance Futures: funding_rate, open_interest (optional, nullable)
    - Fear & Greed: index, label (REQUIRED)

    Returns MarketSnapshot or None on critical failure.
    Uses 60s TTL cache to avoid CoinGecko rate limits when scanning
    multiple markets for the same asset (e.g. 15 BTC markets in one scan).
    """
    # Check cache
    cache_key = f"{coin_id}:{symbol}"
    if cache_key in _cache:
        cached_snapshot, cached_time = _cache[cache_key]
        if time.time() - cached_time < _CACHE_TTL:
            return cached_snapshot

    cg = _get_coingecko()
    bn = _get_coinglass()
    fg = _get_fear_greed()

    # 1. CoinGecko — REQUIRED
    try:
        cg_data = json.loads(cg.get_price(coin_id))
        if "error" in cg_data:
            logger.warning("CoinGecko error for %s: %s", coin_id, cg_data["error"])
            return None
        price_usd = float(cg_data["price_usd"])
        change_24h_pct = float(cg_data["change_24h_pct"])
        market_cap = float(cg_data["market_cap"])
    except Exception as e:
        logger.warning("CoinGecko failed for %s: %s", coin_id, e)
        return None

    # 2. Fear & Greed — REQUIRED
    try:
        fg_data = json.loads(fg.get_current())
        if "error" in fg_data:
            logger.warning("Fear & Greed error: %s", fg_data["error"])
            return None
        fear_greed_index = int(fg_data["index"])
        fear_greed_label = str(fg_data["label"])
    except Exception as e:
        logger.warning("Fear & Greed failed: %s", e)
        return None

    # 3. Binance Futures — OPTIONAL (nullable)
    funding_rate = None
    open_interest = None
    try:
        fr_data = json.loads(bn.get_funding_rate(symbol))
        if "error" not in fr_data:
            funding_rate = float(fr_data.get("funding_rate", 0))
    except Exception:
        pass

    try:
        oi_data = json.loads(bn.get_open_interest(symbol))
        if "error" not in oi_data:
            open_interest = float(oi_data.get("open_interest", 0))
    except Exception:
        pass

    # 4. Derive signal
    signal = derive_signal(change_24h_pct, fear_greed_index, funding_rate)

    snapshot = MarketSnapshot(
        coin_id=coin_id,
        price_usd=price_usd,
        change_24h_pct=change_24h_pct,
        market_cap=market_cap,
        funding_rate=funding_rate,
        open_interest=open_interest,
        fear_greed_index=fear_greed_index,
        fear_greed_label=fear_greed_label,
        signal=signal,
    )
    _cache[cache_key] = (snapshot, time.time())
    return snapshot
