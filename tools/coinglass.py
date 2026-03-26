"""
BinanceFuturesTools — Free derivatives data from Binance Futures public API.

No API key required. Replaces Coinglass (which requires paid plan).
Provides: funding rate, open interest, 24h ticker data.

API docs: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from agno.tools import Toolkit

BASE_URL = "https://fapi.binance.com/fapi/v1"


class CoinglassTools(Toolkit):
    """Binance Futures public API (backwards-compatible name for existing agent imports)."""

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        # api_key kept for backwards compatibility but not used
        super().__init__(
            name="coinglass_tools",
            tools=[
                self.get_funding_rate,
                self.get_open_interest,
                self.get_ticker_24h,
            ],
            **kwargs,
        )

    def get_funding_rate(self, symbol: str = "BTC") -> str:
        """Get current perpetual futures funding rate from Binance.

        Args:
            symbol: Trading symbol (e.g. "BTC", "ETH"). Automatically appends USDT.

        Returns:
            JSON with funding rate data.
        """
        pair = f"{symbol.upper()}USDT"
        try:
            resp = httpx.get(
                f"{BASE_URL}/fundingRate",
                params={"symbol": pair, "limit": "1"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                entry = data[-1]
                return json.dumps({
                    "symbol": symbol,
                    "pair": pair,
                    "funding_rate": float(entry.get("fundingRate", 0)),
                    "funding_time": entry.get("fundingTime"),
                })
            return json.dumps({"symbol": symbol, "data": None, "error": "No data returned"})
        except Exception as e:
            return json.dumps({"error": str(e), "data": None})

    def get_open_interest(self, symbol: str = "BTC") -> str:
        """Get current open interest from Binance Futures.

        Args:
            symbol: Trading symbol (e.g. "BTC", "ETH"). Automatically appends USDT.

        Returns:
            JSON with open interest in contracts and notional value.
        """
        pair = f"{symbol.upper()}USDT"
        try:
            resp = httpx.get(
                f"{BASE_URL}/openInterest",
                params={"symbol": pair},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps({
                "symbol": symbol,
                "pair": pair,
                "open_interest": float(data.get("openInterest", 0)),
                "time": data.get("time"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "data": None})

    def get_ticker_24h(self, symbol: str = "BTC") -> str:
        """Get 24h ticker stats from Binance Futures (volume, price change, etc).

        Args:
            symbol: Trading symbol (e.g. "BTC", "ETH"). Automatically appends USDT.

        Returns:
            JSON with 24h stats: price, volume, high, low, change percent.
        """
        pair = f"{symbol.upper()}USDT"
        try:
            resp = httpx.get(
                f"{BASE_URL}/ticker/24hr",
                params={"symbol": pair},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return json.dumps({
                "symbol": symbol,
                "pair": pair,
                "last_price": float(data.get("lastPrice", 0)),
                "price_change_pct": float(data.get("priceChangePercent", 0)),
                "high_24h": float(data.get("highPrice", 0)),
                "low_24h": float(data.get("lowPrice", 0)),
                "volume_24h": float(data.get("volume", 0)),
                "quote_volume_24h": float(data.get("quoteVolume", 0)),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "data": None})
