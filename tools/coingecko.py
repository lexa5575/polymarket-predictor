"""
CoinGeckoTools — Toolkit for CoinGecko public API (no key required).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from agno.tools import Toolkit

BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoTools(Toolkit):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="coingecko_tools",
            tools=[
                self.get_price,
                self.get_historical,
                self.get_trending,
            ],
            **kwargs,
        )

    def get_price(self, coin_id: str) -> str:
        """Get current price, 24h change, and market cap for a cryptocurrency.

        Args:
            coin_id: CoinGecko coin ID (e.g. "bitcoin", "ethereum").

        Returns:
            JSON with price_usd, change_24h_pct, market_cap.
        """
        try:
            resp = httpx.get(
                f"{BASE_URL}/simple/price",
                params={
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_change": "true",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get(coin_id, {})
            return json.dumps({
                "coin_id": coin_id,
                "price_usd": data.get("usd", 0),
                "change_24h_pct": data.get("usd_24h_change", 0),
                "market_cap": data.get("usd_market_cap", 0),
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_historical(self, coin_id: str, days: int = 30) -> str:
        """Get historical OHLC price data.

        Args:
            coin_id: CoinGecko coin ID (e.g. "bitcoin").
            days: Number of days of history (1, 7, 14, 30, 90, 180, 365).

        Returns:
            JSON array of [timestamp, open, high, low, close] candles.
        """
        try:
            resp = httpx.get(
                f"{BASE_URL}/coins/{coin_id}/ohlc",
                params={"vs_currency": "usd", "days": str(days)},
                timeout=15,
            )
            resp.raise_for_status()
            candles = resp.json()
            return json.dumps({
                "coin_id": coin_id,
                "days": days,
                "candles_count": len(candles),
                "candles": candles[:50],  # limit output size
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_trending(self) -> str:
        """Get currently trending cryptocurrencies on CoinGecko.

        Returns:
            JSON with top trending coins.
        """
        try:
            resp = httpx.get(f"{BASE_URL}/search/trending", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            coins = []
            for item in data.get("coins", [])[:10]:
                coin = item.get("item", {})
                coins.append({
                    "id": coin.get("id"),
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol"),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "price_btc": coin.get("price_btc"),
                })
            return json.dumps({"trending": coins})
        except Exception as e:
            return json.dumps({"error": str(e)})
