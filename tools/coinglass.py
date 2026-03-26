"""
CoinglassTools — Toolkit for Coinglass derivatives data API (v3).

Requires COINGLASS_API_KEY environment variable.
Graceful degradation: returns null data if key is missing or API errors.

API docs: https://docs.coinglass.com/v3.0/reference/general-information-1
Auth header: CG-API-KEY (per v3 docs, replacing older coinglassSecret).
Endpoints: /api/futures/funding/v2/current, /api/futures/openInterest/v2/current
"""

from __future__ import annotations

import json
from os import getenv
from typing import Any

import httpx
from agno.tools import Toolkit

BASE_URL = "https://open-api-v3.coinglass.com/api"


class CoinglassTools(Toolkit):
    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self._api_key = api_key or getenv("COINGLASS_API_KEY", "")
        super().__init__(
            name="coinglass_tools",
            tools=[
                self.get_funding_rate,
                self.get_open_interest,
            ],
            **kwargs,
        )

    def _get_headers(self) -> dict[str, str]:
        """Auth header per Coinglass v3 docs."""
        if self._api_key:
            return {"CG-API-KEY": self._api_key}
        return {}

    def _unavailable(self, reason: str = "Coinglass unavailable") -> str:
        return json.dumps({"error": reason, "data": None})

    def get_funding_rate(self, symbol: str = "BTC") -> str:
        """Get current perpetual futures funding rate for a symbol.

        Uses v2 endpoint per current Coinglass docs.

        Args:
            symbol: Trading symbol (e.g. "BTC", "ETH").

        Returns:
            JSON with funding rate data, or null if unavailable.
        """
        if not self._api_key:
            return self._unavailable("COINGLASS_API_KEY not set")

        try:
            resp = httpx.get(
                f"{BASE_URL}/futures/funding/v2/current",
                params={"symbol": symbol},
                headers=self._get_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            # v3 API uses "success" boolean or "code" string
            if data.get("success") is False or data.get("code") not in (None, "0", 0):
                return self._unavailable(data.get("msg", "API error"))
            return json.dumps({
                "symbol": symbol,
                "data": data.get("data"),
            })
        except Exception as e:
            return self._unavailable(str(e))

    def get_open_interest(self, symbol: str = "BTC") -> str:
        """Get current open interest for a symbol across exchanges.

        Uses v2 endpoint per current Coinglass docs.

        Args:
            symbol: Trading symbol (e.g. "BTC", "ETH").

        Returns:
            JSON with open interest data, or null if unavailable.
        """
        if not self._api_key:
            return self._unavailable("COINGLASS_API_KEY not set")

        try:
            resp = httpx.get(
                f"{BASE_URL}/futures/openInterest/v2/current",
                params={"symbol": symbol},
                headers=self._get_headers(),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") is False or data.get("code") not in (None, "0", 0):
                return self._unavailable(data.get("msg", "API error"))
            return json.dumps({
                "symbol": symbol,
                "data": data.get("data"),
            })
        except Exception as e:
            return self._unavailable(str(e))
