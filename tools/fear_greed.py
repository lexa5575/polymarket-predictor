"""
FearGreedTools — Toolkit for the Crypto Fear & Greed Index.

Uses the free alternative.me API (no key required).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from agno.tools import Toolkit

BASE_URL = "https://api.alternative.me/fng/"


class FearGreedTools(Toolkit):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="fear_greed_tools",
            tools=[
                self.get_current,
                self.get_historical,
            ],
            **kwargs,
        )

    def get_current(self) -> str:
        """Get the current Crypto Fear & Greed Index.

        Returns:
            JSON with index value (0-100) and classification label.
        """
        try:
            resp = httpx.get(BASE_URL, params={"limit": "1"}, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [{}])[0]
            return json.dumps({
                "index": int(data.get("value", 0)),
                "label": data.get("value_classification", "Unknown"),
                "timestamp": data.get("timestamp"),
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_historical(self, days: int = 30) -> str:
        """Get historical Fear & Greed Index values.

        Args:
            days: Number of days of history (default 30, max 365).

        Returns:
            JSON array of {index, label, timestamp} entries.
        """
        try:
            resp = httpx.get(
                BASE_URL,
                params={"limit": str(min(days, 365))},
                timeout=15,
            )
            resp.raise_for_status()
            entries = resp.json().get("data", [])
            results = [
                {
                    "index": int(e.get("value", 0)),
                    "label": e.get("value_classification", "Unknown"),
                    "timestamp": e.get("timestamp"),
                }
                for e in entries
            ]
            return json.dumps({"days": days, "data": results})
        except Exception as e:
            return json.dumps({"error": str(e)})
