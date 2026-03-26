"""
PolymarketTools — Toolkit for Polymarket Gamma API and CLOB API.

Gamma API: market discovery, details, resolution status.
CLOB API: orderbooks and price history (by token_id).

Hardening notes:
- clobTokenIds may arrive as a JSON string or a Python list — always normalized.
- Gamma API does not support `condition_id` as a query param; we fetch-then-filter.
- `tag` param may not be reliable; we post-filter by tags array contents.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from agno.tools import Toolkit

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def _normalize_token_ids(raw: Any) -> list[str]:
    """Normalize clobTokenIds from either a JSON string or a Python list.

    Gamma API may return clobTokenIds as:
    - A Python list: ["0xabc...", "0xdef..."]
    - A JSON string: '["0xabc...", "0xdef..."]'
    - None or empty
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def _normalize_outcome_prices(raw: Any) -> list[float]:
    """Normalize outcomePrices from string or list to list[float]."""
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except (ValueError, TypeError):
            return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return []


def _is_crypto_market(market: dict) -> bool:
    """Check if a market belongs to crypto category via tags array or question heuristic."""
    # Check tags array (documented field)
    tags = market.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            tag_str = str(tag).lower() if isinstance(tag, str) else json.dumps(tag).lower()
            if "crypto" in tag_str or "bitcoin" in tag_str or "ethereum" in tag_str:
                return True
    # Fallback: check question text for crypto keywords
    question = (market.get("question") or "").lower()
    crypto_keywords = ["btc", "bitcoin", "eth", "ethereum", "crypto", "halving", "etf"]
    return any(kw in question for kw in crypto_keywords)


class PolymarketTools(Toolkit):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="polymarket_tools",
            tools=[
                self.get_active_crypto_markets,
                self.get_market_by_id,
                self.find_market,
                self.get_orderbook,
                self.get_price_history,
                self.get_market_with_books,
                self.get_market_resolution,
            ],
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Gamma API — Market discovery
    # ------------------------------------------------------------------

    def get_active_crypto_markets(self, limit: int = 20) -> str:
        """Get active crypto prediction markets from Polymarket.

        Uses the Gamma Events API with tag_slug=crypto-prices to access all
        crypto price markets (5000+), not just the handful from /markets.
        Also fetches general crypto events. Sorts by 24h volume descending.

        Args:
            limit: Maximum number of crypto markets to return (default 20).

        Returns:
            JSON string with list of active crypto markets.
        """
        all_markets = []

        # Source 1: Crypto price events (BTC/ETH/SOL price targets — thousands of markets)
        for tag_slug in ["crypto-prices", "crypto"]:
            try:
                resp = httpx.get(
                    f"{GAMMA_BASE}/events",
                    params={
                        "tag_slug": tag_slug,
                        "active": "true",
                        "closed": "false",
                        "limit": "50",
                    },
                    timeout=20,
                )
                resp.raise_for_status()
                events = resp.json()
                if isinstance(events, list):
                    for event in events:
                        for m in event.get("markets", []):
                            if not m.get("active", True) or m.get("closed", False):
                                continue
                            all_markets.append(m)
            except Exception:
                pass

        # Source 2: General markets search (fallback)
        try:
            resp = httpx.get(
                f"{GAMMA_BASE}/markets",
                params={"active": "true", "closed": "false", "limit": "100"},
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()
            if isinstance(markets, list):
                for m in markets:
                    if _is_crypto_market(m):
                        all_markets.append(m)
        except Exception:
            pass

        # Deduplicate by conditionId
        seen = set()
        unique = []
        for m in all_markets:
            cid = m.get("conditionId", m.get("id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(m)

        # Sort by 24h volume descending
        unique.sort(key=lambda m: float(m.get("volume24hr", 0) or 0), reverse=True)

        # Build results
        results = []
        for m in unique[:limit]:
            token_ids = _normalize_token_ids(m.get("clobTokenIds"))
            results.append({
                "gamma_market_id": m.get("id"),
                "condition_id": m.get("conditionId", ""),
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "end_date": m.get("endDate", ""),
                "volume_24h": float(m.get("volume24hr", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "clob_token_ids": token_ids,
                "outcome_prices": _normalize_outcome_prices(m.get("outcomePrices")),
            })

        return json.dumps(results, indent=2)

    def get_market_by_id(self, gamma_market_id: str) -> str:
        """Get detailed market info by Gamma API market id.

        Args:
            gamma_market_id: The Gamma API market id.

        Returns:
            JSON string with market details including conditionId and clobTokenIds.
        """
        try:
            resp = httpx.get(
                f"{GAMMA_BASE}/markets/{gamma_market_id}",
                timeout=15,
            )
            resp.raise_for_status()
            market = resp.json()
            # Normalize clobTokenIds in-place for downstream safety
            market["clobTokenIds"] = _normalize_token_ids(market.get("clobTokenIds"))
            return json.dumps(market, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def find_market(self, identifier: str) -> str:
        """Find a market by any identifier (gamma_market_id, conditionId, or slug).

        Strategy:
        1. Try direct GET /markets/{identifier} (works for gamma_market_id).
        2. Try slug-based search via GET /markets?slug={identifier}.
        3. Fetch a batch and filter locally by conditionId (Gamma API does not
           support conditionId as a query param).

        Args:
            identifier: Any Polymarket identifier.

        Returns:
            JSON string with market details (clobTokenIds normalized).
        """
        # 1. Direct lookup by gamma_market_id
        try:
            resp = httpx.get(f"{GAMMA_BASE}/markets/{identifier}", timeout=15)
            if resp.status_code == 200:
                market = resp.json()
                market["clobTokenIds"] = _normalize_token_ids(market.get("clobTokenIds"))
                return json.dumps(market, indent=2)
        except Exception:
            pass

        # 2. Search by slug
        try:
            resp = httpx.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": identifier, "limit": "1"},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    market = data[0]
                    market["clobTokenIds"] = _normalize_token_ids(market.get("clobTokenIds"))
                    return json.dumps(market, indent=2)
        except Exception:
            pass

        # 3. Paginated fetch-then-filter by conditionId.
        #    Gamma API does not support conditionId as a query param,
        #    so we paginate until we find the market or exhaust pages.
        try:
            offset = 0
            page_size = 100
            max_pages = 10  # safety cap: 1000 markets max
            for _ in range(max_pages):
                resp = httpx.get(
                    f"{GAMMA_BASE}/markets",
                    params={"limit": str(page_size), "offset": str(offset)},
                    timeout=20,
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                if not isinstance(data, list) or not data:
                    break
                for m in data:
                    if m.get("conditionId") == identifier:
                        m["clobTokenIds"] = _normalize_token_ids(m.get("clobTokenIds"))
                        return json.dumps(m, indent=2)
                if len(data) < page_size:
                    break  # last page
                offset += page_size
        except Exception:
            pass

        return json.dumps({"error": f"Market not found for identifier: {identifier}"})

    def get_market_resolution(self, identifier: str) -> str:
        """Check resolution status of a market.

        Args:
            identifier: Any Polymarket identifier.

        Returns:
            JSON with status (active/closed/resolved) and final_outcome (YES/NO/null).
        """
        market_json = self.find_market(identifier)
        try:
            market = json.loads(market_json)
            if "error" in market:
                return market_json

            closed = market.get("closed", False)
            resolved = market.get("resolved", False)
            outcome_prices = _normalize_outcome_prices(market.get("outcomePrices"))

            status = "active"
            final_outcome = None
            if resolved:
                status = "resolved"
                if len(outcome_prices) >= 2:
                    if outcome_prices[0] == 1.0:
                        final_outcome = "YES"
                    elif outcome_prices[1] == 1.0:
                        final_outcome = "NO"
            elif closed:
                status = "closed"

            return json.dumps({
                "condition_id": market.get("conditionId", ""),
                "question": market.get("question", ""),
                "status": status,
                "final_outcome": final_outcome,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # CLOB API — Orderbook and price history
    # ------------------------------------------------------------------

    def get_orderbook(self, token_id: str) -> str:
        """Get the orderbook for a specific token (YES or NO).

        Args:
            token_id: The CLOB token ID.

        Returns:
            JSON with bids and asks arrays.
        """
        try:
            resp = httpx.get(
                f"{CLOB_BASE}/book",
                params={"token_id": token_id},
                timeout=15,
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_price_history(
        self,
        token_id: str,
        interval: str = "1d",
    ) -> str:
        """Get price history for a token.

        Args:
            token_id: The CLOB token ID.
            interval: Time interval — "1m", "5m", "1h", "1d" (default "1d").

        Returns:
            JSON with price history data points.
        """
        try:
            resp = httpx.get(
                f"{CLOB_BASE}/prices-history",
                params={"market": token_id, "interval": interval},
                timeout=15,
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_market_with_books(self, gamma_market_id: str) -> str:
        """Get market details with full orderbooks for both YES and NO tokens.

        Args:
            gamma_market_id: The Gamma API market id.

        Returns:
            JSON with market info + yes_book + no_book.
        """
        market_json = self.get_market_by_id(gamma_market_id)
        try:
            market = json.loads(market_json)
            if "error" in market:
                return market_json

            token_ids = market.get("clobTokenIds", [])  # already normalized by get_market_by_id
            if len(token_ids) < 2:
                return json.dumps({"error": "Market does not have two token IDs"})

            yes_token_id = token_ids[0]
            no_token_id = token_ids[1]

            yes_book = json.loads(self.get_orderbook(yes_token_id))
            no_book = json.loads(self.get_orderbook(no_token_id))

            return json.dumps({
                "market": market,
                "yes_token_id": yes_token_id,
                "no_token_id": no_token_id,
                "yes_book": yes_book,
                "no_book": no_book,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
