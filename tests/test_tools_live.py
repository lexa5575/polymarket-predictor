"""Live integration tests for tools — requires network access.

Run with: pytest tests/test_tools_live.py -m live
Skip by default with: pytest -m "not live"
"""

import json

import pytest

live = pytest.mark.live


@live
class TestPolymarketLive:
    def test_get_active_crypto_markets(self):
        from tools.polymarket import PolymarketTools

        tools = PolymarketTools()
        result = json.loads(tools.get_active_crypto_markets(limit=3))
        assert isinstance(result, list)
        # May be empty if no crypto markets are active
        if result:
            assert "gamma_market_id" in result[0]


@live
class TestCoinGeckoLive:
    def test_get_bitcoin_price(self):
        from tools.coingecko import CoinGeckoTools

        tools = CoinGeckoTools()
        result = json.loads(tools.get_price("bitcoin"))
        assert "error" not in result
        assert result["price_usd"] > 0


@live
class TestFearGreedLive:
    def test_get_current(self):
        from tools.fear_greed import FearGreedTools

        tools = FearGreedTools()
        result = json.loads(tools.get_current())
        assert "error" not in result
        assert 0 <= result["index"] <= 100
