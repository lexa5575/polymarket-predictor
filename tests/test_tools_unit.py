"""Unit tests for tools — mock httpx, no network required."""

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Coinglass
# ---------------------------------------------------------------------------


class TestBinanceFuturesTools:
    """Tests for CoinglassTools (now backed by Binance Futures public API)."""

    @patch("tools.coinglass.httpx.get")
    def test_funding_rate_success(self, mock_get):
        """Mock a successful Binance funding rate response."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [
            {"symbol": "BTCUSDT", "fundingRate": "0.00010000", "fundingTime": 1700000000000}
        ]
        mock_get.return_value = mock_resp

        from tools.coinglass import CoinglassTools

        tools = CoinglassTools()
        result = json.loads(tools.get_funding_rate("BTC"))
        assert result["symbol"] == "BTC"
        assert result["pair"] == "BTCUSDT"
        assert result["funding_rate"] == pytest.approx(0.0001)

    @patch("tools.coinglass.httpx.get")
    def test_funding_rate_empty_response(self, mock_get):
        """Empty data array → error field."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        from tools.coinglass import CoinglassTools

        tools = CoinglassTools()
        result = json.loads(tools.get_funding_rate("BTC"))
        assert result["data"] is None

    @patch("tools.coinglass.httpx.get")
    def test_open_interest_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"openInterest": "50000.00", "symbol": "BTCUSDT", "time": 1700000000000}
        mock_get.return_value = mock_resp

        from tools.coinglass import CoinglassTools

        tools = CoinglassTools()
        result = json.loads(tools.get_open_interest("BTC"))
        assert result["symbol"] == "BTC"
        assert result["open_interest"] == 50000.0

    @patch("tools.coinglass.httpx.get")
    def test_api_exception(self, mock_get):
        """Network error → graceful degradation."""
        mock_get.side_effect = Exception("Connection refused")

        from tools.coinglass import CoinglassTools

        tools = CoinglassTools()
        result = json.loads(tools.get_funding_rate("BTC"))
        assert "error" in result
        assert result["data"] is None

    def test_no_api_key_still_works(self):
        """Binance API is free — no key needed, should not error on init."""
        from tools.coinglass import CoinglassTools

        tools = CoinglassTools(api_key="")
        # Should not raise — api_key is ignored for Binance
        assert tools is not None


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------


class TestFearGreedParsing:
    @patch("tools.fear_greed.httpx.get")
    def test_parse_current(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"value": "72", "value_classification": "Greed", "timestamp": "1234567890"}]
        }
        mock_get.return_value = mock_resp

        from tools.fear_greed import FearGreedTools

        tools = FearGreedTools()
        result = json.loads(tools.get_current())
        assert result["index"] == 72
        assert result["label"] == "Greed"


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------


class TestCoinGeckoParsing:
    @patch("tools.coingecko.httpx.get")
    def test_parse_price(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "bitcoin": {
                "usd": 67500.0,
                "usd_24h_change": 2.5,
                "usd_market_cap": 1300000000000,
            }
        }
        mock_get.return_value = mock_resp

        from tools.coingecko import CoinGeckoTools

        tools = CoinGeckoTools()
        result = json.loads(tools.get_price("bitcoin"))
        assert result["coin_id"] == "bitcoin"
        assert result["price_usd"] == 67500.0


# ---------------------------------------------------------------------------
# Polymarket — adapter hardening tests
# ---------------------------------------------------------------------------


class TestPolymarketNormalization:
    """Test clobTokenIds normalization from various Gamma API response shapes."""

    def test_normalize_token_ids_from_list(self):
        from tools.polymarket import _normalize_token_ids

        assert _normalize_token_ids(["0xabc", "0xdef"]) == ["0xabc", "0xdef"]

    def test_normalize_token_ids_from_json_string(self):
        from tools.polymarket import _normalize_token_ids

        result = _normalize_token_ids('["0xabc", "0xdef"]')
        assert result == ["0xabc", "0xdef"]

    def test_normalize_token_ids_none(self):
        from tools.polymarket import _normalize_token_ids

        assert _normalize_token_ids(None) == []

    def test_normalize_token_ids_empty_string(self):
        from tools.polymarket import _normalize_token_ids

        assert _normalize_token_ids("") == []

    def test_normalize_outcome_prices_from_string(self):
        from tools.polymarket import _normalize_outcome_prices

        result = _normalize_outcome_prices('["0.65", "0.35"]')
        assert result == [0.65, 0.35]

    def test_normalize_outcome_prices_from_list(self):
        from tools.polymarket import _normalize_outcome_prices

        assert _normalize_outcome_prices([0.7, 0.3]) == [0.7, 0.3]

    def test_normalize_outcome_prices_from_string_list(self):
        from tools.polymarket import _normalize_outcome_prices

        assert _normalize_outcome_prices(["1", "0"]) == [1.0, 0.0]

    def test_is_crypto_market_by_tags(self):
        from tools.polymarket import _is_crypto_market

        assert _is_crypto_market({"tags": ["crypto", "finance"]}) is True
        assert _is_crypto_market({"tags": ["politics"]}) is False

    def test_is_crypto_market_by_question(self):
        from tools.polymarket import _is_crypto_market

        assert _is_crypto_market({"question": "Will BTC exceed $100K?"}) is True
        assert _is_crypto_market({"question": "Will the Lakers win?"}) is False


class TestPolymarketConditionIdLookup:
    """Test find_market with conditionId (fetch-then-filter path)."""

    @patch("tools.polymarket.httpx.get")
    def test_find_by_condition_id(self, mock_get):
        """find_market should fetch batch and filter locally by conditionId."""
        # First call (direct /markets/{id}) returns 404
        resp_404 = MagicMock()
        resp_404.status_code = 404

        # Second call (slug search) returns empty
        resp_empty = MagicMock()
        resp_empty.status_code = 200
        resp_empty.json.return_value = []

        # Third call (batch fetch) returns list with our market
        target_market = {
            "id": "gamma123",
            "conditionId": "0xcond456",
            "question": "BTC $100K?",
            "clobTokenIds": '["0xyes", "0xno"]',  # JSON string format
            "slug": "btc-100k",
        }
        resp_batch = MagicMock()
        resp_batch.status_code = 200
        resp_batch.json.return_value = [
            {"id": "other", "conditionId": "0xother"},
            target_market,
        ]

        mock_get.side_effect = [resp_404, resp_empty, resp_batch]

        from tools.polymarket import PolymarketTools

        tools = PolymarketTools()
        result = json.loads(tools.find_market("0xcond456"))

        assert result["conditionId"] == "0xcond456"
        # clobTokenIds should be normalized from JSON string to list
        assert result["clobTokenIds"] == ["0xyes", "0xno"]

    @patch("tools.polymarket.httpx.get")
    def test_find_market_not_found(self, mock_get):
        resp_404 = MagicMock()
        resp_404.status_code = 404

        resp_empty = MagicMock()
        resp_empty.status_code = 200
        resp_empty.json.return_value = []

        resp_batch = MagicMock()
        resp_batch.status_code = 200
        resp_batch.json.return_value = []

        mock_get.side_effect = [resp_404, resp_empty, resp_batch]

        from tools.polymarket import PolymarketTools

        tools = PolymarketTools()
        result = json.loads(tools.find_market("nonexistent"))
        assert "error" in result


class TestPolymarketResolution:
    """Test get_market_resolution with various outcomePrices formats."""

    @patch("tools.polymarket.httpx.get")
    def test_resolved_yes_from_string_prices(self, mock_get):
        """outcomePrices as JSON string '["1","0"]' → YES wins."""
        market = {
            "id": "g1",
            "conditionId": "0xresolved",
            "resolved": True,
            "closed": True,
            "outcomePrices": '["1","0"]',
            "clobTokenIds": ["t1", "t2"],
            "question": "Did BTC hit $100K?",
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = market
        mock_get.return_value = resp

        from tools.polymarket import PolymarketTools

        tools = PolymarketTools()
        result = json.loads(tools.get_market_resolution("g1"))
        assert result["status"] == "resolved"
        assert result["final_outcome"] == "YES"

    @patch("tools.polymarket.httpx.get")
    def test_resolved_no_from_list_prices(self, mock_get):
        """outcomePrices as Python list [0, 1] → NO wins."""
        market = {
            "id": "g2",
            "conditionId": "0xresno",
            "resolved": True,
            "closed": True,
            "outcomePrices": [0, 1],
            "clobTokenIds": ["t1", "t2"],
            "question": "Did ETH hit $5K?",
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = market
        mock_get.return_value = resp

        from tools.polymarket import PolymarketTools

        tools = PolymarketTools()
        result = json.loads(tools.get_market_resolution("g2"))
        assert result["status"] == "resolved"
        assert result["final_outcome"] == "NO"

    @patch("tools.polymarket.httpx.get")
    def test_active_market(self, mock_get):
        market = {
            "id": "g3",
            "conditionId": "0xactive",
            "resolved": False,
            "closed": False,
            "outcomePrices": '["0.65","0.35"]',
            "clobTokenIds": ["t1", "t2"],
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = market
        mock_get.return_value = resp

        from tools.polymarket import PolymarketTools

        tools = PolymarketTools()
        result = json.loads(tools.get_market_resolution("g3"))
        assert result["status"] == "active"
        assert result["final_outcome"] is None
