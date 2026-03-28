"""Tests for app/market_data_service.py — deterministic market data fetch."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.market_data_service import derive_signal, fetch_market_snapshot


class TestDeriveSignal:
    def test_bullish(self):
        assert derive_signal(3.0, 70, -0.0002) == "Bullish"

    def test_bearish(self):
        assert derive_signal(-3.0, 20, 0.001) == "Bearish"

    def test_neutral_mixed(self):
        assert derive_signal(1.0, 50, None) == "Neutral"

    def test_neutral_one_signal(self):
        assert derive_signal(3.0, 50, None) == "Neutral"

    def test_bullish_without_funding(self):
        assert derive_signal(3.0, 70, None) == "Bullish"

    def test_bearish_without_funding(self):
        assert derive_signal(-3.0, 20, None) == "Bearish"


def _mock_tools(cg_data=None, bn_fr_data=None, bn_oi_data=None, fg_data=None):
    """Helper to create mock tool instances."""
    cg = MagicMock()
    cg.get_price.return_value = json.dumps(cg_data or {"error": "not set"})
    bn = MagicMock()
    bn.get_funding_rate.return_value = json.dumps(bn_fr_data or {"error": "not set"})
    bn.get_open_interest.return_value = json.dumps(bn_oi_data or {"error": "not set"})
    fg = MagicMock()
    fg.get_current.return_value = json.dumps(fg_data or {"error": "not set"})
    return cg, bn, fg


class TestFetchMarketSnapshot:
    @patch("app.market_data_service._get_fear_greed")
    @patch("app.market_data_service._get_coinglass")
    @patch("app.market_data_service._get_coingecko")
    def test_success(self, mock_cg, mock_bn, mock_fg):
        cg, bn, fg = _mock_tools(
            cg_data={"coin_id": "bitcoin", "price_usd": 67000, "change_24h_pct": 2.5, "market_cap": 1.3e12},
            bn_fr_data={"symbol": "BTC", "funding_rate": -0.0001},
            bn_oi_data={"symbol": "BTC", "open_interest": 95000},
            fg_data={"index": 65, "label": "Greed", "timestamp": "123"},
        )
        mock_cg.return_value = cg
        mock_bn.return_value = bn
        mock_fg.return_value = fg

        result = fetch_market_snapshot("bitcoin", "BTC")
        assert result is not None
        assert result.coin_id == "bitcoin"
        assert result.price_usd == 67000
        assert result.funding_rate == pytest.approx(-0.0001)
        assert result.fear_greed_index == 65

    @patch("app.market_data_service._get_fear_greed")
    @patch("app.market_data_service._get_coinglass")
    @patch("app.market_data_service._get_coingecko")
    def test_coingecko_error_returns_none(self, mock_cg, mock_bn, mock_fg):
        cg, bn, fg = _mock_tools(cg_data={"error": "timeout"})
        mock_cg.return_value = cg
        mock_bn.return_value = bn
        mock_fg.return_value = fg
        assert fetch_market_snapshot("bitcoin", "BTC") is None

    @patch("app.market_data_service._get_fear_greed")
    @patch("app.market_data_service._get_coinglass")
    @patch("app.market_data_service._get_coingecko")
    def test_fear_greed_error_returns_none(self, mock_cg, mock_bn, mock_fg):
        cg, bn, fg = _mock_tools(
            cg_data={"coin_id": "bitcoin", "price_usd": 67000, "change_24h_pct": 1.0, "market_cap": 1e12},
            fg_data={"error": "API down"},
        )
        mock_cg.return_value = cg
        mock_bn.return_value = bn
        mock_fg.return_value = fg
        assert fetch_market_snapshot("bitcoin", "BTC") is None

    @patch("app.market_data_service._get_fear_greed")
    @patch("app.market_data_service._get_coinglass")
    @patch("app.market_data_service._get_coingecko")
    def test_binance_error_optional(self, mock_cg, mock_bn, mock_fg):
        cg, bn, fg = _mock_tools(
            cg_data={"coin_id": "bitcoin", "price_usd": 67000, "change_24h_pct": 1.0, "market_cap": 1e12},
            bn_fr_data={"error": "connection refused"},
            bn_oi_data={"error": "connection refused"},
            fg_data={"index": 50, "label": "Neutral", "timestamp": "123"},
        )
        mock_cg.return_value = cg
        mock_bn.return_value = bn
        mock_fg.return_value = fg

        result = fetch_market_snapshot("bitcoin", "BTC")
        assert result is not None
        assert result.funding_rate is None
        assert result.open_interest is None
        assert result.price_usd == 67000

    @patch("app.market_data_service._get_fear_greed")
    @patch("app.market_data_service._get_coinglass")
    @patch("app.market_data_service._get_coingecko")
    def test_error_payload_not_treated_as_zeros(self, mock_cg, mock_bn, mock_fg):
        cg, bn, fg = _mock_tools(cg_data={"error": "timeout"})
        mock_cg.return_value = cg
        mock_bn.return_value = bn
        mock_fg.return_value = fg
        assert fetch_market_snapshot("bitcoin", "BTC") is None
