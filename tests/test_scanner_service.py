"""Tests for app/scanner_service.py — deterministic candidate selection."""

import json
from unittest.mock import patch

import pytest

from app.scanner_service import scan_candidates, _get_spread_and_depth


def _make_market(question, condition_id, token_ids, volume=30000, liquidity=50000):
    return {
        "question": question,
        "condition_id": condition_id,
        "gamma_market_id": "gm1",
        "slug": "test",
        "end_date": "2026-12-31",
        "volume_24h": volume,
        "liquidity": liquidity,
        "clob_token_ids": token_ids,
        "outcome_prices": [0.5, 0.5],
    }


def _make_orderbook(best_bid=0.48, best_ask=0.52, levels=10):
    """Create a valid sorted orderbook with enough depth to pass check_liquidity."""
    # Large sizes to ensure depth_10pct > $10K
    bids = [{"price": str(round(best_bid - i * 0.01, 2)), "size": "50000"} for i in range(levels)]
    asks = [{"price": str(round(best_ask + i * 0.01, 2)), "size": "50000"} for i in range(levels)]
    return json.dumps({"bids": bids, "asks": asks})


class TestGetSpreadAndDepth:
    def test_normal_orderbook(self):
        book = json.loads(_make_orderbook(0.48, 0.52))
        spread, depth = _get_spread_and_depth(book)
        assert spread == pytest.approx(0.04, abs=0.01)
        assert depth > 0

    def test_empty_orderbook(self):
        spread, depth = _get_spread_and_depth({"bids": [], "asks": []})
        assert spread == 1.0
        assert depth == 0.0

    def test_unsorted_orderbook(self):
        """Regression: CLOB returns unsorted data."""
        book = {
            "bids": [{"price": "0.01", "size": "100"}, {"price": "0.48", "size": "500"}],
            "asks": [{"price": "0.99", "size": "100"}, {"price": "0.52", "size": "500"}],
        }
        spread, depth = _get_spread_and_depth(book)
        assert spread == pytest.approx(0.04, abs=0.01)


class TestScanCandidates:
    @patch("app.scanner_service._polymarket")
    def test_returns_supported_only(self, mock_pm):
        """Only BTC/ETH/SOL markets pass."""
        mock_pm.get_active_crypto_markets.return_value = json.dumps([
            _make_market("Will Bitcoin reach $100K?", "0x1", ["t1", "t2"]),
            _make_market("Netherlands FIFA World Cup?", "0x2", ["t3", "t4"]),
            _make_market("EdgeX FDV above $500M?", "0x3", ["t5", "t6"]),
        ])
        mock_pm.get_orderbook.return_value = _make_orderbook()

        result = scan_candidates(max_candidates=10)
        assert len(result) == 1
        assert "Bitcoin" in result[0]["question"]

    @patch("app.scanner_service._polymarket")
    def test_prefilter_wide_spread(self, mock_pm):
        """Wide spread markets are excluded."""
        mock_pm.get_active_crypto_markets.return_value = json.dumps([
            _make_market("Will Bitcoin reach $100K?", "0x1", ["t1", "t2"]),
        ])
        mock_pm.get_orderbook.return_value = json.dumps({
            "bids": [{"price": "0.01", "size": "100"}],
            "asks": [{"price": "0.99", "size": "100"}],
        })

        result = scan_candidates(max_candidates=10)
        assert len(result) == 0

    @patch("app.scanner_service._polymarket")
    def test_empty_markets(self, mock_pm):
        mock_pm.get_active_crypto_markets.return_value = json.dumps([])
        result = scan_candidates(max_candidates=10)
        assert result == []

    @patch("app.scanner_service._polymarket")
    def test_sort_by_tradability(self, mock_pm):
        """Higher volume * depth → ranked first."""
        mock_pm.get_active_crypto_markets.return_value = json.dumps([
            _make_market("BTC dip to $20K?", "0x1", ["t1", "t2"], volume=26000),
            _make_market("BTC reach $200K?", "0x2", ["t3", "t4"], volume=50000),
        ])
        mock_pm.get_orderbook.return_value = _make_orderbook()

        result = scan_candidates(max_candidates=10)
        assert len(result) == 2
        assert result[0]["condition_id"] == "0x2"  # higher volume first

    @patch("app.scanner_service._polymarket")
    def test_no_token_ids_excluded(self, mock_pm):
        mock_pm.get_active_crypto_markets.return_value = json.dumps([
            _make_market("Will Bitcoin reach $100K?", "0x1", []),  # no tokens
        ])
        result = scan_candidates(max_candidates=10)
        assert len(result) == 0
