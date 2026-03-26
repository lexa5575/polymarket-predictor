"""Unit tests for storage/math_utils.py — deterministic financial calculations."""

import pytest

from storage.math_utils import (
    brier_score,
    calculate_entry_price,
    calculate_pnl,
    estimate_slippage,
    fractional_kelly,
    kelly_criterion,
)


class TestKellyCriterion:
    def test_positive_edge(self):
        # p=0.6, market=0.5 → Kelly should be positive
        result = kelly_criterion(0.6, 0.5)
        assert result == pytest.approx(0.2, abs=0.01)

    def test_no_edge(self):
        # p=0.5, market=0.5 → no edge
        assert kelly_criterion(0.5, 0.5) == 0.0

    def test_negative_edge(self):
        # p=0.3, market=0.5 → negative edge → 0
        assert kelly_criterion(0.3, 0.5) == 0.0

    def test_extreme_edge(self):
        # p=0.9, market=0.3 → very strong edge
        result = kelly_criterion(0.9, 0.3)
        assert result > 0.5

    def test_invalid_inputs(self):
        assert kelly_criterion(0.0, 0.5) == 0.0
        assert kelly_criterion(1.0, 0.5) == 0.0
        assert kelly_criterion(0.5, 0.0) == 0.0
        assert kelly_criterion(0.5, 1.0) == 0.0


class TestFractionalKelly:
    def test_quarter_kelly(self):
        assert fractional_kelly(0.2, 0.25) == pytest.approx(0.05)

    def test_zero_kelly(self):
        assert fractional_kelly(0.0) == 0.0

    def test_negative_kelly(self):
        assert fractional_kelly(-0.1) == 0.0


class TestBrierScore:
    def test_perfect_prediction_win(self):
        # Predicted 1.0, outcome 1 → Brier = 0
        assert brier_score(1.0, 1) == 0.0

    def test_perfect_prediction_loss(self):
        # Predicted 0.0, outcome 0 → Brier = 0
        assert brier_score(0.0, 0) == 0.0

    def test_typical_good_prediction(self):
        # Predicted 0.8, outcome 1 → Brier = 0.04
        assert brier_score(0.8, 1) == pytest.approx(0.04)

    def test_bad_prediction(self):
        # Predicted 0.2, outcome 1 → Brier = 0.64
        assert brier_score(0.2, 1) == pytest.approx(0.64)

    def test_random_guess(self):
        # Predicted 0.5, outcome 1 → Brier = 0.25
        assert brier_score(0.5, 1) == pytest.approx(0.25)


class TestEstimateSlippage:
    def test_no_slippage_small_order(self):
        asks = [(0.50, 10000.0)]  # plenty of liquidity
        assert estimate_slippage(100.0, asks) == 0.0

    def test_slippage_large_order(self):
        asks = [(0.50, 500.0), (0.52, 500.0), (0.55, 1000.0)]
        slippage = estimate_slippage(1000.0, asks)
        assert slippage > 0  # should have some slippage

    def test_empty_orderbook(self):
        assert estimate_slippage(100.0, []) == 0.0

    def test_zero_stake(self):
        assert estimate_slippage(0.0, [(0.50, 1000.0)]) == 0.0


class TestCalculateEntryPrice:
    def test_basic(self):
        assert calculate_entry_price(0.50, 0.01) == pytest.approx(0.51)


class TestCalculatePnl:
    def test_win(self):
        # Buy at 0.50, stake $100, win → payout = $200, PnL = +$100
        assert calculate_pnl(0.50, 100.0, True) == pytest.approx(100.0)

    def test_loss(self):
        # Lose → PnL = -stake
        assert calculate_pnl(0.50, 100.0, False) == -100.0

    def test_win_at_higher_price(self):
        # Buy at 0.80, stake $100, win → payout = $125, PnL = +$25
        assert calculate_pnl(0.80, 100.0, True) == pytest.approx(25.0)

    def test_win_at_low_price(self):
        # Buy at 0.20, stake $100, win → payout = $500, PnL = +$400
        assert calculate_pnl(0.20, 100.0, True) == pytest.approx(400.0)
