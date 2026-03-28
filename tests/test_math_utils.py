"""Unit tests for storage/math_utils.py — deterministic financial calculations."""

from datetime import datetime, timezone

import pytest

from schemas.paper_trade import BankrollSnapshot
from storage.math_utils import (
    brier_score,
    calculate_entry_price,
    calculate_pnl,
    check_liquidity,
    check_portfolio_limits,
    compute_edge,
    confidence_to_score,
    determine_risk_rating,
    estimate_slippage,
    fractional_kelly,
    kelly_criterion,
)


def _snap(**overrides) -> BankrollSnapshot:
    """Helper to build a BankrollSnapshot with sensible defaults."""
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        starting_bankroll=10_000.0,
        current_bankroll=7_000.0,  # equity 10K, at_risk 3K
        open_positions=5,
        total_at_risk=3_000.0,
        total_trades=10,
        wins=4,
        losses=3,
        win_rate=0.57,
        total_pnl=0.0,
        avg_brier_score=0.20,
        sharpe_ratio=None,
    )
    defaults.update(overrides)
    return BankrollSnapshot(**defaults)


class TestComputeEdge:
    def test_positive_edge(self):
        assert compute_edge(0.65, 0.51) == pytest.approx(0.14)

    def test_negative_edge(self):
        assert compute_edge(0.40, 0.51) == pytest.approx(-0.11)

    def test_zero_edge(self):
        assert compute_edge(0.50, 0.50) == pytest.approx(0.0)


class TestConfidenceToScore:
    def test_high(self):
        assert confidence_to_score("High") == 1.0

    def test_medium(self):
        assert confidence_to_score("Medium") == 0.7

    def test_low(self):
        assert confidence_to_score("Low") == 0.4

    def test_unknown(self):
        assert confidence_to_score("Unknown") == 0.0

    def test_empty(self):
        assert confidence_to_score("") == 0.0


class TestCheckLiquidity:
    def test_all_ok(self):
        ok, warnings = check_liquidity(depth_10pct=20_000, volume_24h=60_000, spread=0.02)
        assert ok is True
        assert warnings == []

    def test_low_depth(self):
        ok, warnings = check_liquidity(depth_10pct=5_000, volume_24h=60_000, spread=0.02)
        assert ok is False
        assert any("depth" in w.lower() for w in warnings)

    def test_low_volume(self):
        ok, warnings = check_liquidity(depth_10pct=20_000, volume_24h=30_000, spread=0.02)
        assert ok is False
        assert any("volume" in w.lower() for w in warnings)

    def test_wide_spread(self):
        ok, warnings = check_liquidity(depth_10pct=20_000, volume_24h=60_000, spread=0.06)
        assert ok is False
        assert any("spread" in w.lower() for w in warnings)

    def test_multiple_failures(self):
        ok, warnings = check_liquidity(depth_10pct=5_000, volume_24h=30_000, spread=0.06)
        assert ok is False
        assert len(warnings) == 3


class TestDetermineRiskRating:
    def test_low_edge_unacceptable(self):
        assert determine_risk_rating(0.03, True, 0) == "Unacceptable"

    def test_no_liquidity_unacceptable(self):
        assert determine_risk_rating(0.15, False, 0) == "Unacceptable"

    def test_too_many_correlated_unacceptable(self):
        assert determine_risk_rating(0.15, True, 3) == "Unacceptable"

    def test_high_edge_low_correlated_low(self):
        assert determine_risk_rating(0.15, True, 0) == "Low"
        assert determine_risk_rating(0.11, True, 1) == "Low"

    def test_moderate_edge_low_correlated_moderate(self):
        assert determine_risk_rating(0.07, True, 0) == "Moderate"
        assert determine_risk_rating(0.10, True, 1) == "Moderate"

    def test_two_correlated_high(self):
        assert determine_risk_rating(0.15, True, 2) == "High"
        assert determine_risk_rating(0.07, True, 2) == "High"

    def test_boundary_5pct(self):
        assert determine_risk_rating(0.05, True, 0) == "Moderate"
        assert determine_risk_rating(0.049, True, 0) == "Unacceptable"


class TestCheckPortfolioLimits:
    def test_all_ok(self):
        # equity=10K, at_risk=3K, new_stake=500 → total=3.5K (< 6K=60%), reserve=6.5K
        ok, warnings = check_portfolio_limits(_snap(), new_stake=500)
        assert ok is True
        assert warnings == []

    def test_circuit_breaker(self):
        # equity = current_bankroll + total_at_risk = 1K + 3K = 4K < 5K (50%)
        snap = _snap(current_bankroll=1_000.0, total_at_risk=3_000.0)
        ok, warnings = check_portfolio_limits(snap, new_stake=100)
        assert ok is False
        assert any("circuit breaker" in w.lower() for w in warnings)

    def test_max_positions(self):
        snap = _snap(open_positions=10)
        ok, warnings = check_portfolio_limits(snap, new_stake=500)
        assert ok is False
        assert any("positions" in w.lower() for w in warnings)

    def test_capital_at_risk_exceeded(self):
        # equity=10K, at_risk=5K, stake=2K → 7K > 6K (60%)
        snap = _snap(current_bankroll=5_000.0, total_at_risk=5_000.0)
        ok, warnings = check_portfolio_limits(snap, new_stake=2_000)
        assert ok is False
        assert any("60%" in w for w in warnings)

    def test_reserve_breach(self):
        # current_bankroll=1.2K (available cash), stake=500 → remaining=$700 < $1K reserve
        snap = _snap(current_bankroll=1_200.0, total_at_risk=8_800.0)
        ok, warnings = check_portfolio_limits(snap, new_stake=500)
        assert ok is False
        assert any("reserve" in w.lower() for w in warnings)

    def test_snapshot_semantics(self):
        # Verify equity = current_bankroll + total_at_risk
        snap = _snap(current_bankroll=7_000.0, total_at_risk=3_000.0)
        equity = snap.current_bankroll + snap.total_at_risk
        assert equity == 10_000.0
        ok, _ = check_portfolio_limits(snap, new_stake=500)
        assert ok is True


class TestCalculateMtmPnl:
    def test_profit(self):
        # Buy YES at 0.50, current 0.60, stake $100 → shares=200, value=120, PnL=+20
        from storage.math_utils import calculate_mtm_pnl
        assert calculate_mtm_pnl(0.50, 0.60, 100.0) == pytest.approx(20.0)

    def test_loss(self):
        from storage.math_utils import calculate_mtm_pnl
        assert calculate_mtm_pnl(0.50, 0.40, 100.0) == pytest.approx(-20.0)

    def test_breakeven(self):
        from storage.math_utils import calculate_mtm_pnl
        assert calculate_mtm_pnl(0.50, 0.50, 100.0) == pytest.approx(0.0)

    def test_zero_entry(self):
        from storage.math_utils import calculate_mtm_pnl
        assert calculate_mtm_pnl(0.0, 0.50, 100.0) == 0.0


class TestGetExitPrice:
    def test_with_bids(self):
        from storage.math_utils import get_exit_price_from_orderbook
        book = {"bids": [{"price": "0.55", "size": "100"}]}
        assert get_exit_price_from_orderbook(book) == 0.55

    def test_empty_bids(self):
        from storage.math_utils import get_exit_price_from_orderbook
        assert get_exit_price_from_orderbook({"bids": []}) is None

    def test_no_bids_key(self):
        from storage.math_utils import get_exit_price_from_orderbook
        assert get_exit_price_from_orderbook({}) is None


class TestCheckExitConditions:
    def test_resolution_highest_priority(self):
        from storage.math_utils import check_exit_conditions
        # Even with take profit, resolution wins
        should, reason = check_exit_conditions(
            unrealized_pnl=50, stake=100, age_seconds=10,
            market_resolved=True, take_profit_pct=0.10, stop_loss_pct=-0.05, max_hold_seconds=1800,
        )
        assert should is True
        assert reason == "resolution"

    def test_stop_loss_before_take_profit(self):
        from storage.math_utils import check_exit_conditions
        # Negative PnL that hits both (theoretically impossible but tests priority)
        should, reason = check_exit_conditions(
            unrealized_pnl=-10, stake=100, age_seconds=10,
            market_resolved=False, take_profit_pct=0.10, stop_loss_pct=-0.05, max_hold_seconds=1800,
        )
        assert should is True
        assert reason == "stop_loss"

    def test_take_profit(self):
        from storage.math_utils import check_exit_conditions
        should, reason = check_exit_conditions(
            unrealized_pnl=15, stake=100, age_seconds=10,
            market_resolved=False, take_profit_pct=0.10, stop_loss_pct=-0.05, max_hold_seconds=1800,
        )
        assert should is True
        assert reason == "take_profit"

    def test_max_hold(self):
        from storage.math_utils import check_exit_conditions
        should, reason = check_exit_conditions(
            unrealized_pnl=2, stake=100, age_seconds=2000,
            market_resolved=False, take_profit_pct=0.10, stop_loss_pct=-0.05, max_hold_seconds=1800,
        )
        assert should is True
        assert reason == "max_hold"

    def test_no_exit(self):
        from storage.math_utils import check_exit_conditions
        should, reason = check_exit_conditions(
            unrealized_pnl=5, stake=100, age_seconds=600,
            market_resolved=False, take_profit_pct=0.10, stop_loss_pct=-0.05, max_hold_seconds=1800,
        )
        assert should is False
        assert reason is None


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
