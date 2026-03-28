"""E2E test for paper trade flow: open → resolve → PnL → bankroll snapshot.

Uses SQLite in-memory database for isolation (no PostgreSQL needed).
"""

import pytest

from schemas.market import BetDecision
from storage.paper_trades import PaperTradeStore


@pytest.fixture
def store(tmp_path):
    """Create a PaperTradeStore backed by a temporary SQLite database."""
    db_path = tmp_path / "test.db"
    return PaperTradeStore(f"sqlite:///{db_path}")


@pytest.fixture
def sample_decision():
    return BetDecision(
        condition_id="0xabc123",
        market_slug="btc-100k-june-2026",
        token_id="token_yes_123",
        side="YES",
        action="BET",
        estimated_prob_of_side=0.60,
        market_prob_of_side_at_entry=0.48,
        edge=0.12,
        entry_price=0.49,
        slippage_estimate=0.01,
        stake=500.0,
        underlier_group="btc_price",
        rationale="Strong momentum and positive funding rate",
        exit_conditions=["Exit if prob drops below 0.35"],
        confidence="High",
    )


class TestPaperTradeFlow:
    def test_open_trade(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Will BTC exceed $100K by June 2026?")
        assert trade.status == "open"
        assert trade.side == "YES"
        assert trade.stake == 500.0
        assert trade.condition_id == "0xabc123"
        assert trade.underlier_group == "btc_price"

    def test_resolve_trade_win(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Will BTC exceed $100K?")
        resolved = store.resolve_trade(trade.id, "YES")
        assert resolved.status == "won"
        assert resolved.pnl > 0  # profit
        assert resolved.brier_score is not None
        assert resolved.brier_score < 0.25  # better than random

    def test_resolve_trade_loss(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Will BTC exceed $100K?")
        resolved = store.resolve_trade(trade.id, "NO")
        assert resolved.status == "lost"
        assert resolved.pnl == -500.0
        assert resolved.brier_score is not None

    def test_cannot_resolve_twice(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Will BTC exceed $100K?")
        store.resolve_trade(trade.id, "YES")
        with pytest.raises(ValueError, match="already"):
            store.resolve_trade(trade.id, "NO")

    def test_get_open_trades(self, store, sample_decision):
        store.open_trade(sample_decision, "Q1")
        store.open_trade(sample_decision, "Q2")
        assert len(store.get_open_trades()) == 2

    def test_correlated_count(self, store, sample_decision):
        store.open_trade(sample_decision, "Q1")
        store.open_trade(sample_decision, "Q2")
        assert store.get_correlated_count("btc_price") == 2
        assert store.get_correlated_count("eth_price") == 0

    def test_bankroll_snapshot_initial(self, store):
        snapshot = store.get_bankroll_snapshot()
        assert snapshot.current_bankroll == 10_000.0
        assert snapshot.open_positions == 0
        assert snapshot.total_trades == 0

    def test_bankroll_after_trades(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Q1")
        store.resolve_trade(trade.id, "YES")

        snapshot = store.get_bankroll_snapshot()
        assert snapshot.total_trades == 1
        assert snapshot.wins == 1
        assert snapshot.total_pnl > 0
        assert snapshot.win_rate == 1.0

    def test_create_bankroll_snapshot_persists(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Q1")
        store.resolve_trade(trade.id, "YES")

        snapshot = store.create_bankroll_snapshot()
        assert snapshot.total_pnl > 0

    def test_cancel_trade(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Q1")
        cancelled = store.cancel_trade(trade.id)
        assert cancelled.status == "cancelled"
        assert cancelled.pnl == 0.0

    def test_close_trade(self, store, sample_decision):
        """Early exit via close_trade() → status=closed, mark-to-market PnL."""
        trade = store.open_trade(sample_decision, "Q1")
        # Entry at 0.48, exit at 0.55 → profit
        closed = store.close_trade(trade.id, exit_price=0.55, reason="take_profit")
        assert closed.status == "closed"
        assert closed.exit_price == 0.55
        assert closed.exit_reason == "take_profit"
        assert closed.exit_time is not None
        assert closed.pnl > 0  # bought cheap, sold higher
        assert closed.brier_score is None  # no resolution yet

    def test_close_trade_stop_loss(self, store, sample_decision):
        """Early exit at a loss."""
        trade = store.open_trade(sample_decision, "Q1")
        closed = store.close_trade(trade.id, exit_price=0.35, reason="stop_loss")
        assert closed.status == "closed"
        assert closed.exit_reason == "stop_loss"
        assert closed.pnl < 0

    def test_record_resolution_on_closed(self, store, sample_decision):
        """Record market outcome on an already-closed trade for what-if analytics."""
        trade = store.open_trade(sample_decision, "Q1")
        closed = store.close_trade(trade.id, exit_price=0.55, reason="take_profit")
        original_pnl = closed.pnl

        updated = store.record_resolution(trade.id, "YES")
        assert updated.status == "closed"  # status unchanged
        assert updated.pnl == original_pnl  # PnL unchanged
        assert updated.resolved_outcome == "YES"
        assert updated.resolution_time is not None
        assert updated.brier_score is not None

    def test_compute_snapshot_includes_closed(self, store, sample_decision):
        """Closed trades should be included in total_pnl but not in resolution win_rate."""
        trade = store.open_trade(sample_decision, "Q1")
        store.close_trade(trade.id, exit_price=0.55, reason="take_profit")

        snapshot = store.get_bankroll_snapshot()
        assert snapshot.total_pnl != 0.0  # closed trade PnL counted
        assert snapshot.wins == 0  # not a resolution win
        assert snapshot.losses == 0  # not a resolution loss
        assert snapshot.win_rate == 0.0  # resolution-only win_rate

    def test_open_trade_saves_exit_policy(self, store, sample_decision):
        """Exit policy should be snapshot at entry."""
        trade = store.open_trade(sample_decision, "Q1")
        assert trade.take_profit_pct is not None
        assert trade.stop_loss_pct is not None
        assert trade.max_hold_seconds is not None

    def test_get_closed_without_resolution(self, store, sample_decision):
        trade = store.open_trade(sample_decision, "Q1")
        store.close_trade(trade.id, exit_price=0.55, reason="take_profit")

        unresolved = store.get_closed_without_resolution()
        assert len(unresolved) == 1
        assert unresolved[0].id == trade.id

        # After recording resolution, should be empty
        store.record_resolution(trade.id, "YES")
        assert len(store.get_closed_without_resolution()) == 0
