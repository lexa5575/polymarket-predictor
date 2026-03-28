"""Tests for app/monitor.py — position monitoring and exit logic."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from schemas.market import BetDecision
from storage.paper_trades import PaperTradeStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_decision(**overrides) -> BetDecision:
    defaults = dict(
        condition_id="0xmon", market_slug="btc-mon", token_id="tmon_yes",
        side="YES", action="BET", estimated_prob_of_side=0.60,
        market_prob_of_side_at_entry=0.48, edge=0.12,
        entry_price=0.50, slippage_estimate=0.01, stake=500.0,
        underlier_group="btc_price", rationale="Test",
        exit_conditions=[], confidence="High",
    )
    defaults.update(overrides)
    return BetDecision(**defaults)


def _make_orderbook(best_bid: float | None) -> str:
    if best_bid is None:
        return json.dumps({"bids": [], "asks": []})
    return json.dumps({
        "bids": [{"price": str(best_bid), "size": "1000"}],
        "asks": [{"price": str(best_bid + 0.02), "size": "1000"}],
    })


def _make_resolution(status: str = "active", outcome: str | None = None) -> str:
    d = {"status": status}
    if outcome:
        d["final_outcome"] = outcome
    return json.dumps(d)


class _FakePolymarket:
    """Mock PolymarketTools with configurable responses per token/condition."""

    def __init__(self, orderbooks=None, resolutions=None):
        self._orderbooks = orderbooks or {}
        self._resolutions = resolutions or {}

    def get_orderbook(self, token_id: str) -> str:
        return self._orderbooks.get(token_id, _make_orderbook(None))

    def get_market_resolution(self, condition_id: str) -> str:
        return self._resolutions.get(condition_id, _make_resolution("active"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return PaperTradeStore(f"sqlite:///{tmp_path / 'monitor.db'}")


class TestMonitor:
    def _run(self, store, polymarket, monkeypatch):
        """Helper to run monitor with mocked dependencies."""
        import app.monitor as _mod
        monkeypatch.setattr(_mod, "get_paper_trade_store", lambda: store)
        monkeypatch.setattr(_mod, "_polymarket_tools", polymarket)
        return _mod.run_monitor()

    def test_monitor_take_profit(self, store, monkeypatch):
        trade = store.open_trade(_sample_decision(), "Q")
        # Entry at 0.50, current best_bid at 0.60 → +20% → triggers TP (>=10%)
        pm = _FakePolymarket(
            orderbooks={trade.token_id: _make_orderbook(0.60)},
            resolutions={trade.condition_id: _make_resolution("active")},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 1
        assert result["trades_closed"][0]["reason"] == "take_profit"
        assert result["trades_closed"][0]["pnl"] > 0

    def test_monitor_stop_loss(self, store, monkeypatch):
        trade = store.open_trade(_sample_decision(), "Q")
        # Entry at 0.50, current 0.45 → -10% → triggers SL (<= -5%)
        pm = _FakePolymarket(
            orderbooks={trade.token_id: _make_orderbook(0.45)},
            resolutions={trade.condition_id: _make_resolution("active")},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 1
        assert result["trades_closed"][0]["reason"] == "stop_loss"
        assert result["trades_closed"][0]["pnl"] < 0

    def test_monitor_max_hold(self, store, monkeypatch):
        trade = store.open_trade(_sample_decision(), "Q")
        # Make trade old (2 hours > 30 min max_hold)
        with store._session_factory() as session:
            from storage.tables import PaperTradeRow
            row = session.get(PaperTradeRow, trade.id)
            row.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
            session.commit()

        # Price is close to entry (no TP/SL) but old → max_hold
        pm = _FakePolymarket(
            orderbooks={trade.token_id: _make_orderbook(0.51)},
            resolutions={trade.condition_id: _make_resolution("active")},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 1
        assert result["trades_closed"][0]["reason"] == "max_hold"

    def test_monitor_resolution_before_orderbook(self, store, monkeypatch):
        """Resolved market with NO bids → resolve_trade() called, not blocked."""
        trade = store.open_trade(_sample_decision(), "Q")
        pm = _FakePolymarket(
            orderbooks={},  # no bids at all
            resolutions={trade.condition_id: _make_resolution("resolved", "YES")},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 1
        assert result["trades_closed"][0]["reason"] == "resolution"
        assert result["trades_closed"][0]["outcome"] == "YES"

    def test_monitor_resolved_ambiguous_outcome_skip(self, store, monkeypatch):
        """Resolved market with ambiguous outcome → skip entirely (fail-closed), no early exit."""
        trade = store.open_trade(_sample_decision(), "Q")
        # High price would trigger TP, but market is resolved with no clear outcome
        pm = _FakePolymarket(
            orderbooks={trade.token_id: _make_orderbook(0.60)},
            resolutions={trade.condition_id: _make_resolution("resolved", None)},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 0  # NOT closed as take_profit
        assert any("ambiguous" in (t.get("warning") or "") for t in result["trades_open"])

    def test_monitor_no_bids_skip(self, store, monkeypatch):
        """No bids, not resolved → stays open with warning."""
        trade = store.open_trade(_sample_decision(), "Q")
        pm = _FakePolymarket(
            orderbooks={trade.token_id: _make_orderbook(None)},
            resolutions={trade.condition_id: _make_resolution("active")},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 0
        assert any("no bids" in (t.get("warning") or "") for t in result["trades_open"])

    def test_monitor_records_resolution_for_closed(self, store, monkeypatch):
        """Closed trade + later market resolved → record_resolution() called."""
        trade = store.open_trade(_sample_decision(), "Q")
        store.close_trade(trade.id, exit_price=0.55, reason="take_profit")

        pm = _FakePolymarket(
            resolutions={trade.condition_id: _make_resolution("resolved", "YES")},
        )
        self._run(store, pm, monkeypatch)

        # Check that resolution was recorded
        updated = store.get_all_trades()
        assert updated[0].resolved_outcome == "YES"
        assert updated[0].brier_score is not None
        assert updated[0].status == "closed"  # status unchanged

    def test_monitor_legacy_null_policy(self, store, monkeypatch):
        """Trade with NULL exit policy fields → uses defaults."""
        trade = store.open_trade(_sample_decision(), "Q")
        # Simulate legacy trade: NULL out policy fields
        with store._session_factory() as session:
            from storage.tables import PaperTradeRow
            row = session.get(PaperTradeRow, trade.id)
            row.take_profit_pct = None
            row.stop_loss_pct = None
            row.max_hold_seconds = None
            session.commit()

        # High price → should still trigger TP using default 10%
        pm = _FakePolymarket(
            orderbooks={trade.token_id: _make_orderbook(0.60)},
            resolutions={trade.condition_id: _make_resolution("active")},
        )
        result = self._run(store, pm, monkeypatch)
        assert result["closed"] == 1
        assert result["trades_closed"][0]["reason"] == "take_profit"

    def test_monitor_api_error_isolation(self, store, monkeypatch):
        """One trade errors on API call → other trades still processed."""
        t1 = store.open_trade(_sample_decision(token_id="tok1", condition_id="0x1"), "Q1")
        t2 = store.open_trade(_sample_decision(token_id="tok2", condition_id="0x2"), "Q2")

        class _BrokenPolymarket:
            def get_market_resolution(self, cid):
                if cid == "0x1":
                    raise RuntimeError("API down")
                return _make_resolution("active")

            def get_orderbook(self, tid):
                return _make_orderbook(0.60)  # triggers TP

        result = self._run(store, _BrokenPolymarket(), monkeypatch)
        # t1 errored, t2 should still close
        assert result["closed"] == 1
        assert any(t.get("error") is not None for t in result["trades_open"])
