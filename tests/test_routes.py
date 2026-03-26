"""Tests for custom API routes.

Includes:
- Direct store logic tests (no HTTP, for dashboard/settle flow)
- FastAPI TestClient tests for /api/dashboard and /api/settle
"""

import json

import pytest

from schemas.market import BetDecision
from storage.paper_trades import PaperTradeStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_routes.db"
    return PaperTradeStore(f"sqlite:///{db_path}")


@pytest.fixture
def sample_bet_yes():
    return BetDecision(
        condition_id="0xroute_yes",
        market_slug="btc-100k-route-test",
        token_id="token_yes_route",
        side="YES",
        action="BET",
        estimated_prob_of_side=0.62,
        market_prob_of_side_at_entry=0.50,
        edge=0.12,
        entry_price=0.51,
        slippage_estimate=0.01,
        stake=400.0,
        underlier_group="btc_price",
        rationale="Route test YES",
        exit_conditions=["Exit if prob < 0.35"],
        confidence="High",
    )


@pytest.fixture
def sample_bet_no():
    return BetDecision(
        condition_id="0xroute_no",
        market_slug="btc-100k-route-no",
        token_id="token_no_route",
        side="NO",
        action="BET",
        estimated_prob_of_side=0.68,
        market_prob_of_side_at_entry=0.55,
        edge=0.13,
        entry_price=0.42,
        slippage_estimate=0.005,
        stake=300.0,
        underlier_group="btc_price",
        rationale="Route test NO",
        exit_conditions=[],
        confidence="Medium",
    )


# ---------------------------------------------------------------------------
# Direct store logic tests
# ---------------------------------------------------------------------------


class TestDashboardLogic:
    def test_empty_dashboard(self, store):
        snapshot = store.get_bankroll_snapshot()
        assert snapshot.current_bankroll == 10_000.0
        assert snapshot.open_positions == 0

    def test_with_open_trades(self, store, sample_bet_yes):
        store.open_trade(sample_bet_yes, "Will BTC exceed $100K?")
        snapshot = store.get_bankroll_snapshot()
        assert snapshot.open_positions == 1
        assert snapshot.total_at_risk == 400.0

    def test_after_resolution(self, store, sample_bet_yes):
        trade = store.open_trade(sample_bet_yes, "Will BTC exceed $100K?")
        store.resolve_trade(trade.id, "YES")
        snapshot = store.get_bankroll_snapshot()
        assert snapshot.wins == 1
        assert snapshot.total_pnl > 0


class TestSettleLogic:
    def test_settle_win(self, store, sample_bet_yes):
        trade = store.open_trade(sample_bet_yes, "Q1")
        resolved = store.resolve_trade(trade.id, "YES")
        assert resolved.status == "won"
        assert resolved.pnl > 0

        snapshot = store.create_bankroll_snapshot()
        assert snapshot.wins == 1

    def test_settle_loss(self, store, sample_bet_yes):
        trade = store.open_trade(sample_bet_yes, "Q1")
        resolved = store.resolve_trade(trade.id, "NO")
        assert resolved.status == "lost"
        assert resolved.pnl == -400.0

    def test_settle_no_side_win(self, store, sample_bet_no):
        """NO-side trade wins when outcome is NO."""
        trade = store.open_trade(sample_bet_no, "Will BTC exceed $100K?")
        resolved = store.resolve_trade(trade.id, "NO")
        assert resolved.status == "won"
        assert resolved.pnl > 0

    def test_settle_no_side_loss(self, store, sample_bet_no):
        """NO-side trade loses when outcome is YES."""
        trade = store.open_trade(sample_bet_no, "Will BTC exceed $100K?")
        resolved = store.resolve_trade(trade.id, "YES")
        assert resolved.status == "lost"
        assert resolved.pnl == -300.0

    def test_no_open_trades(self, store):
        assert len(store.get_open_trades()) == 0


# ---------------------------------------------------------------------------
# FastAPI TestClient tests
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client(tmp_path, monkeypatch):
    """Create a FastAPI TestClient with a mocked PaperTradeStore."""
    store = PaperTradeStore(f"sqlite:///{tmp_path / 'api_test.db'}")
    monkeypatch.setattr("app.routes.get_paper_trade_store", lambda: store)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.routes import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app), store


class TestDashboardRoute:
    def test_empty_dashboard(self, test_client):
        client, _ = test_client
        resp = client.get("/api/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bankroll"]["current_bankroll"] == 10_000.0
        assert data["bankroll"]["open_positions"] == 0
        assert data["open_positions"] == []

    def test_dashboard_with_trade(self, test_client, sample_bet_yes):
        client, store = test_client
        store.open_trade(sample_bet_yes, "BTC test")

        resp = client.get("/api/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bankroll"]["open_positions"] == 1
        assert len(data["open_positions"]) == 1
        assert data["open_positions"][0]["side"] == "YES"
        assert data["open_positions"][0]["stake"] == 400.0


class TestSettleRoute:
    def test_settle_no_trades(self, test_client):
        client, _ = test_client
        resp = client.post("/api/settle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["settled"] == 0
        assert data["message"] == "No open trades to settle"

    def test_settle_with_resolved_trade(self, test_client, sample_bet_yes, monkeypatch):
        """Settle a trade by mocking Polymarket resolution API."""
        client, store = test_client
        trade = store.open_trade(sample_bet_yes, "BTC test")

        # Mock get_market_resolution to return resolved YES
        monkeypatch.setattr(
            "app.routes._polymarket_tools",
            type("FakeTools", (), {
                "get_market_resolution": lambda self, cid: json.dumps({
                    "status": "resolved",
                    "final_outcome": "YES",
                    "condition_id": cid,
                })
            })(),
        )

        resp = client.post("/api/settle")
        assert resp.status_code == 200
        data = resp.json()
        assert data["settled"] == 1
        assert data["trades"][0]["won"] is True
        assert data["trades"][0]["pnl"] > 0
        assert "bankroll" in data


class TestScanAndFanoutRoute:
    def test_scan_returns_results(self, test_client, monkeypatch):
        """Test /api/scan-and-fanout by monkeypatching the real dependencies it imports."""
        client, store = test_client

        from workflows.prediction_workflow import _emit_tagged_block

        # Mock polymarket_scanner (lazy-imported inside the route function)
        fake_candidate = type("C", (), {
            "condition_id": "0xscan1",
            "question": "Will BTC hit $100K?",
        })()
        fake_scan_content = type("Content", (), {"candidates": [fake_candidate]})()
        fake_scan_result = type("R", (), {"content": fake_scan_content})()

        # Mock prediction_workflow (lazy-imported inside the route function)
        record_block = _emit_tagged_block("RECORD_RESULT", {
            "action": "BET",
            "trade_id": "fake-trade-id",
            "side": "YES",
            "stake": 300,
        })
        fake_wf_result = type("WF", (), {"content": record_block})()

        # Patch the modules that the route function imports at call time
        # The route does: from agents import polymarket_scanner
        #                 from workflows import prediction_workflow
        import agents
        import workflows
        monkeypatch.setattr(
            agents, "polymarket_scanner",
            type("Scanner", (), {"run": lambda self, msg: fake_scan_result})(),
        )
        monkeypatch.setattr(
            workflows, "prediction_workflow",
            type("Workflow", (), {"run": lambda self, **kw: fake_wf_result})(),
        )

        resp = client.post("/api/scan-and-fanout")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scan_completed"] is True
        assert data["candidates_processed"] == 1
        assert data["results"][0]["action"] == "BET"
        assert data["results"][0]["trade_id"] == "fake-trade-id"
        assert data["results"][0]["condition_id"] == "0xscan1"

    def test_scan_no_candidates(self, test_client, monkeypatch):
        """If scanner returns no candidates, route should report that."""
        client, _ = test_client

        fake_scan_content = type("Content", (), {"candidates": []})()
        fake_scan_result = type("R", (), {"content": fake_scan_content})()

        import agents
        monkeypatch.setattr(
            agents, "polymarket_scanner",
            type("Scanner", (), {"run": lambda self, msg: fake_scan_result})(),
        )

        resp = client.post("/api/scan-and-fanout")
        assert resp.status_code == 200
        data = resp.json()
        assert data["candidates_processed"] == 0
