"""Tests for workflow function steps with typed step-to-step handoff.

Tests verify that function steps correctly read typed StepOutput.content
(Pydantic models and dicts) without text/JSON parsing.
"""

import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from agno.workflow.types import StepInput, StepOutput

from schemas.market import (
    BetDecision,
    EventCandidate,
    MarketSnapshot,
    RiskAssessment,
    RiskEstimate,
    SentimentReport,
    TokenBook,
)
from schemas.paper_trade import BankrollSnapshot
import sys

from workflows.prediction_workflow import (
    _safe_bet_decision,
    _safe_risk_assessment,
    _step_content_to_dict,
    _step_content_to_model,
    build_decision,
    compute_edge_and_gate,
    compute_position_sizing,
    conditional_logging,
    ensure_data_quality,
    run_risk_assessment,
)

# workflows/__init__.py shadows the module name with the Workflow object,
# so monkeypatch("workflows.prediction_workflow.X") fails. Get the real module.
_wf_mod = sys.modules["workflows.prediction_workflow"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_token_book(**overrides) -> dict:
    defaults = {"token_id": "t1", "best_bid": 0.49, "best_ask": 0.51, "spread": 0.02, "depth_10pct": 20000.0}
    defaults.update(overrides)
    return defaults


def _make_event_candidate(**overrides) -> EventCandidate:
    defaults = dict(
        gamma_market_id="gm1",
        condition_id="0xtest",
        market_slug="btc-test",
        question="Will BTC exceed $100K?",
        category="crypto",
        end_date="2026-06-30T00:00:00Z",
        yes_book=TokenBook(**_make_token_book(token_id="tyes")),
        no_book=TokenBook(**_make_token_book(token_id="tno", best_bid=0.48, best_ask=0.50)),
        market_prob_yes=0.51,
        volume_24h=50000.0,
        total_liquidity=40000.0,
    )
    defaults.update(overrides)
    return EventCandidate(**defaults)


def _make_market_snapshot(**overrides) -> MarketSnapshot:
    defaults = dict(
        coin_id="bitcoin", price_usd=67000.0, change_24h_pct=2.5,
        market_cap=1300000000000, fear_greed_index=65,
        fear_greed_label="Greed", signal="Bullish",
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


def _make_sentiment_report(**overrides) -> SentimentReport:
    defaults = dict(
        query="btc 100k", sentiment_score=0.4,
        key_narratives=["Bullish momentum"], sources_count=5, confidence=0.7,
    )
    defaults.update(overrides)
    return SentimentReport(**defaults)


def _make_risk_assessment(**overrides) -> RiskAssessment:
    defaults = dict(
        condition_id="0xtest", risk_rating="Moderate", recommended_side="YES",
        estimated_prob_of_side=0.65, market_prob_of_side=0.51, edge=0.14,
        underlier_group="btc_price", warnings=[], liquidity_ok=True, correlated_positions=0,
    )
    defaults.update(overrides)
    return RiskAssessment(**defaults)


def _make_risk_estimate(**overrides) -> RiskEstimate:
    defaults = dict(
        condition_id="0xtest", recommended_side="YES",
        estimated_prob_of_side=0.65, confidence="High",
        underlier_group="btc_price", reasoning="Test reasoning",
        warnings=[],
    )
    defaults.update(overrides)
    return RiskEstimate(**defaults)


def _make_snapshot(**overrides) -> BankrollSnapshot:
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        starting_bankroll=10_000.0,
        current_bankroll=7_000.0,
        open_positions=2,
        total_at_risk=3_000.0,
        total_trades=5,
        wins=2,
        losses=1,
        win_rate=2 / 3,
        total_pnl=0.0,
        avg_brier_score=0.1,
        sharpe_ratio=None,
    )
    defaults.update(overrides)
    return BankrollSnapshot(**defaults)


def _fake_store(snapshot=None, correlated=0):
    """Create a fake store for monkeypatching get_paper_trade_store."""
    snap = snapshot or _make_snapshot()

    class FakeStore:
        def get_bankroll_snapshot(self):
            return snap

        def get_correlated_count(self, group):
            return correlated

    return FakeStore()


def _make_step_input(**step_outputs) -> StepInput:
    """Create StepInput with named step outputs."""
    outputs = {}
    for name, content in step_outputs.items():
        outputs[name] = StepOutput(step_name=name, content=content)
    return StepInput(previous_step_outputs=outputs)


# ---------------------------------------------------------------------------
# _build_token_book tests (orderbook sort order)
# ---------------------------------------------------------------------------


class TestBuildTokenBook:
    """Regression: Polymarket CLOB returns bids/asks in unsorted order."""

    def test_unsorted_bids_asks(self):
        """bids lowest-first, asks highest-first → must still get correct spread."""
        from workflows.prediction_workflow import _build_token_book

        book = {
            "bids": [
                {"price": "0.01", "size": "100"},
                {"price": "0.10", "size": "200"},
                {"price": "0.42", "size": "500"},
            ],
            "asks": [
                {"price": "0.99", "size": "100"},
                {"price": "0.50", "size": "200"},
                {"price": "0.43", "size": "500"},
            ],
        }
        result = _build_token_book(book, "test-token")
        assert result["best_bid"] == 0.42
        assert result["best_ask"] == 0.43
        assert result["spread"] == pytest.approx(0.01, abs=0.001)
        assert result["token_id"] == "test-token"

    def test_already_sorted(self):
        """Already sorted orderbook should also work."""
        from workflows.prediction_workflow import _build_token_book

        book = {
            "bids": [{"price": "0.50", "size": "100"}, {"price": "0.49", "size": "100"}],
            "asks": [{"price": "0.51", "size": "100"}, {"price": "0.52", "size": "100"}],
        }
        result = _build_token_book(book, "tok")
        assert result["best_bid"] == 0.50
        assert result["best_ask"] == 0.51
        assert result["spread"] == pytest.approx(0.01)

    def test_empty_orderbook(self):
        from workflows.prediction_workflow import _build_token_book

        result = _build_token_book({"bids": [], "asks": []}, "tok")
        assert result["best_bid"] == 0.0
        assert result["best_ask"] == 0.0


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestStepContentToDict:
    def test_pydantic_model(self):
        ms = _make_market_snapshot()
        result = _step_content_to_dict(StepOutput(content=ms))
        assert result is not None
        assert result["coin_id"] == "bitcoin"
        assert result["price_usd"] == 67000.0

    def test_dict_passthrough(self):
        d = {"force_skip": True, "sizing_note": "test"}
        assert _step_content_to_dict(StepOutput(content=d)) == d

    def test_json_string(self):
        result = _step_content_to_dict(StepOutput(content='{"key": "value"}'))
        assert result == {"key": "value"}

    def test_none(self):
        assert _step_content_to_dict(None) is None
        assert _step_content_to_dict(StepOutput(content=None)) is None


class TestStepContentToModel:
    def test_correct_model(self):
        ms = _make_market_snapshot()
        result = _step_content_to_model(StepOutput(content=ms), MarketSnapshot)
        assert result is not None
        assert isinstance(result, MarketSnapshot)
        assert result.coin_id == "bitcoin"

    def test_wrong_model_returns_none(self):
        ms = _make_market_snapshot()
        result = _step_content_to_model(StepOutput(content=ms), SentimentReport)
        assert result is None

    def test_none_returns_none(self):
        assert _step_content_to_model(None, MarketSnapshot) is None


# ---------------------------------------------------------------------------
# ensure_data_quality tests
# ---------------------------------------------------------------------------


class TestEnsureDataQuality:
    def test_all_present(self):
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Market Data": _make_market_snapshot(),
            "News & Sentiment": _make_sentiment_report(),
        })
        result = _step_content_to_dict(ensure_data_quality(si))
        assert result["force_skip"] is False
        assert result["event_missing"] is False
        assert result["market_data_missing"] is False
        assert result["sentiment_missing"] is False

    def test_missing_market(self):
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "News & Sentiment": _make_sentiment_report(),
        })
        result = _step_content_to_dict(ensure_data_quality(si))
        assert result["force_skip"] is True
        assert result["market_data_missing"] is True
        assert "MarketSnapshot" in result.get("skip_reason", "")

    def test_missing_event(self):
        si = _make_step_input(**{
            "Market Data": _make_market_snapshot(),
            "News & Sentiment": _make_sentiment_report(),
        })
        result = _step_content_to_dict(ensure_data_quality(si))
        assert result["force_skip"] is True
        assert result["event_missing"] is True

    def test_missing_sentiment_injects_fallback(self):
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Market Data": _make_market_snapshot(),
        })
        result = _step_content_to_dict(ensure_data_quality(si))
        assert result["force_skip"] is False
        assert result["sentiment_missing"] is True
        assert result["sentiment_fallback"]["sentiment_score"] == 0.0

    def test_invalid_market_schema_forces_skip(self):
        """Dict with wrong keys is not a valid MarketSnapshot."""
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Market Data": StepOutput(content={"wrong_key": 123}),  # not MarketSnapshot
            "News & Sentiment": _make_sentiment_report(),
        })
        # _step_content_to_model will return None for invalid schema
        result = _step_content_to_dict(ensure_data_quality(si))
        assert result["force_skip"] is True

    def test_invalid_sentiment_injects_fallback(self):
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Market Data": _make_market_snapshot(),
            "News & Sentiment": StepOutput(content={"wrong_key": 123}),
        })
        result = _step_content_to_dict(ensure_data_quality(si))
        assert result["force_skip"] is False
        assert result["sentiment_missing"] is True
        assert "sentiment_fallback" in result


# ---------------------------------------------------------------------------
# run_risk_assessment tests
# ---------------------------------------------------------------------------


class TestRunRiskAssessment:
    def test_force_skip(self):
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Data Quality": {"force_skip": True, "skip_reason": "Market data missing"},
        })
        result = _step_content_to_model(run_risk_assessment(si), RiskEstimate)
        assert result is not None
        assert result.confidence == "Low"
        assert "Market data" in result.warnings[0]

    def test_missing_event_returns_safe(self):
        si = _make_step_input(**{"Data Quality": {"force_skip": False}})
        result = _step_content_to_model(run_risk_assessment(si), RiskEstimate)
        assert result is not None
        assert result.confidence == "Low"

    def test_agent_exception_returns_safe(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "risk_agent",
            type("Fake", (), {"run": lambda self, msg: (_ for _ in ()).throw(RuntimeError("test"))})(),
        )
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Market Data": _make_market_snapshot(),
            "Data Quality": {"force_skip": False},
        })
        result = _step_content_to_model(run_risk_assessment(si), RiskEstimate)
        assert result is not None
        assert result.confidence == "Low"
        assert "exception" in result.warnings[0].lower()


# ---------------------------------------------------------------------------
# compute_edge_and_gate tests
# ---------------------------------------------------------------------------


class TestComputeEdgeAndGate:
    def _gate(self, si) -> RiskAssessment:
        return _step_content_to_model(compute_edge_and_gate(si), RiskAssessment)

    def test_positive_edge(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=0),
        )
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result is not None
        assert result.edge == pytest.approx(0.14, abs=0.01)
        assert result.risk_rating == "Low"
        assert result.liquidity_ok is True
        assert result.correlated_positions == 0

    def test_no_edge(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=0),
        )
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(estimated_prob_of_side=0.40),
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"
        assert any("edge" in w.lower() for w in result.warnings)

    def test_low_depth(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=0),
        )
        event = _make_event_candidate()
        event.yes_book = TokenBook(token_id="tyes", best_bid=0.49, best_ask=0.51, spread=0.02, depth_10pct=5000.0)
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": event,
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"
        assert result.liquidity_ok is False
        assert any("depth" in w.lower() for w in result.warnings)

    def test_low_volume(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=0),
        )
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(volume_24h=20_000.0),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"
        assert any("volume" in w.lower() for w in result.warnings)

    def test_wide_spread(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=0),
        )
        event = _make_event_candidate()
        event.yes_book = TokenBook(token_id="tyes", best_bid=0.45, best_ask=0.51, spread=0.06, depth_10pct=20000.0)
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": event,
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"
        assert any("spread" in w.lower() for w in result.warnings)

    def test_too_many_correlated(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=3),
        )
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"

    def test_force_skip_passthrough(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": True, "skip_reason": "Missing data"},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"

    def test_missing_estimate(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"

    def test_condition_id_mismatch(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(correlated=0),
        )
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(condition_id="0xwrong"),
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"
        assert any("mismatch" in w.lower() for w in result.warnings)

    def test_db_error_fail_closed(self, monkeypatch):
        def _raise():
            raise RuntimeError("db down")
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: type("Bad", (), {"get_correlated_count": lambda self, g: (_ for _ in ()).throw(RuntimeError("db down"))})(),
        )
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = self._gate(si)
        assert result.risk_rating == "Unacceptable"
        assert any("db error" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# compute_position_sizing tests
# ---------------------------------------------------------------------------


class TestComputePositionSizing:
    def _sizing(self, si) -> dict:
        return _step_content_to_dict(compute_position_sizing(si))

    def test_positive_edge(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Edge & Gate": _make_risk_assessment(),
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is False
        assert result["recommended_stake"] > 0

    def test_dq_force_skip(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": True, "skip_reason": "Missing data"},
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_missing_risk_force_skip(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True
        assert "Risk" in result.get("sizing_note", "")

    def test_missing_event_force_skip(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Edge & Gate": _make_risk_assessment(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_no_edge_force_skip(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Edge & Gate": _make_risk_assessment(estimated_prob_of_side=0.40, market_prob_of_side=0.50, edge=-0.1),
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_missing_orderbook_force_skip(self):
        event = _make_event_candidate()
        event.yes_book = TokenBook(token_id="t1", best_bid=0, best_ask=0, spread=0, depth_10pct=0)
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Edge & Gate": _make_risk_assessment(),
            "Event Scan": event,
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_zero_prob_not_treated_as_missing(self):
        """estimated_prob=0.0 is a valid value (not None)."""
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Edge & Gate": _make_risk_assessment(estimated_prob_of_side=0.0),
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        # Should reach edge check (0.0 <= 0.51), not "missing" check
        assert result["force_skip"] is True
        assert "edge" in result.get("sizing_note", "").lower() or "No positive" in result.get("sizing_note", "")

    def test_invalid_recommended_side_force_skip(self):
        # Literal["YES","NO"] prevents creating RiskAssessment with "MAYBE".
        # A dict with invalid recommended_side will fail model validation,
        # causing sizing to see None → force_skip with "Risk data missing".
        invalid_risk = _make_risk_assessment().model_dump()
        invalid_risk["recommended_side"] = "MAYBE"
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Edge & Gate": invalid_risk,
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True


# ---------------------------------------------------------------------------
# build_decision tests
# ---------------------------------------------------------------------------


def _sizing_ok(**overrides) -> dict:
    """Default sizing dict for build_decision tests."""
    defaults = {
        "force_skip": False,
        "recommended_stake": 500.0,
        "entry_price": 0.52,
        "slippage_estimate": 0.01,
        "bankroll": 7_000.0,
    }
    defaults.update(overrides)
    return defaults


class TestBuildDecision:
    def test_bet_when_all_ok(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result is not None
        assert result.action == "BET"
        assert result.stake == 500.0
        assert result.edge == pytest.approx(0.14)

    def test_skip_when_unacceptable(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(risk_rating="Unacceptable", warnings=["Edge too low"]),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert result.stake == 0.0
        assert "unacceptable" in result.rationale.lower()

    def test_skip_when_low_confidence(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(confidence="Low"),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert "confidence" in result.rationale.lower()

    def test_skip_when_sizing_force_skip(self):
        si = _make_step_input(**{
            "Position Sizing": {"force_skip": True, "sizing_note": "No edge"},
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result is not None
        assert result.action == "SKIP"
        assert result.stake == 0.0

    def test_skip_circuit_breaker(self, monkeypatch):
        snap = _make_snapshot(current_bankroll=1_000.0, total_at_risk=3_000.0)
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(snapshot=snap),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert "circuit breaker" in result.rationale.lower()

    def test_skip_max_positions(self, monkeypatch):
        snap = _make_snapshot(open_positions=10)
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(snapshot=snap),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert "positions" in result.rationale.lower()

    def test_skip_capital_at_risk(self, monkeypatch):
        # equity=10K, at_risk=5K, stake=2K → 7K > 6K (60%)
        snap = _make_snapshot(current_bankroll=5_000.0, total_at_risk=5_000.0)
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(snapshot=snap),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(recommended_stake=2_000.0),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert "60%" in result.rationale

    def test_skip_reserve_breach(self, monkeypatch):
        # current_bankroll=1.2K (available cash), stake=500 → remaining=700 < 1K reserve
        snap = _make_snapshot(current_bankroll=1_200.0, total_at_risk=8_800.0)
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(snapshot=snap),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert "reserve" in result.rationale.lower()

    def test_bet_fields_populated(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "BET"
        assert result.stake > 0
        assert result.entry_price > 0
        assert result.token_id != ""
        assert len(result.exit_conditions) > 0
        assert result.side == "YES"
        assert result.underlier_group == "btc_price"

    def test_skip_fields_zeroed(self, monkeypatch):
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: _fake_store(),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(risk_rating="Unacceptable"),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert result.stake == 0.0
        assert result.entry_price == 0.0
        assert result.token_id == ""
        assert result.exit_conditions == []

    def test_portfolio_store_error_fail_closed(self, monkeypatch):
        def _explode():
            raise RuntimeError("db down")
        monkeypatch.setattr(
            _wf_mod, "get_paper_trade_store",
            lambda: type("Bad", (), {"get_bankroll_snapshot": lambda self: _explode()})(),
        )
        si = _make_step_input(**{
            "Position Sizing": _sizing_ok(),
            "Edge & Gate": _make_risk_assessment(),
            "Risk Assessment": _make_risk_estimate(),
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_model(build_decision(si), BetDecision)
        assert result.action == "SKIP"
        assert "portfolio state" in result.rationale.lower()


# ---------------------------------------------------------------------------
# conditional_logging tests
# ---------------------------------------------------------------------------


class TestConditionalLogging:
    def test_force_skip_from_sizing(self):
        si = _make_step_input(**{
            "Position Sizing": {"force_skip": True, "sizing_note": "No edge"},
            "Decision": _safe_bet_decision(),
        })
        result = _step_content_to_dict(conditional_logging(si))
        assert result["action"] == "SKIP"

    def test_skip_decision(self):
        si = _make_step_input(**{
            "Position Sizing": {"force_skip": False},
            "Decision": _safe_bet_decision(rationale="No edge detected"),
        })
        result = _step_content_to_dict(conditional_logging(si))
        assert result["action"] == "SKIP"

    def test_bet_records_trade(self, tmp_path, monkeypatch):
        from storage.paper_trades import PaperTradeStore

        store = PaperTradeStore(f"sqlite:///{tmp_path / 'test.db'}")
        monkeypatch.setattr(_wf_mod, "get_paper_trade_store", lambda: store)
        monkeypatch.setattr(
            _wf_mod, "logger_agent",
            type("Fake", (), {"run": lambda self, msg: None})(),
        )

        decision = BetDecision(
            condition_id="0xbet", market_slug="btc-bet", token_id="tyes",
            side="YES", action="BET", estimated_prob_of_side=0.65,
            market_prob_of_side_at_entry=0.51, edge=0.14,
            entry_price=0.52, slippage_estimate=0.01, stake=500.0,
            underlier_group="btc_price", rationale="Strong edge",
            exit_conditions=[], confidence="High",
        )
        si = _make_step_input(**{
            "Position Sizing": {"force_skip": False},
            "Decision": decision,
            "Event Scan": _make_event_candidate(),
        })
        result = _step_content_to_dict(conditional_logging(si))
        assert result["action"] == "BET"
        assert result["trade_id"] is not None
        assert store.get_open_trades()[0].side == "YES"

    def test_force_skip_overrides_bet(self, tmp_path, monkeypatch):
        """Even if Decision says BET, force_skip from sizing wins."""
        from storage.paper_trades import PaperTradeStore

        store = PaperTradeStore(f"sqlite:///{tmp_path / 'test.db'}")
        monkeypatch.setattr(_wf_mod, "get_paper_trade_store", lambda: store)

        decision = BetDecision(
            condition_id="0xbet", market_slug="btc-bet", token_id="tyes",
            side="YES", action="BET", estimated_prob_of_side=0.65,
            market_prob_of_side_at_entry=0.51, edge=0.14,
            entry_price=0.52, slippage_estimate=0.01, stake=500.0,
            underlier_group="btc_price", rationale="Strong edge",
            exit_conditions=[], confidence="High",
        )
        si = _make_step_input(**{
            "Position Sizing": {"force_skip": True, "sizing_note": "Slippage too high"},
            "Decision": decision,
        })
        result = _step_content_to_dict(conditional_logging(si))
        assert result["action"] == "SKIP"
        assert len(store.get_open_trades()) == 0

    def test_no_decision(self):
        si = _make_step_input(**{"Position Sizing": {"force_skip": False}})
        result = _step_content_to_dict(conditional_logging(si))
        assert result["action"] == "SKIP"
        assert "No valid" in result["reason"]
