"""Tests for workflow function steps with typed step-to-step handoff.

Tests verify that function steps correctly read typed StepOutput.content
(Pydantic models and dicts) without text/JSON parsing.
"""

import json

import pytest
from pydantic import BaseModel

from agno.workflow.types import StepInput, StepOutput

from schemas.market import (
    BetDecision,
    EventCandidate,
    MarketSnapshot,
    RiskAssessment,
    SentimentReport,
    TokenBook,
)
from workflows.prediction_workflow import (
    _safe_bet_decision,
    _safe_risk_assessment,
    _step_content_to_dict,
    _step_content_to_model,
    compute_position_sizing,
    conditional_logging,
    ensure_data_quality,
    run_decision,
    run_risk_assessment,
)


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


def _make_step_input(**step_outputs) -> StepInput:
    """Create StepInput with named step outputs."""
    outputs = {}
    for name, content in step_outputs.items():
        outputs[name] = StepOutput(step_name=name, content=content)
    return StepInput(previous_step_outputs=outputs)


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
        result = _step_content_to_model(run_risk_assessment(si), RiskAssessment)
        assert result is not None
        assert result.risk_rating == "Unacceptable"
        assert "Market data" in result.warnings[0]

    def test_missing_event_returns_safe(self):
        si = _make_step_input(**{"Data Quality": {"force_skip": False}})
        result = _step_content_to_model(run_risk_assessment(si), RiskAssessment)
        assert result is not None
        assert result.risk_rating == "Unacceptable"

    def test_agent_exception_returns_safe(self, monkeypatch):
        monkeypatch.setattr(
            "workflows.prediction_workflow.risk_agent",
            type("Fake", (), {"run": lambda self, msg: (_ for _ in ()).throw(RuntimeError("test"))})(),
        )
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Market Data": _make_market_snapshot(),
            "Data Quality": {"force_skip": False},
        })
        result = _step_content_to_model(run_risk_assessment(si), RiskAssessment)
        assert result is not None
        assert result.risk_rating == "Unacceptable"
        assert "exception" in result.warnings[0].lower()


# ---------------------------------------------------------------------------
# compute_position_sizing tests
# ---------------------------------------------------------------------------


class TestComputePositionSizing:
    def _sizing(self, si) -> dict:
        return _step_content_to_dict(compute_position_sizing(si))

    def test_positive_edge(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_assessment(),
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
            "Risk Assessment": _make_risk_assessment(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_no_edge_force_skip(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_assessment(estimated_prob_of_side=0.40, market_prob_of_side=0.50, edge=-0.1),
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_missing_orderbook_force_skip(self):
        event = _make_event_candidate()
        event.yes_book = TokenBook(token_id="t1", best_bid=0, best_ask=0, spread=0, depth_10pct=0)
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_assessment(),
            "Event Scan": event,
        })
        result = self._sizing(si)
        assert result["force_skip"] is True

    def test_zero_prob_not_treated_as_missing(self):
        """estimated_prob=0.0 is a valid value (not None)."""
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_assessment(estimated_prob_of_side=0.0),
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        # Should reach edge check (0.0 <= 0.51), not "missing" check
        assert result["force_skip"] is True
        assert "edge" in result.get("sizing_note", "").lower() or "No positive" in result.get("sizing_note", "")

    def test_invalid_recommended_side_force_skip(self):
        si = _make_step_input(**{
            "Data Quality": {"force_skip": False},
            "Risk Assessment": _make_risk_assessment(recommended_side="MAYBE"),
            "Event Scan": _make_event_candidate(),
        })
        result = self._sizing(si)
        assert result["force_skip"] is True
        assert "recommended_side" in result.get("sizing_note", "")


# ---------------------------------------------------------------------------
# run_decision tests
# ---------------------------------------------------------------------------


class TestRunDecision:
    def test_force_skip_from_sizing(self):
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Position Sizing": {"force_skip": True, "sizing_note": "No edge"},
        })
        result = _step_content_to_model(run_decision(si), BetDecision)
        assert result is not None
        assert result.action == "SKIP"
        assert result.stake == 0.0

    def test_missing_event_returns_skip(self):
        si = _make_step_input(**{
            "Position Sizing": {"force_skip": False, "recommended_stake": 500},
        })
        result = _step_content_to_model(run_decision(si), BetDecision)
        assert result is not None
        assert result.action == "SKIP"

    def test_agent_exception_returns_skip(self, monkeypatch):
        monkeypatch.setattr(
            "workflows.prediction_workflow.decision_agent",
            type("Fake", (), {"run": lambda self, msg: (_ for _ in ()).throw(RuntimeError("test"))})(),
        )
        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Risk Assessment": _make_risk_assessment(),
            "Position Sizing": {"force_skip": False},
            "Market Data": _make_market_snapshot(),
            "Data Quality": {"force_skip": False},
        })
        result = _step_content_to_model(run_decision(si), BetDecision)
        assert result is not None
        assert result.action == "SKIP"
        assert "exception" in result.rationale.lower()

    def test_uses_sentiment_fallback_from_dq(self, monkeypatch):
        """When sentiment is None but DQ has fallback, prompt should include it."""
        captured_prompts = []

        class FakeAgent:
            def run(self, msg):
                captured_prompts.append(msg)
                return type("R", (), {"content": _safe_bet_decision()})()

        monkeypatch.setattr("workflows.prediction_workflow.decision_agent", FakeAgent())

        si = _make_step_input(**{
            "Event Scan": _make_event_candidate(),
            "Risk Assessment": _make_risk_assessment(),
            "Position Sizing": {"force_skip": False, "recommended_stake": 500},
            "Market Data": _make_market_snapshot(),
            "Data Quality": {
                "force_skip": False,
                "sentiment_fallback": {"query": "fallback", "sentiment_score": 0.0},
            },
        })
        run_decision(si)
        assert len(captured_prompts) == 1
        assert "fallback" in captured_prompts[0]


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
        monkeypatch.setattr("workflows.prediction_workflow.get_paper_trade_store", lambda: store)
        monkeypatch.setattr(
            "workflows.prediction_workflow.logger_agent",
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
        monkeypatch.setattr("workflows.prediction_workflow.get_paper_trade_store", lambda: store)

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
