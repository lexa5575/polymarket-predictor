"""Tests for workflow function steps: compute_position_sizing and conditional_logging.

These test the deterministic logic without requiring LLM calls or live APIs.
Uses SQLite in-memory for paper trade persistence.
"""

import json

import pytest

from agno.workflow.types import StepInput

from workflows.prediction_workflow import (
    _emit_tagged_block,
    _extract_agent_json,
    _get_tagged_block,
    compute_position_sizing,
    conditional_logging,
)


def _make_step_input(context: str) -> StepInput:
    """Helper to create StepInput from a context string."""
    return StepInput(previous_step_content=context)


# ---------------------------------------------------------------------------
# Tagged block protocol tests
# ---------------------------------------------------------------------------


class TestTaggedBlocks:
    def test_roundtrip(self):
        data = {"action": "BET", "stake": 500}
        block = _emit_tagged_block("TEST", data)
        assert _get_tagged_block(block, "TEST") == data

    def test_missing_tag(self):
        assert _get_tagged_block("no tags here", "TEST") is None

    def test_surrounded_by_text(self):
        data = {"x": 1}
        text = f"some preamble\n{_emit_tagged_block('T', data)}\nsome epilogue"
        assert _get_tagged_block(text, "T") == data


class TestExtractAgentJson:
    def test_finds_json_with_key(self):
        context = 'Some text\n{"estimated_prob_of_side": 0.65, "market_prob_of_side": 0.50}\nMore text'
        result = _extract_agent_json(context, "estimated_prob_of_side")
        assert result is not None
        assert result["estimated_prob_of_side"] == 0.65

    def test_no_key_collision_side_vs_recommended_side(self):
        """Ensure we find the correct JSON block when both 'side' and 'recommended_side' exist."""
        risk_json = '{"recommended_side": "NO", "estimated_prob_of_side": 0.7}'
        decision_json = '{"side": "NO", "action": "BET", "stake": 300}'
        context = f"Risk output: {risk_json}\nDecision output: {decision_json}"

        risk = _extract_agent_json(context, "recommended_side")
        assert risk is not None
        assert risk["recommended_side"] == "NO"

        decision = _extract_agent_json(context, "action")
        assert decision is not None
        assert decision["side"] == "NO"
        assert decision["action"] == "BET"

    def test_not_found(self):
        assert _extract_agent_json("no json here", "missing_key") is None


# ---------------------------------------------------------------------------
# Position Sizing tests
# ---------------------------------------------------------------------------


class TestComputePositionSizing:
    def _parse_sizing(self, step_output) -> dict:
        content = step_output.content if hasattr(step_output, 'content') else str(step_output)
        block = _get_tagged_block(content, "SIZING_DATA")
        assert block is not None, f"No SIZING_DATA block in: {content}"
        return block

    def test_positive_edge_yes_side(self):
        context = (
            '{"estimated_prob_of_side": 0.65, "market_prob_of_side": 0.50, '
            '"recommended_side": "YES"}\n'
            '{"gamma_market_id": "m1", "yes_book": {"token_id": "t1", "best_ask": 0.51, '
            '"best_bid": 0.49, "spread": 0.02, "depth_10pct": 30000}, '
            '"no_book": {"token_id": "t2", "best_ask": 0.50, "best_bid": 0.48, '
            '"spread": 0.02, "depth_10pct": 25000}}'
        )
        sizing = self._parse_sizing(compute_position_sizing(_make_step_input(context)))
        assert sizing["recommended_stake"] > 0
        assert sizing["kelly_fraction_raw"] > 0
        assert sizing["entry_price"] > 0.50  # should be above best_ask
        assert sizing["force_skip"] is False
        assert sizing["recommended_side"] == "YES"

    def test_positive_edge_no_side(self):
        """NO-side should use no_book for slippage/entry, not yes_book."""
        context = (
            '{"estimated_prob_of_side": 0.70, "market_prob_of_side": 0.55, '
            '"recommended_side": "NO"}\n'
            '{"gamma_market_id": "m1", "yes_book": {"token_id": "t1", "best_ask": 0.60, '
            '"best_bid": 0.58, "spread": 0.02, "depth_10pct": 20000}, '
            '"no_book": {"token_id": "t2", "best_ask": 0.42, "best_bid": 0.40, '
            '"spread": 0.02, "depth_10pct": 20000}}'
        )
        sizing = self._parse_sizing(compute_position_sizing(_make_step_input(context)))
        assert sizing["recommended_stake"] > 0
        assert sizing["recommended_side"] == "NO"
        # Entry price should be based on no_book.best_ask (~0.42), not yes_book
        assert sizing["entry_price"] < 0.50

    def test_no_edge_forces_skip(self):
        context = '{"estimated_prob_of_side": 0.45, "market_prob_of_side": 0.50}'
        sizing = self._parse_sizing(compute_position_sizing(_make_step_input(context)))
        assert sizing["recommended_stake"] == 0.0
        assert sizing["force_skip"] is True

    def test_missing_data_forces_skip(self):
        sizing = self._parse_sizing(compute_position_sizing("no useful data"))
        assert sizing["force_skip"] is True

    def test_slippage_exceeds_budget_forces_skip(self):
        """If slippage > 2% of best_ask, stake must be zeroed (hard gate).

        With depth_10pct=10 and a Kelly stake well above $10, the orderbook
        walk will produce slippage far exceeding 2%.
        """
        context = (
            '{"estimated_prob_of_side": 0.90, "market_prob_of_side": 0.50, '
            '"recommended_side": "YES"}\n'
            '{"gamma_market_id": "m1", "yes_book": {"token_id": "t1", "best_ask": 0.51, '
            '"best_bid": 0.49, "spread": 0.02, "depth_10pct": 10}, '
            '"no_book": {"token_id": "t2", "best_ask": 0.50, "best_bid": 0.48, '
            '"spread": 0.02, "depth_10pct": 10}}'
        )
        sizing = self._parse_sizing(compute_position_sizing(_make_step_input(context)))
        # depth_10pct=10 means only $10 in the book. Kelly stake at 90% vs 50%
        # edge is ~$1000+. Walking a $1000 order through a $10 book produces
        # massive slippage, which must trigger the hard gate unconditionally.
        assert sizing["force_skip"] is True
        assert "2%" in sizing.get("sizing_note", "")

    def test_missing_orderbook_for_side_forces_skip(self):
        """If no valid orderbook for the recommended side, force_skip must be True."""
        # EventCandidate has no books at all
        context = (
            '{"estimated_prob_of_side": 0.70, "market_prob_of_side": 0.50, '
            '"recommended_side": "YES"}\n'
            '{"gamma_market_id": "m1"}'  # no yes_book or no_book
        )
        sizing = self._parse_sizing(compute_position_sizing(_make_step_input(context)))
        assert sizing["force_skip"] is True
        assert "orderbook" in sizing.get("sizing_note", "").lower()

    def test_missing_no_book_forces_skip_for_no_side(self):
        """If recommended_side=NO but no_book is missing, force_skip."""
        context = (
            '{"estimated_prob_of_side": 0.70, "market_prob_of_side": 0.50, '
            '"recommended_side": "NO"}\n'
            '{"gamma_market_id": "m1", "yes_book": {"token_id": "t1", "best_ask": 0.55, '
            '"best_bid": 0.53, "spread": 0.02, "depth_10pct": 20000}}'
            # no_book is missing entirely
        )
        sizing = self._parse_sizing(compute_position_sizing(_make_step_input(context)))
        assert sizing["force_skip"] is True
        assert "orderbook" in sizing.get("sizing_note", "").lower()


# ---------------------------------------------------------------------------
# Conditional Logging tests
# ---------------------------------------------------------------------------


class TestConditionalLogging:
    def _parse_record(self, step_output) -> dict:
        content = step_output.content if hasattr(step_output, 'content') else str(step_output)
        block = _get_tagged_block(content, "RECORD_RESULT")
        assert block is not None, f"No RECORD_RESULT block in: {content}"
        return block

    def test_skip_from_agent(self):
        context = '{"action": "SKIP", "rationale": "No edge"}'
        record = self._parse_record(conditional_logging(_make_step_input(context)))
        assert record["action"] == "SKIP"
        assert record["trade_id"] is None

    def test_skip_from_sizing_force(self):
        sizing_block = _emit_tagged_block("SIZING_DATA", {
            "force_skip": True,
            "sizing_note": "Slippage too high",
            "recommended_stake": 0,
        })
        context = f'{sizing_block}\n{{"action": "BET", "stake": 500}}'
        record = self._parse_record(conditional_logging(_make_step_input(context)))
        assert record["action"] == "SKIP"
        assert "Slippage" in record.get("reason", "")

    def test_bet_with_zero_stake(self):
        context = '{"action": "BET", "stake": 0, "side": "YES"}'
        record = self._parse_record(conditional_logging(_make_step_input(context)))
        assert record["action"] == "SKIP"
        assert "zero stake" in record.get("reason", "").lower()

    def test_agent_zero_stake_not_overridden_by_sizing(self):
        """REGRESSION: If agent says stake=0, sizing.recommended_stake must NOT override it.

        This ensures explicit zero from the Decision Agent is respected,
        not silently replaced by the sizing fallback via falsy `or` logic.
        """
        sizing_block = _emit_tagged_block("SIZING_DATA", {
            "force_skip": False,
            "recommended_stake": 500,  # sizing says bet $500
            "entry_price": 0.51,
            "slippage_estimate": 0.01,
        })
        # Agent explicitly says stake=0 (meaning "I changed my mind, don't bet")
        decision_json = json.dumps({
            "action": "BET",
            "condition_id": "0xzero",
            "market_slug": "zero-test",
            "token_id": "tzero",
            "side": "YES",
            "stake": 0,  # explicit zero — must be respected
            "estimated_prob_of_side": 0.6,
            "market_prob_of_side_at_entry": 0.5,
            "edge": 0.1,
            "entry_price": 0.51,
            "slippage_estimate": 0.01,
            "underlier_group": "btc_price",
            "rationale": "Changed mind",
            "confidence": "Low",
        })
        context = f"{sizing_block}\n{decision_json}"
        record = self._parse_record(conditional_logging(_make_step_input(context)))
        assert record["action"] == "SKIP"
        assert "zero stake" in record.get("reason", "").lower()

    def test_bet_records_trade_yes(self, tmp_path, monkeypatch):
        """BET YES creates a paper trade in DB."""
        from storage.paper_trades import PaperTradeStore

        test_store = PaperTradeStore(f"sqlite:///{tmp_path / 'test.db'}")
        monkeypatch.setattr("workflows.prediction_workflow.get_paper_trade_store", lambda: test_store)
        monkeypatch.setattr(
            "workflows.prediction_workflow.logger_agent",
            type("Fake", (), {"run": lambda self, msg: type("R", (), {"content": "ok"})()})(),
        )

        sizing_block = _emit_tagged_block("SIZING_DATA", {
            "force_skip": False, "recommended_stake": 450, "entry_price": 0.51, "slippage_estimate": 0.01,
        })
        decision_json = json.dumps({
            "action": "BET", "condition_id": "0xyes", "market_slug": "btc-yes-test",
            "token_id": "tyes", "side": "YES", "estimated_prob_of_side": 0.65,
            "market_prob_of_side_at_entry": 0.50, "edge": 0.15, "entry_price": 0.51,
            "slippage_estimate": 0.01, "stake": 450, "underlier_group": "btc_price",
            "rationale": "Test", "exit_conditions": [], "confidence": "High",
        })
        event_json = '{"question": "Will BTC exceed $100K?"}'
        context = f"{sizing_block}\n{decision_json}\n{event_json}"

        record = self._parse_record(conditional_logging(_make_step_input(context)))
        assert record["action"] == "BET"
        assert record["trade_id"] is not None
        assert record["side"] == "YES"
        assert record["stake"] == 450

        trades = test_store.get_open_trades()
        assert len(trades) == 1
        assert trades[0].side == "YES"

    def test_bet_records_trade_no(self, tmp_path, monkeypatch):
        """BET NO creates a paper trade with side=NO."""
        from storage.paper_trades import PaperTradeStore

        test_store = PaperTradeStore(f"sqlite:///{tmp_path / 'test.db'}")
        monkeypatch.setattr("workflows.prediction_workflow.get_paper_trade_store", lambda: test_store)
        monkeypatch.setattr(
            "workflows.prediction_workflow.logger_agent",
            type("Fake", (), {"run": lambda self, msg: type("R", (), {"content": "ok"})()})(),
        )

        sizing_block = _emit_tagged_block("SIZING_DATA", {
            "force_skip": False, "recommended_stake": 300, "entry_price": 0.42, "slippage_estimate": 0.005,
        })
        decision_json = json.dumps({
            "action": "BET", "condition_id": "0xno", "market_slug": "btc-no-test",
            "token_id": "tno", "side": "NO", "estimated_prob_of_side": 0.70,
            "market_prob_of_side_at_entry": 0.55, "edge": 0.15, "entry_price": 0.42,
            "slippage_estimate": 0.005, "stake": 300, "underlier_group": "btc_price",
            "rationale": "Bearish test", "exit_conditions": [], "confidence": "Medium",
        })
        context = f"{sizing_block}\n{decision_json}"

        record = self._parse_record(conditional_logging(_make_step_input(context)))
        assert record["action"] == "BET"
        assert record["side"] == "NO"
        assert record["stake"] == 300

        trades = test_store.get_open_trades()
        assert len(trades) == 1
        assert trades[0].side == "NO"
        assert trades[0].entry_fill_price == 0.42

    def test_unclear_action(self):
        record = self._parse_record(conditional_logging("some random text"))
        assert record["action"] == "SKIP"
