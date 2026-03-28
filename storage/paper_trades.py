"""
Paper trade CRUD operations.

PaperTradeStore is the single point of truth for all trade persistence.
The conditional_logging workflow function step is the sole caller of open_trade().
"""

from __future__ import annotations

import statistics
import uuid
from datetime import datetime, timezone
from typing import Literal

from sqlalchemy.orm import Session

from schemas.market import BetDecision
from schemas.paper_trade import BankrollSnapshot, PaperTrade
from storage.exit_policy import MAX_HOLD_SECONDS, STOP_LOSS_PCT, TAKE_PROFIT_PCT, compute_max_hold
from storage.math_utils import brier_score, calculate_mtm_pnl, calculate_pnl
from storage.tables import BankrollSnapshotRow, PaperTradeRow, init_db

STARTING_BANKROLL = 1_000.0


class PaperTradeStore:
    """CRUD for paper trades backed by PostgreSQL via SQLAlchemy."""

    def __init__(self, db_url: str) -> None:
        self._session_factory = init_db(db_url)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def open_trade(self, decision: BetDecision, question: str, end_date: str = "") -> PaperTrade:
        """Record a new paper trade from a BET decision."""
        trade_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        max_hold = compute_max_hold(end_date) if end_date else MAX_HOLD_SECONDS

        row = PaperTradeRow(
            id=trade_id,
            created_at=now,
            condition_id=decision.condition_id,
            market_slug=decision.market_slug,
            token_id=decision.token_id,
            question=question,
            side=decision.side,
            underlier_group=decision.underlier_group,
            market_prob_at_entry=decision.market_prob_of_side_at_entry,
            estimated_prob=decision.estimated_prob_of_side,
            edge=decision.edge,
            stake=decision.stake,
            entry_fill_price=decision.entry_price,
            status="open",
            exit_conditions=decision.exit_conditions,
            take_profit_pct=TAKE_PROFIT_PCT,
            stop_loss_pct=STOP_LOSS_PCT,
            max_hold_seconds=max_hold,
        )

        with self._session_factory() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._row_to_model(row)

    def resolve_trade(
        self,
        trade_id: str,
        outcome: Literal["YES", "NO"],
    ) -> PaperTrade:
        """Resolve an open trade and compute PnL + Brier score."""
        with self._session_factory() as session:
            row: PaperTradeRow | None = session.get(PaperTradeRow, trade_id)
            if row is None:
                raise ValueError(f"Trade {trade_id} not found")
            if row.status != "open":
                raise ValueError(f"Trade {trade_id} is already {row.status}")

            won = outcome == row.side
            row.resolved_outcome = outcome
            row.resolution_time = datetime.now(timezone.utc)
            row.status = "won" if won else "lost"
            row.pnl = calculate_pnl(row.entry_fill_price, row.stake, won)
            row.brier_score = brier_score(
                row.estimated_prob,
                1 if won else 0,
            )
            session.commit()
            session.refresh(row)
            return self._row_to_model(row)

    def cancel_trade(self, trade_id: str) -> PaperTrade:
        """Cancel an open trade (e.g. market voided)."""
        with self._session_factory() as session:
            row: PaperTradeRow | None = session.get(PaperTradeRow, trade_id)
            if row is None:
                raise ValueError(f"Trade {trade_id} not found")
            row.status = "cancelled"
            row.pnl = 0.0
            session.commit()
            session.refresh(row)
            return self._row_to_model(row)

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        reason: str,
    ) -> PaperTrade:
        """Close an open trade at mark-to-market price (early exit).

        PnL = shares * exit_price - stake, where shares = stake / entry_price.
        Does NOT set brier_score (no binary outcome yet — use record_resolution later).
        """
        with self._session_factory() as session:
            row: PaperTradeRow | None = session.get(PaperTradeRow, trade_id)
            if row is None:
                raise ValueError(f"Trade {trade_id} not found")
            if row.status != "open":
                raise ValueError(f"Trade {trade_id} is already {row.status}")

            row.status = "closed"
            row.exit_price = exit_price
            row.exit_reason = reason
            row.exit_time = datetime.now(timezone.utc)
            row.pnl = calculate_mtm_pnl(row.entry_fill_price, exit_price, row.stake)
            # brier_score stays None until market resolves (record_resolution)
            session.commit()
            session.refresh(row)
            return self._row_to_model(row)

    def record_resolution(
        self,
        trade_id: str,
        outcome: Literal["YES", "NO"],
    ) -> PaperTrade:
        """Record market outcome on an already-closed trade (for what-if analytics).

        Sets resolved_outcome, resolution_time, brier_score.
        Does NOT change status, pnl, exit_price, or exit_reason.
        """
        with self._session_factory() as session:
            row: PaperTradeRow | None = session.get(PaperTradeRow, trade_id)
            if row is None:
                raise ValueError(f"Trade {trade_id} not found")
            if row.status != "closed":
                raise ValueError(f"Trade {trade_id} has status {row.status}, expected 'closed'")

            won = outcome == row.side
            row.resolved_outcome = outcome
            row.resolution_time = datetime.now(timezone.utc)
            row.brier_score = brier_score(row.estimated_prob, 1 if won else 0)
            session.commit()
            session.refresh(row)
            return self._row_to_model(row)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list[PaperTrade]:
        with self._session_factory() as session:
            rows = (
                session.query(PaperTradeRow)
                .filter(PaperTradeRow.status == "open")
                .all()
            )
            return [self._row_to_model(r) for r in rows]

    def get_trades_by_condition(self, condition_id: str) -> list[PaperTrade]:
        with self._session_factory() as session:
            rows = (
                session.query(PaperTradeRow)
                .filter(PaperTradeRow.condition_id == condition_id)
                .all()
            )
            return [self._row_to_model(r) for r in rows]

    def get_correlated_count(self, underlier_group: str) -> int:
        """Count open trades in the same underlier group."""
        with self._session_factory() as session:
            return (
                session.query(PaperTradeRow)
                .filter(
                    PaperTradeRow.status == "open",
                    PaperTradeRow.underlier_group == underlier_group,
                )
                .count()
            )

    def get_closed_without_resolution(self) -> list[PaperTrade]:
        """Get closed trades that don't yet have a market resolution recorded."""
        with self._session_factory() as session:
            rows = (
                session.query(PaperTradeRow)
                .filter(
                    PaperTradeRow.status == "closed",
                    PaperTradeRow.resolved_outcome.is_(None),
                )
                .all()
            )
            return [self._row_to_model(r) for r in rows]

    def get_all_trades(self) -> list[PaperTrade]:
        with self._session_factory() as session:
            rows = session.query(PaperTradeRow).order_by(PaperTradeRow.created_at).all()
            return [self._row_to_model(r) for r in rows]

    # ------------------------------------------------------------------
    # Bankroll
    # ------------------------------------------------------------------

    def get_bankroll_snapshot(self) -> BankrollSnapshot:
        """Compute current bankroll state without persisting."""
        return self._compute_snapshot()

    def create_bankroll_snapshot(self) -> BankrollSnapshot:
        """Compute current bankroll state and persist to DB."""
        snapshot = self._compute_snapshot()
        row = BankrollSnapshotRow(
            timestamp=snapshot.timestamp,
            starting_bankroll=snapshot.starting_bankroll,
            current_bankroll=snapshot.current_bankroll,
            open_positions=snapshot.open_positions,
            total_at_risk=snapshot.total_at_risk,
            total_trades=snapshot.total_trades,
            wins=snapshot.wins,
            losses=snapshot.losses,
            win_rate=snapshot.win_rate,
            total_pnl=snapshot.total_pnl,
            avg_brier_score=snapshot.avg_brier_score,
            sharpe_ratio=snapshot.sharpe_ratio,
        )
        with self._session_factory() as session:
            session.add(row)
            session.commit()
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_snapshot(self) -> BankrollSnapshot:
        with self._session_factory() as session:
            all_rows: list[PaperTradeRow] = session.query(PaperTradeRow).all()

        open_rows = [r for r in all_rows if r.status == "open"]
        # finished = all trades with realized PnL (binary resolution + early close)
        finished = [r for r in all_rows if r.status in ("won", "lost", "closed")]
        # resolution-only for backwards-compatible win_rate
        resolved = [r for r in all_rows if r.status in ("won", "lost")]
        wins = [r for r in resolved if r.status == "won"]
        losses = [r for r in resolved if r.status == "lost"]

        total_pnl = sum(r.pnl or 0.0 for r in finished)
        total_at_risk = sum(r.stake for r in open_rows)

        # brier_score from all trades that have it (won/lost + closed with recorded resolution)
        brier_scores = [r.brier_score for r in all_rows if r.brier_score is not None]
        avg_brier = statistics.mean(brier_scores) if brier_scores else 0.0

        # sharpe from all finished trades
        pnl_list = [r.pnl for r in finished if r.pnl is not None]
        sharpe = None
        if len(pnl_list) >= 2:
            mean_pnl = statistics.mean(pnl_list)
            std_pnl = statistics.stdev(pnl_list)
            if std_pnl > 0:
                sharpe = mean_pnl / std_pnl

        n_resolved = len(resolved)

        return BankrollSnapshot(
            timestamp=datetime.now(timezone.utc),
            starting_bankroll=STARTING_BANKROLL,
            current_bankroll=STARTING_BANKROLL + total_pnl - total_at_risk,
            open_positions=len(open_rows),
            total_at_risk=total_at_risk,
            total_trades=len(all_rows),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / n_resolved if n_resolved > 0 else 0.0,
            total_pnl=total_pnl,
            avg_brier_score=avg_brier,
            sharpe_ratio=sharpe,
        )

    @staticmethod
    def _row_to_model(row: PaperTradeRow) -> PaperTrade:
        return PaperTrade(
            id=row.id,
            created_at=row.created_at,
            condition_id=row.condition_id,
            market_slug=row.market_slug,
            token_id=row.token_id,
            question=row.question,
            side=row.side,
            underlier_group=row.underlier_group,
            market_prob_at_entry=row.market_prob_at_entry,
            estimated_prob=row.estimated_prob,
            edge=row.edge,
            stake=row.stake,
            entry_fill_price=row.entry_fill_price,
            status=row.status,
            resolution_time=row.resolution_time,
            resolved_outcome=row.resolved_outcome,
            pnl=row.pnl,
            brier_score=row.brier_score,
            exit_conditions=row.exit_conditions or [],
            exit_price=row.exit_price,
            exit_reason=row.exit_reason,
            exit_time=row.exit_time,
            take_profit_pct=row.take_profit_pct,
            stop_loss_pct=row.stop_loss_pct,
            max_hold_seconds=row.max_hold_seconds,
        )
