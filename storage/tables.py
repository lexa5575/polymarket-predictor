"""
SQLAlchemy ORM models for paper trading persistence.

This module is the **sole owner** of the database schema for paper trades
and bankroll snapshots. All CRUD operations in paper_trades.py use these models.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class PaperTradeRow(Base):
    __tablename__ = "paper_trades"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    condition_id = Column(String, nullable=False, index=True)
    market_slug = Column(String, nullable=False)
    token_id = Column(String, nullable=False)
    question = Column(String, nullable=False)
    side = Column(String, nullable=False)  # "YES" | "NO"
    underlier_group = Column(String, nullable=False, index=True)
    market_prob_at_entry = Column(Float, nullable=False)
    estimated_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)
    entry_fill_price = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="open", index=True)
    resolution_time = Column(DateTime(timezone=True), nullable=True)
    resolved_outcome = Column(String, nullable=True)  # "YES" | "NO" | None
    pnl = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    exit_conditions = Column(JSON, nullable=False, default=list)


class BankrollSnapshotRow(Base):
    __tablename__ = "bankroll_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    starting_bankroll = Column(Float, nullable=False, default=10_000.0)
    current_bankroll = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False)
    total_at_risk = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    wins = Column(Integer, nullable=False)
    losses = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False)
    avg_brier_score = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)


def init_db(db_url: str) -> sessionmaker[Session]:
    """Create tables if they don't exist and return a session factory."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
