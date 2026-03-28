"""
Paper trading domain models.

These models define the persistence schema for simulated trades
and bankroll tracking. Source of truth is PostgreSQL; memos/ are
read-only audit artifacts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PaperTrade(BaseModel):
    """A single paper trade record."""

    id: str = Field(..., description="UUID")
    created_at: datetime
    condition_id: str = Field(..., description="Polymarket condition ID (FK)")
    market_slug: str
    token_id: str
    question: str
    side: Literal["YES", "NO"]
    underlier_group: str = Field(
        ...,
        description="Correlation group: btc_price, eth_price, etf, regulation, other",
    )
    market_prob_at_entry: float = Field(
        ...,
        description="P(side wins) per market at entry",
    )
    estimated_prob: float = Field(
        ...,
        description="P(side wins) per our estimate",
    )
    edge: float
    stake: float = Field(..., description="USD amount risked")
    entry_fill_price: float = Field(
        ...,
        description="best_ask + slippage estimate",
    )
    status: Literal["open", "won", "lost", "closed", "cancelled", "expired"] = Field(
        default="open",
    )
    resolution_time: datetime | None = None
    resolved_outcome: Literal["YES", "NO"] | None = None
    pnl: float | None = Field(
        None,
        description="Binary: stake * (1/entry_price - 1) or -stake. MTM: shares * exit_price - stake.",
    )
    brier_score: float | None = Field(
        None,
        description="(estimated_prob - outcome_binary)^2",
    )
    exit_conditions: list[str] = Field(default_factory=list)
    # Exit tracking (mark-to-market early close)
    exit_price: float | None = Field(None, description="Mark-to-market exit price (best_bid)")
    exit_reason: str | None = Field(None, description="take_profit/stop_loss/max_hold")
    exit_time: datetime | None = Field(None, description="When position was closed")
    # Exit policy snapshot (saved at entry for reproducibility)
    take_profit_pct: float | None = Field(None, description="TP threshold at entry")
    stop_loss_pct: float | None = Field(None, description="SL threshold at entry")
    max_hold_seconds: float | None = Field(None, description="Max hold time at entry")


class BankrollSnapshot(BaseModel):
    """Point-in-time snapshot of paper trading performance."""

    timestamp: datetime
    starting_bankroll: float = Field(default=10_000.0)
    current_bankroll: float
    open_positions: int
    total_at_risk: float = Field(
        ...,
        description="Sum of stakes across open trades",
    )
    total_trades: int
    wins: int
    losses: int
    win_rate: float = Field(..., ge=0.0, le=1.0)
    total_pnl: float
    avg_brier_score: float
    sharpe_ratio: float | None = None
