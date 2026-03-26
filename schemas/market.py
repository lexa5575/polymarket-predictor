"""
Inter-agent data contracts for Polymarket Predictor.

These Pydantic models define the structured output for each agent
and serve as the formal handoff format between pipeline steps.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TokenBook(BaseModel):
    """Orderbook snapshot for a single token (YES or NO)."""

    token_id: str = Field(..., description="CLOB token ID")
    best_bid: float = Field(..., description="Highest bid price")
    best_ask: float = Field(..., description="Lowest ask price")
    spread: float = Field(..., description="best_ask - best_bid")
    depth_10pct: float = Field(
        ...,
        description="Total $ liquidity within 10% of midpoint",
    )


class EventCandidate(BaseModel):
    """Output of Polymarket Agent (single_market mode).

    Contains full market details with orderbooks for both YES and NO tokens.
    Used as input to the prediction workflow pipeline.
    """

    gamma_market_id: str = Field(..., description="Gamma API market id")
    condition_id: str = Field(..., description="Primary key — Polymarket condition ID")
    market_slug: str = Field(..., description="Human-readable slug for logs/memos")
    question: str = Field(..., description='E.g. "Will BTC exceed $100K by June 2026?"')
    category: str = Field(default="crypto")
    end_date: datetime = Field(..., description="Market resolution date")
    yes_book: TokenBook = Field(..., description="Full orderbook for YES token")
    no_book: TokenBook = Field(..., description="Full orderbook for NO token")
    market_prob_yes: float = Field(
        ...,
        description="Midpoint implied probability of YES outcome",
    )
    volume_24h: float = Field(..., description="24-hour volume in USD")
    total_liquidity: float = Field(
        ...,
        description="yes_book.depth_10pct + no_book.depth_10pct",
    )


class BatchScanResult(BaseModel):
    """Output of Polymarket Scanner agent (batch_scan mode).

    Returns a ranked list of EventCandidates for fan-out processing.
    """

    candidates: list[EventCandidate] = Field(
        ...,
        description="Ranked list of market candidates",
    )
    total_scanned: int = Field(..., description="Total markets evaluated")
    filters_applied: dict[str, str] = Field(
        default_factory=dict,
        description="Filters used during scan (e.g. category=crypto)",
    )


class MarketSnapshot(BaseModel):
    """Output of Market Data Agent.

    Aggregates crypto market data from CoinGecko, Coinglass, and Fear&Greed.
    Nullable fields indicate external source was unavailable (graceful degradation).
    """

    coin_id: str = Field(..., description='CoinGecko coin ID, e.g. "bitcoin"')
    price_usd: float
    change_24h_pct: float
    market_cap: float
    funding_rate: float | None = Field(
        None,
        description="Current perpetual funding rate (None if Coinglass unavailable)",
    )
    open_interest: float | None = Field(
        None,
        description="Open interest in USD (None if Coinglass unavailable)",
    )
    fear_greed_index: int = Field(..., ge=0, le=100)
    fear_greed_label: str = Field(
        ...,
        description='"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"',
    )
    signal: Literal["Bullish", "Neutral", "Bearish"]


class SentimentReport(BaseModel):
    """Output of News Agent (Grok + Exa web search)."""

    query: str = Field(..., description="Search query used")
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="-1.0 (extremely bearish) to +1.0 (extremely bullish)",
    )
    key_narratives: list[str] = Field(
        ...,
        max_length=5,
        description="Top narratives driving sentiment",
    )
    sources_count: int = Field(..., description="Number of sources analyzed")
    confidence: float = Field(..., ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    """Output of Risk Agent.

    Provides qualitative risk analysis. Kelly/slippage/stake are computed
    by the deterministic compute_position_sizing function step, not by this agent.
    All probabilities are relative to recommended_side.
    """

    condition_id: str
    risk_rating: Literal["Low", "Moderate", "High", "Unacceptable"]
    recommended_side: Literal["YES", "NO"] = Field(
        ...,
        description="Which side to bet on",
    )
    estimated_prob_of_side: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Our estimated P(recommended_side wins)",
    )
    market_prob_of_side: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Market implied P(recommended_side wins)",
    )
    edge: float = Field(
        ...,
        description="estimated_prob_of_side - market_prob_of_side",
    )
    underlier_group: str = Field(
        ...,
        description='"btc_price", "eth_price", "etf", "regulation", "other"',
    )
    warnings: list[str] = Field(default_factory=list)
    liquidity_ok: bool
    correlated_positions: int = Field(
        ...,
        description="Open trades in the same underlier_group",
    )


class BetDecision(BaseModel):
    """Output of Decision Agent — the final actionable decision.

    All probabilities are relative to the chosen side.
    """

    condition_id: str
    market_slug: str
    token_id: str = Field(..., description="Token ID of the chosen side")
    side: Literal["YES", "NO"]
    action: Literal["BET", "SKIP"]
    estimated_prob_of_side: float = Field(
        ...,
        description="P(side wins) per our estimate",
    )
    market_prob_of_side_at_entry: float = Field(
        ...,
        description="P(side wins) per market at time of entry",
    )
    edge: float = Field(..., description="estimated - market for chosen side")
    entry_price: float = Field(
        ...,
        description="best_ask + slippage (0 if SKIP)",
    )
    slippage_estimate: float = Field(default=0.0)
    stake: float = Field(..., description="USD amount to bet (0 if SKIP)")
    underlier_group: str = Field(
        ...,
        description="Inherited from RiskAssessment",
    )
    rationale: str
    exit_conditions: list[str] = Field(default_factory=list)
    confidence: Literal["High", "Medium", "Low"]
