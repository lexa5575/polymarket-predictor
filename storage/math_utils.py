"""
Deterministic math utilities for paper trading.

All financial calculations live here — never inside an LLM agent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas.paper_trade import BankrollSnapshot


def kelly_criterion(estimated_prob: float, market_prob: float) -> float:
    """Compute the raw Kelly fraction.

    Kelly f* = (p * b - q) / b
    where p = estimated_prob, q = 1 - p, b = (1 / market_prob) - 1

    Returns 0 if there is no positive edge or inputs are invalid.
    """
    if market_prob <= 0 or market_prob >= 1 or estimated_prob <= 0 or estimated_prob >= 1:
        return 0.0

    b = (1.0 / market_prob) - 1.0  # decimal odds - 1
    if b <= 0:
        return 0.0

    q = 1.0 - estimated_prob
    f = (estimated_prob * b - q) / b
    return max(f, 0.0)


def fractional_kelly(kelly: float, fraction: float = 0.25) -> float:
    """Apply fractional Kelly (default quarter-Kelly) for conservative sizing."""
    return max(kelly * fraction, 0.0)


def brier_score(estimated_prob: float, outcome: int) -> float:
    """Compute the Brier score for a single prediction.

    Args:
        estimated_prob: Our predicted probability (0-1) for the chosen side.
        outcome: 1 if the chosen side won, 0 otherwise.

    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    return (estimated_prob - outcome) ** 2


def estimate_slippage(
    stake: float,
    orderbook_asks: list[tuple[float, float]],
) -> float:
    """Estimate slippage by walking the orderbook ask side.

    Args:
        stake: USD amount we want to fill.
        orderbook_asks: List of (price, size_usd) tuples sorted by price ascending.

    Returns:
        Estimated average fill price minus best ask, as a fraction.
        Returns 0 if the orderbook can fully absorb the stake at best ask.
    """
    if not orderbook_asks or stake <= 0:
        return 0.0

    best_ask = orderbook_asks[0][0]
    remaining = stake
    total_cost = 0.0

    for price, size_usd in orderbook_asks:
        fill = min(remaining, size_usd)
        total_cost += fill * price
        remaining -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        # Not enough liquidity — estimate with worst price + 5% premium
        worst_price = orderbook_asks[-1][0] if orderbook_asks else best_ask
        total_cost += remaining * worst_price * 1.05

    avg_fill_price = total_cost / stake
    return avg_fill_price - best_ask


def calculate_entry_price(best_ask: float, slippage: float) -> float:
    """Compute the expected entry fill price."""
    return best_ask + slippage


def compute_edge(estimated_prob: float, market_prob: float) -> float:
    """Edge = estimated probability - market price."""
    return estimated_prob - market_prob


def confidence_to_score(confidence: str) -> float:
    """Map qualitative LLM confidence to numeric score."""
    return {"High": 1.0, "Medium": 0.7, "Low": 0.4}.get(confidence, 0.0)


def check_liquidity(
    depth_10pct: float,
    volume_24h: float,
    spread: float,
) -> tuple[bool, list[str]]:
    """Composite liquidity check per mandate.md and risk_policy.md.

    Returns (liquidity_ok, warnings).
    Rules:
    - depth_10pct >= $10K  (mandate.md:17, :26)
    - volume_24h >= $50K   (mandate.md:17)
    - spread <= 0.05 (5%)  (mandate.md:24, risk_policy.md:25)
    """
    warnings: list[str] = []
    if depth_10pct < 10_000:
        warnings.append(f"Orderbook depth ${depth_10pct:,.0f} below $10K minimum")
    if volume_24h < 50_000:
        warnings.append(f"24h volume ${volume_24h:,.0f} below $50K minimum")
    if spread > 0.05:
        warnings.append(f"Spread {spread:.1%} exceeds 5% limit")
    return len(warnings) == 0, warnings


def determine_risk_rating(
    edge: float,
    liquidity_ok: bool,
    correlated: int,
) -> str:
    """Deterministic risk rating.

    | edge   | liquidity | correlated | rating       |
    |--------|-----------|------------|--------------|
    | < 5%   | any       | any        | Unacceptable |
    | >= 5%  | no        | any        | Unacceptable |
    | >= 5%  | yes       | >= 3       | Unacceptable |
    | > 10%  | yes       | 0-1        | Low          |
    | 5-10%  | yes       | 0-1        | Moderate     |
    | any>=5 | yes       | 2          | High         |
    """
    if edge < 0.05 or not liquidity_ok or correlated >= 3:
        return "Unacceptable"
    if edge > 0.10 and correlated < 2:
        return "Low"
    if correlated < 2:
        return "Moderate"
    # edge >= 5%, liquidity ok, exactly 2 correlated
    return "High"


def check_portfolio_limits(
    snapshot: "BankrollSnapshot",
    new_stake: float,
) -> tuple[bool, list[str]]:
    """Portfolio-level gates from mandate.md and risk_policy.md.

    Returns (ok, warnings).

    BankrollSnapshot semantics (paper_trades.py:200-203):
      current_bankroll = starting_bankroll + total_pnl - total_at_risk
    So current_bankroll is AVAILABLE CASH (already net of open positions).
    Equity = current_bankroll + total_at_risk = starting_bankroll + total_pnl.

    Rules:
    - equity >= 50% starting                          (risk_policy.md:20 circuit breaker)
    - open_positions < 10                              (mandate.md:16)
    - total_at_risk + new_stake <= equity * 0.60       (risk_policy.md:21)
    - current_bankroll - new_stake >= 10% starting     (mandate.md:18 reserve)
    """
    warnings: list[str] = []
    equity = snapshot.current_bankroll + snapshot.total_at_risk
    available_cash = snapshot.current_bankroll

    if equity < snapshot.starting_bankroll * 0.50:
        warnings.append(
            f"Circuit breaker: equity ${equity:,.0f} below 50% of "
            f"starting ${snapshot.starting_bankroll:,.0f}"
        )
    if snapshot.open_positions >= 10:
        warnings.append(f"Max open positions reached: {snapshot.open_positions}/10")
    if snapshot.total_at_risk + new_stake > equity * 0.60:
        warnings.append(
            f"Total at risk ${snapshot.total_at_risk + new_stake:,.0f} "
            f"exceeds 60% of equity ${equity:,.0f}"
        )
    remaining_cash = available_cash - new_stake
    reserve_min = snapshot.starting_bankroll * 0.10
    if remaining_cash < reserve_min:
        warnings.append(
            f"Remaining cash ${remaining_cash:,.0f} below "
            f"10% reserve ${reserve_min:,.0f}"
        )
    return len(warnings) == 0, warnings


def calculate_pnl(entry_price: float, stake: float, won: bool) -> float:
    """Compute profit/loss for a resolved binary bet.

    In prediction markets:
    - If won: payout = stake / entry_price, so PnL = payout - stake
    - If lost: PnL = -stake

    Args:
        entry_price: Price paid per share (0 < entry_price <= 1).
        stake: USD amount risked.
        won: Whether the chosen side resolved as the outcome.

    Returns:
        PnL in USD (positive for profit, negative for loss).
    """
    if won and entry_price > 0:
        payout = stake / entry_price
        return payout - stake
    return -stake
