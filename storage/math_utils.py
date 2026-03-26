"""
Deterministic math utilities for paper trading.

All financial calculations live here — never inside an LLM agent.
"""

from __future__ import annotations


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
