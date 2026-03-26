# Kelly Criterion for Prediction Markets

## Overview
The Kelly Criterion determines the optimal bet size to maximize long-term
bankroll growth. In prediction markets, it accounts for the edge between
our estimated probability and the market's implied probability.

## Formula

```
f* = (p * b - q) / b
```

Where:
- `f*` = fraction of bankroll to bet (raw Kelly)
- `p` = our estimated probability of the chosen side winning
- `q` = 1 - p (probability of losing)
- `b` = (1 / market_price) - 1 (decimal odds minus 1)

## Fractional Kelly

Raw Kelly is mathematically optimal but assumes perfect probability estimates.
In practice, we use **fractional Kelly (0.25x)** to account for estimation error.

```
actual_bet = f* × 0.25 × bankroll
```

### Why 0.25x?
- Full Kelly is too aggressive for imperfect probability estimates
- Quarter Kelly reduces variance by ~75% while capturing ~50% of growth rate
- This matches the risk tolerance for paper trading phase

## Example

If we estimate P(YES) = 0.60 and market price for YES is 0.45:
- b = (1/0.45) - 1 = 1.222
- f* = (0.60 × 1.222 - 0.40) / 1.222 = 0.273 (27.3% raw Kelly)
- Fractional: 0.273 × 0.25 = 0.068 (6.8% of bankroll)
- With $10K bankroll: $683 bet size

## Constraints
- Never exceed 20% of bankroll per bet regardless of Kelly output
- Minimum bet: 2% of bankroll ($200)
- If Kelly suggests negative fraction → SKIP (no edge)
