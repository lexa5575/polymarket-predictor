# Prediction Market Basics

## How Polymarket Works

Polymarket is a binary prediction market on the Polygon blockchain.
Each market has a YES token and a NO token. Prices range from $0 to $1.

- If YES resolves true: YES token = $1, NO token = $0
- If NO resolves true: YES token = $0, NO token = $1
- YES price + NO price ≈ $1 (minus spread)

## Edge and Mispricing

**Edge** = our estimated probability minus the market's implied probability.
- Positive edge on YES: we think P(YES) > market_price_YES
- Positive edge on NO: we think P(NO) > market_price_NO

Minimum edge threshold: **5%** (avoid noise and transaction costs).

## Execution

### Entry
- Buy YES token at best_ask price (or NO token at its best_ask)
- Account for slippage: walking the orderbook for larger bets
- Entry price = best_ask + estimated_slippage

### PnL Calculation
- If won: PnL = stake × (1/entry_price - 1)
- If lost: PnL = -stake
- Example: buy YES at 0.45 for $500. If YES resolves:
  - Win: $500 × (1/0.45 - 1) = $500 × 1.222 = +$611
  - Lose: -$500

## Calibration — Brier Score

The Brier Score measures prediction accuracy:
```
Brier = (estimated_probability - outcome)²
```
Where outcome = 1 if our side won, 0 if lost.

- Perfect calibration: Brier = 0
- Random guessing (50%): Brier = 0.25
- Our target: Brier < 0.25 (better than random)

## Key Risks

1. **Liquidity risk** — thin orderbooks mean high slippage
2. **Resolution risk** — ambiguous outcomes can delay resolution
3. **Smart contract risk** — Polygon blockchain dependencies
4. **Information asymmetry** — insiders may have edge near resolution
5. **Correlation risk** — related events move together (BTC price cluster)
