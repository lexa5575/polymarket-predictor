# Risk Policy

## Position Limits
- Maximum single bet: 20% of bankroll ($2,000)
- Minimum bet size: 2% of bankroll ($200)
- Maximum 3 bets on correlated events (same underlier_group)

## Position Sizing — Kelly Criterion
- Use **fractional Kelly (0.25x)** for conservative sizing
- Kelly f* = (p × b − q) / b, where p = estimated_prob, b = (1/market_prob) − 1
- Apply 0.25 multiplier to raw Kelly for actual stake
- Never exceed maximum single bet limit regardless of Kelly output

## Conviction Tiers
- High conviction (edge > 15%): up to 15-20% of bankroll
- Standard (edge 10-15%): up to 5-15% of bankroll
- Exploratory (edge 5-10%): up to 2-5% of bankroll

## Portfolio Risk
- Circuit breaker: if bankroll drops below 50% of starting ($5,000) → pause all new bets
- Maximum total capital at risk: 60% of bankroll across all open positions
- Correlation limit: no more than 3 open bets in the same underlier_group

## Liquidity & Execution
- Do not bet if bid-ask spread > 5%
- Slippage budget: maximum 2% of stake amount
- Entry price = best_ask + estimated_slippage (walk the orderbook)
- For NO side: entry price = no_token best_ask + estimated_slippage

## Rebalancing
- Review all open positions every 4 hours (automated scan)
- Settlement check every 12 hours
- Exit conditions from the prediction workflow are monitored continuously

> **Note:** Risk ratings, edge, liquidity checks, and portfolio gates are computed
> deterministically by code in the prediction workflow, not by LLM agents.
