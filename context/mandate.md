# Prediction Market Mandate

> All constraints below are enforced by deterministic code in the prediction workflow.

## Fund Overview
- **Bankroll:** $10,000 (paper trading)
- **Asset Class:** Polymarket prediction markets — crypto category only
- **Benchmark:** Calibrated probability forecasting (target Brier Score < 0.25)
- **Horizon:** Event-specific (days to months)

## Focus Areas (Phase 1)
- BTC/ETH price targets by specific dates
- Crypto ETF approvals and regulatory decisions
- Bitcoin halving-related events
- Major protocol upgrades and forks

## Constraints
- Maximum 10 open positions at any time
- Minimum market liquidity: $50K 24h volume AND orderbook depth > $10K
- Must maintain 10% bankroll reserve ($1,000) — never risk more than 90%
- Minimum edge: |estimated_prob - market_prob| >= 5%

## Prohibited (Phase 1)
- Political events
- Sports events
- Markets with bid-ask spread > 5%
- Markets resolving in less than 24 hours
- Markets with less than $10K orderbook depth

## Approved Event Categories
Crypto price targets, ETF approvals, regulatory decisions, halving events,
protocol upgrades, exchange listings, stablecoin events
