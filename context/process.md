# Prediction Process

## Evaluation Pipeline
1. **Event Scan** — Polymarket Agent scans active crypto markets, extracts odds and liquidity
2. **Data Collection** (parallel):
   - Market Data Agent — crypto prices, funding rates, open interest, Fear & Greed index
   - News Agent — sentiment from X/Twitter, news, narratives (Grok + Exa search)
3. **Risk Assessment** — Risk Agent evaluates edge, recommends side (YES/NO), rates risk
4. **Position Sizing** — Deterministic step: Kelly criterion, slippage estimate, entry price
5. **Decision** — Decision Agent: BET YES / BET NO / SKIP with stake and exit conditions
6. **Logging** — If BET: record paper trade in DB + write audit memo. If SKIP: trace only.

## Decision Framework
- **BET YES:** Positive edge on YES outcome, acceptable risk, sufficient liquidity
- **BET NO:** Positive edge on NO outcome, acceptable risk, sufficient liquidity
- **SKIP:** Insufficient edge (<5%), unacceptable risk, poor liquidity, or mandate violation

## Execution Model
- YES bet: buy YES token at best_ask + slippage. PnL if won = stake × (1/entry_price − 1).
- NO bet: buy NO token at best_ask + slippage. PnL if won = stake × (1/entry_price − 1).
- Loss in both cases: −stake.

## Documentation
Every BET decision is recorded with: event question, condition_id, side, estimated
probability, market probability, edge, stake, entry price, rationale, exit conditions,
and confidence level. SKIP decisions are logged as trace events only.

## Source of Truth
- **PostgreSQL paper_trades table** = operational source of truth for all trades
- **memos/ directory** = human-readable audit artifacts (read-only after creation)
