# Prediction Process

## Evaluation Pipeline
1. **Event Scan** — Polymarket Agent scans active crypto markets, extracts odds and liquidity
2. **Data Collection** (parallel):
   - Market Data Agent — crypto prices, funding rates, open interest, Fear & Greed index
   - News Agent — sentiment from X/Twitter, news, narratives (Grok + Exa search)
3. **Risk Assessment** — Risk Agent (LLM) estimates probability, confidence, and reasoning
4. **Edge & Gate** — Deterministic code: computes edge, liquidity checks, correlation limits, risk rating
5. **Position Sizing** — Deterministic step: Kelly criterion, slippage estimate, entry price
6. **Decision** — Deterministic code: portfolio gates (circuit breaker, max positions, capital limits) → BET/SKIP
7. **Logging** — If BET: record paper trade in DB + write audit memo. If SKIP: trace only.

> **Principle:** LLM provides the forecast (probability + confidence). All math, risk checks,
> and BET/SKIP decisions are made by deterministic Python code.

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
