# Sample Questions — Polymarket Predictor

Use these to test each architecture and agent. Questions are grouped by which agent/team/workflow they target.

---

## Route Team (Single-agent dispatch)

These should route to exactly one specialist.

1. **What crypto prediction markets are active on Polymarket?** → Polymarket Agent
2. **What's the current Bitcoin price and 24h change?** → Market Data Agent
3. **What's the sentiment around BTC reaching $100K?** → News Agent
4. **What's the risk of betting YES on BTC $100K by June 2026?** → Risk Agent
5. **What past prediction memos do we have on file?** → Knowledge Agent
6. **How does Kelly Criterion work for prediction markets?** → Knowledge Agent
7. **Write an audit memo for our last BTC bet.** → Logger Agent
8. **What is the estimated edge on the ETH $5K prediction market?** → Risk Agent
9. **What's the Crypto Fear & Greed Index right now?** → Market Data Agent
10. **What's the funding rate for ETH?** → Market Data Agent

---

## Coordinate Team (Chair orchestrates dynamically)

Open-ended prediction questions that benefit from multi-agent analysis.

11. **Should we bet on BTC exceeding $100K by June 2026?**
12. **Analyze the ETH ETF approval prediction market.**
13. **Is there edge in any BTC price prediction markets right now?**
14. **Should we bet YES or NO on the crypto regulation market?**
15. **What's the best crypto prediction opportunity right now?**

---

## Broadcast Team (Parallel independent evaluation)

High-stakes decisions where you want independent agent opinions.

16. **Full team review: BTC $100K by June 2026 — what is the edge and risk?**
17. **All agents: evaluate the ETH $5K prediction market.**
18. **Committee analysis: is BTC $150K mispriced?**
19. **Independent evaluation: BTC halving impact prediction market.**

---

## Task Team (Autonomous decomposition)

Complex multi-step tasks that require planning and parallel work.

20. **Analyze the top 5 crypto prediction markets and rank them by edge.**
21. **Build a ranked watchlist across BTC and ETH events, max $3K hypothetical risk.**
22. **Review all open positions and recommend which to exit.**
23. **Compare BTC $100K, $120K, and $150K prediction markets — where's the best edge?**
24. **Evaluate all ETF-related prediction markets and rank by edge.**

---

## Prediction Workflow (Deterministic pipeline)

Full prediction reviews that follow the fixed 7-step process.

25. **Run full prediction analysis on the BTC $100K market.**
26. **Process the ETH $5K prediction market through the pipeline.**
27. **Standard review: evaluate the BTC halving impact market.**
28. **Run full analysis on the crypto regulation prediction market.**

---

## Stress Tests (Edge cases and mandate compliance)

Questions that test risk limits, mandate rules, and paper trading logic.

29. **Should we bet $3,000 on one market?** (Exceeds 20% max bet)
30. **We have 3 open BTC price bets. Can we add another?** (Tests correlation limit)
31. **Analyze a market with $5K 24h volume.** (Below $50K liquidity minimum)
32. **What's the maximum we can bet on any single event?** (Tests Layer 1 knowledge)
33. **What happens if our bankroll drops below $5,000?** (Tests circuit breaker)
34. **A market has 8% bid-ask spread — should we still bet?** (Tests spread limit)

---

## Knowledge & Memory Tests

Questions that test the three-layer knowledge system.

35. **What does our research say about BTC price prediction events?** (Layer 2 — RAG)
36. **Pull up our last BTC prediction memo.** (Layer 3 — FileTools)
37. **What are our betting position limits?** (Layer 1 — should be in every agent's context)
38. **What event categories do we cover?** (Layer 2 — RAG discovery)
39. **Summarize all past prediction decisions.** (Layer 3 — multi-memo retrieval)
40. **What is our prediction market mandate?** (Layer 1 — system prompt context)
