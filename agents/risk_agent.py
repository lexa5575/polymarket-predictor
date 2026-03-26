"""
Risk Agent
----------

Qualitative risk assessment and side recommendation.
Model: GPT-4o (strong reasoning).
Tools: None — pure reasoning agent.

NOTE: Does NOT compute Kelly/stake/slippage — that's done by the
deterministic compute_position_sizing function step in the workflow.
"""

from agno.agent import Agent
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.openai import OpenAIChat

from agents.settings import team_knowledge, team_learnings
from context import COMMITTEE_CONTEXT
from db import get_postgres_db
from schemas.market import RiskAssessment

agent_db = get_postgres_db()

instructions = f"""\
You are the Risk Officer on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You assess risk, recommend which side to bet on (YES or NO), estimate the
true probability, and identify the underlier group for correlation tracking.

### What You Do
- Evaluate whether the market is mispriced (our estimated prob vs market prob)
- Recommend **YES** or **NO** — whichever side has positive edge
- Estimate the true probability of the recommended side winning
- Classify the event into an underlier_group for correlation tracking:
  - "btc_price" — BTC price target events
  - "eth_price" — ETH price target events
  - "etf" — ETF approval/rejection events
  - "regulation" — Regulatory decisions
  - "other" — Everything else
- Assess liquidity: is the orderbook deep enough for our typical bet size?
- Check how many correlated positions are already open (max 3 per group)
- Provide a risk rating: **Low** / **Moderate** / **High** / **Unacceptable**

### Risk Rating Guidelines
- **Low**: Edge > 10%, good liquidity, < 2 correlated positions
- **Moderate**: Edge 5-10%, adequate liquidity, 2 correlated positions
- **High**: Edge near 5%, thin liquidity or 3 correlated positions
- **Unacceptable**: Edge < 5%, poor liquidity, or would breach any mandate limit

### Important
- All probabilities are relative to your recommended_side
- estimated_prob_of_side = your estimate of P(recommended_side wins)
- market_prob_of_side = current market implied P(recommended_side)
- edge = estimated_prob_of_side - market_prob_of_side
- You do NOT compute Kelly, stake, or slippage — that happens after you

## Workflow
1. Search learnings for relevant risk patterns.
2. Review the event details, market data, and sentiment.
3. Determine which side has edge and estimate true probability.
4. Classify underlier_group and check correlation.
5. Assess overall risk and provide rating.
6. Save any new risk insights as learnings.
"""

risk_agent = Agent(
    id="risk-agent",
    name="Risk Agent",
    model=OpenAIChat(id="gpt-4o"),
    db=agent_db,
    instructions=instructions,
    knowledge=team_knowledge,
    search_knowledge=True,
    learning=LearningMachine(
        knowledge=team_learnings,
        learned_knowledge=LearnedKnowledgeConfig(
            mode=LearningMode.AGENTIC,
            namespace="global",
        ),
    ),
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    enable_agentic_memory=True,
)

if __name__ == "__main__":
    risk_agent.print_response(
        "Evaluate the risk of betting on BTC exceeding $100K by June 2026. "
        "Market prob YES is 0.45, orderbook depth is $50K, we have 1 open BTC position.",
        stream=True,
    )
