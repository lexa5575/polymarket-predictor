"""
Risk Agent
----------

Probability estimation only. Model: GPT-4o (strong reasoning).
Tools: None — pure reasoning agent.

NOTE: This agent returns only RiskEstimate (estimated_prob, confidence, reasoning).
Edge, risk rating, liquidity, and correlation checks are computed deterministically
by the compute_edge_and_gate workflow step.
"""

from agno.agent import Agent
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.openai import OpenAIChat

from agents.settings import team_knowledge, team_learnings
from schemas.market import RiskEstimate
from context import COMMITTEE_CONTEXT
from db import get_postgres_db

agent_db = get_postgres_db()

instructions = f"""\
You are the Risk Analyst on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You estimate the probability that a given side of a prediction market will win,
and express your confidence in that estimate. You do NOT compute edge, risk ratings,
or position sizes — those are handled by deterministic code after you.

### What You Do
- Evaluate whether the market is mispriced based on available evidence
- Recommend **YES** or **NO** — whichever side you believe is more likely to win
- Estimate the true probability of the recommended side winning (0.0 to 1.0)
- Express your confidence in this estimate: **High**, **Medium**, or **Low**
- Classify the event into an underlier_group for correlation tracking:
  - "btc_price" — BTC price target events
  - "eth_price" — ETH price target events
  - "etf" — ETF approval/rejection events
  - "regulation" — Regulatory decisions
  - "other" — Everything else
- Explain your reasoning clearly
- Flag any qualitative warnings (e.g. "Event too close to resolution", "Ambiguous resolution criteria")

### Important
- estimated_prob_of_side = your estimate of P(recommended_side wins)
- You do NOT compute edge, risk rating, Kelly, stake, or slippage — that happens after you
- Focus on the quality of your probability estimate and the reasoning behind it

## Workflow
1. Search learnings for relevant patterns.
2. Review the event details, market data, and sentiment.
3. Determine which side is more likely and estimate true probability.
4. Classify underlier_group for correlation tracking.
5. Provide clear reasoning for your estimate.
6. Save any new insights as learnings.
"""

risk_agent = Agent(
    id="risk-agent",
    name="Risk Agent",
    model=OpenAIChat(id="gpt-4.1"),
    db=agent_db,
    instructions=instructions + (
        "\n\nCRITICAL: Your response must be a single raw JSON object with ALL required fields."
        "\nNo markdown. No explanation outside JSON."
        "\nRequired: condition_id, recommended_side, estimated_prob_of_side,"
        "\nconfidence, underlier_group, reasoning, warnings (array)."
        "\n\nIf market data is missing or unclear, return a COMPLETE valid JSON with:"
        "\nrecommended_side=YES, estimated_prob_of_side=0.5,"
        "\nconfidence=Low, underlier_group=other,"
        "\nreasoning=\"Insufficient data for reliable estimate\","
        "\nwarnings=[\"Market data missing\"]."
    ),
    output_schema=RiskEstimate,
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
    markdown=False,
    enable_agentic_memory=True,
)

if __name__ == "__main__":
    risk_agent.print_response(
        "Evaluate the probability of BTC exceeding $100K by June 2026. "
        "Market prob YES is 0.45, orderbook depth is $50K, we have 1 open BTC position.",
        stream=True,
    )
