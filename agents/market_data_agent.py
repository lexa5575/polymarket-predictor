"""
Market Data Agent
-----------------

Aggregates crypto market data from CoinGecko, Coinglass, and Fear & Greed Index.
Graceful degradation if Coinglass is unavailable (fields become None).
Tools: CoinGeckoTools, CoinglassTools, FearGreedTools.
"""

from agno.agent import Agent
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.anthropic import Claude

from agents.settings import COINGLASS_API_KEY, team_knowledge, team_learnings
from context import COMMITTEE_CONTEXT
from db import get_postgres_db
from schemas.market import MarketSnapshot
from tools.coingecko import CoinGeckoTools
from tools.coinglass import CoinglassTools
from tools.fear_greed import FearGreedTools

agent_db = get_postgres_db()

instructions = f"""\
You are the Market Data Analyst on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You aggregate crypto market data to provide context for prediction decisions.

### What You Do
- Fetch current crypto prices, 24h changes, and market caps (CoinGecko)
- Retrieve perpetual funding rates and open interest (Coinglass — may be unavailable)
- Check the Crypto Fear & Greed Index for overall market sentiment
- Provide a market signal: **Bullish** / **Neutral** / **Bearish**

### Signal Guidelines
- Bullish: Positive funding + rising OI + Greed index + price uptrend
- Bearish: Negative funding + declining OI + Fear index + price downtrend
- Neutral: Mixed signals or insufficient data

### Important
- If Coinglass data is unavailable, set funding_rate and open_interest to null
- Never fabricate data — report what you can actually retrieve
- Focus on the coin most relevant to the prediction market being evaluated

## Workflow
1. Search learnings for relevant patterns and past insights.
2. Use CoinGecko for price and market data.
3. Use Coinglass for derivatives data (if available).
4. Use Fear & Greed Index for sentiment.
5. Synthesize into a clear market signal.
6. Save any new patterns or insights as learnings.
"""

market_data_agent = Agent(
    id="market-data-agent",
    name="Market Data Agent",
    model=Claude(id="claude-haiku-4-5-20251001"),
    db=agent_db,
    instructions=instructions,
    tools=[
        CoinGeckoTools(),
        CoinglassTools(api_key=COINGLASS_API_KEY),
        FearGreedTools(),
    ],
    # No output_schema — xAI doesn't reliably support it; ensure_data_quality validates
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
    market_data_agent.print_response(
        "What's the current market data for Bitcoin?",
        stream=True,
    )
