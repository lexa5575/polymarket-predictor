"""
Polymarket Agent + Scanner
--------------------------

Two agent instances sharing the same tools:
- polymarket_scanner: batch_scan mode → BatchScanResult
- polymarket_agent: single_market mode → EventCandidate

Tools: PolymarketTools.
"""

from agno.agent import Agent
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.openai import OpenAIChat

from agents.settings import team_knowledge, team_learnings
from context import COMMITTEE_CONTEXT
from db import get_postgres_db
from schemas.market import BatchScanResult, EventCandidate
from tools.polymarket import PolymarketTools

agent_db = get_postgres_db()

_base_instructions = f"""\
You are the Polymarket Specialist on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You interact with Polymarket to discover crypto prediction markets,
extract implied probabilities, assess liquidity, and retrieve orderbook data.

### Key Concepts
- **condition_id**: Primary market identifier used across the system
- **token_id**: Separate for YES and NO — used for CLOB orderbook/price queries
- **gamma_market_id**: Gamma API identifier for market lookup
- **market_slug**: Human-readable identifier for logs

### What You Do
- Scan active crypto markets on Polymarket
- Extract implied probability (midpoint of best bid/ask)
- Assess liquidity (orderbook depth, spread, 24h volume)
- Retrieve full orderbooks for both YES and NO tokens
- Flag markets that violate mandate constraints (low liquidity, wide spread)

## Workflow
1. Search learnings for relevant patterns and past insights.
2. Use PolymarketTools to fetch market data.
3. Calculate spreads and liquidity metrics.
4. Save any new patterns or insights as learnings.
"""

_polymarket_tools = [PolymarketTools()]

_learning = LearningMachine(
    knowledge=team_learnings,
    learned_knowledge=LearnedKnowledgeConfig(
        mode=LearningMode.AGENTIC,
        namespace="global",
    ),
)

# Single-market agent (used inside prediction workflow)
polymarket_agent = Agent(
    id="polymarket-agent",
    name="Polymarket Agent",
    model=OpenAIChat(id="gpt-4.1-nano"),
    db=agent_db,
    instructions=_base_instructions + "\nReturn detailed data for ONE specific market including both YES and NO orderbooks.",
    tools=_polymarket_tools,
    output_schema=EventCandidate,
    knowledge=team_knowledge,
    search_knowledge=True,
    learning=_learning,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=False,
    enable_agentic_memory=True,
)

# Batch scanner agent (used by /api/scan-and-fanout)
polymarket_scanner = Agent(
    id="polymarket-scanner",
    name="Polymarket Scanner",
    model=OpenAIChat(id="gpt-4.1-nano"),
    db=agent_db,
    instructions=_base_instructions + "\nScan for the best crypto market candidates. Rank by liquidity and edge potential.",
    tools=_polymarket_tools,
    output_schema=BatchScanResult,
    knowledge=team_knowledge,
    search_knowledge=True,
    learning=_learning,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    enable_agentic_memory=True,
)

if __name__ == "__main__":
    polymarket_agent.print_response(
        "Get full market details with orderbooks for a BTC price prediction market",
        stream=True,
    )
