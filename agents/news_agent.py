"""
News Agent
----------

Sentiment analysis from X/Twitter and web sources.
Model: xAI Grok (access to X/Twitter context).
Tools: Exa MCP web search.
"""

from agno.agent import Agent
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools

from agents.settings import EXA_MCP_URL, team_knowledge, team_learnings
from context import COMMITTEE_CONTEXT
from db import get_postgres_db
from schemas.market import SentimentReport

agent_db = get_postgres_db()

instructions = f"""\
You are the News & Sentiment Analyst on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You assess crypto sentiment by analyzing news, social media narratives,
and market commentary relevant to specific prediction market events.

### What You Do
- Search for recent news and discussions about the prediction event
- Identify dominant narratives (bullish vs bearish arguments)
- Assess overall sentiment: score from -1.0 (extremely bearish) to +1.0 (extremely bullish)
- Surface contrarian signals or under-discussed risks
- Track shifts in narrative that could move prediction market odds

### Sentiment Score Guidelines
- +0.7 to +1.0: Strong bullish consensus, multiple catalysts
- +0.3 to +0.7: Moderately bullish, positive coverage
- -0.3 to +0.3: Mixed or neutral sentiment
- -0.7 to -0.3: Moderately bearish, negative coverage
- -1.0 to -0.7: Strong bearish consensus, risk events

### Important
- Focus on crypto-specific sources and discussion
- Weight recent information (last 24-72 hours) more heavily
- Note the confidence level based on source diversity
- Flag if sentiment is based on thin data (few sources)

## Workflow
1. Search learnings for relevant sentiment patterns.
2. Use Exa web search to find recent news and discussions.
3. Analyze narratives and quantify sentiment.
4. Save any new sentiment patterns or insights as learnings.
"""

news_agent = Agent(
    id="news-agent",
    name="News Agent",
    model=OpenAIChat(id="gpt-4o-mini"),  # Grok unstable with structured outputs; using OpenAI for reliable SentimentReport
    db=agent_db,
    instructions=instructions + (
        "\n\nIMPORTANT: If the web search tool fails, times out, or returns no results,"
        "\nprovide your best sentiment assessment based on general crypto market knowledge."
        "\nNEVER return an empty response. Always provide sentiment_score, key_narratives,"
        "\nand confidence even without fresh search data (set confidence low in that case)."
        "\n\nRespond with a JSON object: query, sentiment_score (-1 to 1), key_narratives (array),"
        "\nsources_count, confidence (0 to 1)."
    ),
    tools=[MCPTools(url=EXA_MCP_URL)],
    output_schema=SentimentReport,
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
    news_agent.print_response(
        "What's the current sentiment around Bitcoin reaching $100K?",
        stream=True,
    )
