"""
Route Team
----------

Routes each question to exactly one specialist.
Best for: quick, targeted questions.
"""

from agno.models.openai import OpenAIChat
from agno.team import Team, TeamMode

from agents import (
    knowledge_agent,
    logger_agent,
    market_data_agent,
    news_agent,
    polymarket_agent,
    risk_agent,
)

route_team = Team(
    id="route-team",
    name="Crypto Team - Route",
    mode=TeamMode.route,
    model=OpenAIChat(id="gpt-4o"),
    members=[
        polymarket_agent,
        market_data_agent,
        news_agent,
        risk_agent,
        knowledge_agent,
        logger_agent,
    ],
    instructions=[
        "Route each question to exactly one specialist:",
        "- Polymarket odds/events/markets → Polymarket Agent",
        "- Crypto prices/funding/OI/Fear&Greed → Market Data Agent",
        "- News/sentiment/X-Twitter → News Agent",
        "- Risk/edge/probability assessment → Risk Agent",
        "- Past memos/research/strategies → Knowledge Agent",
        "- Write an audit memo → Logger Agent",
        "For actual BET/SKIP decisions with trade execution, use the prediction workflow — not this team.",
    ],
    show_members_responses=True,
    markdown=True,
)
