"""
Coordinate Team
---------------

Chair (GPT-4o) dynamically orchestrates agents based on the question.
Best for: open-ended crypto prediction questions.
"""

from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
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
from agents.settings import team_learnings

coordinate_team = Team(
    id="coordinate-team",
    name="Crypto Team - Coordinate",
    mode=TeamMode.coordinate,
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
        "You are the coordinator of a crypto prediction team.",
        "Dynamically decide which agents to consult based on the question.",
        "For prediction evaluations: start with Polymarket + Market Data agents, then News, then Risk.",
        "Always consult the Risk Agent before drawing any conclusions.",
        "Provide ANALYTICAL RECOMMENDATIONS ONLY — probability estimates, edge, risk rating.",
        "NEVER output final BET/SKIP decisions or stake sizes. Those are computed by the prediction workflow.",
        "Your role is to gather and synthesize information, not to execute trades.",
    ],
    learning=LearningMachine(
        knowledge=team_learnings,
        learned_knowledge=LearnedKnowledgeConfig(
            mode=LearningMode.AGENTIC,
            namespace="global",
        ),
    ),
    show_members_responses=True,
    markdown=True,
)
