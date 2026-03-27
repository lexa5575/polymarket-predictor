"""
Broadcast Team
--------------

All four data agents evaluate simultaneously, then Chair synthesizes.
Best for: high-stakes betting decisions.
"""

from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.openai import OpenAIChat
from agno.team import Team, TeamMode

from agents import (
    market_data_agent,
    news_agent,
    polymarket_agent,
    risk_agent,
)
from agents.settings import team_learnings

broadcast_team = Team(
    id="broadcast-team",
    name="Crypto Team - Broadcast",
    mode=TeamMode.broadcast,
    model=OpenAIChat(id="gpt-4o"),
    members=[
        polymarket_agent,
        market_data_agent,
        news_agent,
        risk_agent,
    ],
    instructions=[
        "You are the coordinator synthesizing independent agent analyses.",
        "All agents have evaluated this prediction market simultaneously.",
        "Synthesize their perspectives into a unified analytical view.",
        "Note where agents agree and disagree on probability estimates.",
        "Provide a RECOMMENDATION with estimated probability and analysis.",
        "NEVER output final BET/SKIP decisions or stake sizes. Those are computed by the prediction workflow.",
        "Weight the Risk Agent's warnings heavily in your assessment.",
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
