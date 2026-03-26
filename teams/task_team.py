"""
Task Team
---------

Chair autonomously decomposes complex tasks with dependencies.
Best for: multi-step prediction analysis and batch evaluations.
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

task_team = Team(
    id="task-team",
    name="Crypto Team - Tasks",
    mode=TeamMode.tasks,
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
        "Decompose complex prediction tasks into sub-tasks with dependencies.",
        "Assign each sub-task to the most appropriate agent.",
        "Parallelize independent tasks (e.g., market data + news sentiment).",
        "Ensure risk assessment happens after data collection is complete.",
        "Logging should be the final step after all analysis is complete.",
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
