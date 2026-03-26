"""
Polymarket Predictor
--------------------

A multi-agent system for crypto prediction markets on Polymarket.

Run:
    python -m app.main
"""

import os
from pathlib import Path

from agno.os import AgentOS

from agents import (
    knowledge_agent,
    logger_agent,
    market_data_agent,
    news_agent,
    polymarket_agent,
    polymarket_scanner,
    risk_agent,
)
from app.routes import router
from db import get_postgres_db
from teams import broadcast_team, coordinate_team, route_team, task_team
from workflows import prediction_workflow

IS_DEBUG = os.getenv("RUNTIME_ENV", "") == "dev"

# ---------------------------------------------------------------------------
# Create AgentOS
# ---------------------------------------------------------------------------

# Public agents — visible in UI and directly callable
_agents = [
    polymarket_scanner,
    polymarket_agent,
    market_data_agent,
    news_agent,
    risk_agent,
    knowledge_agent,
    logger_agent,
]

# decision_agent is internal to the prediction workflow.
# Only register it in AgentOS when running in debug/dev mode.
if IS_DEBUG:
    from agents import decision_agent
    _agents.append(decision_agent)

agent_os = AgentOS(
    name="Polymarket Predictor",
    tracing=True,
    scheduler=True,
    scheduler_poll_interval=15,
    db=get_postgres_db(),
    agents=_agents,
    teams=[coordinate_team, route_team, broadcast_team, task_team],
    workflows=[prediction_workflow],
    config=str(Path(__file__).parent / "config.yaml"),
)

app = agent_os.get_app()
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    agent_os.serve(app="app.main:app", port=port, reload=IS_DEBUG)
