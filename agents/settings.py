"""
Agent Settings
--------------

Shared instances used across all agents: knowledge bases, paths, URLs, and stores.
Import from here — never recreate these.
"""

from __future__ import annotations

from os import getenv
from pathlib import Path

from db import create_knowledge, db_url

# ---------------------------------------------------------------------------
# Knowledge bases — instantiated ONCE, imported everywhere
# ---------------------------------------------------------------------------
team_knowledge = create_knowledge("Crypto Knowledge", "crypto_team_knowledge")
team_learnings = create_knowledge("Crypto Learnings", "crypto_team_learnings")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MEMOS_DIR = Path(__file__).parent.parent / "memos"

# ---------------------------------------------------------------------------
# API keys & model config
# ---------------------------------------------------------------------------
EXA_API_KEY = getenv("EXA_API_KEY", "")
EXA_MCP_URL = f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}&tools=web_search_exa"

COINGLASS_API_KEY = getenv("COINGLASS_API_KEY", "")

# xAI model ID — configurable via env for easy upgrades
XAI_MODEL_ID = getenv("XAI_MODEL_ID", "grok-4-1-fast-non-reasoning")

# ---------------------------------------------------------------------------
# Paper trade store — lazy factory to avoid import-time DB connection
# ---------------------------------------------------------------------------
_paper_trade_store = None


def get_paper_trade_store():
    """Return a singleton PaperTradeStore instance (lazy init)."""
    global _paper_trade_store
    if _paper_trade_store is None:
        from storage.paper_trades import PaperTradeStore

        _paper_trade_store = PaperTradeStore(db_url)
    return _paper_trade_store
