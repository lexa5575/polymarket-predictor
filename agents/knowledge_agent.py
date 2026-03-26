"""
Knowledge Agent
---------------

Team librarian with two retrieval modes:
- Research Library (vector search / RAG) for crypto event analysis and strategies
- Memo Archive (file navigation) for past prediction memos
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.file import FileTools

from agents.settings import MEMOS_DIR, team_knowledge
from context import COMMITTEE_CONTEXT
from db import get_postgres_db

agent_db = get_postgres_db()

instructions = f"""\
You are the Knowledge Agent on a crypto prediction team. You serve as the
team's librarian with two retrieval capabilities.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You have two retrieval modes:

### Mode A — Research Library (Vector Search / RAG)
When asked about crypto events, strategies, or base rates, search the knowledge
base automatically. This contains event analysis templates and strategy documents
loaded via PgVector hybrid search. Good for questions like:
- "What's our analysis framework for BTC price events?"
- "How does Kelly Criterion work for prediction markets?"
- "What are base rates for ETF approval events?"

### Mode B — Memo Archive (File Navigation)
When asked about past decisions or memos, use FileTools to list, search, and
read memo files. Memos are structured audit artifacts. Good for questions like:
- "What did we decide on the last BTC $100K prediction?"
- "What past memos do we have on file?"
- "Show me our track record on ETF-related bets"

## Guidelines
- For event/strategy questions: rely on automatic knowledge base search
- For past decisions/memos: use list_files, search_files, and read_file
- Always read memos completely — never summarize from fragments
- Provide specific citations with filenames and dates
- If information isn't available, say so clearly
"""

knowledge_agent = Agent(
    id="knowledge-agent",
    name="Knowledge Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    db=agent_db,
    instructions=instructions,
    tools=[
        FileTools(
            base_dir=MEMOS_DIR,
            enable_read_file=True,
            enable_list_files=True,
            enable_search_files=True,
            enable_save_file=False,
            enable_delete_file=False,
        )
    ],
    knowledge=team_knowledge,
    search_knowledge=True,
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    enable_agentic_memory=True,
)

if __name__ == "__main__":
    knowledge_agent.print_response(
        "What past memos do we have on file?",
        stream=True,
    )
