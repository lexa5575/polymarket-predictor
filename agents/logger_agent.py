"""
Logger Agent
------------

Writes human-readable audit memos to memos/ directory.
Does NOT write to paper_trades DB — that's done by conditional_logging().
Tools: FileTools (save + read memos).
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.file import FileTools

from agents.settings import MEMOS_DIR
from context import COMMITTEE_CONTEXT
from db import get_postgres_db

agent_db = get_postgres_db()

instructions = f"""\
You are the Logger on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You write human-readable audit memos for BET decisions. These memos are
stored in the memos/ directory and serve as the team's historical record.

### What You Do
- Take the BET decision details and write a structured memo
- Include all relevant analysis from other agents
- Format for human readability
- Save the memo with the correct naming convention

### Memo Format
Every memo must include these sections:
1. **Event** — question, condition_id, market_slug, end_date
2. **Decision** — BET YES/NO, stake, entry_price, confidence
3. **Edge Analysis** — estimated_prob vs market_prob, edge
4. **Market Context** — price data, sentiment, Fear & Greed
5. **Risk Assessment** — risk rating, underlier_group, warnings
6. **Exit Conditions** — when to reconsider or exit
7. **Rationale** — the reasoning behind the decision

### File Naming Convention
Save memos as: `{{market_slug}}_{{date}}_{{side}}.md`
Examples: `btc-100k-june-2026_2026-03-26_yes.md`

### Important
- You do NOT record trades in the database — that happens elsewhere
- You ONLY write memo files for audit purposes
- Read existing memos to maintain consistent format
"""

logger_agent = Agent(
    id="logger-agent",
    name="Logger Agent",
    model=OpenAIChat(id="gpt-4.1-nano"),
    db=agent_db,
    instructions=instructions,
    tools=[
        FileTools(
            base_dir=MEMOS_DIR,
            enable_save_file=True,
            enable_read_file=True,
            enable_list_files=True,
            enable_search_files=True,
            enable_delete_file=False,
        )
    ],
    add_datetime_to_context=True,
    add_history_to_context=True,
    num_history_runs=5,
    markdown=True,
    enable_agentic_memory=True,
)

if __name__ == "__main__":
    logger_agent.print_response(
        "Write a memo for a BET YES on BTC exceeding $100K by June 2026. "
        "Stake $500, entry price 0.48, edge 12%, confidence High.",
        stream=True,
    )
