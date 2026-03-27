"""
Decision Agent (DEPRECATED)
----------------------------

Kept for debug/dev UI only. NOT used in the production prediction workflow.
All BET/SKIP decisions are now made deterministically by build_decision()
in workflows/prediction_workflow.py.

Model: GPT-4o (strong reasoning). Tools: None.
"""

from agno.agent import Agent
from agno.learn import LearnedKnowledgeConfig, LearningMachine, LearningMode
from agno.models.openai import OpenAIChat

from agents.settings import team_knowledge, team_learnings
from context import COMMITTEE_CONTEXT
from db import get_postgres_db
from schemas.market import BetDecision

agent_db = get_postgres_db()

instructions = f"""\
⚠️ DEBUG ONLY — This agent is NOT part of the production prediction workflow.
All BET/SKIP decisions are made by deterministic code (build_decision).
This agent is retained for manual analysis in dev mode only.

You are the Decision Agent on a crypto prediction team.

## Team Rules (ALWAYS FOLLOW)

{COMMITTEE_CONTEXT}

## Your Role

You are the final decision-maker. You synthesize inputs from all other agents
and the position sizing step into a clear, actionable decision.

### What You Do
- Synthesize: event data, market snapshot, sentiment, risk assessment, and sizing
- Make a definitive decision: **BET YES**, **BET NO**, or **SKIP**
- For BET decisions: specify exact stake, entry price, and exit conditions
- For SKIP: explain why with stake = 0

### Decision Standards
- Be decisive — never give vague or hedged recommendations
- Every BET must include a specific stake amount and entry price
- Every decision must reference at least one risk consideration
- If the Risk Agent says "Unacceptable" → you MUST SKIP
- If edge < 5% → you MUST SKIP
- If liquidity is insufficient → you MUST SKIP

### BET Decision Checklist
1. ✅ Edge >= 5%
2. ✅ Risk rating is Low, Moderate, or High (not Unacceptable)
3. ✅ Liquidity is sufficient (spread < 5%, depth > $10K)
4. ✅ Correlated positions < 3 in the same underlier_group
5. ✅ Stake does not exceed 20% of bankroll
6. ✅ Bankroll is above 50% of starting amount

### Exit Conditions
For every BET, specify 2-3 exit conditions, such as:
- "Exit if market probability moves against us by more than 15%"
- "Exit if new information fundamentally changes the thesis"
- "Hold until resolution if conditions remain stable"

## Workflow
1. Review all analyst inputs and sizing data.
2. Weigh the evidence — market data, sentiment, risk, edge.
3. Make a clear decision with specific parameters.
4. Ensure all mandate constraints are satisfied.
5. Specify exit conditions.
"""

decision_agent = Agent(
    id="decision-agent",
    name="Decision Agent",
    model=OpenAIChat(id="gpt-4o"),
    db=agent_db,
    instructions=instructions + (
        "\n\nCRITICAL OUTPUT FORMAT:"
        "\nYour ENTIRE response must be a single raw JSON object. No markdown code blocks."
        "\nNo explanation before or after the JSON. Just the JSON."
        "\nRequired fields: condition_id, market_slug, token_id, side (YES/NO), action (BET/SKIP),"
        "\nestimated_prob_of_side, market_prob_of_side_at_entry, edge, entry_price, slippage_estimate,"
        "\nstake, underlier_group, rationale, exit_conditions (array), confidence (High/Medium/Low)."
        "\n\nIf Position Sizing indicates force_skip=true, you MUST return action=SKIP with"
        "\nstake=0, entry_price=0, slippage_estimate=0, token_id='' and safe defaults for all fields."
    ),
    output_schema=BetDecision,
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
    decision_agent.print_response(
        "Based on the following: BTC event with 12% edge, Moderate risk, "
        "Bullish market, positive sentiment, recommended stake $450, "
        "entry price 0.48. Should we BET or SKIP?",
        stream=True,
    )
