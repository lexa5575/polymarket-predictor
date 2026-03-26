# CLAUDE.md

This file provides context for Claude Code when working with this repository.

## Project Overview

**Polymarket Predictor** — A multi-agent system built on Agno that analyzes crypto prediction markets on Polymarket and produces BET YES / BET NO / SKIP recommendations with Kelly-based position sizing. Phase 1: paper trading only.

## Architecture

```
AgentOS (app/main.py)
├── Agents (7+1 specialists)
│   ├── Polymarket Scanner      — batch scan for crypto markets (GPT-4o-mini)
│   ├── Polymarket Agent        — single market details + orderbooks (GPT-4o-mini)
│   ├── Market Data Agent       — CoinGecko + Coinglass + Fear&Greed (GPT-4o-mini)
│   ├── News Agent              — sentiment from X/Twitter + web (Grok via xAI)
│   ├── Risk Agent              — probability estimation + risk rating (GPT-4o)
│   ├── Knowledge Agent         — RAG research + memo archive (GPT-4o-mini)
│   ├── Logger Agent            — audit memos only, no DB writes (GPT-4o-mini)
│   └── Decision Agent          — BET/SKIP with stake (GPT-4o, workflow-only)
│
├── Teams (4 analytical-only architectures)
│   ├── Coordinate Team         — dynamic orchestration
│   ├── Route Team              — single-agent dispatch
│   ├── Broadcast Team          — parallel evaluation
│   └── Task Team               — autonomous decomposition
│   NOTE: Teams are analytical only. All BET/SKIP decisions go through the workflow.
│
├── Workflow
│   └── Prediction Pipeline     — Event Scan → Data+News → Risk → Sizing → Decision → Record
│       ├── compute_position_sizing (deterministic: Kelly, slippage, entry price)
│       └── conditional_logging (sole DB writer for paper trades)
│
├── Custom Routes
│   ├── POST /api/scan-and-fanout  — batch scan + fan-out workflow runs
│   ├── POST /api/settle           — check resolved markets, update trades
│   └── GET  /api/dashboard        — bankroll snapshot + open positions
│
├── Schemas (Pydantic contracts)
│   ├── EventCandidate, BatchScanResult, MarketSnapshot, SentimentReport
│   ├── RiskAssessment, BetDecision
│   └── PaperTrade, BankrollSnapshot, PredictionRequest
│
├── Storage (deterministic, no LLM)
│   ├── PaperTradeStore — CRUD for paper trades (sole DB owner)
│   ├── math_utils      — Kelly, Brier Score, PnL, slippage
│   └── SQLAlchemy ORM  — paper_trades + bankroll_snapshots tables
│
└── Three-Layer Knowledge
    ├── Layer 1: Static Context   — mandate, risk policy, process (always in prompt)
    ├── Layer 2: Research Library  — crypto event templates, strategies (PgVector RAG)
    └── Layer 3: Memo Archive     — past prediction memos (FileTools, audit only)
```

## Key Design Decisions

1. **decision_agent is workflow-internal** — not registered in production AgentOS, only in dev mode
2. **Teams are analytical only** — never produce BET/SKIP decisions or stake sizes
3. **conditional_logging is the sole DB writer** — paper trades go through it exclusively
4. **Math is deterministic** — Kelly, Brier, PnL, slippage computed in Python, never by LLM
5. **Source of truth**: PostgreSQL paper_trades table. `memos/` = read-only audit artifacts
6. **Structured handoff**: workflow function steps use tagged JSON blocks, not regex on text

## Environment Variables

Required:
- `OPENAI_API_KEY` — for GPT-4o / GPT-4o-mini models
- `XAI_API_KEY` — for Grok (News Agent)
- `EXA_API_KEY` — for Exa web search

Optional:
- `COINGLASS_API_KEY` — for derivatives data (graceful degradation without it)
- `XAI_MODEL_ID` — override Grok model (default: grok-2-latest)
- `RUNTIME_ENV=dev` — enables hot-reload + registers decision_agent in UI
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, `DB_DATABASE`

## Commands

```bash
# Docker development
docker compose up -d --build

# Load knowledge base
python -m app.load_knowledge --recreate

# Run API server
python -m app.main

# Run tests
pytest tests/test_math_utils.py tests/test_paper_trade.py tests/test_workflow_functions.py
pytest tests/test_tools_unit.py tests/test_routes.py
pytest tests/test_tools_live.py -m live  # optional, requires network
```

## Conventions

- **Agent IDs are kebab-case** and match `config.yaml` keys
- **Never duplicate knowledge instances** — import from `agents.settings`
- **All instructions include `COMMITTEE_CONTEXT`** via f-string (Layer 1)
- **GPT-4o for reasoning agents**, GPT-4o-mini for data agents, Grok for News
- **Polymarket IDs**: `condition_id` = primary key, `token_id` = CLOB API, `gamma_market_id` = Gamma API
