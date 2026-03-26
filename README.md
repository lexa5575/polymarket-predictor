# Polymarket Predictor

Polymarket Predictor is a multi-agent system built on Agno/AgentOS for analyzing crypto prediction markets on Polymarket. It produces structured `BET YES` / `BET NO` / `SKIP` recommendations with deterministic sizing, paper-trade recording, and audit memos.

Phase 1 is analytics plus paper trading only.

## Architecture

```text
AgentOS
├── Agents
│   ├── Polymarket Scanner   ── batch market scan
│   ├── Polymarket Agent     ── single-market details + orderbooks
│   ├── Market Data Agent    ── CoinGecko + Coinglass + Fear & Greed
│   ├── News Agent           ── Grok + Exa sentiment/news
│   ├── Risk Agent           ── qualitative risk + side recommendation
│   ├── Knowledge Agent      ── RAG over research + memo archive
│   ├── Logger Agent         ── audit memos only
│   └── Decision Agent       ── workflow-internal final decision
│
├── Teams (analytical only)
│   ├── Coordinate Team
│   ├── Route Team
│   ├── Broadcast Team
│   └── Task Team
│
└── Workflow
    └── Prediction Pipeline  ── Scan → Data+News → Risk → Sizing → Decision → Record
```

## Key Rules

- Teams are analytical only. Final actionable `BET/SKIP` decisions go through the workflow.
- `conditional_logging()` is the sole writer to `paper_trades`.
- Kelly, slippage, entry price, PnL, and Brier Score are deterministic Python logic, not LLM output.
- Phase 1 does not place real trades.

## Quick Start

### 1. Clone and configure

```sh
git clone <your-repo-url> polymarket-predictor
cd polymarket-predictor

cp example.env .env
```

Fill in the keys you plan to use:

```env
OPENAI_API_KEY=...
XAI_API_KEY=...
EXA_API_KEY=...
COINGLASS_API_KEY=...
```

### 2. Start services

```sh
docker compose up -d --build
```

This starts PostgreSQL with pgvector and the API server.

### 3. Load research into the knowledge base

```sh
docker exec -it polymarket-predictor-api python -m app.load_knowledge --recreate
```

### 4. Open the UI

1. Open [os.agno.com](https://os.agno.com)
2. Add a local OS pointing to `http://localhost:8000`
3. Connect

## What To Run

### Full workflow

Use the workflow for real pipeline decisions:

```text
Run full prediction analysis on the BTC $100K market
```

### Analytical teams

Use teams for research and synthesis, not for final execution:

```text
Analyze the BTC $100K prediction market — what's the edge?
```

### Useful API routes

- `POST /api/scan-and-fanout`
- `POST /api/settle`
- `GET /api/dashboard`

## Local Development

```sh
# install dependencies
uv sync --extra dev

# run tests
uv run pytest tests/test_math_utils.py tests/test_paper_trade.py tests/test_workflow_functions.py
uv run pytest tests/test_tools_unit.py tests/test_routes.py

# optional live tests
uv run pytest tests/test_tools_live.py -m live

# run app locally
uv run python -m app.main
```

## Project Structure

```text
polymarket-predictor/
├── agents/
├── app/
├── context/
├── db/
├── research/
├── schemas/
├── scripts/
├── storage/
├── teams/
├── tests/
├── tools/
├── workflows/
├── compose.yaml
├── Dockerfile
└── pyproject.toml
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | GPT-4o / GPT-4o-mini |
| `XAI_API_KEY` | Yes | Grok for the News Agent |
| `EXA_API_KEY` | Yes | Exa MCP web search |
| `COINGLASS_API_KEY` | No | Funding/OI data, graceful degradation if missing |
| `XAI_MODEL_ID` | No | Override default Grok model |
| `RUNTIME_ENV` | No | Set `dev` to expose the internal `decision_agent` in UI |
| `DB_HOST` | No | PostgreSQL host |
| `DB_PORT` | No | PostgreSQL port |
| `DB_USER` | No | PostgreSQL user |
| `DB_PASS` | No | PostgreSQL password |
| `DB_DATABASE` | No | PostgreSQL database |

## Deployment Notes

- Docker image name: `polymarket-predictor`
- Compose services: `polymarket-predictor-db`, `polymarket-predictor-api`
- If you deploy to Railway or another platform, update the service name to `polymarket-predictor`

## Learn More

- [Agno GitHub](https://github.com/agno-agi/agno)
- [Agno Documentation](https://docs.agno.com)
- [AgentOS Documentation](https://docs.agno.com/agent-os/introduction)
