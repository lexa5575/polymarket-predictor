#!/usr/bin/env python3
"""One-time migration: add exit tracking columns to paper_trades table.

Run on VPS to preserve existing data:
    python3 scripts/migrate_add_exit_columns.py

Safe to run multiple times (IF NOT EXISTS).
"""

import os
import sys

from sqlalchemy import create_engine, text

# Build DB URL from env (same logic as db/url.py)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "ai")
DB_PASS = os.getenv("DB_PASS", "ai")
DB_DATABASE = os.getenv("DB_DATABASE", "ai")
DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"

ALTER_STATEMENTS = [
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_price FLOAT",
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_reason VARCHAR",
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_time TIMESTAMPTZ",
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS take_profit_pct FLOAT",
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS stop_loss_pct FLOAT",
    "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS max_hold_seconds FLOAT",
]

BACKFILL = """
UPDATE paper_trades
SET take_profit_pct = 0.10, stop_loss_pct = -0.05, max_hold_seconds = 1800
WHERE status = 'open' AND take_profit_pct IS NULL
"""


def main():
    print(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_DATABASE}...")
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        for stmt in ALTER_STATEMENTS:
            print(f"  {stmt}")
            conn.execute(text(stmt))

        print(f"  Backfilling exit policy for open trades...")
        result = conn.execute(text(BACKFILL))
        print(f"  Updated {result.rowcount} rows")

    print("Migration complete.")


if __name__ == "__main__":
    main()
