"""
Monitor Worker
--------------

Standalone process that checks open positions every 60 seconds.
Runs independently of AgentOS — no event-loop blocking, no startup-hook issues.

Usage:
    python -m app.monitor_worker
"""

import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("monitor.worker")

MONITOR_INTERVAL = int(os.getenv("MONITOR_INTERVAL", "60"))


def main() -> None:
    logger.info("Monitor worker starting (interval=%ds)", MONITOR_INTERVAL)

    # Import lazily so DB/settings init happens after env vars are set
    from app.monitor import run_monitor

    while True:
        try:
            result = run_monitor()
            if result["closed"]:
                logger.info(
                    "Closed %d trades: %s",
                    result["closed"],
                    [t["reason"] for t in result["trades_closed"]],
                )
            else:
                logger.debug(
                    "Checked %d trades, none closed", result["checked"],
                )
        except Exception:
            logger.exception("Monitor pass failed")

        time.sleep(MONITOR_INTERVAL)


if __name__ == "__main__":
    main()
