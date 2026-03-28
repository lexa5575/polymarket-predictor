"""Exit policy for short-term trading strategy.

Saved per-trade at entry for reproducibility.
v1: global constants + dynamic max_hold.
v2: strategy profiles passed from entry layer.
"""

from __future__ import annotations

from datetime import datetime, timezone

TAKE_PROFIT_PCT = 0.10      # +10% of stake
STOP_LOSS_PCT = -0.05       # -5% of stake
MAX_HOLD_SECONDS = 1800     # default 30 min (fallback for legacy trades)

# Dynamic max_hold bounds
MIN_HOLD_SECONDS = 7200     # 2 hours minimum
MAX_HOLD_CAP = 86400        # 24 hours maximum
HOLD_FRACTION = 0.40        # hold up to 40% of time to resolution


def compute_max_hold(end_date_str: str) -> float:
    """Compute dynamic max_hold based on time to resolution.

    Holds up to 40% of time remaining, clamped to 2h-24h range.
    For a market resolving in 3 days: 3*24*0.4 = 28.8h → capped at 24h.
    For a market resolving in 6 hours: 6*0.4 = 2.4h → ok.
    For a market resolving in 2 hours: 2*0.4 = 0.8h → floored at 2h.
    """
    try:
        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        seconds_to_resolution = (end_date - now).total_seconds()
        if seconds_to_resolution <= 0:
            return float(MIN_HOLD_SECONDS)
        dynamic = seconds_to_resolution * HOLD_FRACTION
        return float(max(MIN_HOLD_SECONDS, min(dynamic, MAX_HOLD_CAP)))
    except (ValueError, TypeError):
        return float(MAX_HOLD_SECONDS)  # fallback to default
