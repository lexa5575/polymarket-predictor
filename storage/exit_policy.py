"""Default exit policy for scalp strategy.

Saved per-trade at entry for reproducibility.
v1: global constants. v2: strategy profiles passed from entry layer.
"""

TAKE_PROFIT_PCT = 0.10      # +10% of stake
STOP_LOSS_PCT = -0.05       # -5% of stake
MAX_HOLD_SECONDS = 1800     # 30 minutes
