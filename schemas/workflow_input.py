"""
Workflow input schema for the prediction pipeline.

Used by both the AgentOS API and the scheduler to trigger workflow runs.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    """Input schema for the prediction workflow and scheduler runs.

    - ``single_market``: analyse one specific market (requires at least one ID).
    - ``batch_scan``: scan for candidates matching *category* and return up to
      *max_candidates* results.
    """

    mode: Literal["single_market", "batch_scan"]
    condition_id: str | None = Field(
        None,
        description="Polymarket condition ID",
    )
    gamma_market_id: str | None = Field(
        None,
        description="Gamma API market id",
    )
    market_slug: str | None = Field(
        None,
        description="Human-readable market slug",
    )
    category: str = Field(
        default="crypto",
        description="Category filter for batch_scan mode",
    )
    max_candidates: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max candidates to return in batch_scan mode",
    )

    @model_validator(mode="after")
    def check_single_market_has_id(self) -> Self:
        if self.mode == "single_market":
            if not any([self.condition_id, self.gamma_market_id, self.market_slug]):
                raise ValueError(
                    "single_market mode requires at least one of: "
                    "condition_id, gamma_market_id, or market_slug"
                )
        return self
