"""Price prediction request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    coin: str = Field(default="bitcoin", description="CoinGecko coin ID")
    price_target: float = Field(..., description="Target price in USD")
    direction: Literal["above", "below"] = Field(..., description="Predict if price will be above or below target")
    timeframe: str = Field(default="24h", description="Prediction timeframe (e.g. 1h, 4h, 24h, 7d)")


class PricePrediction(BaseModel):
    coin: str
    current_price: float
    price_target: float
    direction: str
    timeframe: str
    prediction: Literal["YES", "NO"]
    estimated_probability: float = Field(ge=0.0, le=1.0)
    confidence: Literal["High", "Medium", "Low"]
    signal: str
    fear_greed_index: int
    sentiment_score: float
    rationale: str
    market_data: dict
    sentiment: dict
