"""Core domain types shared across the whole engine.

Kept dependency-light (pydantic + enums) so every module — indicators,
confluence, ML, API — speaks the same language.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NewsSentiment(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Vote(BaseModel):
    """A single analysis module's opinion.

    bias:     -1.0 (max bearish) .. +1.0 (max bullish)
    strength:  0.0 (ignore me) .. 1.0 (high conviction) — used as the weight
    """

    module: str
    bias: float = Field(ge=-1.0, le=1.0)
    strength: float = Field(ge=0.0, le=1.0)
    note: str = ""

    @property
    def weighted(self) -> float:
        return self.bias * self.strength


class Signal(BaseModel):
    """The final, tradeable output for one symbol/timeframe."""

    symbol: str
    timeframe: str
    exchange: str
    generated_at: datetime

    direction: Direction
    action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    probability_up: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel

    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward: float

    # transparency: what drove the call
    votes: list[Vote] = Field(default_factory=list)
    news_sentiment: NewsSentiment = NewsSentiment.NEUTRAL
    rationale: str = ""

    class Config:
        use_enum_values = True


class Candle(BaseModel):
    timestamp: int  # ms epoch
    open: float
    high: float
    low: float
    close: float
    volume: float
