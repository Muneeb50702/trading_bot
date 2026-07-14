"""Signal orchestration: the single entry point that produces a full signal.

Pipeline: fetch OHLCV+OI -> confluence -> ML blend -> news -> build signal
-> risk plan. Returns a `SignalResult` bundling everything the API/UI needs.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.data.market import market_data
from app.engine import confluence, signal_builder
from app.ml.predictor import blend_probability
from app.news.sentiment import news_analyzer
from app.risk.manager import PositionPlan, risk_manager
from app.schemas import Signal

log = get_logger(__name__)


@dataclass
class SignalResult:
    signal: Signal
    position_plan: PositionPlan
    ml_probability: float | None
    news_score: float
    news_headlines: list[str]

    def to_dict(self) -> dict:
        return {
            # mode="json" -> datetimes/enums become JSON-safe primitives so the
            # result can be stored in a JSON column and sent over WebSocket.
            "signal": self.signal.model_dump(mode="json"),
            "position_plan": asdict(self.position_plan),
            "ml_probability": self.ml_probability,
            "news_score": self.news_score,
            "news_headlines": self.news_headlines,
        }


async def generate_signal(
    symbol: str, timeframe: str, *, exchange: str | None = None,
    leverage: int | None = None,
) -> SignalResult:
    exchange = exchange or settings.default_exchange
    df = await market_data.fetch_ohlcv(symbol, timeframe, exchange=exchange)

    conf_prob, votes = confluence.analyze(df)
    final_prob, ml_prob = blend_probability(df, conf_prob)
    sentiment, news_score, headlines = await news_analyzer.analyze(symbol)

    signal = signal_builder.build_signal(
        df, final_prob, votes,
        symbol=symbol, timeframe=timeframe, exchange=exchange,
        news=sentiment, leverage=leverage,
    )
    plan = risk_manager.evaluate(signal)
    return SignalResult(signal, plan, ml_prob, news_score, headlines)


async def scan(
    symbols: list[str] | None = None, timeframes: list[str] | None = None,
    *, exchange: str | None = None,
) -> list[SignalResult]:
    """Generate signals across the full symbol × timeframe grid, concurrently."""
    symbols = symbols or settings.default_symbols
    timeframes = timeframes or settings.default_timeframes
    tasks = [
        generate_signal(s, tf, exchange=exchange)
        for s in symbols for tf in timeframes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[SignalResult] = []
    for r in results:
        if isinstance(r, Exception):
            log.warning("scan item failed: %s", r)
        else:
            out.append(r)
    return out
