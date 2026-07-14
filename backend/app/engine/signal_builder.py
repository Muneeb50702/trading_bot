"""Turn a fused probability into a concrete, tradeable Signal.

Responsibilities:
  * decide Direction + Action (Buy/Sell/No-Trade) from probability + thresholds
  * place Entry / Stop-Loss / TP1-3 using ATR-scaled risk distance
  * compute Confidence (probability × vote-agreement × news alignment)
  * classify Risk Level from volatility, leverage and confidence
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.core.config import settings
from app.schemas import (
    Action,
    Direction,
    NewsSentiment,
    RiskLevel,
    Signal,
    Vote,
)
from app.ta import core


def _confidence(prob: float, votes: list[Vote], direction: Direction,
                news: NewsSentiment) -> float:
    base = abs(prob - 0.5) * 2  # 0 at coin-flip, 1 at certainty

    # agreement: share of (weighted) conviction pointing the signalled way
    agree = disagree = 0.0
    for v in votes:
        if direction == Direction.UP and v.bias > 0:
            agree += v.strength
        elif direction == Direction.DOWN and v.bias < 0:
            agree += v.strength
        elif v.bias != 0:
            disagree += v.strength
    agreement = agree / (agree + disagree) if (agree + disagree) else 0.5

    news_align = 1.0
    if news == NewsSentiment.BULLISH:
        news_align = 1.08 if direction == Direction.UP else 0.9
    elif news == NewsSentiment.BEARISH:
        news_align = 1.08 if direction == Direction.DOWN else 0.9

    conf = (0.55 * base + 0.45 * agreement) * news_align
    return round(min(max(conf, 0.0), 1.0), 4)


def _risk_level(atr_pct: float, leverage: int, confidence: float) -> RiskLevel:
    # volatility- and leverage-aware, softened by confidence
    score = atr_pct * 100 * 0.5 + leverage * 0.4 + (1 - confidence) * 3
    if score < 3.0:
        return RiskLevel.LOW
    if score < 6.0:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def build_signal(
    df: pd.DataFrame,
    probability_up: float,
    votes: list[Vote],
    *,
    symbol: str,
    timeframe: str,
    exchange: str,
    news: NewsSentiment = NewsSentiment.NEUTRAL,
    leverage: int | None = None,
    atr_sl_mult: float | None = None,
    tp_r_multiples: list[float] | None = None,
) -> Signal:
    leverage = leverage or settings.default_leverage
    atr_sl_mult = atr_sl_mult or settings.atr_sl_mult
    tp_r_multiples = tp_r_multiples or settings.tp_r_multiples

    entry = float(df["close"].iloc[-1])
    atr_val = float(core.atr(df["high"], df["low"], df["close"]).iloc[-1])
    atr_pct = atr_val / entry if entry else 0.0

    # direction from probability
    if probability_up >= 0.5:
        direction = Direction.UP
    else:
        direction = Direction.DOWN

    confidence = _confidence(probability_up, votes, direction, news)

    # Buy / Sell / No-Trade gating
    if probability_up >= settings.long_threshold and confidence >= settings.min_confidence:
        action = Action.BUY
    elif probability_up <= settings.short_threshold and confidence >= settings.min_confidence:
        action = Action.SELL
    else:
        action = Action.NO_TRADE
        direction = direction if abs(probability_up - 0.5) > 0.03 else Direction.NEUTRAL

    risk_dist = max(atr_val * atr_sl_mult, entry * 0.001)  # floor to avoid zero-risk
    if direction == Direction.UP:
        stop = entry - risk_dist
        tps = [entry + risk_dist * r for r in tp_r_multiples]
    else:
        stop = entry + risk_dist
        tps = [entry - risk_dist * r for r in tp_r_multiples]

    rr = round(abs(tps[0] - entry) / risk_dist, 2)
    risk_level = _risk_level(atr_pct, leverage, confidence)

    top = sorted(votes, key=lambda v: v.strength, reverse=True)[:3]
    rationale = (
        f"P(up)={probability_up:.1%}, {direction.value} @ {confidence:.0%} conf. "
        f"Drivers: " + ", ".join(f"{v.module}({v.note})" for v in top)
    )

    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        exchange=exchange,
        generated_at=datetime.now(timezone.utc),
        direction=direction,
        action=action,
        confidence=confidence,
        probability_up=round(probability_up, 4),
        risk_level=risk_level,
        entry_price=round(entry, 8),
        stop_loss=round(stop, 8),
        take_profit_1=round(tps[0], 8),
        take_profit_2=round(tps[1], 8),
        take_profit_3=round(tps[2], 8),
        risk_reward=rr,
        votes=votes,
        news_sentiment=news,
        rationale=rationale,
    )
