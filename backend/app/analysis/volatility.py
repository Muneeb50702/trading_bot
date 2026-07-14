"""Volatility & band-based analysis modules."""
from __future__ import annotations

import pandas as pd

from app.analysis.base import clamp, register, safe_last
from app.schemas import Vote
from app.ta import core


@register("bollinger", weight=0.9)
def bollinger(df: pd.DataFrame) -> Vote:
    upper, mid, lower = core.bollinger(df["close"])
    px = safe_last(df["close"])
    u, m, l = safe_last(upper), safe_last(mid), safe_last(lower)
    width = (u - l) / m if m else 0
    # %B position within bands: >1 breakout up, <0 breakout down.
    pct_b = (px - l) / (u - l) if u != l else 0.5
    if pct_b > 1:
        bias = 0.7          # upper-band breakout -> momentum long
    elif pct_b < 0:
        bias = -0.7
    else:
        bias = clamp((pct_b - 0.5) * 1.4)
    # squeeze (narrow width) lowers conviction — expansion raises it.
    strength = clamp(0.3 + width * 8, 0.2, 1)
    return Vote(module="bollinger", bias=bias, strength=strength,
                note=f"%B={pct_b:.2f} width={width:.3f}")


@register("atr_regime", weight=0.5)
def atr_regime(df: pd.DataFrame) -> Vote:
    """ATR itself is directionless, but a volatility *expansion* in the
    direction of the last candle adds conviction; contraction removes it."""
    a = core.atr(df["high"], df["low"], df["close"])
    a_now, a_avg = safe_last(a), safe_last(a.rolling(20).mean())
    last = df["close"].iloc[-1] - df["open"].iloc[-1]
    expanding = a_now > a_avg
    bias = clamp((1.0 if last > 0 else -1.0) * (0.4 if expanding else 0.1))
    return Vote(module="atr_regime", bias=bias, strength=0.4 if expanding else 0.2,
                note=f"ATR {'expanding' if expanding else 'contracting'}")
