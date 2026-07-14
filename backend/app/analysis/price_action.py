"""Price-action & candlestick-pattern analysis."""
from __future__ import annotations

import pandas as pd

from app.analysis.base import clamp, register, safe_last
from app.schemas import Vote
from app.ta import core


@register("market_structure", weight=1.2)
def market_structure(df: pd.DataFrame, lookback: int = 20) -> Vote:
    """Higher-highs/higher-lows => uptrend structure, and vice-versa."""
    highs = df["high"].tail(lookback)
    lows = df["low"].tail(lookback)
    mid = lookback // 2
    recent_hh = highs.iloc[mid:].max() > highs.iloc[:mid].max()
    recent_hl = lows.iloc[mid:].min() > lows.iloc[:mid].min()
    recent_lh = highs.iloc[mid:].max() < highs.iloc[:mid].max()
    recent_ll = lows.iloc[mid:].min() < lows.iloc[:mid].min()
    if recent_hh and recent_hl:
        bias, note = 1.0, "HH+HL uptrend"
    elif recent_lh and recent_ll:
        bias, note = -1.0, "LH+LL downtrend"
    else:
        bias, note = clamp((int(recent_hh) - int(recent_ll)) * 0.5), "mixed structure"
    return Vote(module="market_structure", bias=bias, strength=0.8 if abs(bias) == 1 else 0.4, note=note)


@register("support_resistance", weight=0.8)
def support_resistance(df: pd.DataFrame, lookback: int = 50) -> Vote:
    """Proximity to recent swing high/low — closer to support = bullish reaction."""
    window = df.tail(lookback)
    hi, lo, px = window["high"].max(), window["low"].min(), safe_last(df["close"])
    rng = hi - lo
    if rng <= 0:
        return Vote(module="support_resistance", bias=0, strength=0.2, note="flat range")
    pos = (px - lo) / rng  # 0 at support, 1 at resistance
    # near support -> bullish bounce bias; near resistance -> bearish
    bias = clamp((0.5 - pos) * 1.6)
    return Vote(module="support_resistance", bias=bias, strength=0.5,
                note=f"pos in range={pos:.2f}")


@register("candlestick", weight=0.8)
def candlestick(df: pd.DataFrame) -> Vote:
    """Detect the most recent high-signal candlestick pattern."""
    o, h, l, c = (df[x] for x in ("open", "high", "low", "close"))
    o1, h1, l1, c1 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o2, c2 = o.iloc[-2], c.iloc[-2]
    body = abs(c1 - o1)
    rng = max(h1 - l1, 1e-9)
    upper_wick = h1 - max(c1, o1)
    lower_wick = min(c1, o1) - l1

    bias, note, strength = 0.0, "no pattern", 0.3
    # Bullish / bearish engulfing
    if c1 > o1 and c2 < o2 and c1 >= o2 and o1 <= c2:
        bias, note, strength = 0.8, "bullish engulfing", 0.75
    elif c1 < o1 and c2 > o2 and c1 <= o2 and o1 >= c2:
        bias, note, strength = -0.8, "bearish engulfing", 0.75
    # Hammer / shooting star
    elif lower_wick > body * 2 and upper_wick < body:
        bias, note, strength = 0.6, "hammer", 0.6
    elif upper_wick > body * 2 and lower_wick < body:
        bias, note, strength = -0.6, "shooting star", 0.6
    # Doji = indecision
    elif body / rng < 0.1:
        bias, note, strength = 0.0, "doji (indecision)", 0.25
    return Vote(module="candlestick", bias=bias, strength=strength, note=note)
