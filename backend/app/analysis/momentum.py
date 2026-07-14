"""Momentum & oscillator analysis modules."""
from __future__ import annotations

import pandas as pd

from app.analysis.base import clamp, register, safe_last
from app.schemas import Vote
from app.ta import core


@register("rsi", weight=1.0)
def rsi(df: pd.DataFrame) -> Vote:
    r = core.rsi(df["close"])
    val = safe_last(r, 50)
    # Distance from 50 = momentum direction; extremes are mean-reversion warnings.
    bias = clamp((val - 50) / 25)
    if val > 70:
        bias, note = clamp(bias * 0.4), f"RSI {val:.1f} overbought"
    elif val < 30:
        bias, note = clamp(bias * 0.4), f"RSI {val:.1f} oversold"
    else:
        note = f"RSI {val:.1f}"
    strength = clamp(abs(val - 50) / 30, 0.2, 1)
    return Vote(module="rsi", bias=bias, strength=strength, note=note)


@register("macd", weight=1.1)
def macd(df: pd.DataFrame) -> Vote:
    line, signal, hist = core.macd(df["close"])
    h, l, px = safe_last(hist), safe_last(line), safe_last(df["close"])
    prev_h = safe_last(hist.iloc[:-1]) if len(hist.dropna()) > 1 else h
    bias = clamp((1.0 if h > 0 else -1.0) * 0.7 + (0.3 if h > prev_h else -0.3))
    strength = clamp(0.3 + abs(h) / px * 200, 0, 1) if px else 0.3
    return Vote(module="macd", bias=bias, strength=strength,
                note=f"hist={h:.4g} {'rising' if h > prev_h else 'falling'}")


@register("stoch_rsi", weight=0.8)
def stoch_rsi(df: pd.DataFrame) -> Vote:
    k, d = core.stoch_rsi(df["close"])
    kk, dd = safe_last(k, 50), safe_last(d, 50)
    cross = 1.0 if kk > dd else -1.0
    if kk > 80:
        bias, note = -0.4, f"StochRSI {kk:.0f} overbought"
    elif kk < 20:
        bias, note = 0.4, f"StochRSI {kk:.0f} oversold"
    else:
        bias, note = clamp((kk - 50) / 40 + cross * 0.3), f"StochRSI K={kk:.0f} D={dd:.0f}"
    return Vote(module="stoch_rsi", bias=bias, strength=0.6, note=note)


@register("stochastic", weight=0.7)
def stochastic(df: pd.DataFrame) -> Vote:
    k, d = core.stochastic(df["high"], df["low"], df["close"])
    kk, dd = safe_last(k, 50), safe_last(d, 50)
    bias = clamp((kk - 50) / 40 + (0.25 if kk > dd else -0.25))
    return Vote(module="stochastic", bias=bias, strength=0.5,
                note=f"Stoch K={kk:.0f} D={dd:.0f}")
