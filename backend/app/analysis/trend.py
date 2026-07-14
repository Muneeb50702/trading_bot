"""Trend & moving-average based analysis modules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analysis.base import clamp, register, safe_last
from app.schemas import Vote
from app.ta import core


@register("ema_stack", weight=1.2)
def ema_stack(df: pd.DataFrame) -> Vote:
    """Fast>Mid>Slow EMA alignment = trend; price vs EMAs = participation."""
    c = df["close"]
    e_fast, e_mid, e_slow = core.ema(c, 9), core.ema(c, 21), core.ema(c, 50)
    f, m, s, px = safe_last(e_fast), safe_last(e_mid), safe_last(e_slow), safe_last(c)
    stacked_up = f > m > s
    stacked_dn = f < m < s
    bias = 1.0 if stacked_up else -1.0 if stacked_dn else clamp((f - s) / s * 20)
    # strength grows with EMA separation (normalised by price)
    sep = abs(f - s) / px if px else 0
    strength = clamp(0.35 + sep * 40, 0, 1)
    return Vote(module="ema_stack", bias=bias, strength=strength,
                note=f"EMA9={f:.4g} EMA21={m:.4g} EMA50={s:.4g}")


@register("sma_cross", weight=0.9)
def sma_cross(df: pd.DataFrame) -> Vote:
    c = df["close"]
    fast, slow = core.sma(c, 20), core.sma(c, 50)
    f, s = safe_last(fast), safe_last(slow)
    diff = (f - s) / s if s else 0
    return Vote(module="sma_cross", bias=clamp(diff * 30), strength=clamp(0.3 + abs(diff) * 25, 0, 1),
                note=f"SMA20={f:.4g} SMA50={s:.4g}")


@register("adx_trend", weight=1.1)
def adx_trend(df: pd.DataFrame) -> Vote:
    """ADX gives trend *strength*; +DI/-DI gives direction."""
    adx_, plus_di, minus_di = core.adx(df["high"], df["low"], df["close"])
    a, p, m = safe_last(adx_, 0), safe_last(plus_di), safe_last(minus_di)
    bias = clamp((p - m) / max(p + m, 1e-9))
    # ADX<20 = no trend (low strength); >40 = strong
    strength = clamp((a - 15) / 35, 0, 1)
    return Vote(module="adx_trend", bias=bias, strength=strength,
                note=f"ADX={a:.1f} +DI={p:.1f} -DI={m:.1f}")


@register("supertrend", weight=1.2)
def supertrend(df: pd.DataFrame) -> Vote:
    _, direction = core.supertrend(df["high"], df["low"], df["close"])
    d = safe_last(direction, 0)
    # persistence of current direction => stronger conviction
    run = 0
    for v in reversed(direction.dropna().to_list()):
        if v == d:
            run += 1
        else:
            break
    strength = clamp(0.4 + run * 0.04, 0, 1)
    return Vote(module="supertrend", bias=float(np.sign(d)), strength=strength,
                note=f"dir={'up' if d > 0 else 'down'} run={run}")


@register("ichimoku", weight=1.0)
def ichimoku(df: pd.DataFrame) -> Vote:
    ich = core.ichimoku(df["high"], df["low"], df["close"])
    px = safe_last(df["close"])
    span_a, span_b = safe_last(ich["span_a"]), safe_last(ich["span_b"])
    conv, base = safe_last(ich["conversion"]), safe_last(ich["base"])
    cloud_top, cloud_bot = max(span_a, span_b), min(span_a, span_b)
    if px > cloud_top:
        cloud_bias = 1.0
    elif px < cloud_bot:
        cloud_bias = -1.0
    else:
        cloud_bias = 0.0
    tk_bias = 1.0 if conv > base else -1.0 if conv < base else 0.0
    bias = clamp(0.65 * cloud_bias + 0.35 * tk_bias)
    strength = 0.75 if cloud_bias != 0 else 0.4
    return Vote(module="ichimoku", bias=bias, strength=strength,
                note=f"price {'>' if px > cloud_top else '<' if px < cloud_bot else 'in'} cloud")
