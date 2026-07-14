"""Smart Money Concepts: swings, BOS, CHOCH, Order Blocks, Fair Value Gaps.

These emulate the structural read that discretionary "smart money" traders use:
where liquidity was taken, where institutional demand/supply sits, and whether
the trend is continuing (BOS) or reversing (CHOCH).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analysis.base import clamp, register, safe_last
from app.schemas import Vote


def _swings(df: pd.DataFrame, left: int = 2, right: int = 2):
    """Return lists of (index, price) fractal swing highs and lows."""
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    swing_high, swing_low = [], []
    for i in range(left, n - right):
        window_h = highs[i - left : i + right + 1]
        window_l = lows[i - left : i + right + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == left):
            swing_high.append((i, highs[i]))
        if lows[i] == window_l.min() and (window_l.argmin() == left):
            swing_low.append((i, lows[i]))
    return swing_high, swing_low


@register("bos_choch", weight=1.3)
def bos_choch(df: pd.DataFrame) -> Vote:
    """Break of Structure (trend continuation) vs Change of Character (reversal)."""
    sh, sl = _swings(df)
    if len(sh) < 2 or len(sl) < 2:
        return Vote(module="bos_choch", bias=0, strength=0.2, note="insufficient swings")
    px = safe_last(df["close"])
    last_sh = sh[-1][1]
    last_sl = sl[-1][1]
    prev_sh = sh[-2][1]
    prev_sl = sl[-2][1]

    uptrend = last_sh > prev_sh and last_sl > prev_sl
    downtrend = last_sh < prev_sh and last_sl < prev_sl

    if px > last_sh:  # broke above last swing high
        bias = 1.0
        note = "bullish BOS" if uptrend else "bullish CHOCH (reversal up)"
        strength = 0.85
    elif px < last_sl:  # broke below last swing low
        bias = -1.0
        note = "bearish BOS" if downtrend else "bearish CHOCH (reversal down)"
        strength = 0.85
    else:
        bias = clamp((int(uptrend) - int(downtrend)) * 0.4)
        note = "within structure"
        strength = 0.4
    return Vote(module="bos_choch", bias=bias, strength=strength, note=note)


@register("order_block", weight=1.1)
def order_block(df: pd.DataFrame, impulse_mult: float = 1.5) -> Vote:
    """Find the most recent order block and check if price is reacting to it.

    Bullish OB: last down-candle before an up-impulse -> demand zone.
    """
    o, c, h, l = (df[x].to_numpy() for x in ("open", "close", "high", "low"))
    body = np.abs(c - o)
    avg_body = body[-50:].mean() if len(body) >= 50 else body.mean()
    px = c[-1]
    n = len(df)

    for i in range(n - 2, max(n - 30, 1), -1):
        impulse = body[i] > avg_body * impulse_mult
        if not impulse:
            continue
        if c[i] > o[i] and c[i - 1] < o[i - 1]:  # bullish impulse after down candle
            zone_lo, zone_hi = l[i - 1], h[i - 1]
            in_zone = zone_lo <= px <= zone_hi * 1.001
            return Vote(module="order_block", bias=0.7 if in_zone else 0.4, strength=0.7 if in_zone else 0.4,
                        note=f"bullish OB {zone_lo:.4g}-{zone_hi:.4g}" + (" (tapped)" if in_zone else ""))
        if c[i] < o[i] and c[i - 1] > o[i - 1]:  # bearish impulse after up candle
            zone_lo, zone_hi = l[i - 1], h[i - 1]
            in_zone = zone_lo * 0.999 <= px <= zone_hi
            return Vote(module="order_block", bias=-0.7 if in_zone else -0.4, strength=0.7 if in_zone else 0.4,
                        note=f"bearish OB {zone_lo:.4g}-{zone_hi:.4g}" + (" (tapped)" if in_zone else ""))
    return Vote(module="order_block", bias=0, strength=0.2, note="no recent OB")


@register("fair_value_gap", weight=0.9)
def fair_value_gap(df: pd.DataFrame) -> Vote:
    """3-candle imbalance (FVG). An unfilled gap pulls price toward it."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    px = df["close"].iloc[-1]
    n = len(df)
    for i in range(n - 1, max(n - 20, 2), -1):
        # bullish FVG: gap between candle[i-2].high and candle[i].low
        if l[i] > h[i - 2]:
            gap_lo, gap_hi = h[i - 2], l[i]
            unfilled = px > gap_hi
            return Vote(module="fair_value_gap", bias=0.6 if unfilled else 0.3, strength=0.6,
                        note=f"bullish FVG {gap_lo:.4g}-{gap_hi:.4g}")
        if h[i] < l[i - 2]:
            gap_lo, gap_hi = h[i], l[i - 2]
            unfilled = px < gap_lo
            return Vote(module="fair_value_gap", bias=-0.6 if unfilled else -0.3, strength=0.6,
                        note=f"bearish FVG {gap_lo:.4g}-{gap_hi:.4g}")
    return Vote(module="fair_value_gap", bias=0, strength=0.2, note="no recent FVG")


@register("open_interest", weight=0.8)
def open_interest(df: pd.DataFrame) -> Vote:
    """Price+OI confluence (if OI column present).

    Rising price + rising OI = new longs (bullish); rising price + falling OI
    = short covering (weaker). Falls back to neutral when OI unavailable.
    """
    if "open_interest" not in df.columns or df["open_interest"].isna().all():
        return Vote(module="open_interest", bias=0, strength=0.0, note="OI unavailable")
    oi = df["open_interest"]
    d_oi = safe_last(oi.diff().rolling(5).mean())
    d_px = safe_last(df["close"].diff().rolling(5).mean())
    if d_px > 0 and d_oi > 0:
        bias, note = 0.7, "price↑ OI↑ new longs"
    elif d_px < 0 and d_oi > 0:
        bias, note = -0.7, "price↓ OI↑ new shorts"
    elif d_px > 0 and d_oi < 0:
        bias, note = 0.3, "price↑ OI↓ short covering"
    else:
        bias, note = -0.3, "price↓ OI↓ long liquidation"
    return Vote(module="open_interest", bias=clamp(bias), strength=0.6, note=note)
