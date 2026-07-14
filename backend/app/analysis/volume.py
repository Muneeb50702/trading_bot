"""Volume-based analysis modules: VWAP, OBV, Volume Profile."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analysis.base import clamp, register, safe_last
from app.schemas import Vote
from app.ta import core


@register("vwap", weight=1.0)
def vwap(df: pd.DataFrame) -> Vote:
    v = core.vwap(df)
    px, vw = safe_last(df["close"]), safe_last(v)
    dist = (px - vw) / vw if vw else 0
    return Vote(module="vwap", bias=clamp(dist * 25), strength=clamp(0.3 + abs(dist) * 20, 0.2, 1),
                note=f"price {'above' if px > vw else 'below'} VWAP ({dist * 100:.2f}%)")


@register("obv", weight=0.9)
def obv(df: pd.DataFrame) -> Vote:
    o = core.obv(df["close"], df["volume"])
    slope = o.diff().rolling(10).mean()
    s = safe_last(slope)
    norm = s / (df["volume"].rolling(10).mean().iloc[-1] + 1e-9)
    return Vote(module="obv", bias=clamp(norm), strength=clamp(0.3 + abs(norm) * 0.5, 0.2, 1),
                note=f"OBV {'accumulating' if s > 0 else 'distributing'}")


@register("volume_profile", weight=1.0)
def volume_profile(df: pd.DataFrame) -> Vote:
    centers, profile, poc = core.volume_profile(df)
    px = safe_last(df["close"])
    # Price above the point-of-control (value area) = acceptance higher.
    bias = clamp((px - poc) / poc * 20) if poc else 0
    # Are we near a high-volume node (support/resistance) -> lower conviction to break.
    nearest = int(np.argmin(np.abs(centers - px)))
    node_strength = profile[nearest] / (profile.max() + 1e-9)
    strength = clamp(0.5 - node_strength * 0.3 + abs(bias) * 0.3, 0.2, 1)
    return Vote(module="volume_profile", bias=bias, strength=strength,
                note=f"POC={poc:.4g} price {'>' if px > poc else '<'} POC")
