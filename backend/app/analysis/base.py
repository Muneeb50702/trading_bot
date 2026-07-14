"""Analysis-module framework.

An *analysis module* looks at an OHLCV DataFrame and returns one `Vote`
(bias in [-1,1], strength in [0,1]). Modules self-register via @register so
the confluence engine can discover the full bank automatically.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from app.schemas import Vote

# name -> (callable, default weight). Weight scales a module's influence in
# the confluence blend (some modules are structurally more reliable).
REGISTRY: dict[str, tuple[Callable[[pd.DataFrame], Vote], float]] = {}


def register(name: str, weight: float = 1.0):
    def deco(fn: Callable[[pd.DataFrame], Vote]):
        REGISTRY[name] = (fn, weight)
        return fn

    return deco


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def safe_last(series: pd.Series, default: float = 0.0) -> float:
    """Last non-NaN value or a default (guards short/early series)."""
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else default
