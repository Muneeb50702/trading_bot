"""Vectorized technical-analysis primitives (NumPy/pandas, no TA-Lib).

Every function is pure: it takes Series/arrays and returns Series/arrays.
The higher-level *analysis modules* build on these to produce Votes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (used by RSI/ATR/ADX)."""
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stoch_rsi(
    close: pd.Series, length: int = 14, k: int = 3, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    r = rsi(close, length)
    lo = r.rolling(length).min()
    hi = r.rolling(length).max()
    stoch = (r - lo) / (hi - lo).replace(0, np.nan) * 100
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    k_line = (close - lo) / (hi - lo).replace(0, np.nan) * 100
    return k_line, k_line.rolling(d).mean()


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return rma(true_range(high, low, close), length)


def bollinger(
    close: pd.Series, length: int = 20, mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, length)
    std = close.rolling(length).std(ddof=0)
    return mid + mult * std, mid, mid - mult * std


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    atr_ = rma(tr, length)
    plus_di = 100 * rma(pd.Series(plus_dm, index=high.index), length) / atr_
    minus_di = 100 * rma(pd.Series(minus_dm, index=high.index), length) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return rma(dx, length), plus_di, minus_di


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, mult: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Returns (supertrend_line, direction) where direction is +1 up / -1 down.

    The trailing-band update is inherently recursive (each band depends on the
    prior one), so it can't be fully vectorized — but we loop over raw NumPy
    arrays instead of pandas .iloc, which is ~50-100x faster and matters a lot
    for backtests/scans that call this thousands of times.
    """
    atr_ = atr(high, low, close, length)
    hl2 = ((high + low) / 2).to_numpy()
    upper = (hl2 + mult * atr_.to_numpy())
    lower = (hl2 - mult * atr_.to_numpy())
    close_a = close.to_numpy()
    atr_a = atr_.to_numpy()
    n = len(close_a)

    st = np.empty(n)
    dir_ = np.empty(n)
    prev_dir = 1
    fu = fl = np.nan
    for i in range(n):
        cu, cl = upper[i], lower[i]
        if i == 0 or np.isnan(atr_a[i]):
            fu, fl, prev_dir = cu, cl, 1
            st[i], dir_[i] = cl, prev_dir
            continue
        fu = cu if (cu < fu or close_a[i - 1] > fu) else fu
        fl = cl if (cl > fl or close_a[i - 1] < fl) else fl
        if close_a[i] > fu:
            prev_dir = 1
        elif close_a[i] < fl:
            prev_dir = -1
        st[i], dir_[i] = (fl if prev_dir == 1 else fu), prev_dir
    return pd.Series(st, index=close.index), pd.Series(dir_, index=close.index)


def ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, pd.Series]:
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    return {"conversion": conv, "base": base, "span_a": span_a, "span_b": span_b}


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def vwap(df: pd.DataFrame, window: int = 96) -> pd.Series:
    """Rolling (anchored) VWAP over the last `window` candles.

    A rolling window keeps VWAP intraday-relevant; a cumulative VWAP drifts
    arbitrarily far from price on long, trending series.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(window, min_periods=1).sum()
    vv = df["volume"].rolling(window, min_periods=1).sum()
    return pv / vv.replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()


def volume_profile(
    df: pd.DataFrame, bins: int = 24
) -> tuple[np.ndarray, np.ndarray, float]:
    """Histogram of traded volume by price. Returns (levels, volumes, poc_price)."""
    prices = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    vols = df["volume"].to_numpy()
    lo, hi = prices.min(), prices.max()
    if hi <= lo:
        return np.array([lo]), np.array([vols.sum()]), lo
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(prices, edges) - 1, 0, bins - 1)
    profile = np.zeros(bins)
    np.add.at(profile, idx, vols)
    centers = (edges[:-1] + edges[1:]) / 2
    poc = float(centers[int(profile.argmax())])
    return centers, profile, poc
