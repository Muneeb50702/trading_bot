"""Feature engineering for the probability model.

Produces a purely-numeric feature matrix from OHLCV so the same code path
serves both offline training and live inference (last row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.ta import core

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "rsi", "rsi_slope",
    "macd_hist", "macd_hist_slope",
    "stoch_k",
    "ema_fast_rel", "ema_slow_rel", "ema_align",
    "adx", "di_diff",
    "bb_pctb", "bb_width",
    "atr_pct",
    "vwap_rel",
    "obv_slope",
    "supertrend_dir",
    "vol_rel",
    "close_pos",  # position of close within candle range
    # --- added for stronger predictive power ---
    "ret_vol",        # recent realised volatility of returns
    "ret_accel",      # momentum acceleration (Δ of short return)
    "ema_slope",      # slope of the slow EMA (trend gradient)
    "range_pos_50",   # position within the last 50-bar high/low range
    "streak",         # signed consecutive up/down candle count
    "body_ratio",     # candle body vs range (conviction)
    "upper_wick",     # rejection from above
    "lower_wick",     # rejection from below
    "vol_z",          # volume z-score vs recent norm
    "dist_high_20",   # distance below the 20-bar high
    "dist_low_20",    # distance above the 20-bar low
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = (df[x] for x in ("close", "high", "low", "open", "volume"))
    f = pd.DataFrame(index=df.index)

    f["ret_1"] = c.pct_change(1)
    f["ret_3"] = c.pct_change(3)
    f["ret_5"] = c.pct_change(5)
    f["ret_10"] = c.pct_change(10)

    r = core.rsi(c)
    f["rsi"] = r / 100
    f["rsi_slope"] = r.diff(3) / 100

    macd_line, sig, hist = core.macd(c)
    f["macd_hist"] = hist / c
    f["macd_hist_slope"] = hist.diff(2) / c

    k, _ = core.stochastic(h, l, c)
    f["stoch_k"] = k / 100

    ema_f, ema_s = core.ema(c, 9), core.ema(c, 50)
    f["ema_fast_rel"] = (c - ema_f) / c
    f["ema_slow_rel"] = (c - ema_s) / c
    f["ema_align"] = np.sign(ema_f - ema_s)

    adx_, plus_di, minus_di = core.adx(h, l, c)
    f["adx"] = adx_ / 100
    f["di_diff"] = (plus_di - minus_di) / 100

    upper, mid, lower = core.bollinger(c)
    f["bb_pctb"] = (c - lower) / (upper - lower).replace(0, np.nan)
    f["bb_width"] = (upper - lower) / mid

    f["atr_pct"] = core.atr(h, l, c) / c

    f["vwap_rel"] = (c - core.vwap(df)) / c

    ob = core.obv(c, v)
    f["obv_slope"] = ob.diff(5) / (v.rolling(5).mean() + 1e-9)

    _, st_dir = core.supertrend(h, l, c)
    f["supertrend_dir"] = st_dir

    f["vol_rel"] = v / v.rolling(20).mean()

    f["close_pos"] = (c - l) / (h - l).replace(0, np.nan)

    # --- added features ---
    ret1 = c.pct_change(1)
    f["ret_vol"] = ret1.rolling(14).std()
    f["ret_accel"] = ret1 - ret1.shift(3)

    ema_slow = core.ema(c, 50)
    f["ema_slope"] = ema_slow.diff(5) / c

    lo50, hi50 = l.rolling(50).min(), h.rolling(50).max()
    f["range_pos_50"] = (c - lo50) / (hi50 - lo50).replace(0, np.nan)

    up = (c > o).astype(int).replace(0, -1)
    # signed run-length of same-direction candles, normalised
    grp = (up != up.shift()).cumsum()
    f["streak"] = (up.groupby(grp).cumcount() + 1) * up / 10.0

    rng = (h - l).replace(0, np.nan)
    f["body_ratio"] = (c - o).abs() / rng
    f["upper_wick"] = (h - c.combine(o, max)) / rng
    f["lower_wick"] = (c.combine(o, min) - l) / rng

    f["vol_z"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)

    hi20, lo20 = h.rolling(20).max(), l.rolling(20).min()
    f["dist_high_20"] = (hi20 - c) / c
    f["dist_low_20"] = (c - lo20) / c

    return f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def build_labeled_dataset(
    df: pd.DataFrame, horizon: int = 3, atr_filter: float = 0.25
) -> tuple[pd.DataFrame, pd.Series]:
    """Features X and binary label y (1 = price up over `horizon` bars).

    Bars whose forward move is smaller than `atr_filter * ATR` are dropped as
    noise, so the model learns on decisive moves only.
    """
    feats = build_features(df)
    fwd_ret = df["close"].shift(-horizon) - df["close"]
    atr_val = core.atr(df["high"], df["low"], df["close"])
    decisive = fwd_ret.abs() > (atr_val * atr_filter)
    y = (fwd_ret > 0).astype(int)

    mask = feats.notna().all(axis=1) & fwd_ret.notna() & decisive
    return feats[mask], y[mask]
