"""Historical dataset builder for model training.

ccxt returns at most ~1500 candles per call, so to train on real history we
page backwards through time and stitch the pages together. Then we build one
combined, labelled feature matrix across many symbols so the model learns
patterns that generalise rather than memorising one pair.
"""
from __future__ import annotations

import asyncio

import pandas as pd

from app.core.logging import get_logger
from app.data.market import market_data
from app.ml.features import build_labeled_dataset

log = get_logger(__name__)

# rough milliseconds-per-candle for pagination stepping
_TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


async def fetch_history(
    symbol: str, timeframe: str, *, exchange: str | None = None,
    target_candles: int = 15_000, page: int = 1500,
) -> pd.DataFrame:
    """Page backwards to assemble up to `target_candles` of OHLCV history."""
    client = market_data._client(exchange or "binanceusdm")
    step = _TF_MS.get(timeframe, 900_000)
    # start `target_candles` back from now
    now = client.milliseconds()
    since = now - target_candles * step

    rows: list[list] = []
    cursor = since
    while len(rows) < target_candles:
        batch = await client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=page)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
        if cursor >= now:
            break
        await asyncio.sleep(client.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    df = df.astype({c: "float64" for c in ["open", "high", "low", "close", "volume"]})
    log.info("fetched %d candles for %s %s", len(df), symbol, timeframe)
    return df


async def build_training_set(
    symbols: list[str], timeframe: str, *, target_candles: int = 15_000,
    horizon: int = 3,
) -> tuple[pd.DataFrame, pd.Series, list[pd.DataFrame]]:
    """Combined labelled (X, y) across all symbols, plus per-symbol raw frames.

    Returns the raw frames too so a walk-forward backtest can reuse them.
    """
    frames = await asyncio.gather(*[
        fetch_history(s, timeframe, target_candles=target_candles) for s in symbols
    ])
    X_parts, y_parts = [], []
    for sym, df in zip(symbols, frames):
        if len(df) < 300:
            log.warning("skipping %s: only %d candles", sym, len(df))
            continue
        X, y = build_labeled_dataset(df, horizon=horizon)
        X_parts.append(X)
        y_parts.append(y)
    if not X_parts:
        raise ValueError("no usable data fetched")
    return pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True), list(frames)
