"""Backtesting API."""
from __future__ import annotations

from fastapi import APIRouter, Query

from app.backtest.engine import run_backtest
from app.data.market import market_data

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.get("/run")
async def backtest(
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("15m"),
    exchange: str | None = Query(None),
    candles: int = Query(1000, le=1500),
    hold_bars: int = Query(24, le=200),
    min_confidence: float = Query(0.55, ge=0.0, le=1.0),
):
    df = await market_data.fetch_ohlcv(symbol, timeframe, exchange=exchange, limit=candles)
    report = run_backtest(
        df, symbol=symbol, timeframe=timeframe,
        hold_bars=hold_bars, min_confidence=min_confidence,
    )
    return report.to_dict()
