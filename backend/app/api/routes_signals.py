"""Signals API: on-demand generation, full-grid scan, and history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SignalRecord
from app.db.session import get_session
from app.notifications.ws import manager
from app.signals.service import SignalResult, generate_signal, scan

router = APIRouter(prefix="/api/signals", tags=["signals"])


async def _persist(session: AsyncSession, result: SignalResult) -> int:
    s = result.signal
    rec = SignalRecord(
        symbol=s.symbol, timeframe=s.timeframe, exchange=s.exchange,
        direction=str(s.direction), action=str(s.action), confidence=s.confidence,
        probability_up=s.probability_up, risk_level=str(s.risk_level),
        entry_price=s.entry_price, stop_loss=s.stop_loss,
        take_profit_1=s.take_profit_1, take_profit_2=s.take_profit_2,
        take_profit_3=s.take_profit_3, risk_reward=s.risk_reward,
        news_sentiment=str(s.news_sentiment), payload=result.to_dict(),
    )
    session.add(rec)
    await session.commit()
    await session.refresh(rec)
    return rec.id


@router.get("/generate")
async def generate(
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("5m"),
    exchange: str | None = Query(None),
    persist: bool = Query(True),
    session: AsyncSession = Depends(get_session),
):
    result = await generate_signal(symbol, timeframe, exchange=exchange)
    data = result.to_dict()
    if persist:
        data["id"] = await _persist(session, result)
        await manager.broadcast({"type": "signal", "data": data})
    return data


@router.get("/scan")
async def scan_grid(
    symbols: str | None = Query(None, description="comma-separated, e.g. BTC/USDT,ETH/USDT"),
    timeframes: str | None = Query(None, description="comma-separated, e.g. 5m,15m"),
    exchange: str | None = Query(None),
    persist: bool = Query(False),
    session: AsyncSession = Depends(get_session),
):
    sym = [s.strip() for s in symbols.split(",")] if symbols else None
    tfs = [t.strip() for t in timeframes.split(",")] if timeframes else None
    results = await scan(sym, tfs, exchange=exchange)
    out = []
    for r in results:
        d = r.to_dict()
        if persist:
            d["id"] = await _persist(session, r)
        out.append(d)
    if persist and out:
        await manager.broadcast({"type": "scan", "count": len(out)})
    return {"count": len(out), "signals": out}


@router.get("/history")
async def history(
    symbol: str | None = Query(None),
    limit: int = Query(50, le=500),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(SignalRecord).order_by(SignalRecord.created_at.desc()).limit(limit)
    if symbol:
        stmt = stmt.where(SignalRecord.symbol == symbol)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": r.id, "symbol": r.symbol, "timeframe": r.timeframe,
            "action": r.action, "direction": r.direction, "confidence": r.confidence,
            "probability_up": r.probability_up, "risk_level": r.risk_level,
            "entry_price": r.entry_price, "stop_loss": r.stop_loss,
            "take_profit_1": r.take_profit_1, "take_profit_2": r.take_profit_2,
            "take_profit_3": r.take_profit_3, "risk_reward": r.risk_reward,
            "news_sentiment": r.news_sentiment, "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
