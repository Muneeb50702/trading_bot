"""Background scanner: periodically regenerates signals and pushes them live.

Runs inside the API process. Each cycle scans the default grid, persists new
signals, broadcasts them over WebSocket, and fires notifications for actionable
(Buy/Sell) signals — the always-on brain behind the dashboard.
"""
from __future__ import annotations

import asyncio

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import SignalRecord
from app.db.session import SessionLocal
from app.notifications.service import notifier
from app.notifications.ws import manager
from app.schemas import Action
from app.signals.service import scan

log = get_logger(__name__)


class Scanner:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False

    async def _cycle(self) -> None:
        results = await scan()
        async with SessionLocal() as session:
            for r in results:
                s = r.signal
                rec = SignalRecord(
                    symbol=s.symbol, timeframe=s.timeframe, exchange=s.exchange,
                    direction=str(s.direction), action=str(s.action), confidence=s.confidence,
                    probability_up=s.probability_up, risk_level=str(s.risk_level),
                    entry_price=s.entry_price, stop_loss=s.stop_loss,
                    take_profit_1=s.take_profit_1, take_profit_2=s.take_profit_2,
                    take_profit_3=s.take_profit_3, risk_reward=s.risk_reward,
                    news_sentiment=str(s.news_sentiment), payload=r.to_dict(),
                )
                session.add(rec)
            await session.commit()

        for r in results:
            await manager.broadcast({"type": "signal", "data": r.to_dict()})
            if str(r.signal.action) in (Action.BUY.value, Action.SELL.value):
                await notifier.dispatch(r.signal, ["telegram"])
        log.info("scan cycle complete: %d signals", len(results))

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._cycle()
            except Exception as exc:
                log.warning("scan cycle failed: %s", exc)
            await asyncio.sleep(settings.scan_interval_seconds)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("background scanner started (every %ds)", settings.scan_interval_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()


scanner = Scanner()
