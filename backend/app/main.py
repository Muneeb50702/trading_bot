"""FastAPI application entry point.

Wires routers, WebSocket live feed, DB init, model load, and the optional
background scanner into one ASGI app.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_admin,
    routes_auth,
    routes_backtest,
    routes_model,
    routes_settings,
    routes_signals,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.data.market import market_data
from app.db.session import init_db
from app.ml.predictor import try_load_model
from app.notifications.ws import manager
from app.signals.scanner import scanner

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %s (%s)", settings.app_name, settings.env)
    await init_db()
    if try_load_model("default"):
        log.info("probability model loaded")
    else:
        log.info("no trained model found — running on confluence only")
    if os.getenv("BOT_ENABLE_SCANNER", "0") == "1":
        scanner.start()
    yield
    await scanner.stop()
    await market_data.close()


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("BOT_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (routes_auth, routes_signals, routes_backtest, routes_model,
          routes_admin, routes_settings):
    app.include_router(r.router)


@app.get("/")
async def root():
    return {"name": settings.app_name, "version": "1.0.0", "status": "online",
            "docs": "/docs"}


@app.websocket("/ws/signals")
async def ws_signals(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; clients may send pings
    except WebSocketDisconnect:
        await manager.disconnect(ws)
