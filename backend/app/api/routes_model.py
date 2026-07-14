"""ML model API: train / retrain on live history, inspect status."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import require_admin
from app.data.market import market_data
from app.ml.predictor import get_model
from app.core.config import settings

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status")
async def status():
    m = get_model()
    return {"trained": m.is_trained, "meta": m.meta, "blend_weight": settings.ml_blend_weight}


@router.post("/train", dependencies=[Depends(require_admin)])
async def train(
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("15m"),
    candles: int = Query(1500, le=1500),
    horizon: int = Query(3, ge=1, le=12),
):
    """Pull recent history and (re)train the probability model on it."""
    df = await market_data.fetch_ohlcv(symbol, timeframe, limit=candles)
    model = get_model()
    try:
        metrics = model.train(df, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    model.save("default")
    return {"status": "trained", "metrics": metrics}
