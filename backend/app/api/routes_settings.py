"""User settings & encrypted exchange API-key management."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.security import encrypt_secret
from app.db.models import ExchangeApiKey, User, UserSettings
from app.db.session import get_session

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingsIn(BaseModel):
    symbols: list[str] | None = None
    timeframes: list[str] | None = None
    exchange: str | None = None
    leverage: int | None = None
    risk_per_trade_pct: float | None = None
    notify_channels: list[str] | None = None


async def _get_settings(session: AsyncSession, user: User) -> UserSettings:
    s = (await session.execute(
        select(UserSettings).where(UserSettings.user_id == user.id)
    )).scalar_one_or_none()
    if not s:
        s = UserSettings(user_id=user.id, symbols=[], timeframes=[])
        session.add(s)
        await session.commit()
        await session.refresh(s)
    return s


@router.get("")
async def get_settings(user: User = Depends(get_current_user),
                       session: AsyncSession = Depends(get_session)):
    s = await _get_settings(session, user)
    return {"symbols": s.symbols, "timeframes": s.timeframes, "exchange": s.exchange,
            "leverage": s.leverage, "risk_per_trade_pct": s.risk_per_trade_pct,
            "notify_channels": s.notify_channels}


@router.put("")
async def update_settings(body: SettingsIn, user: User = Depends(get_current_user),
                          session: AsyncSession = Depends(get_session)):
    s = await _get_settings(session, user)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(s, field, value)
    await session.commit()
    return {"status": "updated"}


class ApiKeyIn(BaseModel):
    exchange: str
    api_key: str
    api_secret: str


@router.post("/api-keys")
async def add_api_key(body: ApiKeyIn, user: User = Depends(get_current_user),
                      session: AsyncSession = Depends(get_session)):
    """Store exchange credentials encrypted at rest (Fernet)."""
    session.add(ExchangeApiKey(
        user_id=user.id, exchange=body.exchange,
        api_key_enc=encrypt_secret(body.api_key),
        api_secret_enc=encrypt_secret(body.api_secret),
    ))
    await session.commit()
    return {"status": "stored", "exchange": body.exchange}


@router.get("/api-keys")
async def list_api_keys(user: User = Depends(get_current_user),
                        session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        select(ExchangeApiKey).where(ExchangeApiKey.user_id == user.id)
    )).scalars().all()
    # never return secrets — only masked metadata
    return [{"id": r.id, "exchange": r.exchange, "created_at": r.created_at.isoformat()}
            for r in rows]
