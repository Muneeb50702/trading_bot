"""Admin & analytics: users, health, performance analytics, audit log."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin
from app.db.models import AuditLog, SignalRecord, Trade, User
from app.db.session import get_session
from app.ml.predictor import get_model
from app.risk.manager import risk_manager

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/health")
async def health(session: AsyncSession = Depends(get_session)):
    n_users = (await session.execute(select(func.count(User.id)))).scalar()
    n_signals = (await session.execute(select(func.count(SignalRecord.id)))).scalar()
    halted, reason = risk_manager.kill_switch()
    return {
        "status": "ok",
        "model_trained": get_model().is_trained,
        "users": n_users,
        "signals_stored": n_signals,
        "risk": {"halted": halted, "reason": reason,
                 "daily_pnl": risk_manager.state.daily_pnl,
                 "consecutive_losses": risk_manager.state.consecutive_losses},
    }


@router.get("/analytics", dependencies=[Depends(require_admin)])
async def analytics(session: AsyncSession = Depends(get_session)):
    since = datetime.now(timezone.utc) - timedelta(days=7)
    rows = (await session.execute(
        select(SignalRecord).where(SignalRecord.created_at >= since)
    )).scalars().all()

    total = len(rows)
    by_action: dict[str, int] = {}
    conf_sum = 0.0
    for r in rows:
        by_action[r.action] = by_action.get(r.action, 0) + 1
        conf_sum += r.confidence

    closed = (await session.execute(
        select(Trade).where(Trade.status == "CLOSED")
    )).scalars().all()
    wins = [t for t in closed if (t.pnl or 0) > 0]
    pnl = sum(t.pnl or 0 for t in closed)

    return {
        "signals_7d": total,
        "by_action": by_action,
        "avg_confidence": round(conf_sum / total, 4) if total else 0,
        "trades_closed": len(closed),
        "win_rate": round(len(wins) / len(closed), 4) if closed else 0,
        "net_pnl": round(pnl, 2),
    }


@router.get("/users", dependencies=[Depends(require_admin)])
async def list_users(session: AsyncSession = Depends(get_session)):
    users = (await session.execute(select(User))).scalars().all()
    return [{"id": u.id, "email": u.email, "role": u.role, "active": u.is_active,
             "totp": u.totp_enabled, "created_at": u.created_at.isoformat()} for u in users]


@router.get("/audit", dependencies=[Depends(require_admin)])
async def audit(limit: int = 100, session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    )).scalars().all()
    return [{"action": a.action, "detail": a.detail, "user_id": a.user_id,
             "created_at": a.created_at.isoformat()} for a in rows]
