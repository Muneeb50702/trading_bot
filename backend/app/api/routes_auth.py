"""Auth: register, login (with optional TOTP), enable 2FA."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core import security
from app.db.models import AuditLog, User, UserSettings
from app.db.session import get_session
from app.core.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str
    totp_code: str | None = None


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


@router.post("/register", response_model=TokenOut)
async def register(body: RegisterIn, session: AsyncSession = Depends(get_session)):
    exists = (await session.execute(select(User).where(User.email == body.email))).scalar_one_or_none()
    if exists:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "email already registered")
    # first user becomes admin
    count = (await session.execute(select(func.count(User.id)))).scalar() or 0
    user = User(
        email=body.email,
        hashed_password=security.hash_password(body.password),
        role="admin" if count == 0 else "user",
    )
    session.add(user)
    await session.flush()
    session.add(UserSettings(
        user_id=user.id, symbols=settings.default_symbols,
        timeframes=settings.default_timeframes, exchange=settings.default_exchange,
    ))
    session.add(AuditLog(user_id=user.id, action="register", detail=body.email))
    await session.commit()
    return TokenOut(access_token=security.create_access_token(user.email, user.role), role=user.role)


@router.post("/login", response_model=TokenOut)
async def login(body: LoginIn, session: AsyncSession = Depends(get_session)):
    user = (await session.execute(select(User).where(User.email == body.email))).scalar_one_or_none()
    if not user or not security.verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    if user.totp_enabled:
        if not body.totp_code or not security.verify_totp(user.totp_secret or "", body.totp_code):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing 2FA code")
    session.add(AuditLog(user_id=user.id, action="login", detail=user.email))
    await session.commit()
    return TokenOut(access_token=security.create_access_token(user.email, user.role), role=user.role)


class TotpSetupOut(BaseModel):
    secret: str
    otpauth_uri: str


@router.post("/2fa/setup", response_model=TotpSetupOut)
async def setup_2fa(user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    secret = security.new_totp_secret()
    user.totp_secret = secret
    await session.commit()
    return TotpSetupOut(secret=secret, otpauth_uri=security.totp_uri(secret, user.email))


class TotpVerifyIn(BaseModel):
    code: str


@router.post("/2fa/enable")
async def enable_2fa(body: TotpVerifyIn, user: User = Depends(get_current_user),
                     session: AsyncSession = Depends(get_session)):
    if not user.totp_secret or not security.verify_totp(user.totp_secret, body.code):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid code")
    user.totp_enabled = True
    await session.commit()
    return {"enabled": True}


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"email": user.email, "role": user.role, "totp_enabled": user.totp_enabled}
