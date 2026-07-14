"""Multi-channel notifications: Telegram, Email, and WebSocket push.

Each channel is best-effort and independent — a failing channel logs and is
skipped, never blocking signal delivery. WhatsApp can be added via the same
interface (Twilio) by implementing another `_send_*` method.
"""
from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas import Signal

log = get_logger(__name__)


def format_signal(sig: Signal) -> str:
    emoji = {"BUY": "🟢", "SELL": "🔴", "NO_TRADE": "⚪"}.get(str(sig.action), "")
    return (
        f"{emoji} {sig.action} {sig.symbol} [{sig.timeframe}]\n"
        f"Direction: {sig.direction} | Confidence: {sig.confidence:.0%} | Risk: {sig.risk_level}\n"
        f"Entry: {sig.entry_price}\nSL: {sig.stop_loss}\n"
        f"TP1: {sig.take_profit_1} | TP2: {sig.take_profit_2} | TP3: {sig.take_profit_3}\n"
        f"RR: {sig.risk_reward} | News: {sig.news_sentiment}\n"
        f"P(up): {sig.probability_up:.0%}"
    )


class Notifier:
    async def send_telegram(self, text: str) -> bool:
        token, chat = settings.telegram_bot_token, settings.telegram_chat_id
        if not token or not chat:
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat, "text": text},
                )
                return r.status_code == 200
        except Exception as exc:
            log.warning("telegram send failed: %s", exc)
            return False

    def send_email(self, subject: str, body: str, to: str | None = None) -> bool:
        if not settings.smtp_host or not settings.smtp_user:
            return False
        to = to or settings.smtp_user
        try:
            msg = MIMEText(body)
            msg["Subject"], msg["From"], msg["To"] = subject, settings.smtp_user, to
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as s:
                s.starttls()
                s.login(settings.smtp_user, settings.smtp_password)
                s.send_message(msg)
            return True
        except Exception as exc:
            log.warning("email send failed: %s", exc)
            return False

    async def dispatch(self, sig: Signal, channels: list[str]) -> dict[str, bool]:
        text = format_signal(sig)
        results: dict[str, bool] = {}
        if "telegram" in channels:
            results["telegram"] = await self.send_telegram(text)
        if "email" in channels:
            results["email"] = self.send_email(f"Signal {sig.action} {sig.symbol}", text)
        return results


notifier = Notifier()
