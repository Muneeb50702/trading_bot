"""Risk management: position sizing, RR, daily-loss & consecutive-loss guards.

Stateful per account: tracks realised daily P/L and loss streaks so the engine
can veto new trades ("kill switch") exactly as the requirements demand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from app.core.config import settings
from app.schemas import Action, Signal


@dataclass
class RiskConfig:
    account_balance: float = settings.account_balance
    risk_per_trade_pct: float = settings.risk_per_trade_pct
    max_daily_loss_pct: float = settings.max_daily_loss_pct
    max_consecutive_losses: int = settings.max_consecutive_losses
    leverage: int = settings.default_leverage


@dataclass
class RiskState:
    day: date = field(default_factory=date.today)
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0


@dataclass
class PositionPlan:
    allowed: bool
    reason: str
    position_size: float = 0.0        # in quote currency (notional)
    quantity: float = 0.0            # in base currency
    risk_amount: float = 0.0         # cash at risk if SL hit
    risk_reward: float = 0.0
    leverage: int = 1


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.state = RiskState()

    def _roll_day(self) -> None:
        today = date.today()
        if today != self.state.day:
            self.state = RiskState(day=today)

    def kill_switch(self) -> tuple[bool, str]:
        """Is trading currently halted?"""
        self._roll_day()
        cfg, st = self.config, self.state
        max_loss = cfg.account_balance * cfg.max_daily_loss_pct / 100
        if st.daily_pnl <= -max_loss:
            return True, f"daily loss limit hit ({st.daily_pnl:.2f} <= -{max_loss:.2f})"
        if st.consecutive_losses >= cfg.max_consecutive_losses:
            return True, f"{st.consecutive_losses} consecutive losses"
        return False, "ok"

    def evaluate(self, signal: Signal) -> PositionPlan:
        halted, reason = self.kill_switch()
        if halted:
            return PositionPlan(allowed=False, reason=f"HALTED: {reason}")
        if signal.action == Action.NO_TRADE:
            return PositionPlan(allowed=False, reason="signal is No-Trade")

        cfg = self.config
        entry, stop = signal.entry_price, signal.stop_loss
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return PositionPlan(allowed=False, reason="invalid stop distance")

        risk_amount = cfg.account_balance * cfg.risk_per_trade_pct / 100
        quantity = risk_amount / risk_per_unit           # size so SL == risk_amount
        notional = quantity * entry
        max_notional = cfg.account_balance * cfg.leverage
        if notional > max_notional:                      # cap to available leverage
            notional = max_notional
            quantity = notional / entry
            risk_amount = quantity * risk_per_unit

        return PositionPlan(
            allowed=True,
            reason="ok",
            position_size=round(notional, 2),
            quantity=round(quantity, 8),
            risk_amount=round(risk_amount, 2),
            risk_reward=signal.risk_reward,
            leverage=cfg.leverage,
        )

    def record_trade(self, pnl: float) -> None:
        """Update daily P/L and loss streak after a trade closes."""
        self._roll_day()
        self.state.daily_pnl += pnl
        self.state.trades_today += 1
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0


risk_manager = RiskManager()
