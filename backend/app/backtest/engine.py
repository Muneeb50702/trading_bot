"""Event-driven backtester.

Replays historical candles, generates a signal on a rolling window at each step,
simulates the trade (entry next-open, exit at SL or TP1 by intrabar high/low),
and reports win rate, P/L, profit factor and max drawdown — the metrics the
dashboard's Backtesting panel needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.engine import confluence, signal_builder
from app.schemas import Action


@dataclass
class Trade:
    entry_index: int
    direction: str
    entry: float
    stop: float
    target: float
    exit_index: int
    exit_price: float
    r_multiple: float
    won: bool


@dataclass
class BacktestReport:
    symbol: str
    timeframe: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_r: float
    avg_r: float
    profit_factor: float
    max_drawdown_r: float
    final_equity_r: float
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def run_backtest(
    df: pd.DataFrame, *, symbol: str, timeframe: str, exchange: str = "backtest",
    window: int = 200, warmup: int = 120, step: int = 1, hold_bars: int = 24,
    min_confidence: float = 0.55,
) -> BacktestReport:
    trades: list[Trade] = []
    n = len(df)
    i = warmup
    while i < n - 1:
        win = df.iloc[max(0, i - window): i + 1]
        prob, votes = confluence.analyze(win)
        sig = signal_builder.build_signal(
            win, prob, votes, symbol=symbol, timeframe=timeframe, exchange=exchange
        )
        if sig.action == Action.NO_TRADE or sig.confidence < min_confidence:
            i += step
            continue

        # enter at next candle open
        entry = float(df["open"].iloc[i + 1])
        long = sig.action == Action.BUY
        risk = abs(entry - sig.stop_loss) or entry * 0.001
        stop = entry - risk if long else entry + risk
        target = entry + risk if long else entry - risk  # TP1 == 1R

        exit_idx, exit_px, r, won = i + 1 + hold_bars, entry, 0.0, False
        for j in range(i + 1, min(i + 1 + hold_bars, n)):
            hi, lo = df["high"].iloc[j], df["low"].iloc[j]
            if long:
                if lo <= stop:
                    exit_idx, exit_px, r, won = j, stop, -1.0, False; break
                if hi >= target:
                    exit_idx, exit_px, r, won = j, target, 1.0, True; break
            else:
                if hi >= stop:
                    exit_idx, exit_px, r, won = j, stop, -1.0, False; break
                if lo <= target:
                    exit_idx, exit_px, r, won = j, target, 1.0, True; break
        else:
            # timed exit: mark-to-market in R
            exit_px = float(df["close"].iloc[min(i + hold_bars, n - 1)])
            r = ((exit_px - entry) if long else (entry - exit_px)) / risk
            won = r > 0

        trades.append(Trade(i + 1, "LONG" if long else "SHORT", entry, stop,
                            target, exit_idx, exit_px, round(r, 3), won))
        i = max(exit_idx, i + step)  # no overlapping trades

    return _report(trades, symbol, timeframe)


def _report(trades: list[Trade], symbol: str, timeframe: str) -> BacktestReport:
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    total_r = sum(t.r_multiple for t in trades)
    gross_win = sum(t.r_multiple for t in wins)
    gross_loss = abs(sum(t.r_multiple for t in losses))

    equity, curve, peak, max_dd = 0.0, [], 0.0, 0.0
    for t in trades:
        equity += t.r_multiple
        curve.append(round(equity, 3))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    n = len(trades)
    return BacktestReport(
        symbol=symbol, timeframe=timeframe, trades=n, wins=len(wins), losses=len(losses),
        win_rate=round(len(wins) / n, 4) if n else 0.0,
        total_r=round(total_r, 3),
        avg_r=round(total_r / n, 3) if n else 0.0,
        profit_factor=round(gross_win / gross_loss, 3) if gross_loss else float("inf") if gross_win else 0.0,
        max_drawdown_r=round(max_dd, 3),
        final_equity_r=round(equity, 3),
        equity_curve=curve,
    )
