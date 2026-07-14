"""Offline smoke test: exercises the whole engine on synthetic OHLCV.

No network / no exchange needed. Verifies indicators -> confluence -> ML ->
signal builder -> backtester all run and produce coherent output.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def synth_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # trend + cycles + noise -> semi-realistic price
    t = np.arange(n)
    price = 100 + np.cumsum(rng.normal(0.05, 1.0, n)) + 8 * np.sin(t / 40)
    price = np.maximum(price, 1.0)
    close = price
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.6, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.6, n))
    vol = np.abs(rng.normal(1000, 300, n)) * (1 + 0.5 * np.abs(np.sin(t / 25)))
    ts = (np.arange(n) * 300_000).astype("int64")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol,
    })


def main() -> int:
    from app.engine import confluence, signal_builder
    from app.ml.model import ProbabilityModel
    from app.ml.predictor import blend_probability, get_model
    from app.backtest.engine import run_backtest

    df = synth_ohlcv()
    print(f"synthetic candles: {len(df)}")

    prob, votes = confluence.analyze(df)
    print(f"\n[confluence] P(up)={prob:.3f} from {len(votes)} modules")
    for v in sorted(votes, key=lambda x: -x.strength)[:6]:
        print(f"   {v.module:<18} bias={v.bias:+.2f} str={v.strength:.2f}  {v.note}")

    sig = signal_builder.build_signal(df, prob, votes, symbol="BTC/USDT",
                                      timeframe="5m", exchange="synthetic")
    print(f"\n[signal] {sig.action} {sig.direction} conf={sig.confidence:.2%} "
          f"risk={sig.risk_level}")
    print(f"   entry={sig.entry_price:.2f} SL={sig.stop_loss:.2f} "
          f"TP1={sig.take_profit_1:.2f} TP2={sig.take_profit_2:.2f} "
          f"TP3={sig.take_profit_3:.2f} RR={sig.risk_reward}")

    # train + blend
    model = get_model()
    metrics = model.train(df, horizon=3)
    print(f"\n[ml] trained: {metrics}")
    final, ml = blend_probability(df, prob)
    print(f"[ml] blended P(up)={final:.3f} (ml={ml:.3f}, confluence={prob:.3f})")

    report = run_backtest(df, symbol="BTC/USDT", timeframe="5m", warmup=150,
                          window=150, hold_bars=20)
    print(f"\n[backtest] trades={report.trades} win_rate={report.win_rate:.1%} "
          f"total_R={report.total_r} PF={report.profit_factor} "
          f"maxDD={report.max_drawdown_r}R")

    assert 0 <= prob <= 1
    assert 0 <= sig.confidence <= 1
    assert sig.take_profit_1 != sig.entry_price
    print("\n✅ ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
