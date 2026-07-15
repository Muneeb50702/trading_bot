"""Fetch once, sweep prediction horizons, report which gives the best edge."""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from app.data.market import market_data  # noqa: E402
from app.ml.dataset import fetch_history  # noqa: E402
from app.ml.features import build_labeled_dataset  # noqa: E402
from app.ml.model import ProbabilityModel  # noqa: E402
import pandas as pd  # noqa: E402

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


async def main() -> int:
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 15_000
    print(f"Fetching {target} × {len(SYMBOLS)} on {tf} …")
    frames = await asyncio.gather(*[fetch_history(s, tf, target_candles=target) for s in SYMBOLS])
    await market_data.close()

    print(f"\n{'horizon':>7} | {'samples':>8} | {'acc':>6} | {'auc':>6} | {'conf_acc':>8} | {'conf_share':>10}")
    print("-" * 60)
    best = None
    for horizon in (2, 3, 4, 5, 8):
        Xs, ys = [], []
        for df in frames:
            X, y = build_labeled_dataset(df, horizon=horizon)
            Xs.append(X); ys.append(y)
        X = pd.concat(Xs, ignore_index=True); y = pd.concat(ys, ignore_index=True)
        m = ProbabilityModel().train_on_matrix(X, y)
        ca = m["high_conf_accuracy"]
        print(f"{horizon:>7} | {m['n_total']:>8,} | {m['test_accuracy']:>6.1%} | "
              f"{m['test_auc']:>6.3f} | {(ca or 0):>8.1%} | {m['high_conf_share']:>10.1%}")
        score = (ca or 0) * m["high_conf_share"] + m["test_accuracy"] * 0.3
        if best is None or score > best[1]:
            best = (horizon, score, m)
    print("-" * 60)
    print(f"BEST horizon = {best[0]}  (acc {best[2]['test_accuracy']:.1%}, "
          f"confident-acc {best[2]['high_conf_accuracy'] or 0:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
