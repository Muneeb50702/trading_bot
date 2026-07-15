"""Train the probability model on a large, multi-pair historical dataset.

Usage:
    python scripts/train_model.py                 # defaults below
    python scripts/train_model.py 15m 15000 3     # timeframe candles horizon

Reports HONEST out-of-sample metrics (test slice never seen by model or
calibrator) and saves the model to models/default.joblib.
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from app.core.logging import get_logger  # noqa: E402
from app.data.market import market_data  # noqa: E402
from app.ml.dataset import build_training_set  # noqa: E402
from app.ml.model import ProbabilityModel  # noqa: E402

log = get_logger("train")

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


async def main() -> int:
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "15m"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 15_000
    horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    print(f"\nFetching ~{target} candles × {len(SYMBOLS)} pairs on {timeframe} …")
    try:
        X, y, _frames = await build_training_set(
            SYMBOLS, timeframe, target_candles=target, horizon=horizon
        )
    finally:
        await market_data.close()

    print(f"Dataset: {len(X):,} labelled samples · {X.shape[1]} features · "
          f"base rate {y.mean():.3f}")

    model = ProbabilityModel()
    metrics = model.train_on_matrix(X, y, meta_extra={"timeframe": timeframe,
                                                      "horizon": horizon,
                                                      "symbols": SYMBOLS})
    path = model.save("default")

    print("\n" + "=" * 60)
    print("  HONEST OUT-OF-SAMPLE RESULTS (test slice)")
    print("=" * 60)
    print(f"  Samples (train/test) : {metrics['n_train']:,} / {metrics['n_test']:,}")
    print(f"  Directional accuracy : {metrics['test_accuracy']:.1%}")
    print(f"  ROC-AUC              : {metrics['test_auc']:.4f}   (0.5 = coin flip)")
    print(f"  Precision / Recall   : {metrics['test_precision']:.1%} / {metrics['test_recall']:.1%}")
    print(f"  F1                   : {metrics['test_f1']:.4f}")
    print(f"  Brier score          : {metrics['brier']:.4f}   (lower = better calibrated)")
    if metrics["high_conf_accuracy"]:
        print(f"  Accuracy on CONFIDENT calls : {metrics['high_conf_accuracy']:.1%} "
              f"(covers {metrics['high_conf_share']:.0%} of bars)")
    print(f"  Base rate (up freq)  : {metrics['base_rate']:.1%}")
    print(f"\n  Top features         : {', '.join(metrics['top_features'][:6])}")
    print(f"  Saved                : {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
