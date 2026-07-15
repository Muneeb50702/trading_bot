"""LightGBM probability model with time-series-honest training + calibration.

Predicts P(up) from the engineered feature vector. Chosen over a deep net
because indicator features are tabular: LightGBM trains in seconds, calibrates
well, serves sub-millisecond, and produces honest probabilities — matching the
"probability-based, no 90% guarantee" mandate. The interface is model-agnostic
so a PyTorch/TF model can be dropped in behind it later.

Training discipline (why the reported numbers are trustworthy):
  * TIME-ORDERED splits (no shuffling) -> no look-ahead leakage.
  * A separate calibration slice fits an isotonic map so probabilities mean
    what they say (a 0.6 really is ~60% historically).
  * All headline metrics are computed on a FINAL TEST slice the model and the
    calibrator never saw.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.ml.features import FEATURE_COLUMNS, build_labeled_dataset

log = get_logger(__name__)

MODEL_DIR = Path(os.getenv("BOT_MODEL_DIR", "./models"))


class ProbabilityModel:
    def __init__(self) -> None:
        self._model = None
        self._calibrator = None
        self._meta: dict = {}

    # ------------------------------------------------------------ train (matrix)
    def train_on_matrix(self, X: pd.DataFrame, y: pd.Series, *, meta_extra: dict | None = None) -> dict:
        """Train on a prebuilt, TIME-ORDERED (X, y). Returns honest test metrics."""
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        X = X[FEATURE_COLUMNS].reset_index(drop=True)
        y = y.reset_index(drop=True)
        n = len(X)
        if n < 500 or y.nunique() < 2:
            raise ValueError(f"insufficient/degenerate data (n={n})")

        # time-ordered: 70% train, 15% calibrate, 15% test
        i_tr, i_cal = int(n * 0.70), int(n * 0.85)
        X_tr, y_tr = X.iloc[:i_tr], y.iloc[:i_tr]
        X_cal, y_cal = X.iloc[i_tr:i_cal], y.iloc[i_tr:i_cal]
        X_te, y_te = X.iloc[i_cal:], y.iloc[i_cal:]

        model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.02, num_leaves=48, max_depth=-1,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])

        # isotonic calibration fitted on the calibration slice
        cal_raw = model.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_raw, y_cal.to_numpy())

        # honest metrics on the untouched test slice
        te_raw = model.predict_proba(X_te)[:, 1]
        te_cal = calibrator.predict(te_raw)
        pred = (te_cal > 0.5).astype(int)

        # accuracy among only high-confidence calls (how it's actually used)
        hi = (np.abs(te_cal - 0.5) >= 0.10)
        hi_acc = float(accuracy_score(y_te[hi], pred[hi])) if hi.sum() > 20 else None

        metrics = {
            "n_total": int(n), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "test_auc": round(float(roc_auc_score(y_te, te_cal)), 4),
            "test_precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "test_recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "test_f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
            "brier": round(float(brier_score_loss(y_te, te_cal)), 4),
            "high_conf_accuracy": round(hi_acc, 4) if hi_acc else None,
            "high_conf_share": round(float(hi.mean()), 4),
            "base_rate": round(float(y.mean()), 4),
        }
        if meta_extra:
            metrics.update(meta_extra)

        self._model, self._calibrator, self._meta = model, calibrator, metrics
        # store top feature importances for transparency
        imp = sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                     key=lambda t: -t[1])[:10]
        self._meta["top_features"] = [f"{k}({int(val)})" for k, val in imp]
        log.info("trained model: acc=%s auc=%s (n=%d)",
                 metrics["test_accuracy"], metrics["test_auc"], n)
        return metrics

    # ------------------------------------------------------------ train (single df)
    def train(self, df: pd.DataFrame, horizon: int = 3) -> dict:
        """Convenience: build features from one frame then train."""
        X, y = build_labeled_dataset(df, horizon=horizon)
        return self.train_on_matrix(X, y, meta_extra={"horizon": horizon})

    # ---------------------------------------------------------------- predict
    def predict_proba(self, features_row: pd.Series | pd.DataFrame) -> float | None:
        if self._model is None:
            return None
        if isinstance(features_row, pd.Series):
            features_row = features_row.to_frame().T
        X = features_row[FEATURE_COLUMNS].fillna(0.0)
        raw = float(self._model.predict_proba(X)[:, 1][-1])
        if self._calibrator is not None:
            return float(self._calibrator.predict([raw])[0])
        return raw

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def meta(self) -> dict:
        return self._meta

    # ------------------------------------------------------------- persistence
    def save(self, name: str = "default") -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump({"model": self._model, "calibrator": self._calibrator, "meta": self._meta}, path)
        return path

    def load(self, name: str = "default") -> bool:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            return False
        blob = joblib.load(path)
        self._model = blob["model"]
        self._calibrator = blob.get("calibrator")
        self._meta = blob.get("meta", {})
        log.info("loaded model %s: %s", name, self._meta.get("test_accuracy"))
        return True
