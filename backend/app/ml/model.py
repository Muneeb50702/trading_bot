"""LightGBM probability model wrapper.

Predicts P(up) from the engineered feature vector. Chosen over a deep net
because indicator features are tabular: LightGBM trains in seconds, calibrates
well, serves sub-millisecond, and produces honest probabilities — matching the
"probability-based, no 90% guarantee" mandate. The interface is model-agnostic
so a PyTorch/TF model can be dropped in behind it later.
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
        self._meta: dict = {}

    # ------------------------------------------------------------------ train
    def train(self, df: pd.DataFrame, horizon: int = 3) -> dict:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y = build_labeled_dataset(df, horizon=horizon)
        if len(X) < 200 or y.nunique() < 2:
            raise ValueError(f"insufficient/degenerate data for training (n={len(X)})")

        # time-ordered split (no shuffle) to avoid look-ahead leakage
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        proba = model.predict_proba(X_val)[:, 1]
        metrics = {
            "n_samples": int(len(X)),
            "val_accuracy": round(float(accuracy_score(y_val, proba > 0.5)), 4),
            "val_auc": round(float(roc_auc_score(y_val, proba)), 4),
            "horizon": horizon,
        }
        self._model = model
        self._meta = metrics
        log.info("trained model: %s", metrics)
        return metrics

    # ---------------------------------------------------------------- predict
    def predict_proba(self, features_row: pd.Series | pd.DataFrame) -> float | None:
        if self._model is None:
            return None
        if isinstance(features_row, pd.Series):
            features_row = features_row.to_frame().T
        X = features_row[FEATURE_COLUMNS].fillna(0.0)
        return float(self._model.predict_proba(X)[:, 1][-1])

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
        joblib.dump({"model": self._model, "meta": self._meta}, path)
        return path

    def load(self, name: str = "default") -> bool:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            return False
        blob = joblib.load(path)
        self._model, self._meta = blob["model"], blob.get("meta", {})
        log.info("loaded model %s: %s", name, self._meta)
        return True
