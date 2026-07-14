"""Blends the deterministic confluence probability with the ML model output.

Degrades gracefully: with no trained model, returns the confluence probability
untouched, so the bot is fully operational before any training happens.
"""
from __future__ import annotations

import pandas as pd

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.features import build_features
from app.ml.model import ProbabilityModel

log = get_logger(__name__)

# process-wide singleton; loaded at startup, retrained via API/scheduler
_model = ProbabilityModel()


def get_model() -> ProbabilityModel:
    return _model


def try_load_model(name: str = "default") -> bool:
    return _model.load(name)


def blend_probability(df: pd.DataFrame, confluence_prob: float) -> tuple[float, float | None]:
    """Return (final_prob, ml_prob). ml_prob is None when no model is loaded."""
    if not _model.is_trained:
        return confluence_prob, None
    try:
        feats = build_features(df).iloc[[-1]]
        ml_prob = _model.predict_proba(feats)
    except Exception as exc:
        log.warning("ML prediction failed, using confluence only: %s", exc)
        return confluence_prob, None
    if ml_prob is None:
        return confluence_prob, None
    w = settings.ml_blend_weight
    return w * ml_prob + (1 - w) * confluence_prob, ml_prob
