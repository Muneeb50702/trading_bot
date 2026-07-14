"""Confluence engine: run the full analysis bank and fuse votes -> probability.

This is the *deterministic* probability estimate. The ML layer later refines
it, but the bot is fully functional on this alone (works with zero training).
"""
from __future__ import annotations

import math

import pandas as pd

import app.analysis  # noqa: F401  (triggers module registration)
from app.analysis.base import REGISTRY
from app.core.logging import get_logger
from app.schemas import Vote

log = get_logger(__name__)


def run_modules(df: pd.DataFrame) -> list[Vote]:
    """Execute every registered analysis module, tolerating individual failures."""
    votes: list[Vote] = []
    for name, (fn, _weight) in REGISTRY.items():
        try:
            votes.append(fn(df))
        except Exception as exc:  # a broken module must never kill a signal
            log.warning("module %s failed: %s", name, exc)
    return votes


def confluence_score(votes: list[Vote]) -> float:
    """Weighted, module-weight-aware net bias in [-1, 1].

    net = Σ(bias·strength·module_weight) / Σ(strength·module_weight)
    """
    num = den = 0.0
    for v in votes:
        w = REGISTRY.get(v.module, (None, 1.0))[1]
        num += v.weighted * w
        den += v.strength * w
    return num / den if den else 0.0


def score_to_probability(score: float, sharpness: float = 1.8) -> float:
    """Squash net bias [-1,1] into P(up) via a deliberately gentle logistic.

    Calibrated so even full agreement (score=±1) tops out near 0.86, never
    implying certainty — the requirements forbid guaranteeing high accuracy.
    Typical net bias (~0.3-0.5) maps to a humble 0.63-0.71.
    """
    return 1.0 / (1.0 + math.exp(-sharpness * score))


def analyze(df: pd.DataFrame) -> tuple[float, list[Vote]]:
    """Return (probability_up, votes) for a candle frame."""
    votes = run_modules(df)
    score = confluence_score(votes)
    return score_to_probability(score), votes
