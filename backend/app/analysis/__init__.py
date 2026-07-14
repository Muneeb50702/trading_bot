"""Importing this package registers every analysis module in the REGISTRY."""
from app.analysis import (  # noqa: F401
    momentum,
    price_action,
    smc,
    trend,
    volatility,
    volume,
)
from app.analysis.base import REGISTRY, register  # noqa: F401

__all__ = ["REGISTRY", "register"]
