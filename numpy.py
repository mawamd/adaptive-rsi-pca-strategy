"""Lightweight subset of NumPy functionality used in tests."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.mean(values))


def median(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(statistics.median(values))


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def sqrt(value: float) -> float:
    return float(math.sqrt(value))


bool_ = bool


def isscalar(obj: object) -> bool:
    return isinstance(obj, (int, float, bool))
