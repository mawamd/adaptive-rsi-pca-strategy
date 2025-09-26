"""Minimal subset of scipy.stats required for testing."""

from __future__ import annotations

import math
from typing import Sequence


def chi2_contingency(table: Sequence[Sequence[float]]):
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError("contingency table must not be empty")

    row_totals = [float(sum(row)) for row in table]
    col_totals = [float(sum(table[r][c] for r in range(rows))) for c in range(cols)]
    grand_total = float(sum(row_totals))
    if grand_total == 0:
        raise ValueError("grand total must be positive")

    expected = [
        [row_totals[r] * col_totals[c] / grand_total for c in range(cols)]
        for r in range(rows)
    ]

    chi2 = 0.0
    for r in range(rows):
        for c in range(cols):
            exp = expected[r][c]
            if exp == 0:
                continue
            obs = float(table[r][c])
            chi2 += (obs - exp) ** 2 / exp

    dof = (rows - 1) * (cols - 1)
    if dof == 1:
        p_value = math.erfc(math.sqrt(chi2 / 2))
    else:
        p_value = math.exp(-chi2 / 2)

    return chi2, p_value, dof, expected


__all__ = ["chi2_contingency"]
