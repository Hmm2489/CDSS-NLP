"""Certainty-factor arithmetic (MYCIN, Shortliffe & Buchanan 1975).

A certainty factor (CF) lies in [-1, 1]: 1 = definitely true, -1 = definitely
false, 0 = no information. Combining is commutative and associative, so the
order evidence arrives in doesn't change the result.
"""

from __future__ import annotations

from functools import reduce


def combine(x: float, y: float) -> float:
    """Combine two independent CFs for the same hypothesis."""
    if x >= 0 and y >= 0:
        return x + y * (1 - x)
    if x <= 0 and y <= 0:
        return x + y * (1 + x)
    denom = 1 - min(abs(x), abs(y))
    # Full certainty for and against at once (e.g. "no fever ... fever"): treat as unknown.
    return 0.0 if denom == 0 else (x + y) / denom


def combine_all(values) -> float:
    return reduce(combine, values, 0.0)
