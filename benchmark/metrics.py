"""Shared metrics for the benchmark ladder. Grows as later rungs need more
(hit rate, precision@k, etc.) -- only what Rung 0/1 need is here so far.
"""

from __future__ import annotations


def branching_ratio_error(estimated: float, published: float = 0.81) -> float:
    """Relative error of an estimated branching ratio against a reference
    (default: Filimonov & Sornette 2012's ~0.81 on E-mini S&P 500 ticks).
    """
    return abs(estimated - published) / published
