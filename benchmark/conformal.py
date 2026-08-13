"""Block-bootstrap calibrated confidence bands (Kunsch 1989 moving-block
bootstrap) for the market-generator comparison suite.

Named precisely, not "conformal prediction": textbook split-conformal
prediction relies on exchangeable (i.i.d.-like) data, which real market
returns are not -- they're autocorrelated by construction (that's what
volatility clustering *is*). Calling this "conformal" in a diagnostics
writeup would overclaim a guarantee the method doesn't have. What this
actually is: resample the real reference data via a moving-block
bootstrap (which preserves its own short-range dependence structure,
unlike an i.i.d. resample) to learn how much a statistic naturally
bounces around across equally-sized real samples of the same process,
then use that as a calibrated pass/fail band for synthetic data. Still a
rigorous, standard, well-understood technique -- just correctly named.

Deliberately decoupled from benchmark/stylized_facts.py (generic
stat_fn/distance_fn callables, not stylized-facts-specific types) --
mirrors how benchmark/cv.py's purging logic stays decoupled from the
specific models it's used to score. benchmark/generator_ladder.py wires
this module and stylized_facts.py together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def moving_block_bootstrap(x: np.ndarray, block_size: int, n_out: int, seed: int | None = None) -> np.ndarray:
    """Concatenates randomly-chosen contiguous length-`block_size` blocks
    of `x` (with replacement across block choices) until reaching at least
    `n_out` points, then trims to exactly `n_out`. Preserves `x`'s own
    short-range dependence structure within each block -- an i.i.d.
    resample would destroy exactly the autocorrelation this module exists
    to characterize.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not 1 <= block_size <= n:
        raise ValueError(f"block_size must be in [1, {n}], got {block_size}")
    if n_out < 1:
        raise ValueError("n_out must be >= 1")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n_out // block_size)  # ceil division
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    blocks = [x[s : s + block_size] for s in starts]
    return np.concatenate(blocks)[:n_out]


@dataclass
class ConformalBand:
    reference_stat: Any
    null_distances: np.ndarray
    threshold: float
    alpha: float


def _order_statistic_threshold(null_distances: np.ndarray, alpha: float) -> float:
    """Vovk et al.'s finite-sample-valid quantile: the
    ceil((n+1)(1-alpha))-th order statistic, not a naive percentile (which
    is anti-conservative at small n_bootstrap). Clamped to the largest
    observed distance if (n+1)(1-alpha) would exceed n -- with this few
    bootstrap draws the target alpha can't be achieved exactly; using the
    max is the conservative fallback, not a silent under-coverage.
    """
    n = len(null_distances)
    k = min(math.ceil((n + 1) * (1 - alpha)), n)
    return float(np.sort(null_distances)[k - 1])


def calibrate_band(
    reference: np.ndarray,
    stat_fn: Callable[[np.ndarray], Any],
    distance_fn: Callable[[Any, Any], float],
    alpha: float,
    n_bootstrap: int = 500,
    block_size: int = 30,
    resample_length: int | None = None,
    seed: int | None = None,
) -> ConformalBand:
    """Learns how much `distance_fn(stat_fn(resample), stat_fn(reference))`
    naturally varies across `n_bootstrap` block-bootstrap resamples of
    `reference` against itself (no generator involved yet) -- the
    real-data-only calibration step. `threshold` is then the distance a
    generator's own synthetic output must not exceed to "pass" at
    confidence level `1 - alpha`.

    `resample_length` MUST match the length of whatever will actually be
    scored against this band (a generator's synthetic path), not the
    reference's own length, which was this function's original (wrong)
    default. Sample statistics like ACF/kurtosis have sampling variance
    that shrinks with N -- a band calibrated from resamples as long as a
    ~23,000-point reference is far too tight for anything scored at
    ~2,000 points, and would reject real data at that shorter length too.
    Confirmed the hard way: real ~2,000-point contiguous subsamples of the
    same real reference series -- genuinely real data, by construction the
    "most realistic" input possible -- failed a same-length-mismatched
    band on raw_return_acf and leverage_curve 0/20 times, matching every
    generator's failure rate on those same facts almost exactly (see
    diagnostics/2026-08-13-conformal-band-length-mismatch/findings.md).
    Defaults to `len(reference)` only for backward compatibility with
    same-length use cases; callers scoring shorter candidates must pass
    the candidate length explicitly.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    reference = np.asarray(reference, dtype=float)
    reference_stat = stat_fn(reference)
    resample_length = resample_length if resample_length is not None else len(reference)

    rng = np.random.default_rng(seed)
    null_distances = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = moving_block_bootstrap(
            reference, block_size, resample_length, seed=int(rng.integers(0, 2**32 - 1))
        )
        null_distances[i] = distance_fn(stat_fn(resample), reference_stat)

    threshold = _order_statistic_threshold(null_distances, alpha)
    return ConformalBand(reference_stat=reference_stat, null_distances=null_distances, threshold=threshold, alpha=alpha)


def covered(observed_distance: float, band: ConformalBand) -> bool:
    return observed_distance <= band.threshold


def coverage_rate(observed_distances: np.ndarray, band: ConformalBand) -> float:
    observed_distances = np.asarray(observed_distances, dtype=float)
    return float(np.mean(observed_distances <= band.threshold))
