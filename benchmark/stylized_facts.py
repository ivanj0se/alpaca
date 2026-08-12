"""Cont (2001)'s "stylized facts of asset returns" checklist, as pure
functions over a returns array. This is the shared measuring stick every
generator in the market-generator comparison suite (generators/,
models/tcn_forecaster.py) gets scored against -- the generative analogue
of benchmark/ladder.py's shared NLL scoring for the predictive rungs.

Five checks, all well-documented properties of real equity/index returns
that a naive generator (e.g. i.i.d. Gaussian) fails to reproduce:
  1. Absence of linear autocorrelation in raw returns (near zero at every lag)
  2. Volatility clustering (autocorrelation of |returns| is positive and
     decays slowly -- unlike fact 1)
  3. Fat tails (excess kurtosis > 0 -- large moves are more common than a
     Gaussian would predict)
  4. Leverage effect (negative correlation between a return and *future*
     volatility -- down moves increase near-term volatility more than up
     moves)
  5. Aggregational Gaussianity (the return distribution looks more
     Gaussian -- lower excess kurtosis -- at coarser time scales)

Before trusting this module to judge any generator, diagnostics/ must
confirm compute_stylized_facts reproduces these signs on REAL market
data -- this is this initiative's own trust gate, the same role Rung 1's
Hawkes replication plays for the predictive ladder. If real data doesn't
show these signs through this code, the code has a bug, not the market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (biased-but-consistent
    estimator: covariance at each lag divided by the full-sample variance,
    the same convention statsmodels' acf() uses).
    """
    x = np.asarray(x, dtype=float)
    if len(x) <= max_lag:
        raise ValueError(f"need more than max_lag={max_lag} observations, got {len(x)}")
    centered = x - x.mean()
    denom = np.sum(centered**2)
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(centered[:-lag] * centered[lag:]) / denom for lag in range(1, max_lag + 1)])


def raw_return_acf_curve(returns: np.ndarray, max_lag: int) -> np.ndarray:
    """Fact 1: should be ~0 at every lag for real returns."""
    return acf(returns, max_lag)


def volatility_clustering_curve(returns: np.ndarray, max_lag: int) -> np.ndarray:
    """Fact 2: ACF of |returns| -- should be positive and decay slowly,
    unlike raw_return_acf_curve.
    """
    return acf(np.abs(np.asarray(returns, dtype=float)), max_lag)


def excess_kurtosis(returns: np.ndarray) -> float:
    """Fact 3: Fisher excess kurtosis (0 for a Gaussian). Real returns are
    reliably > 0 (leptokurtic / fat-tailed).
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 4:
        raise ValueError("need at least 4 observations for a kurtosis estimate")
    return float(kurtosis(returns, fisher=True))


def leverage_effect_curve(returns: np.ndarray, max_lag: int) -> np.ndarray:
    """Fact 4: corr(r_t, |r_{t+k}|) for k=1..max_lag -- should be negative
    for real equities (a down move raises near-term volatility more than
    an equal-sized up move does).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) <= max_lag:
        raise ValueError(f"need more than max_lag={max_lag} observations, got {len(r)}")
    r_c = r - r.mean()
    abs_r_c = np.abs(r) - np.abs(r).mean()
    denom = np.sqrt(np.sum(r_c**2) * np.sum(abs_r_c**2))
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(r_c[:-lag] * abs_r_c[lag:]) / denom for lag in range(1, max_lag + 1)])


def aggregational_gaussianity_curve(returns: np.ndarray, scales: tuple[int, ...]) -> np.ndarray:
    """Fact 5: excess kurtosis of returns aggregated (non-overlapping sum)
    over each scale in `scales` -- should decline toward 0 as scale grows
    (the return distribution looks more Gaussian at coarser horizons).
    """
    r = np.asarray(returns, dtype=float)
    out = np.empty(len(scales))
    for i, s in enumerate(scales):
        n_full = len(r) // s
        if n_full < 4:
            raise ValueError(f"scale={s} leaves only {n_full} aggregated points, need >= 4")
        aggregated = r[: n_full * s].reshape(n_full, s).sum(axis=1)
        out[i] = float(kurtosis(aggregated, fisher=True))
    return out


@dataclass
class StylizedFactsSummary:
    raw_return_acf: np.ndarray
    volatility_clustering_acf: np.ndarray
    excess_kurtosis: float
    leverage_curve: np.ndarray
    aggregational_kurtosis: np.ndarray


FACT_NAMES = (
    "raw_return_acf",
    "volatility_clustering_acf",
    "excess_kurtosis",
    "leverage_curve",
    "aggregational_kurtosis",
)


def compute_stylized_facts(
    returns: np.ndarray,
    max_lag: int = 50,
    leverage_max_lag: int | None = None,
    agg_scales: tuple[int, ...] = (1, 5, 15, 30),
) -> StylizedFactsSummary:
    leverage_max_lag = leverage_max_lag if leverage_max_lag is not None else max_lag
    return StylizedFactsSummary(
        raw_return_acf=raw_return_acf_curve(returns, max_lag),
        volatility_clustering_acf=volatility_clustering_curve(returns, max_lag),
        excess_kurtosis=excess_kurtosis(returns),
        leverage_curve=leverage_effect_curve(returns, leverage_max_lag),
        aggregational_kurtosis=aggregational_gaussianity_curve(returns, agg_scales),
    )


def fact_distance(a: StylizedFactsSummary, b: StylizedFactsSummary, fact_name: str) -> float:
    """Mean absolute difference over the lag/scale grid for curve-valued
    facts; plain absolute difference for the scalar excess_kurtosis.
    """
    if fact_name not in FACT_NAMES:
        raise ValueError(f"unknown fact_name {fact_name!r}, expected one of {FACT_NAMES}")
    va, vb = getattr(a, fact_name), getattr(b, fact_name)
    if fact_name == "excess_kurtosis":
        return abs(float(va) - float(vb))
    return float(np.mean(np.abs(np.asarray(va) - np.asarray(vb))))
