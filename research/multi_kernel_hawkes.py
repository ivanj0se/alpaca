"""Sum-of-exponentials (multi-timescale) Hawkes kernel:
lambda(t) = mu + sum_k alpha_k * sum_{t_i<t} exp(-beta_k*(t-t_i)).

events/hawkes.py deliberately uses a single exponential -- the right
starting choice for tractability and for validating against Filimonov &
Sornette's published branching ratio, which it did. But Hardiman, Bercot
& Bouchaud (2013) found real order flow's self-excitation kernel is
actually a power law (two regimes: decay exponent ~-1.15 below ~10^3s,
~-1.45 above), which a single exponential structurally cannot represent
-- it has exactly one relaxation timescale. This approximates the
power-law with a small, fixed grid of exponential timescales (the
standard practical technique, e.g. Bacry/Mastromatteo/Muzy's survey),
fitting only the amplitudes via MLE -- freely fitting both amplitudes
and timescales for K>1 components is poorly identified (the same class
of instability multistart was built to catch in
events/hawkes.py::fit_hawkes_exponential_multistart, worse with more
free parameters), so the timescales are fixed at a data-adaptive
geometric grid instead.

The actual research question this exists to answer: is the already-found
IEX-vs-SIP branching-ratio gap
(diagnostics/2026-08-11-sip-consolidated-tape-check/) concentrated at a
specific timescale (evidence of venue-specific microstructure noise --
e.g. quote-stuffing or internal matching artifacts local to one venue)
or spread proportionally across all of them (evidence of a more uniform,
scale-independent inflation)? This requires having both a single-venue
and a consolidated feed for the same real instrument/period to ask --
this project has both.

Does NOT modify events/hawkes.py, which is trust-gated and validated
against a published replication target -- this is a separate,
exploratory extension living alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class MultiKernelFitResult:
    mu: float
    alphas: np.ndarray  # (K,)
    betas: np.ndarray  # (K,) -- fixed, not fitted
    loglik: float
    converged: bool
    n_events: int

    @property
    def branching_ratio_total(self) -> float:
        return float(np.sum(self.alphas / self.betas))

    @property
    def branching_ratio_per_component(self) -> np.ndarray:
        return self.alphas / self.betas


def default_timescale_grid(event_times: np.ndarray, n_components: int = 3, span_decades: float = 1.0) -> np.ndarray:
    """Data-adaptive geometric grid of beta_k (decay rates), centered on
    1/median(inter-event gap) -- the same data-adaptive scale
    events/hawkes.py::fit_hawkes_exponential already uses as its single
    beta0 default -- spanning `span_decades` orders of magnitude faster
    and slower on either side.
    """
    event_times = np.sort(np.asarray(event_times, dtype=float))
    median_gap = float(np.median(np.diff(event_times)))
    if median_gap <= 0:
        raise ValueError("median inter-event gap must be positive")
    center_beta = 1.0 / median_gap
    exponents = np.linspace(-span_decades, span_decades, n_components)
    return center_beta * 10.0**exponents


def _recursive_sums(event_times: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """(K, N): R_k(i) = sum_{j<i} exp(-beta_k*(t_i-t_j)), the same Ozaki
    recursion events/hawkes.py uses, computed independently per component
    -- each has its own decay rate and thus its own recursion, which is
    why this stays O(N*K), not worse.
    """
    n = len(event_times)
    k = len(betas)
    r = np.zeros((k, n))
    for i in range(1, n):
        dt = event_times[i] - event_times[i - 1]
        r[:, i] = np.exp(-betas * dt) * (1.0 + r[:, i - 1])
    return r


def _neg_log_likelihood(params: np.ndarray, event_times: np.ndarray, betas: np.ndarray, T: float) -> float:
    mu = params[0]
    alphas = params[1:]
    if mu <= 0 or np.any(alphas <= 0):
        return np.inf
    r = _recursive_sums(event_times, betas)  # (K, N)
    intensities = mu + np.sum(alphas[:, None] * r, axis=0)
    if np.any(intensities <= 0):
        return np.inf
    sum_log_intensity = np.sum(np.log(intensities))
    compensator = mu * T + np.sum((alphas / betas)[:, None] * (1.0 - np.exp(-betas[:, None] * (T - event_times))))
    return -(sum_log_intensity - compensator)


def fit_multi_kernel_hawkes(
    event_times: np.ndarray,
    betas: np.ndarray,
    alpha0_grid: tuple[float, ...] = (0.02, 0.05, 0.1, 0.3, 0.7, 1.5, 3.0, 7.0, 15.0),
    mu0: float | None = None,
    T: float | None = None,
) -> MultiKernelFitResult:
    """MLE fit of mu and alpha_1..alpha_K via L-BFGS-B, betas fixed.
    Multistart over alpha0 (same starting value applied uniformly across
    all K components), same selection rule as
    fit_hawkes_exponential_multistart: keep the highest-log-likelihood
    converged fit, fall back to the best available if none converged.
    """
    event_times = np.asarray(event_times, dtype=float)
    n = len(event_times)
    if n < 2:
        raise ValueError("need at least 2 events to fit a Hawkes process")
    betas = np.asarray(betas, dtype=float)
    if np.any(betas <= 0):
        raise ValueError("betas must all be positive")
    if T is None:
        T = float(event_times[-1])
    if mu0 is None:
        mu0 = max(n / T * 0.5, 1e-6)

    bounds = [(1e-12, None)] * (1 + len(betas))
    fits = []
    for a0 in alpha0_grid:
        x0 = np.concatenate([[mu0], np.full(len(betas), a0)])
        result = minimize(_neg_log_likelihood, x0, args=(event_times, betas, T), method="L-BFGS-B", bounds=bounds)
        fits.append(
            MultiKernelFitResult(
                mu=float(result.x[0]),
                alphas=result.x[1:].copy(),
                betas=betas,
                loglik=float(-result.fun),
                converged=bool(result.success),
                n_events=n,
            )
        )

    converged = [f for f in fits if f.converged]
    candidates = converged or fits
    return max(candidates, key=lambda f: f.loglik)
