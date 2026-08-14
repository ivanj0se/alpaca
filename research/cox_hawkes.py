"""Hawkes process with a time-varying, covariate-driven baseline instead
of a constant mu: lambda(t) = mu(t) + alpha*sum_{t_i<t} exp(-beta(t-t_i)),
mu(t) = mu0 * exp(gamma * x(t)).

The Cox-Hawkes literature (doubly-stochastic self-exciting processes)
typically makes mu(t) an UNOBSERVED latent stochastic process (a Gaussian
process prior, inferred via MCMC/variational methods -- real machinery,
but abstract: the "exogenous" driver is never actually tied to anything
real-world). This substitutes something real and already-measured
instead: x(t) is this project's own validated RPCA cross-sectional
common factor (rpca/rolling_rpca.py, already shown to beat the linear
factor model on real data), standardized. Because x(t) is OBSERVED (not
latent), this is technically a Hawkes process with an observed
time-varying covariate baseline, not a full latent Cox process -- a
simpler, more tractable model that still operationalizes the same idea:
separate self-excitation (the alpha term) from a genuinely exogenous,
externally-measured driver (the mu(t) term), rather than lumping
everything into one constant mu the way events/hawkes.py's baseline
model does.

Does NOT modify events/hawkes.py or rpca/rolling_rpca.py -- both
trust-gated, validated modules -- this is a separate extension that
calls into rpca/rolling_rpca.py's already-tested decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _recursive_sum(event_times: np.ndarray, beta: float) -> np.ndarray:
    """Same O(N) Ozaki recursion as events/hawkes.py -- reimplemented
    locally (a few lines) rather than importing a private (underscore-
    prefixed) function from another module.
    """
    n = len(event_times)
    r = np.zeros(n)
    for i in range(1, n):
        dt = event_times[i] - event_times[i - 1]
        r[i] = np.exp(-beta * dt) * (1.0 + r[i - 1])
    return r


@dataclass
class CoxHawkesFitResult:
    mu0: float
    gamma: float
    alpha: float
    beta: float  # fixed, not fitted
    loglik: float
    converged: bool
    n_events: int

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta


def _neg_log_likelihood(
    params: np.ndarray,
    event_times: np.ndarray,
    event_covariate: np.ndarray,
    beta: float,
    T: float,
    grid_covariate: np.ndarray,
    grid_durations: np.ndarray,
) -> float:
    mu0, gamma, alpha = params
    if mu0 <= 0 or alpha <= 0:
        return np.inf

    r = _recursive_sum(event_times, beta)
    baseline_at_events = mu0 * np.exp(gamma * event_covariate)
    intensities = baseline_at_events + alpha * r
    if np.any(intensities <= 0) or not np.all(np.isfinite(intensities)):
        return np.inf
    sum_log_intensity = np.sum(np.log(intensities))

    baseline_term = mu0 * np.exp(gamma * grid_covariate)
    if not np.all(np.isfinite(baseline_term)):
        return np.inf
    baseline_compensator = np.sum(baseline_term * grid_durations)
    excitation_compensator = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - event_times)))

    return -(sum_log_intensity - baseline_compensator - excitation_compensator)


def fit_cox_hawkes(
    event_times: np.ndarray,
    event_covariate: np.ndarray,
    beta: float,
    grid_covariate: np.ndarray,
    grid_durations: np.ndarray,
    T: float | None = None,
    alpha0_grid: tuple[float, ...] = (0.02, 0.1, 0.3, 0.7, 1.5, 3.0, 7.0),
    gamma0_grid: tuple[float, ...] = (0.0, 0.5, -0.5, 1.5, -1.5),
    mu0_0: float | None = None,
) -> CoxHawkesFitResult:
    """MLE fit of (mu0, gamma, alpha) via L-BFGS-B, beta fixed (reuse an
    already-validated single-exponential beta rather than refitting the
    excitation timescale jointly -- keeps the model comparison to "does
    adding the covariate-driven baseline help" isolated from "did the
    excitation timescale itself change"). Multistart over both alpha0 and
    gamma0 (gamma can be positive, negative, or ~0 a priori -- no reason
    to assume the sign in advance), same converged-preferred selection
    rule as events/hawkes.py::fit_hawkes_exponential_multistart.

    `grid_covariate`/`grid_durations` describe a piecewise-constant
    approximation of x(t) over [0, T] (e.g. one value per RPCA-decomposition
    timestep, held constant until the next) -- the discretization the
    baseline compensator integral is computed against.
    """
    event_times = np.asarray(event_times, dtype=float)
    event_covariate = np.asarray(event_covariate, dtype=float)
    n = len(event_times)
    if n < 2:
        raise ValueError("need at least 2 events to fit a Hawkes process")
    if len(event_covariate) != n:
        raise ValueError("event_covariate must have one value per event")
    if beta <= 0:
        raise ValueError("beta must be positive")
    grid_covariate = np.asarray(grid_covariate, dtype=float)
    grid_durations = np.asarray(grid_durations, dtype=float)
    if len(grid_covariate) != len(grid_durations):
        raise ValueError("grid_covariate and grid_durations must have the same length")
    if T is None:
        T = float(event_times[-1])
    if mu0_0 is None:
        mu0_0 = max(n / T * 0.5, 1e-6)

    bounds = [(1e-12, None), (None, None), (1e-12, None)]
    fits = []
    for a0 in alpha0_grid:
        for g0 in gamma0_grid:
            x0 = np.array([mu0_0, g0, a0])
            result = minimize(
                _neg_log_likelihood,
                x0,
                args=(event_times, event_covariate, beta, T, grid_covariate, grid_durations),
                method="L-BFGS-B",
                bounds=bounds,
            )
            fits.append(
                CoxHawkesFitResult(
                    mu0=float(result.x[0]),
                    gamma=float(result.x[1]),
                    alpha=float(result.x[2]),
                    beta=beta,
                    loglik=float(-result.fun),
                    converged=bool(result.success),
                    n_events=n,
                )
            )

    converged = [f for f in fits if f.converged]
    candidates = converged or fits
    return max(candidates, key=lambda f: f.loglik)
