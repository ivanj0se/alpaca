"""From-scratch exponential-kernel Hawkes process: MLE fitting, branching
ratio, and exact simulation. This is the Rung 1 trust gate -- nothing
downstream in the benchmark ladder is trusted until this reproduces a
published result (Filimonov & Sornette 2012, branching ratio ~0.81 on
E-mini S&P 500 tick data) on real SPY data
(tests/replication/test_hawkes_branching_ratio_replication.py).

Intensity: lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))
Branching ratio n = alpha / beta = expected number of direct offspring per
event; n < 1 required for a stationary process. mu, alpha, beta > 0.

Log-likelihood uses the Ozaki (1979) O(N) recursive form rather than the
naive O(N^2) double sum -- see fit_hawkes_exponential's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class HawkesFitResult:
    mu: float
    alpha: float
    beta: float
    loglik: float
    converged: bool
    n_events: int


def branching_ratio(fit_result: HawkesFitResult) -> float:
    return fit_result.alpha / fit_result.beta


def _recursive_sum(event_times: np.ndarray, beta: float) -> np.ndarray:
    """R(i) = sum_{j<i} exp(-beta * (t_i - t_j)), computed in O(N) via the
    Ozaki recursion R(i) = exp(-beta * (t_i - t_{i-1})) * (1 + R(i-1)),
    R(1) = 0.
    """
    n = len(event_times)
    r = np.zeros(n)
    for i in range(1, n):
        dt = event_times[i] - event_times[i - 1]
        r[i] = np.exp(-beta * dt) * (1.0 + r[i - 1])
    return r


def _neg_log_likelihood(params: np.ndarray, event_times: np.ndarray, T: float) -> float:
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return np.inf

    r = _recursive_sum(event_times, beta)
    intensities = mu + alpha * r
    if np.any(intensities <= 0):
        return np.inf

    sum_log_intensity = np.sum(np.log(intensities))
    compensator = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - event_times)))
    return -(sum_log_intensity - compensator)


def fit_hawkes_exponential(
    event_times: np.ndarray,
    mu0: float | None = None,
    alpha0: float = 0.5,
    beta0: float | None = None,
    T: float | None = None,
) -> HawkesFitResult:
    """MLE fit of mu, alpha, beta via L-BFGS-B with positivity bounds.
    `event_times` must be sorted, non-negative seconds-since-first-event
    (see events/price_events.event_times_array). `T` defaults to the last
    event time (a slight underestimate of the true observation window, but
    standard practice absent explicit window bounds).

    `beta0` defaults to 1/median(inter-event gap) rather than a fixed
    constant. A fixed beta0=1.0 (1-second decay) is badly scaled for any
    event stream sparser than ~1/second -- confirmed on real SPY
    minute-bar-derived events (median gap 480s): exp(-beta0 * gap) ~ 3e-209
    at beta0=1.0, meaning d(loglik)/d(alpha) ~ 0 at the starting point.
    The optimizer had no gradient signal there and reported converged=True
    after barely moving from (alpha0, beta0=1.0) -- not a real MLE optimum,
    just a numerically flat region around the starting guess. A
    data-adaptive default puts the optimizer somewhere the likelihood
    surface actually has curvature.
    """
    event_times = np.asarray(event_times, dtype=float)
    n = len(event_times)
    if n < 2:
        raise ValueError("need at least 2 events to fit a Hawkes process")
    if T is None:
        T = float(event_times[-1])

    if mu0 is None:
        mu0 = max(n / T * 0.5, 1e-6)
    if beta0 is None:
        median_gap = float(np.median(np.diff(event_times)))
        beta0 = 1.0 / median_gap if median_gap > 0 else 1.0

    x0 = np.array([mu0, alpha0, beta0])
    bounds = [(1e-10, None), (1e-10, None), (1e-10, None)]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(event_times, T),
        method="L-BFGS-B",
        bounds=bounds,
    )

    mu, alpha, beta = result.x
    return HawkesFitResult(
        mu=float(mu),
        alpha=float(alpha),
        beta=float(beta),
        loglik=float(-result.fun),
        converged=bool(result.success),
        n_events=n,
    )


def fit_hawkes_exponential_multistart(
    event_times: np.ndarray,
    alpha0_grid: tuple[float, ...] = (0.1, 0.5, 0.9, 2.0),
    T: float | None = None,
) -> HawkesFitResult:
    """Refits from several `alpha0` starting points and keeps the one with
    the highest achieved log-likelihood among converged fits.

    Exists because the L-BFGS-B likelihood surface for this model is not
    always well-conditioned enough for a single default start to be
    trustworthy -- confirmed on real SPY tick-level event streams
    (diagnostics/2026-08-11-real-tick-hawkes-replication/findings.md):
    refitting the same event set from different `alpha0` starting points
    landed on branching-ratio estimates ranging from ~0.03 to >1.6 for
    several moderate-density event definitions, each individually
    reporting `converged=True`. The earlier
    diagnostics/2026-08-11-hawkes-optimizer-initialization-bug/ fix (a
    data-adaptive `beta0`) made the *sparse* bar-proxy case robust to this,
    but doesn't guarantee a well-conditioned surface at every event
    density -- this is the general-purpose hardening `fit_hawkes_exponential`
    itself doesn't attempt (it fits once, from whatever `alpha0` the caller
    picked). Falls back to whichever fit is available if none converged.
    """
    fits = [fit_hawkes_exponential(event_times, alpha0=a0, T=T) for a0 in alpha0_grid]
    converged = [f for f in fits if f.converged]
    candidates = converged or fits
    return max(candidates, key=lambda f: f.loglik)


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, seed: int | None = None) -> np.ndarray:
    """Exact simulation via the cluster/branching representation: immigrants
    arrive as a homogeneous Poisson(mu) process on [0, T]; each event
    (immigrant or offspring) spawns Poisson(alpha/beta)-many direct
    offspring, each at parent_time + Exponential(beta). Requires
    alpha/beta < 1 for a finite (stationary) process -- see e.g. Moller &
    Rasmussen (2005) or Laub, Taimre & Pollett's Hawkes process tutorial.
    Returns a sorted array of event times in [0, T].
    """
    if alpha >= beta:
        raise ValueError("alpha must be < beta (branching ratio < 1) for a stationary process")

    rng = np.random.default_rng(seed)
    events: list[float] = []

    n_immigrants = rng.poisson(mu * T)
    queue = list(rng.uniform(0, T, n_immigrants))

    while queue:
        parent_t = queue.pop()
        if parent_t >= T:
            continue
        events.append(parent_t)
        n_children = rng.poisson(alpha / beta)
        if n_children > 0:
            offsets = rng.exponential(1.0 / beta, n_children)
            child_times = parent_t + offsets
            queue.extend(child_times[child_times < T].tolist())

    return np.sort(np.array(events))
