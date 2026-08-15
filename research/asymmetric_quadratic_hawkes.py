"""Sign-asymmetric quadratic Hawkes: the genuine leverage-effect
extension research/quadratic_hawkes.py's own real-data diagnostics entry
(diagnostics/2026-08-14-quadratic-hawkes-real-fit/) flagged as missing.

That earlier model's intensity included a kappa*L1(t)^2 term -- squaring
a signed linear feedback of past marks. Squaring erases sign: a strong
up-trend and a strong down-trend of the same magnitude produce IDENTICAL
extra intensity under that model, no matter how large or significant
kappa turns out to be. The real, classic leverage effect (negative past
returns raise future volatility MORE than positive past returns of the
same size) is a sign-ASYMMETRIC relationship that a purely-quadratic term
cannot represent by construction, regardless of its coefficient.

This model breaks that symmetry the minimal way: two separate
coefficients instead of one, selected by L1's sign at the moment it
matters --
    lambda(t) = lambda0 + kappa(t)*L1(t)^2 + alpha*L2(t)
    kappa(t) = kappa_minus if L1(t) < 0 else kappa_plus

L1(t) never changes sign between events (it's mark_i * exp(-beta*(t-t_i))
for a FIXED, already-realized mark_i, decaying purely toward zero -- sign
is set once, at the moment the mark arrives, and only magnitude decays
after that), so kappa(t) is piecewise-constant, taking exactly one value
per inter-event interval -- the compensator integral stays exactly as
closed-form/tractable as the symmetric model's, just with the
appropriate kappa selected per interval rather than one shared value.

The actual test this model exists to run: is kappa_minus > kappa_plus,
and is that difference statistically significant relative to the
symmetric model (kappa_minus == kappa_plus, i.e.
research/quadratic_hawkes.py) already fit on the same data? That
comparison is the real leverage-effect test; a nonzero shared kappa
alone (already found) is not.

Does NOT modify research/quadratic_hawkes.py or events/hawkes.py, both
separately tested/trust-gated -- reimplements the shared recursion
helpers locally (a few lines) rather than importing private functions
cross-module, same discipline as every other research/ extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class AsymmetricQuadraticHawkesFitResult:
    lambda0: float
    kappa_minus: float  # applies when L1(t) < 0 -- a recent net DOWN-trend; the classic leverage term
    kappa_plus: float  # applies when L1(t) >= 0 -- a recent net UP-trend
    alpha: float
    beta_leverage: float  # fixed, not fitted
    beta: float  # fixed, not fitted
    loglik: float
    converged: bool
    n_events: int

    @property
    def branching_ratio(self) -> float:
        """Only the standard linear (L2) component has a simple
        Poisson-offspring branching interpretation -- see
        research/quadratic_hawkes.py::QuadraticHawkesFitResult.branching_ratio
        for why the quadratic terms don't.
        """
        return self.alpha / self.beta

    @property
    def leverage_asymmetry(self) -> float:
        """kappa_minus - kappa_plus. Positive = the classic leverage
        direction (down-trends trigger more future intensity than
        up-trends of the same magnitude). Zero = no asymmetry (the model
        collapses to research/quadratic_hawkes.py's symmetric one).
        Negative = the OPPOSITE of the classic effect -- a real,
        reportable possibility, not assumed away.
        """
        return self.kappa_minus - self.kappa_plus


def _recursive_sum(event_times: np.ndarray, beta: float) -> np.ndarray:
    n = len(event_times)
    r = np.zeros(n)
    for i in range(1, n):
        dt = event_times[i] - event_times[i - 1]
        r[i] = np.exp(-beta * dt) * (1.0 + r[i - 1])
    return r


def _recursive_signed_sum(event_times: np.ndarray, marks: np.ndarray, beta: float) -> np.ndarray:
    n = len(event_times)
    r = np.zeros(n)
    for i in range(1, n):
        dt = event_times[i] - event_times[i - 1]
        r[i] = np.exp(-beta * dt) * (marks[i - 1] + r[i - 1])
    return r


def _neg_log_likelihood(
    params: np.ndarray,
    event_times: np.ndarray,
    marks: np.ndarray,
    beta_leverage: float,
    beta: float,
    T: float,
) -> float:
    lambda0, kappa_minus, kappa_plus, alpha = params
    if lambda0 <= 0 or kappa_minus < 0 or kappa_plus < 0 or alpha < 0:
        return np.inf

    r1 = _recursive_signed_sum(event_times, marks, beta_leverage)  # L1(t_i^-)
    r2 = _recursive_sum(event_times, beta)  # L2(t_i^-)
    kappa_at_event = np.where(r1 < 0, kappa_minus, kappa_plus)
    intensities = lambda0 + kappa_at_event * r1**2 + alpha * r2
    if np.any(intensities <= 0) or not np.all(np.isfinite(intensities)):
        return np.inf
    sum_log_intensity = np.sum(np.log(intensities))

    # m[i] = L1(t_i^+), the value right after event i fires -- fixed sign
    # for the whole interval that follows, so kappa is selected once per
    # interval, same closed-form integral as the symmetric model's.
    m = r1 + marks
    dt = np.diff(event_times)
    dt_full = np.concatenate([dt, [T - event_times[-1]]])
    kappa_at_interval = np.where(m < 0, kappa_minus, kappa_plus)
    quad_compensator = np.sum(kappa_at_interval * m**2 * (1.0 - np.exp(-2.0 * beta_leverage * dt_full)) / (2.0 * beta_leverage))

    linear_compensator = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - event_times)))
    compensator = lambda0 * T + quad_compensator + linear_compensator

    return -(sum_log_intensity - compensator)


def fit_asymmetric_quadratic_hawkes(
    event_times: np.ndarray,
    marks: np.ndarray,
    beta_leverage: float,
    beta: float,
    kappa0_grid: tuple[float, ...] = (0.0, 0.02, 0.1, 0.3),
    alpha0_grid: tuple[float, ...] = (0.1, 0.5, 1.5),
    lambda0_0: float | None = None,
    T: float | None = None,
) -> AsymmetricQuadraticHawkesFitResult:
    """MLE fit of (lambda0, kappa_minus, kappa_plus, alpha) via L-BFGS-B,
    beta_leverage and beta both fixed. Multistart over kappa_minus0,
    kappa_plus0 (same grid for both, independently) and alpha0 -- kept
    modest (4x4x3=48 combinations) since this has one more free parameter
    than research/quadratic_hawkes.py's fit_quadratic_hawkes; same
    converged-preferred selection rule as every other Hawkes-family
    fitter in this project.
    """
    event_times = np.asarray(event_times, dtype=float)
    marks = np.asarray(marks, dtype=float)
    n = len(event_times)
    if n < 2:
        raise ValueError("need at least 2 events to fit a Hawkes process")
    if len(marks) != n:
        raise ValueError("marks must have one value per event")
    if beta_leverage <= 0:
        raise ValueError("beta_leverage must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if T is None:
        T = float(event_times[-1])
    if lambda0_0 is None:
        lambda0_0 = max(n / T * 0.5, 1e-6)

    bounds = [(1e-12, None), (0.0, None), (0.0, None), (1e-12, None)]
    fits = []
    for km0 in kappa0_grid:
        for kp0 in kappa0_grid:
            for a0 in alpha0_grid:
                x0 = np.array([lambda0_0, km0, kp0, a0])
                result = minimize(
                    _neg_log_likelihood,
                    x0,
                    args=(event_times, marks, beta_leverage, beta, T),
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                fits.append(
                    AsymmetricQuadraticHawkesFitResult(
                        lambda0=float(result.x[0]),
                        kappa_minus=float(result.x[1]),
                        kappa_plus=float(result.x[2]),
                        alpha=float(result.x[3]),
                        beta_leverage=beta_leverage,
                        beta=beta,
                        loglik=float(-result.fun),
                        converged=bool(result.success),
                        n_events=n,
                    )
                )

    converged = [f for f in fits if f.converged]
    candidates = converged or fits
    return max(candidates, key=lambda f: f.loglik)


def simulate_asymmetric_quadratic_hawkes(
    lambda0: float,
    kappa_minus: float,
    kappa_plus: float,
    alpha: float,
    beta_leverage: float,
    beta: float,
    T: float,
    marks_pool: np.ndarray,
    seed: int | None = None,
    max_events: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Ogata thinning, same structure and same efficiency argument
    as research/quadratic_hawkes.py::simulate_quadratic_hawkes: since
    kappa_minus,kappa_plus,alpha>=0 and L1(t) cannot change sign between
    events (decay only shrinks its magnitude), the piecewise-selected
    kappa(t) is fixed within each inter-event interval and lambda(t)
    remains monotonically non-increasing between events -- the current
    intensity is still a tight, valid upper bound, not a loose one.

    `max_events` is mandatory here too, for the same reason as the
    symmetric model: no simple stability guarantee exists for a squared
    feedback term, doubly true here since whichever of kappa_minus/
    kappa_plus is larger can push the process into an explosive regime
    the OTHER branch alone would not.
    """
    marks_pool = np.asarray(marks_pool, dtype=float)
    if len(marks_pool) == 0:
        raise ValueError("marks_pool must be non-empty")
    if beta_leverage <= 0 or beta <= 0:
        raise ValueError("beta_leverage and beta must both be positive")
    if lambda0 <= 0 or kappa_minus < 0 or kappa_plus < 0 or alpha < 0:
        raise ValueError("lambda0 must be positive; kappa_minus, kappa_plus, and alpha must be non-negative")

    rng = np.random.default_rng(seed)
    t = 0.0
    l1 = 0.0
    l2 = 0.0
    events: list[float] = []
    marks_out: list[float] = []

    while t < T:
        kappa_now = kappa_minus if l1 < 0 else kappa_plus
        bound = lambda0 + kappa_now * l1**2 + alpha * l2
        candidate_gap = rng.exponential(1.0 / bound)
        candidate_t = t + candidate_gap
        if candidate_t >= T:
            break

        decay1 = np.exp(-beta_leverage * candidate_gap)
        decay2 = np.exp(-beta * candidate_gap)
        l1_candidate = l1 * decay1  # same sign as l1 (or 0), decay never flips sign
        l2_candidate = l2 * decay2
        kappa_candidate = kappa_minus if l1_candidate < 0 else kappa_plus
        true_intensity = lambda0 + kappa_candidate * l1_candidate**2 + alpha * l2_candidate

        if rng.uniform(0.0, 1.0) <= true_intensity / bound:
            mark = float(rng.choice(marks_pool))
            l1 = l1_candidate + mark
            l2 = l2_candidate + 1.0
            events.append(candidate_t)
            marks_out.append(mark)
            t = candidate_t
            if len(events) > max_events:
                raise RuntimeError(
                    f"simulated more than max_events={max_events} events before reaching T={T} -- likely an "
                    "unstable/explosive parameter regime; reduce kappa_minus/kappa_plus/alpha or raise "
                    "max_events deliberately, don't silently proceed"
                )
        else:
            l1 = l1_candidate
            l2 = l2_candidate
            t = candidate_t

    return np.array(events), np.array(marks_out)
