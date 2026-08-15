"""Quadratic Hawkes: a self-excitation model whose intensity depends on
the SIGN pattern of past moves, not just their magnitude/count -- the one
mechanism every model built earlier in this research extension
structurally lacks. events/hawkes.py, research/multi_kernel_hawkes.py,
and research/cox_hawkes.py all have intensity that only ever goes UP
after an event, regardless of which direction that event moved the
price. That's exactly why Tier 3's ablation found those models weakest
on the leverage effect (past NEGATIVE returns predict future volatility
more than past positive returns of the same size) -- there's no term in
any of those models capable of telling the two apart.

Intensity: lambda(t) = lambda0 + kappa*L1(t)^2 + alpha*L2(t)
    L1(t) = sum_{t_i<t} m_i * exp(-beta_leverage*(t-t_i))   -- SIGNED linear
            feedback of past marks (e.g. signed return z-scores). A run of
            same-signed marks (a trend) makes |L1| grow; alternating signs
            (chop) keep it near zero -- squaring then converts "how much of
            a trend" into "how much extra intensity," independent of the
            trend's direction. This is what creates an asymmetric response
            to past sign, i.e. the leverage effect, and is loosely
            inspired by the mechanism in Blanc, Donier & Bouchaud (2017)'s
            quadratic Hawkes model for the leverage/Zumbach effects -- NOT
            a literal reproduction of their exact parameterization or
            calibration method (this project doesn't have that paper's
            full equations on hand), but an independently-derived model
            built to capture the same qualitative "squared linear feedback
            of signed returns" idea in a form this project can fit and
            simulate exactly, the same spirit as
            research/multi_kernel_hawkes.py's honest "approximates, doesn't
            reproduce" relationship to Hardiman/Bercot/Bouchaud.
    L2(t) = sum_{t_i<t} exp(-beta*(t-t_i))   -- the ordinary, sign-blind
            self-excitation term every earlier model in this project
            already has (identical in form to events/hawkes.py's).

beta_leverage and beta are both FIXED (not fitted), same discipline as
every other extension in this project (research/multi_kernel_hawkes.py,
research/cox_hawkes.py) -- jointly fitting decay rates and amplitudes is
poorly identified, worse here with two decay rates instead of one.
lambda0>0, kappa>=0, alpha>=0 (both feedback terms must stay non-negative
contributions so the intensity can't go negative; L1^2 is always >=0 so
kappa's sign is what would need to flip to make the term negative, which
this model doesn't allow -- the ASYMMETRY comes from squaring a SIGNED
quantity, not from letting the coefficient itself go negative).

Does NOT modify events/hawkes.py, research/multi_kernel_hawkes.py, or
research/cox_hawkes.py -- all separately trust-gated -- this is a fourth,
independent extension living alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class QuadraticHawkesFitResult:
    lambda0: float
    kappa: float  # quadratic/leverage coefficient
    alpha: float  # standard linear self-excitation coefficient
    beta_leverage: float  # decay rate of the signed L1 kernel -- fixed, not fitted
    beta: float  # decay rate of the standard L2 kernel -- fixed, not fitted
    loglik: float
    converged: bool
    n_events: int

    @property
    def branching_ratio(self) -> float:
        """Only the standard linear (L2) component has a simple
        Poisson-offspring branching interpretation -- the quadratic term
        doesn't (offspring counts from a squared-signed-feedback term
        aren't simply Poisson(kappa/beta_leverage) the way linear Hawkes
        offspring are), so this deliberately reports only the L2 half,
        not a combined "total branching ratio" that would be misleading.
        """
        return self.alpha / self.beta


def _recursive_sum(event_times: np.ndarray, beta: float) -> np.ndarray:
    """Same O(N) Ozaki recursion as events/hawkes.py -- reimplemented
    locally rather than importing a private function from another module.
    """
    n = len(event_times)
    r = np.zeros(n)
    for i in range(1, n):
        dt = event_times[i] - event_times[i - 1]
        r[i] = np.exp(-beta * dt) * (1.0 + r[i - 1])
    return r


def _recursive_signed_sum(event_times: np.ndarray, marks: np.ndarray, beta: float) -> np.ndarray:
    """R(i) = sum_{j<i} m_j * exp(-beta*(t_i-t_j)) -- the SIGNED
    generalization of _recursive_sum: instead of every past event
    contributing a flat +1 to the running sum, each contributes its own
    (signed, real-valued) mark. R(i) = exp(-beta*dt) * (m_{i-1} + R(i-1)),
    R(0) = 0 -- identical recursion structure, just replacing the "+1" of
    the unsigned version with "+m_{i-1})".
    """
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
    lambda0, kappa, alpha = params
    if lambda0 <= 0 or kappa < 0 or alpha < 0:
        return np.inf

    r1 = _recursive_signed_sum(event_times, marks, beta_leverage)  # L1(t_i^-)
    r2 = _recursive_sum(event_times, beta)  # L2(t_i^-)
    intensities = lambda0 + kappa * r1**2 + alpha * r2
    if np.any(intensities <= 0) or not np.all(np.isfinite(intensities)):
        return np.inf
    sum_log_intensity = np.sum(np.log(intensities))

    # Between consecutive events, L1(t) = M_i * exp(-beta_leverage*(t-t_i))
    # where M_i = L1(t_i^+) = r1[i] + marks[i] is L1's value right after
    # event i fires -- a fixed real number decaying purely exponentially
    # until the next event, so integral of L1(t)^2 over each inter-event
    # gap has an exact closed form: M_i^2 * (1-exp(-2*beta_leverage*dt)) /
    # (2*beta_leverage). Same closed-form-compensator discipline as every
    # other Hawkes-family fitter in this project.
    m = r1 + marks
    dt = np.diff(event_times)
    dt_full = np.concatenate([dt, [T - event_times[-1]]])
    quad_compensator = kappa * np.sum(m**2 * (1.0 - np.exp(-2.0 * beta_leverage * dt_full)) / (2.0 * beta_leverage))

    linear_compensator = (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - event_times)))
    compensator = lambda0 * T + quad_compensator + linear_compensator

    return -(sum_log_intensity - compensator)


def fit_quadratic_hawkes(
    event_times: np.ndarray,
    marks: np.ndarray,
    beta_leverage: float,
    beta: float,
    kappa0_grid: tuple[float, ...] = (0.0, 0.02, 0.1, 0.3, 0.7, 1.5, 3.0),
    alpha0_grid: tuple[float, ...] = (0.02, 0.1, 0.3, 0.7, 1.5, 3.0),
    lambda0_0: float | None = None,
    T: float | None = None,
) -> QuadraticHawkesFitResult:
    """MLE fit of (lambda0, kappa, alpha) via L-BFGS-B, beta_leverage and
    beta both fixed. Multistart over both kappa0 and alpha0 (kappa0
    includes 0.0 deliberately -- "no leverage effect" is a real, live
    possibility a priori, not something to rule out by construction of
    the starting grid), same converged-preferred selection rule as
    events/hawkes.py::fit_hawkes_exponential_multistart.
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

    bounds = [(1e-12, None), (0.0, None), (1e-12, None)]
    fits = []
    for k0 in kappa0_grid:
        for a0 in alpha0_grid:
            x0 = np.array([lambda0_0, k0, a0])
            result = minimize(
                _neg_log_likelihood,
                x0,
                args=(event_times, marks, beta_leverage, beta, T),
                method="L-BFGS-B",
                bounds=bounds,
            )
            fits.append(
                QuadraticHawkesFitResult(
                    lambda0=float(result.x[0]),
                    kappa=float(result.x[1]),
                    alpha=float(result.x[2]),
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


def stability_heuristic(kappa: float, alpha: float, beta_leverage: float, beta: float, marks_pool: np.ndarray) -> float:
    """Approximate, NOT rigorously proven, generalization of the standard
    linear-Hawkes branching ratio (alpha/beta < 1) to this quadratic
    model. Unlike the linear case -- where alpha/beta < 1 is an exact,
    well-known stationarity condition -- a squared self-referential
    feedback term doesn't have an equally simple guarantee, and this
    project doesn't have a derivation of the exact one (the literature
    this model is loosely inspired by almost certainly has a more careful
    argument for it). Derived from a "diagonal-only" mean-field
    approximation: E[L1(t)^2] ~ E[event rate]*E[mark^2]/(2*beta_leverage),
    ignoring cross-terms between different events' marks -- giving an
    "effective branching ratio" of kappa*E[mark^2]/(2*beta_leverage) +
    alpha/beta, finite/stable when < 1.

    CONFIRMED EMPIRICALLY TO UNDERESTIMATE real risk, sometimes by 3-5x,
    for two specific cases the diagonal-only approximation misses: (1)
    non-zero-mean or highly-correlated mark distributions, where the
    ignored cross-terms are NOT negligible (roughly
    E[mark]^2 * L2(t)^2 in a rough sense -- itself a self-reinforcing,
    quadratic-in-the-mean-field-intensity term, not a small correction);
    (2) degenerate marks_pool (e.g. a single repeated value), where
    L1(t) becomes numerically IDENTICAL to L2(t) rather than an
    independent-ish fluctuating quantity, making the kappa*L1^2 term
    behave like kappa*L2^2 -- much more explosive than the diagonal
    approximation assumes, with no cancellation to temper it. Both
    failure modes were found the hard way building this module's own
    test suite (values reported "safe" at heuristic~0.4 by this formula
    hit the max_events cap in practice; the real boundary for those
    specific cases was closer to heuristic~0.15-0.2). Treat this as a
    rough, non-degenerate-marks-only signal, not a certificate -- always
    confirm any new parameter combination directly with a modest
    max_events cap before trusting it for a long run, especially with
    skewed or low-diversity mark distributions.
    """
    marks_pool = np.asarray(marks_pool, dtype=float)
    mean_sq_mark = float(np.mean(marks_pool**2)) if len(marks_pool) > 0 else 0.0
    return kappa * mean_sq_mark / (2.0 * beta_leverage) + alpha / beta


def simulate_quadratic_hawkes(
    lambda0: float,
    kappa: float,
    alpha: float,
    beta_leverage: float,
    beta: float,
    T: float,
    marks_pool: np.ndarray,
    seed: int | None = None,
    max_events: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Ogata (1981) thinning simulation -- the quadratic term makes
    this process NOT expressible via the cluster/branching representation
    events/hawkes.py's simulate_hawkes uses (offspring counts from a
    squared-feedback term aren't simply Poisson-distributed the way
    linear Hawkes offspring are), so thinning/rejection sampling is the
    correct tool here, not a shortcut.

    Efficient AND exact (not just a valid upper bound) because of a
    property specific to this intensity form: kappa>=0 and alpha>=0 means
    L1(t)^2 and L2(t) are both purely-decaying (monotonically
    non-increasing) between consecutive events -- no new marks arrive to
    push them back up until the next event fires. So lambda(t) is itself
    monotonically non-increasing between events, meaning its value at the
    CURRENT time is always a tight upper bound for the rest of that
    inter-event stretch, not a loose global bound. Every rejected
    candidate still advances the clock and decays the state exactly, so
    the next bound is exactly the intensity at the new current time --
    no wasted margin beyond the randomness inherent to thinning itself.

    `marks_pool` is resampled with replacement for each new synthetic
    event -- an empirical bootstrap of the real mark distribution rather
    than assuming a parametric form (e.g. symmetric +/-1), matching this
    project's calibrate-from-real-data discipline elsewhere
    (generators/hawkes_jump_diffusion.py::calibrate_jump_and_background_std).

    `max_events` is a mandatory defensive cap, checked every accepted
    event, not an afterthought: found the hard way while building this
    module's own test suite that a seemingly modest-looking kappa (e.g.
    1.5 with beta_leverage=0.5) can put the process into a genuinely
    explosive regime that hangs for a very long time before naturally
    reaching T -- unlike a near-critical but SUBcritical linear Hawkes
    (bounded, if high-variance, per events/hawkes.py::simulate_hawkes's
    own docstring), a quadratic feedback term has no equivalent guarantee
    at all once stability_heuristic's approximate threshold is crossed.
    Raises clearly instead of hanging; see stability_heuristic for how to
    sanity-check parameters before a long real-data run.
    """
    marks_pool = np.asarray(marks_pool, dtype=float)
    if len(marks_pool) == 0:
        raise ValueError("marks_pool must be non-empty")
    if beta_leverage <= 0 or beta <= 0:
        raise ValueError("beta_leverage and beta must both be positive")
    if lambda0 <= 0 or kappa < 0 or alpha < 0:
        raise ValueError("lambda0 must be positive; kappa and alpha must be non-negative")

    rng = np.random.default_rng(seed)
    t = 0.0
    l1 = 0.0
    l2 = 0.0
    events: list[float] = []
    marks_out: list[float] = []

    while t < T:
        bound = lambda0 + kappa * l1**2 + alpha * l2
        candidate_gap = rng.exponential(1.0 / bound)
        candidate_t = t + candidate_gap
        if candidate_t >= T:
            break

        decay1 = np.exp(-beta_leverage * candidate_gap)
        decay2 = np.exp(-beta * candidate_gap)
        l1_candidate = l1 * decay1
        l2_candidate = l2 * decay2
        true_intensity = lambda0 + kappa * l1_candidate**2 + alpha * l2_candidate

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
                    f"unstable/explosive parameter regime (stability_heuristic="
                    f"{stability_heuristic(kappa, alpha, beta_leverage, beta, marks_pool):.3f}, want well below 1); "
                    "reduce kappa/alpha or raise max_events deliberately, don't silently proceed"
                )
        else:
            l1 = l1_candidate
            l2 = l2_candidate
            t = candidate_t

    return np.array(events), np.array(marks_out)
