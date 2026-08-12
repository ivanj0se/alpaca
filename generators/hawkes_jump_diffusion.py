"""Hawkes-branching-ratio ablation generator (Rung G2): a direct,
falsifiable test of the project's own core research question. Rather than
reporting the fitted branching ratio as a bare number, simulate synthetic
price paths WITH the real fitted self-excitation and WITHOUT it (a pure
Poisson control, same expected event rate), then let
benchmark/stylized_facts.py + benchmark/conformal.py say which one
actually looks more like a real market.

Reuses events/hawkes.py's already-tested, trust-gated
fit_hawkes_exponential_multistart and simulate_hawkes rather than
reimplementing or modifying them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from events.hawkes import HawkesFitResult, fit_hawkes_exponential_multistart, simulate_hawkes
from events.price_events import event_times_array, tick_events_from_recorder
from features.returns import build_feature_frame, session_boundary_mask
from generators.path import GeneratedPath
from ingest.storage import read_bars, read_ticks


def session_elapsed_seconds(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Seconds elapsed since the first timestamp, with each
    session-boundary gap (overnight/weekend/holiday -- see
    features.returns.session_boundary_mask) replaced by the median
    within-session inter-event gap rather than either its real huge value
    (which would dilute a fitted Hawkes mu with market-closed dead time,
    the same class of bug session_boundary_mask fixed for returns -- see
    diagnostics/2026-08-11-session-boundary-returns/) or zero (which would
    fake a burst of simultaneous events exactly at every session
    boundary). This is a deliberate approximation, not exact session-clock
    arithmetic -- documented as such in the generator's own diagnostics
    entry.
    """
    ts = pd.DatetimeIndex(timestamps).sort_values()
    if len(ts) < 2:
        return np.zeros(len(ts))
    raw_seconds = (ts - ts[0]).total_seconds().to_numpy()
    gaps = np.diff(raw_seconds)
    boundary = session_boundary_mask(ts).to_numpy()[1:]
    if boundary.any() and (~boundary).any():
        typical_gap = float(np.median(gaps[~boundary]))
        gaps = np.where(boundary, typical_gap, gaps)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def fit_real_hawkes_params(
    data_dir: Path,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    session_compress: bool = True,
) -> HawkesFitResult:
    """Live-refits mu/alpha/beta against real backfilled tick data rather
    than hardcoding a stale snapshot from diagnostics/ -- the always-on
    recorder (ingest/tick_recorder.py) keeps accumulating real ticks, so a
    fresh fit is both more correct and more current than any number baked
    into this file. Mirrors the exact methodology already validated in
    diagnostics/2026-08-11-real-tick-hawkes-replication/ and
    diagnostics/2026-08-11-sip-consolidated-tape-check/ (same
    sigma_threshold event definition, same fit_hawkes_exponential_multistart).
    """
    ticks = read_ticks(data_dir, tickers=[ticker])
    if ticks.empty:
        raise ValueError(f"no real tick data for {ticker} in {data_dir}")
    events = tick_events_from_recorder(ticks, sigma_threshold=sigma_threshold)
    times = pd.to_datetime(events[events["ticker"] == ticker]["timestamp"]).sort_values()
    if len(times) < 20:
        raise ValueError(f"only {len(times)} events for {ticker} -- not enough to fit a Hawkes process")

    if session_compress:
        seconds = session_elapsed_seconds(pd.DatetimeIndex(times))
    else:
        seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()

    return fit_hawkes_exponential_multistart(seconds)


@dataclass
class HawkesAblationArm:
    label: str  # "control" (branching_ratio=0) or "treatment" (real fitted self-excitation)
    mu: float
    alpha: float
    beta: float


def build_ablation_arms(fit: HawkesFitResult) -> tuple[HawkesAblationArm, HawkesAblationArm]:
    """Control (pure Poisson, branching_ratio=0) and treatment (real
    fitted self-excitation) arms, with E[total events] = mu/(1-branching_ratio)
    held EQUAL between them. This is the single most important
    correctness point in this module: using the same mu for both arms
    would give the treatment arm ~1/(1-branching_ratio) times more total
    events than the control (at branching_ratio=0.997, ~300x), confounding
    "self-excitation" with "event density" -- the resulting stylized-facts
    gap would then be attributable to either, not specifically to
    temporal clustering. Scaling mu_control up to match keeps every other
    factor equal; only the *temporal pattern* (clustered vs. perfectly
    homogeneous) differs between arms.
    """
    branching_ratio = fit.alpha / fit.beta
    if not 0 <= branching_ratio < 1:
        raise ValueError(f"branching_ratio={branching_ratio} outside [0, 1) -- refit did not converge to a stationary process")
    mu_control = fit.mu / (1.0 - branching_ratio)
    control = HawkesAblationArm(label="control", mu=mu_control, alpha=0.0, beta=fit.beta)
    treatment = HawkesAblationArm(label="treatment", mu=fit.mu, alpha=fit.alpha, beta=fit.beta)
    return control, treatment


def simulate_hawkes_bounded(mu: float, alpha: float, beta: float, T: float, max_events: int, seed: int | None = None) -> np.ndarray:
    """Thin wrapper around events.hawkes.simulate_hawkes with a defensive
    event-count cap. simulate_hawkes is provably bounded by T (every event
    -- immigrant or offspring -- must be < T to ever be appended, so the
    loop always terminates) -- it cannot hang forever. But a near-critical
    branching ratio has astronomically higher cluster-size VARIANCE than
    its mean (Galton-Watson: variance ~ n/(1-n)^3, blowing up as
    branching_ratio n -> 1), so an unlucky draw can still produce an
    unreasonably large event count. Raise clearly rather than silently
    proceeding with a run that ties up unbounded memory/time. Does not
    modify events/hawkes.py itself -- that module is trust-gated.
    """
    events = simulate_hawkes(mu, alpha, beta, T, seed=seed)
    if len(events) > max_events:
        raise RuntimeError(
            f"simulated {len(events)} events, exceeding max_events={max_events} "
            f"(mu={mu:.6g}, alpha={alpha:.6g}, beta={beta:.6g}, T={T:.1f}) -- likely a "
            "near-critical branching ratio producing a runaway cluster; reduce T or raise "
            "max_events deliberately, don't silently proceed"
        )
    return events


def calibrate_jump_and_background_std(returns: np.ndarray, sigma_threshold: float) -> tuple[float, float]:
    """background_std/jump_std calibrated from disjoint subsets of real
    minute-bar returns (below vs. at/above sigma_threshold, the same
    threshold used to define Hawkes events), so the background diffusion
    and event jumps never double-count the same real observations.
    """
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1)
    if std == 0:
        raise ValueError("returns have zero variance -- cannot calibrate a threshold split")
    z = np.abs(returns) / std
    below = returns[z < sigma_threshold]
    at_or_above = returns[z >= sigma_threshold]
    if len(below) < 2 or len(at_or_above) < 2:
        raise ValueError("threshold split leaves too few observations on one side to calibrate std from")
    return float(below.std(ddof=1)), float(at_or_above.std(ddof=1))


def hawkes_events_to_returns(
    event_seconds: np.ndarray, T: float, bar_seconds: float, background_std: float, jump_std: float, seed: int | None = None
) -> np.ndarray:
    """Converts a Hawkes event-time point process into a synthetic
    minute-bar log-return series: background Gaussian diffusion at every
    bar (calibrated to real sub-threshold returns) plus an extra
    random-sign jump (calibrated to real at/above-threshold returns) at
    bars containing at least one event. Needed because a jump-only
    construction leaves long exactly-flat stretches between events (~29s
    average real gap vs. 60s bars) that would itself distort
    volatility/kurtosis measurements.
    """
    rng = np.random.default_rng(seed)
    n_bars = max(1, int(np.ceil(T / bar_seconds)))
    returns = rng.normal(0.0, background_std, n_bars)
    if len(event_seconds) > 0:
        bar_idx = np.clip((np.asarray(event_seconds) / bar_seconds).astype(int), 0, n_bars - 1)
        signs = rng.choice([-1.0, 1.0], size=len(bar_idx))
        magnitudes = np.abs(rng.normal(0.0, jump_std, size=len(bar_idx)))
        np.add.at(returns, bar_idx, signs * magnitudes)
    return returns


def generate_ablation_paths(
    data_dir: Path,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    T_days: float = 5.0,
    bar_seconds: float = 60.0,
    n_sims: int = 25,
    max_events: int = 200_000,
    seed: int = 0,
) -> dict[str, list[GeneratedPath]]:
    """End-to-end: fit real Hawkes params, build the constant-total-rate
    control/treatment arms, simulate n_sims independent realizations of
    each, convert to minute-bar synthetic return series. Returns
    {"control": [...], "treatment": [...]}.
    """
    fit = fit_real_hawkes_params(data_dir, ticker=ticker, sigma_threshold=sigma_threshold)
    control, treatment = build_ablation_arms(fit)

    bars = read_bars(data_dir, tickers=[ticker])
    frame = build_feature_frame(bars, vol_window=15, volume_window=15)
    background_std, jump_std = calibrate_jump_and_background_std(frame["log_return"].to_numpy(), sigma_threshold)

    session_seconds_per_day = 6.5 * 3600  # NYSE regular session -- matches the session-compressed fit's own units
    T = T_days * session_seconds_per_day

    rng = np.random.default_rng(seed)
    results: dict[str, list[GeneratedPath]] = {"control": [], "treatment": []}
    for arm in (control, treatment):
        for i in range(n_sims):
            sim_seed = int(rng.integers(0, 2**32 - 1))
            events = simulate_hawkes_bounded(arm.mu, arm.alpha, arm.beta, T, max_events, seed=sim_seed)
            returns = hawkes_events_to_returns(
                events, T, bar_seconds, background_std, jump_std, seed=sim_seed
            )
            results[arm.label].append(
                GeneratedPath(
                    generator_id=f"hawkes_{arm.label}",
                    log_returns=returns,
                    seed=sim_seed,
                    params={"mu": arm.mu, "alpha": arm.alpha, "beta": arm.beta, "T_days": T_days},
                )
            )
    return results
