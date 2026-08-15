"""Tier 3: controlled ablation of the two self-excitation EXTENSIONS
(research/multi_kernel_hawkes.py, research/cox_hawkes.py) against the
already-scored single-exponential Hawkes treatment arm
(generators/hawkes_jump_diffusion.py, overall_score=0.896 in the
original generator-comparison run -- see CLAUDE.md's "Market-generator
comparison suite" section). Question: does modeling real structure this
project has since found (multiple genuine self-excitation timescales;
a real, externally-measured exogenous baseline) produce measurably MORE
realistic synthetic paths than the single-exponential, constant-mu
treatment arm already does -- or does the single-exponential fit already
capture everything the shared stylized-facts harness can detect at this
sample length?

Reuses generators/hawkes_jump_diffusion.py's already-tested
session_elapsed_seconds, calibrate_jump_and_background_std and
hawkes_events_to_returns (the event-time -> synthetic-return-series
conversion is identical regardless of which point-process extension
produced the event times) rather than duplicating them. Does not modify
that module, research/multi_kernel_hawkes.py, or research/cox_hawkes.py
-- all separately tested and trust-gated in their own right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from events.price_events import tick_events_from_recorder
from features.returns import build_feature_frame
from generators.hawkes_jump_diffusion import (
    calibrate_jump_and_background_std,
    hawkes_events_to_returns,
    session_elapsed_seconds,
)
from generators.path import GeneratedPath
from ingest.storage import read_bars, read_ticks
from research.cox_hawkes import CoxHawkesFitResult, fit_cox_hawkes, simulate_cox_hawkes
from research.multi_kernel_hawkes import (
    MultiKernelFitResult,
    default_timescale_grid,
    fit_multi_kernel_hawkes,
    simulate_multi_kernel_hawkes,
)

SESSION_SECONDS_PER_DAY = 6.5 * 3600  # NYSE regular session -- matches hawkes_jump_diffusion.py's own convention


def fit_real_multi_kernel_params(
    data_dir: Path,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    n_components: int = 3,
    span_decades: float = 1.0,
) -> MultiKernelFitResult:
    """Live-refits the multi-kernel Hawkes on real backfilled tick data --
    same discipline as hawkes_jump_diffusion.py::fit_real_hawkes_params
    (a fresh fit, not a snapshot baked into this file). Session-compressed
    elapsed time, same as the single-kernel generator, for a like-for-like
    T convention between the two.
    """
    ticks = read_ticks(data_dir, tickers=[ticker])
    if ticks.empty:
        raise ValueError(f"no real tick data for {ticker} in {data_dir}")
    events = tick_events_from_recorder(ticks, sigma_threshold=sigma_threshold)
    times = pd.to_datetime(events[events["ticker"] == ticker]["timestamp"]).sort_values()
    if len(times) < 20:
        raise ValueError(f"only {len(times)} events for {ticker} -- not enough to fit a multi-kernel Hawkes process")

    seconds = session_elapsed_seconds(pd.DatetimeIndex(times))
    betas = default_timescale_grid(seconds, n_components=n_components, span_decades=span_decades)
    return fit_multi_kernel_hawkes(seconds, betas=betas)


def simulate_multi_kernel_hawkes_bounded(
    mu: float, alphas: np.ndarray, betas: np.ndarray, T: float, max_events: int, seed: int | None = None
) -> np.ndarray:
    """Same defensive event-count cap as
    hawkes_jump_diffusion.py::simulate_hawkes_bounded, generalized to K
    components -- a near-critical total branching ratio has the same
    high-variance-cluster-size risk regardless of how many kernels share
    it.
    """
    events = simulate_multi_kernel_hawkes(mu, alphas, betas, T, seed=seed)
    if len(events) > max_events:
        raise RuntimeError(
            f"simulated {len(events)} events, exceeding max_events={max_events} "
            f"(mu={mu:.6g}, alphas={alphas}, betas={betas}, T={T:.1f}) -- likely a "
            "near-critical total branching ratio producing a runaway cluster"
        )
    return events


def generate_multi_kernel_paths(
    data_dir: Path,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    T_days: float = 5.0,
    bar_seconds: float = 60.0,
    n_sims: int = 25,
    max_events: int = 200_000,
    seed: int = 0,
) -> list[GeneratedPath]:
    """End-to-end multi-kernel treatment arm: fit real multi-timescale
    params, simulate n_sims independent realizations, convert to
    minute-bar synthetic returns via the same jump-diffusion conversion
    the single-kernel generator uses.
    """
    fit = fit_real_multi_kernel_params(data_dir, ticker=ticker, sigma_threshold=sigma_threshold)

    bars = read_bars(data_dir, tickers=[ticker])
    frame = build_feature_frame(bars, vol_window=15, volume_window=15)
    background_std, jump_std = calibrate_jump_and_background_std(frame["log_return"].to_numpy(), sigma_threshold)

    T = T_days * SESSION_SECONDS_PER_DAY
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_sims):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        events = simulate_multi_kernel_hawkes_bounded(fit.mu, fit.alphas, fit.betas, T, max_events, seed=sim_seed)
        returns = hawkes_events_to_returns(events, T, bar_seconds, background_std, jump_std, seed=sim_seed)
        paths.append(
            GeneratedPath(
                generator_id="hawkes_multi_kernel",
                log_returns=returns,
                seed=sim_seed,
                params={
                    "mu": fit.mu,
                    "alphas": fit.alphas.tolist(),
                    "betas": fit.betas.tolist(),
                    "T_days": T_days,
                },
            )
        )
    return paths


def fit_real_cox_hawkes_params(
    data_dir: Path,
    common_factor: pd.Series,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
) -> tuple[CoxHawkesFitResult, np.ndarray, np.ndarray]:
    """Live-refits the Cox-Hawkes RPCA-baseline model on real data,
    mirroring the exact alignment methodology validated in
    diagnostics/2026-08-13-cox-hawkes-rpca-baseline/findings.md
    (pd.merge_asof(direction="backward"), NOT the fragile reindex/union
    chain that hit a real pandas duplicate-index edge case there).
    `common_factor` is passed in rather than recomputed here -- RPCA
    decomposition is itself expensive and already validated elsewhere
    (rpca/rolling_rpca.py); this function's job is only the Hawkes side.

    Returns (fit, grid_covariate, grid_durations) -- the grid is returned
    alongside the fit because generate_cox_hawkes_paths needs the SAME
    real covariate discretization to simulate forward from.

    Uses raw calendar time throughout (not session-compressed), matching
    exactly how the real diagnostics fit was performed -- an explicit,
    documented inconsistency with the multi-kernel/single-kernel
    generators' session-compressed convention, carried over rather than
    silently changed, since changing it would make this arm's real
    fitted mu0/gamma no longer match the already-validated diagnostics
    result.
    """
    ticks = read_ticks(data_dir, tickers=[ticker])
    if ticks.empty:
        raise ValueError(f"no real tick data for {ticker} in {data_dir}")
    events = tick_events_from_recorder(ticks, sigma_threshold=sigma_threshold)
    event_timestamps = pd.DatetimeIndex(
        pd.to_datetime(events[events["ticker"] == ticker]["timestamp"])
    ).sort_values()
    mask = (event_timestamps >= common_factor.index.min()) & (event_timestamps <= common_factor.index.max())
    event_timestamps = event_timestamps[mask]
    if len(event_timestamps) < 20:
        raise ValueError(f"only {len(event_timestamps)} events within RPCA coverage -- not enough to fit")

    events_frame = pd.DataFrame({"event_time": event_timestamps}).sort_values("event_time")
    cf_frame = pd.DataFrame(
        {"cf_time": common_factor.index, "cf_value": common_factor.to_numpy()}
    ).sort_values("cf_time")
    merged = pd.merge_asof(events_frame, cf_frame, left_on="event_time", right_on="cf_time", direction="backward")
    event_covariate = np.nan_to_num(merged["cf_value"].to_numpy(), nan=0.0)

    t = (event_timestamps - event_timestamps[0]).total_seconds().to_numpy()
    T = float(t[-1])
    grid_ts = (common_factor.index - event_timestamps[0]).total_seconds().to_numpy()
    in_range = (grid_ts >= 0) & (grid_ts <= T)
    grid_ts = grid_ts[in_range]
    grid_vals = common_factor.to_numpy()[in_range]
    grid_durations = np.clip(np.diff(np.concatenate([grid_ts, [T]])), 0, None)

    from events.hawkes import fit_hawkes_exponential_multistart

    standard_fit = fit_hawkes_exponential_multistart(t)
    cox_fit = fit_cox_hawkes(
        t, event_covariate, beta=standard_fit.beta, grid_covariate=grid_vals, grid_durations=grid_durations, T=T
    )
    return cox_fit, grid_vals, grid_durations


def generate_cox_hawkes_paths(
    data_dir: Path,
    common_factor: pd.Series,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    T_days: float = 5.0,
    bar_seconds: float = 60.0,
    n_sims: int = 25,
    max_events: int = 200_000,
    seed: int = 0,
) -> list[GeneratedPath]:
    """End-to-end Cox-Hawkes treatment arm. x(t) is a genuinely exogenous,
    externally-measured driver in this design (real RPCA history), not
    something the generator invents -- so only the point-process
    randomness (immigrant draws + branching) varies across the n_sims
    independent realizations, all conditioned on the SAME real covariate
    window. This is a deliberate modeling choice: it tests "does knowing
    the real exogenous history produce a more realistic path," not "can
    we invent plausible synthetic exogenous histories" (a different,
    harder question this generator does not attempt).

    Truncates the real covariate grid (itself indexed in raw calendar
    seconds -- see fit_real_cox_hawkes_params) to exactly
    T_days*SESSION_SECONDS_PER_DAY seconds of real covariate history --
    T_days uses the SAME session-trading-day convention as
    generate_multi_kernel_paths and hawkes_jump_diffusion.py's
    generate_ablation_paths, deliberately, NOT calendar days: every arm
    evaluated by benchmark/generator_ladder.py must produce the exact
    same bar count, since calibrate_reference_bands calibrates its bands
    at one specific path_length and scoring a different-length path
    against them silently measures the length mismatch, not realism
    (the exact bug already caught once, see
    diagnostics/2026-08-13-conformal-band-length-mismatch/). An earlier
    version of this function used calendar days (86400s) here AND
    rounded UP to "at least T_requested" rather than clipping the final
    segment -- together those produced paths 5x longer than every other
    arm, and a spuriously bad realism score that was actually just this
    same length-mismatch bug wearing a new hat (see
    diagnostics/2026-08-14-tier3-hawkes-extensions-ablation/). Clips the
    final included grid segment's duration so T_actual always lands
    EXACTLY on the target, never over.
    """
    fit, grid_vals, grid_durations = fit_real_cox_hawkes_params(
        data_dir, common_factor, ticker=ticker, sigma_threshold=sigma_threshold
    )

    T_requested = T_days * SESSION_SECONDS_PER_DAY
    cum = np.cumsum(grid_durations)
    total_available = float(cum[-1]) if len(cum) > 0 else 0.0
    if total_available < T_requested * 0.5:
        raise ValueError(
            f"real covariate window only covers {total_available:.0f}s, less than half the requested "
            f"{T_requested:.0f}s ({T_days} days) -- extend the RPCA common-factor series first"
        )
    n_keep = int(np.searchsorted(cum, T_requested, side="left")) + 1
    n_keep = min(n_keep, len(grid_durations))
    grid_vals = grid_vals[:n_keep]
    grid_durations = grid_durations[:n_keep].copy()
    overshoot = float(cum[n_keep - 1]) - T_requested
    if overshoot > 0:
        grid_durations[-1] -= overshoot
    T_actual = float(grid_durations.sum())

    bars = read_bars(data_dir, tickers=[ticker])
    frame = build_feature_frame(bars, vol_window=15, volume_window=15)
    background_std, jump_std = calibrate_jump_and_background_std(frame["log_return"].to_numpy(), sigma_threshold)

    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_sims):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        events = simulate_cox_hawkes(
            fit.mu0, fit.gamma, fit.alpha, fit.beta, grid_vals, grid_durations, seed=sim_seed
        )
        if len(events) > max_events:
            raise RuntimeError(f"simulated {len(events)} events, exceeding max_events={max_events}")
        returns = hawkes_events_to_returns(events, T_actual, bar_seconds, background_std, jump_std, seed=sim_seed)
        paths.append(
            GeneratedPath(
                generator_id="cox_hawkes_rpca",
                log_returns=returns,
                seed=sim_seed,
                params={"mu0": fit.mu0, "gamma": fit.gamma, "alpha": fit.alpha, "beta": fit.beta, "T_days": T_days},
            )
        )
    return paths
