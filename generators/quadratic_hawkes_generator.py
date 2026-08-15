"""Real-data wiring for research/quadratic_hawkes.py (Tier 6): fits the
model against real signed tick-level marks, the one input the other two
extensions (multi-kernel, Cox-Hawkes) never needed, since this is the
first model in this project whose intensity depends on which DIRECTION
past price moves went, not just when they happened.

A separate module from generators/hawkes_extensions_generator.py
(Tier 3's multi-kernel/Cox-Hawkes wiring) rather than folded into it --
that module's own docstring specifically scopes itself to "the two
self-excitation extensions," and this one needs a genuinely different
real-data input (signed marks, not just event times), not just another
arm sharing the same inputs.

Reuses generators/hawkes_jump_diffusion.py's already-tested
session_elapsed_seconds, calibrate_jump_and_background_std and
hawkes_events_to_returns, same as generators/hawkes_extensions_generator.py.
Does not modify events/price_events.py or research/quadratic_hawkes.py --
both separately tested/trust-gated -- this reimplements the signed
z-score event definition locally (a few lines) rather than importing
private logic from another module, the same discipline already used
throughout this project's research/ extensions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.returns import build_feature_frame, session_boundary_mask
from generators.hawkes_jump_diffusion import (
    calibrate_jump_and_background_std,
    hawkes_events_to_returns,
    session_elapsed_seconds,
)
from generators.path import GeneratedPath
from ingest.storage import read_bars, read_ticks
from research.quadratic_hawkes import QuadraticHawkesFitResult, fit_quadratic_hawkes, simulate_quadratic_hawkes


def signed_tick_events(ticks_df: pd.DataFrame, ticker: str, sigma_threshold: float = 2.0) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Same event definition as events/price_events.py::tick_events_from_recorder
    (per-ticker log-return z-score >= sigma_threshold, session-boundary
    changes excluded) but returns the SIGNED z-score as the mark instead
    of discarding the sign via abs(). Returns (event_times, signed_marks),
    both sorted by time.
    """
    group = ticks_df[ticks_df["ticker"] == ticker].sort_values("timestamp")
    if group.empty:
        raise ValueError(f"no real tick data for {ticker}")
    timestamps = pd.DatetimeIndex(group["timestamp"])
    price = group["price"].to_numpy()
    if len(price) < 3:
        raise ValueError(f"only {len(price)} ticks for {ticker} -- not enough to compute returns")

    log_returns = np.diff(np.log(price))
    boundary = session_boundary_mask(timestamps).to_numpy()[1:]
    log_returns = log_returns[~boundary]
    event_times_all = timestamps.to_numpy()[1:][~boundary]

    std = log_returns.std(ddof=1)
    if std == 0:
        raise ValueError(f"{ticker} returns have zero variance -- cannot z-score")
    signed_z = log_returns / std
    event_mask = np.abs(signed_z) >= sigma_threshold
    if not event_mask.any():
        raise ValueError(f"no events for {ticker} at sigma_threshold={sigma_threshold}")

    return pd.DatetimeIndex(event_times_all[event_mask]), signed_z[event_mask]


def fit_real_quadratic_hawkes_params(
    data_dir: Path,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    beta_leverage: float | None = None,
    beta: float | None = None,
) -> tuple[QuadraticHawkesFitResult, np.ndarray]:
    """Live-refits against real backfilled tick data, same discipline as
    every other fit_real_*_params function in this project (a fresh fit,
    not a snapshot). beta_leverage/beta default to the SAME data-adaptive
    scale (1/median inter-event gap, session-compressed) -- no strong
    prior for why the leverage kernel's natural timescale would differ
    from the standard clustering kernel's, so start them equal rather
    than introduce an unmotivated asymmetry; a real diagnostics run can
    always try a second beta_leverage grid point if this default turns
    out to be a poor fit.

    Returns (fit, marks) -- marks are returned alongside the fit because
    downstream generation needs the SAME real marks as its
    empirical-bootstrap marks_pool (research/quadratic_hawkes.py's
    simulate_quadratic_hawkes takes marks_pool explicitly, doesn't invent
    one).
    """
    ticks = read_ticks(data_dir, tickers=[ticker])
    if ticks.empty:
        raise ValueError(f"no real tick data for {ticker} in {data_dir}")
    event_times, marks = signed_tick_events(ticks, ticker, sigma_threshold=sigma_threshold)
    if len(event_times) < 20:
        raise ValueError(f"only {len(event_times)} events for {ticker} -- not enough to fit a quadratic Hawkes process")

    seconds = session_elapsed_seconds(event_times)
    if beta_leverage is None or beta is None:
        median_gap = float(np.median(np.diff(np.sort(seconds))))
        default_beta = 1.0 / median_gap
        beta_leverage = beta_leverage if beta_leverage is not None else default_beta
        beta = beta if beta is not None else default_beta

    fit = fit_quadratic_hawkes(seconds, marks, beta_leverage=beta_leverage, beta=beta)
    return fit, marks


def generate_quadratic_hawkes_paths(
    data_dir: Path,
    ticker: str = "SPY",
    sigma_threshold: float = 2.0,
    T_days: float = 5.0,
    bar_seconds: float = 60.0,
    n_sims: int = 25,
    max_events: int = 200_000,
    seed: int = 0,
) -> list[GeneratedPath]:
    """End-to-end quadratic-Hawkes treatment arm, same shape as
    generate_ablation_paths/generate_multi_kernel_paths/
    generate_cox_hawkes_paths -- fit real params, simulate n_sims
    independent realizations (marks resampled from the real observed
    marks each time, so every realization gets its own draw, not one
    fixed replayed sequence -- unlike Tier 3's Cox-Hawkes arm, which
    deliberately replays one fixed real covariate path since x(t) there
    is a genuinely exogenous driver; a jump's own SIGN isn't exogenous in
    the same sense, it's part of what this model is generating), convert
    to minute-bar synthetic returns via the same jump-diffusion
    conversion every other arm in this project uses.
    """
    fit, marks = fit_real_quadratic_hawkes_params(data_dir, ticker=ticker, sigma_threshold=sigma_threshold)

    bars = read_bars(data_dir, tickers=[ticker])
    frame = build_feature_frame(bars, vol_window=15, volume_window=15)
    background_std, jump_std = calibrate_jump_and_background_std(frame["log_return"].to_numpy(), sigma_threshold)

    T = T_days * (6.5 * 3600)
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_sims):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        events, _ = simulate_quadratic_hawkes(
            fit.lambda0, fit.kappa, fit.alpha, fit.beta_leverage, fit.beta, T, marks, seed=sim_seed, max_events=max_events
        )
        returns = hawkes_events_to_returns(events, T, bar_seconds, background_std, jump_std, seed=sim_seed)
        paths.append(
            GeneratedPath(
                generator_id="quadratic_hawkes",
                log_returns=returns,
                seed=sim_seed,
                params={
                    "lambda0": fit.lambda0,
                    "kappa": fit.kappa,
                    "alpha": fit.alpha,
                    "beta_leverage": fit.beta_leverage,
                    "beta": fit.beta,
                    "T_days": T_days,
                },
            )
        )
    return paths
