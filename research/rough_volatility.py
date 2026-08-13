"""Empirical check of the rough-volatility prediction implied by this
project's own near-critical Hawkes fits.

El Euch, Fukasawa & Rosenbaum (2018, Finance & Stochastics) prove that a
NEARLY-UNSTABLE, heavy-tailed Hawkes process (branching ratio -> 1)
converges, in an appropriate macroscopic scaling limit, to a rough
volatility process. Gatheral, Jaisson & Rosenbaum (2018, "Volatility is
Rough") found this empirically across many real assets: realized
log-volatility's Hurst exponent H ~ 0.1 -- much rougher than standard
Brownian motion's H=0.5. Our own live-refit SPY branching ratio
(0.95-0.997, diagnostics/2026-08-11-real-tick-hawkes-replication/) sits
exactly in the regime this theorem requires. This module checks whether
real SPY realized volatility is actually as rough as the theory predicts
for THIS specific instrument/venue, rather than assuming the published
H~0.1 transfers automatically -- and whether IEX-only vs SIP-consolidated
feeds give a different answer.

Adapted to this project's data scale: Gatheral et al.'s original study
used DAILY realized volatility (from 5-min intraday returns) over YEARS
of data. We have ~1 month of real backfilled data, not enough daily
observations for a robust daily-lag regression -- this uses an INTRADAY
block-realized-vol construction instead (same structure-function
methodology, finer/data-appropriate scale, in the spirit of
Bennedsen/Lunde/Pakkanen-style intraday roughness estimation). An
explicit, documented adaptation, not a silent substitution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.returns import session_boundary_mask


def block_realized_vol(log_returns: pd.Series, block_minutes: int) -> pd.Series:
    """Non-overlapping realized volatility: sqrt(sum(r_i^2)) within each
    disjoint block_minutes-wide window. Non-overlapping is essential: an
    overlapping/rolling window (like features/returns.py's realized_vol
    feature column) shares most of its underlying returns between
    adjacent estimates, injecting strong short-lag autocorrelation BY
    CONSTRUCTION that would masquerade as roughness in a Hurst estimate --
    never substitute that feature column for this function. Drops any
    trailing partial block (fewer than half a full block's worth of
    observations) rather than including a noisy, under-sampled estimate.
    """
    idx = log_returns.index
    if len(idx) < 2:
        return pd.Series(dtype=float)
    block_id = (idx - idx[0]) // pd.Timedelta(minutes=block_minutes)
    grouped = log_returns.groupby(block_id.to_numpy())
    rv = grouped.apply(lambda r: np.sqrt(np.sum(r.to_numpy() ** 2)))
    counts = grouped.size()
    min_count = max(1, int(block_minutes * 0.5))
    return rv[counts >= min_count]


def block_realized_vol_by_session(log_returns: pd.Series, block_minutes: int) -> list[pd.Series]:
    """One block_realized_vol Series PER TRADING SESSION, computed
    independently -- avoids any cross-session-boundary contamination in
    the later structure-function analysis. A lag-1 step must always mean
    block_minutes of real, contiguous trading time within a single
    session, never a step across an overnight/weekend gap (reuses
    features.returns.session_boundary_mask, the same fix already
    validated for the return series itself -- see
    diagnostics/2026-08-11-session-boundary-returns/).
    """
    idx = log_returns.index
    if len(idx) < 2:
        return []
    boundary = session_boundary_mask(idx)
    session_id = boundary.cumsum()
    results = []
    for _, group in log_returns.groupby(session_id.to_numpy()):
        rv = block_realized_vol(group, block_minutes)
        if len(rv) > 5:
            results.append(rv)
    return results


@dataclass
class RoughnessEstimate:
    hurst: float
    lags: np.ndarray
    log_lags: np.ndarray
    log_moments: np.ndarray
    r_squared: float
    n_sessions: int


def estimate_roughness(session_rvs: list[pd.Series], lags: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 15, 20)) -> RoughnessEstimate:
    """Structure-function / smoothness estimator (Gatheral, Jaisson &
    Rosenbaum 2018): for log(realized_vol), E[|log(RV_{t+lag}) -
    log(RV_t)|] ~ lag^H for Hurst exponent H. Pools lag-differences across
    multiple independent trading sessions (each contributing its own
    within-session lag pairs, never crossing a session boundary) rather
    than requiring one long contiguous series. Fits
    log(E[|diff|]) vs log(lag) via OLS; the slope is the Hurst estimate.
    """
    if not session_rvs:
        raise ValueError("no sessions provided")
    used_lags, log_moments_list = [], []
    for lag in lags:
        diffs_for_lag = []
        for rv in session_rvs:
            log_rv = np.log(rv.to_numpy())
            if lag < len(log_rv):
                diffs_for_lag.append(np.abs(log_rv[lag:] - log_rv[:-lag]))
        if not diffs_for_lag:
            continue
        pooled = np.concatenate(diffs_for_lag)
        used_lags.append(lag)
        log_moments_list.append(np.log(pooled.mean()))

    if len(used_lags) < 2:
        raise ValueError("not enough valid lags to fit a Hurst exponent")

    lags_arr = np.array(used_lags)
    log_lags = np.log(lags_arr)
    log_moments = np.array(log_moments_list)
    slope, intercept = np.polyfit(log_lags, log_moments, 1)
    predicted = slope * log_lags + intercept
    ss_res = np.sum((log_moments - predicted) ** 2)
    ss_tot = np.sum((log_moments - log_moments.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return RoughnessEstimate(
        hurst=float(slope),
        lags=lags_arr,
        log_lags=log_lags,
        log_moments=log_moments,
        r_squared=r_squared,
        n_sessions=len(session_rvs),
    )
