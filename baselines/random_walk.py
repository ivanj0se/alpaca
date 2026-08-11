"""Rung 0: vol-matched random-walk null. Establishes the sanity floor every
higher rung must beat -- if a metric can't tell real data apart from a GBM
path matched to the same drift/volatility, it isn't measuring anything.

Also the source of the pipeline's specificity check for Rung 1
(events/hawkes.py): a GBM path's increments are i.i.d., so a price-event
point process built from it has no self-excitation -- a Hawkes fit should
recover a branching ratio near 0, in contrast to the real market's ~0.81.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GBMParams:
    mu: float  # per-step log-return mean
    sigma: float  # per-step log-return std


def estimate_gbm_params(prices: pd.Series) -> GBMParams:
    """Estimate per-step drift/volatility from real log returns, so the null
    is matched to the real data's scale rather than arbitrary.
    """
    if len(prices) < 3:
        raise ValueError("need at least 3 price points to estimate GBM parameters")
    log_returns = np.diff(np.log(prices.to_numpy()))
    return GBMParams(mu=float(np.mean(log_returns)), sigma=float(np.std(log_returns, ddof=1)))


def simulate_gbm(
    params: GBMParams,
    n_steps: int,
    n_sims: int = 1,
    s0: float = 100.0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate `n_sims` GBM price paths, shape (n_sims, n_steps + 1)
    including the starting price s0 as column 0.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=params.mu, scale=params.sigma, size=(n_sims, n_steps))
    log_paths = np.cumsum(shocks, axis=1)
    return s0 * np.exp(np.hstack([np.zeros((n_sims, 1)), log_paths]))


def simulate_matched_to(prices: pd.Series, n_sims: int = 1, seed: int | None = None) -> np.ndarray:
    """Simulate paths matched in length, starting price, and drift/volatility
    to a real price series -- the standard null for comparing any downstream
    metric against.
    """
    params = estimate_gbm_params(prices)
    n_steps = len(prices) - 1
    return simulate_gbm(params, n_steps=n_steps, n_sims=n_sims, s0=float(prices.iloc[0]), seed=seed)


def make_fold_scorer(returns: pd.Series):
    """Adapter for benchmark/ladder.py's evaluate_rung -- Rung 0, the
    temporal lane's sanity floor. No dynamics at all: estimates a single
    constant variance from the training fold's returns (i.i.d. Gaussian, no
    time-varying volatility, no factor conditioning -- the simplest
    possible baseline) and scores held-out returns' Gaussian NLL against
    it. Rung 2a's GARCH must beat this by modeling volatility clustering;
    if it can't, volatility clustering isn't adding real predictive value
    over just "returns are noisy with roughly this much variance."
    """

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float | None:
        if len(train_idx) < 20 or len(test_idx) == 0:
            return None
        train_returns = returns.iloc[train_idx].to_numpy()
        test_returns = returns.iloc[test_idx].to_numpy()

        variance = float(np.var(train_returns, ddof=1))
        if variance <= 0 or not np.isfinite(variance):
            return None

        nlls = 0.5 * (np.log(2 * np.pi * variance) + test_returns**2 / variance)
        return float(np.mean(nlls))

    return score
