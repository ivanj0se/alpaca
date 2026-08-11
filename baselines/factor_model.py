"""Rung 2b: single-factor (CAPM-style) linear decomposition --
returns_i = alpha_i + beta_i * market_return + residual_i.

The linear-model version of what Rung 3 (Robust PCA) and Rung 4 (TCN-VAE)
do more richly: residual_i is the idiosyncratic, unexplained-by-the-market
component -- the same conceptual target as the sparse component in RPCA and
the reconstruction residual in the TCN-VAE. A closed-form OLS is used
rather than a library call -- the math is elementary (single predictor,
well-tested via unit tests recovering known alpha/beta on synthetic data)
and this avoids pulling in statsmodels' formula API for something this
simple.

Expect low beta and R^2 at minute-bar frequency (confirmed on real
AAPL-vs-SPY data during development: beta~0.4, R^2~0.03, well below the
~1.0-1.3 beta and much higher R^2 typical at daily frequency) -- this is
the well-documented Epps effect (Epps 1979): measured co-movement between
assets drops sharply as sampling frequency increases, due to microstructure
noise and asynchronous trading, not a bug in the fit. A low R^2 here
reflects that, not evidence of high market endogeneity by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorModelFit:
    ticker: str
    alpha: float
    beta: float
    residuals: pd.Series
    r_squared: float


def fit_linear_factor_model(returns: pd.Series, market_returns: pd.Series, ticker: str = "") -> FactorModelFit:
    """OLS of `returns` on `market_returns` (aligned on index; unaligned
    timestamps are dropped rather than assumed pre-aligned).
    """
    aligned = pd.DataFrame({"y": returns, "x": market_returns}).dropna()
    if len(aligned) < 3:
        raise ValueError("need at least 3 aligned observations to fit a factor model")

    x = aligned["x"].to_numpy()
    y = aligned["y"].to_numpy()
    x_mean, y_mean = x.mean(), y.mean()
    x_centered = x - x_mean
    denom = np.sum(x_centered**2)
    if denom == 0:
        raise ValueError("market_returns has zero variance in the aligned window -- cannot estimate beta")

    beta = float(np.sum(x_centered * (y - y_mean)) / denom)
    alpha = float(y_mean - beta * x_mean)
    predicted = alpha + beta * x
    residuals = y - predicted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return FactorModelFit(
        ticker=ticker,
        alpha=alpha,
        beta=beta,
        residuals=pd.Series(residuals, index=aligned.index),
        r_squared=r_squared,
    )


def factor_model_anomaly_score(fit: FactorModelFit) -> pd.Series:
    """Standardized (z-scored) idiosyncratic residual magnitude -- the
    Rung 2b anomaly score.
    """
    std = fit.residuals.std(ddof=1)
    if std == 0:
        return fit.residuals.abs() * 0.0
    return (fit.residuals / std).abs()


def make_fold_scorer(returns: pd.Series, market_returns: pd.Series):
    """Adapter for benchmark/ladder.py's evaluate_rung. `returns` and
    `market_returns` must already share the exact same index -- the CV
    splitter (benchmark/cv.py) generates train/test indices positionally
    against whatever single index `t1` was built from, so this scorer
    cannot re-align/drop rows internally without silently invalidating
    those positions (caught during development: internal re-alignment here
    shrank the effective series and threw IndexError deep inside a fold).
    Align the universe's returns onto a common index once, upstream, before
    building `t1` and calling evaluate_rung -- alignment is a
    data-preparation concern, not a per-model one.

    Fits alpha/beta and the in-sample residual variance on the training
    fold (homoskedastic assumption -- unlike GARCH, this baseline doesn't
    model time-varying variance, which is exactly the kind of gap Rung 3/4
    are expected to close), then scores the test fold's residuals (actual -
    predicted) against that fixed variance. Returns None (fold skipped)
    rather than raising if a fold can't be fit.
    """
    if not returns.index.equals(market_returns.index):
        raise ValueError(
            "returns and market_returns must share the exact same index for make_fold_scorer "
            "-- align them upstream (e.g. with a shared reindex/dropna) before calling"
        )
    aligned = pd.DataFrame({"y": returns, "x": market_returns})

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float | None:
        train = aligned.iloc[train_idx]
        test = aligned.iloc[test_idx]
        if len(train) < 3 or len(test) == 0:
            return None
        try:
            fit = fit_linear_factor_model(train["y"], train["x"])
        except ValueError:
            return None

        residual_var = fit.residuals.var(ddof=1)
        if not np.isfinite(residual_var) or residual_var <= 0:
            return None

        predicted = fit.alpha + fit.beta * test["x"].to_numpy()
        residuals = test["y"].to_numpy() - predicted
        nlls = 0.5 * (np.log(2 * np.pi * residual_var) + residuals**2 / residual_var)
        return float(np.mean(nlls))

    return score


def fit_universe_factor_models(
    returns_by_ticker: dict[str, pd.Series], market_returns: pd.Series
) -> dict[str, FactorModelFit]:
    """Fit one single-factor model per ticker against a common market proxy
    (e.g. SPY) -- the natural unit of comparison for the whole basket.
    """
    return {
        ticker: fit_linear_factor_model(r, market_returns, ticker=ticker)
        for ticker, r in returns_by_ticker.items()
    }
