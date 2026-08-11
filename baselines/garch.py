"""Rung 2a: GARCH(1,1) -- a "boring baseline" anomaly score every fancier
model must beat. Anomaly score = how extreme a return is relative to the
model's own time-varying volatility estimate (a standardized residual).
This is a much lower bar than reconstruction-based approaches, but a real
one: does the fancy stuff actually add anything a univariate GARCH doesn't
already capture?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


@dataclass
class GarchFitResult:
    model_result: ARCHModelResult
    standardized_residuals: pd.Series
    scale: float


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, rescale: bool = True) -> GarchFitResult:
    """Fit GARCH(p,q) with a zero mean (consistent with treating returns as
    close to a random walk -- see the endogenous/exogenous discussion in
    docs/architecture.md) to `returns`. `rescale` scales returns so their
    std is ~10 before fitting, which `arch` recommends for numerical
    stability. A *fixed* multiplier (e.g. the commonly-used 100x, tuned for
    daily-bar-scale returns) under-scales intraday minute returns, whose
    std is roughly 20x smaller (vol scales with sqrt(time), and there are
    ~390 minutes/trading day) -- confirmed via arch's own DataScaleWarning
    on real data during development. Standardized residuals are scale
    invariant, so this only affects optimizer numerics, not the score.
    """
    if len(returns) < 20:
        raise ValueError("need at least 20 observations for a stable GARCH(1,1) fit")
    std = returns.std()
    scale = 10.0 / std if (rescale and std > 0) else 1.0
    model = arch_model(returns * scale, vol="Garch", p=p, q=q, mean="Zero", dist="normal")
    result = model.fit(disp="off")
    standardized_residuals = pd.Series(
        result.resid / result.conditional_volatility, index=returns.index
    )
    return GarchFitResult(model_result=result, standardized_residuals=standardized_residuals, scale=scale)


def garch_anomaly_score(fit: GarchFitResult) -> pd.Series:
    """|standardized residual|: how many conditional-volatility standard
    deviations a return is from zero. This is the Rung 2a anomaly score.
    """
    return fit.standardized_residuals.abs()


def forecast_variance_path(fit: GarchFitResult, horizon: int) -> np.ndarray:
    """Variance forecast for each of the next `horizon` steps ahead from
    the end of the training data (in the original returns' scale), without
    refitting -- the standard way to score a whole held-out fold against a
    single fitted model rather than repeatedly one-step-forecasting.
    """
    forecast = fit.model_result.forecast(horizon=horizon, reindex=False)
    variance_scaled = forecast.variance.to_numpy()[-1, :]
    return variance_scaled / (fit.scale**2)


def forecast_one_step_variance(fit: GarchFitResult) -> float:
    """Next-period conditional variance forecast, for scoring a single
    out-of-sample return without refitting.
    """
    return float(forecast_variance_path(fit, horizon=1)[0])


def out_of_sample_neg_log_likelihood(fit: GarchFitResult, next_return: float) -> float:
    """Gaussian negative log-likelihood of a single out-of-sample return
    under the model's one-step-ahead variance forecast (mean assumed 0,
    matching the fitted model). Used by benchmark/ladder.py to compare
    Rung 2 against later rungs on shared, ground-truth-free footing:
    "how well does this model predict data it hasn't seen."
    """
    variance = forecast_one_step_variance(fit)
    return 0.5 * (np.log(2 * np.pi * variance) + next_return**2 / variance)


def make_fold_scorer(returns: pd.Series):
    """Adapter for benchmark/ladder.py's evaluate_rung: given positional
    train/test indices into `returns`, fit GARCH on the training fold and
    return the mean out-of-sample NLL over the test fold, scored against a
    forecast_variance_path (no refitting mid-fold). Returns None (fold
    skipped) if the training slice is too short or the fit doesn't
    converge, rather than raising and aborting the whole ladder run.
    """

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float | None:
        train_returns = returns.iloc[train_idx]
        test_returns = returns.iloc[test_idx]
        if len(train_returns) < 20:
            return None
        try:
            fit = fit_garch(train_returns)
        except Exception:
            return None
        if not fit.model_result.convergence_flag == 0:
            return None

        variances = forecast_variance_path(fit, horizon=len(test_returns))
        actual = test_returns.to_numpy()
        nlls = 0.5 * (np.log(2 * np.pi * variances) + actual**2 / variances)
        return float(np.mean(nlls))

    return score
