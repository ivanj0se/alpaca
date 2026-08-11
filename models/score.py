"""Reconstruction-residual scoring for the trained TCN-VAE -- the anomaly
score for Rung 4, and the temporal lane's out-of-sample NLL for
benchmark/ladder.py.
"""

from __future__ import annotations

import numpy as np
import torch

from models.tcn_vae import TCNVAE
from models.train import TrainConfig, train


@torch.no_grad()
def reconstruction_residuals(model: TCNVAE, windows: np.ndarray) -> np.ndarray:
    """Element-wise reconstruction residual (recon - actual), same shape as
    `windows` (n_windows, window_len, n_features) -- can be positive or
    negative, unlike the MSE anomaly score below. This is what
    make_fold_scorer's NLL is computed against, matching the other rungs'
    residual-under-Gaussian methodology (GARCH's return residual, RPCA's
    projection residual) rather than scoring an always-positive quantity.
    """
    model.eval()
    x = torch.from_numpy(windows).float()
    recon, _, _ = model(x)
    return (recon - x).numpy()


def reconstruction_residual(model: TCNVAE, windows: np.ndarray) -> np.ndarray:
    """Per-window mean squared reconstruction error -- the Rung 4 anomaly
    score. Shape: (n_windows,).
    """
    residuals = reconstruction_residuals(model, windows)
    return (residuals**2).mean(axis=(1, 2))


def standardize_windows(
    train_windows: np.ndarray, test_windows: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature z-score, fit on train only (avoids leaking test-fold
    statistics) and applied to both. Necessary before joint multi-channel
    training/scoring: confirmed on real AAPL data that log_return and
    realized_vol have std ~0.0007-0.001 while volume_zscore (already
    z-scored by construction) has std ~1.04 -- about 1000x larger. Without
    standardizing, both the MSE training loss and the pooled-variance NLL
    are dominated by volume_zscore, so the model effectively learns to
    reconstruct volume_zscore at the expense of the tiny-scale return
    features, and the resulting NLL isn't comparable to GARCH's (which
    scores log_return alone, at its own natural tiny-variance scale).
    """
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, eps)
    return (train_windows - mean) / std, (test_windows - mean) / std


def make_fold_scorer(windows: np.ndarray, config: TrainConfig | None = None, primary_channel: int = 0):
    """Adapter for benchmark/ladder.py's evaluate_rung -- Rung 4's entry in
    the temporal lane (must beat Rung 2a's GARCH).

    Standardizes each fold for *training* (fit on train only, see
    standardize_windows) so the loss isn't dominated by whichever feature
    has the largest raw scale, then trains a fresh TCNVAE on the
    standardized windows. But the final NLL is computed on only
    `primary_channel`'s residual (default 0 = log_return, matching
    features/returns.build_feature_frame's fixed column order), *un*-standardized
    back to its original raw-return scale.

    This two-step handling is necessary, not cosmetic: NLL is not
    scale-invariant. A Gaussian's log-density grows without bound as
    variance shrinks, so scoring a standardized (~unit-variance) quantity
    is never comparable to scoring GARCH's raw tiny-variance return,
    independent of the training-scale fix above -- confirmed empirically:
    even after standardizing all channels for training, a naive pooled NLL
    over the standardized residuals still wasn't comparable to GARCH's raw
    NLL. Scoring only the un-standardized primary channel puts both models
    on the same footing: how well each predicts/reconstructs the
    instrument's own raw-scale log return, with TCN-VAE allowed the
    (legitimate, still history-only, no lookahead) advantage of also using
    realized_vol and volume_zscore as additional input signal.
    """
    config = config or TrainConfig()

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float | None:
        if len(train_idx) < 20 or len(test_idx) == 0:
            return None
        train_raw, test_raw = windows[train_idx], windows[test_idx]
        train_windows, test_windows = standardize_windows(train_raw, test_raw)

        channel_std = float(train_raw[..., primary_channel].std())
        if channel_std == 0 or not np.isfinite(channel_std):
            return None

        result = train(train_windows, config=config)
        model = result.model

        train_residuals_raw = reconstruction_residuals(model, train_windows)[..., primary_channel] * channel_std
        residual_var = max(float(np.var(train_residuals_raw, ddof=1)), 1e-12)

        test_residuals_raw = reconstruction_residuals(model, test_windows)[..., primary_channel] * channel_std
        nlls = 0.5 * (np.log(2 * np.pi * residual_var) + test_residuals_raw**2 / residual_var)
        return float(np.mean(nlls))

    return score
