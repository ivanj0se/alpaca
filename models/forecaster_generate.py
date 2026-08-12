"""Free-running (ancestral) sampling from a trained TCNForecaster --
generates a genuinely new synthetic path by feeding the model's own
sampled output back in as input at each step, unlike models/tcn_vae.py's
decode() which only ever reconstructs a real, already-observed window.

Unlike the reconstruction VAE (permanently anchored to real per-position
h_seq, structurally bounded), free-running generation has no real anchor
after the seed window -- errors can compound step to step (the standard
autoregressive-generation "exposure bias" failure mode: the model is
trained on real history via teacher forcing but run at generation time on
its own, possibly-imperfect, prior outputs). Validate a bounded
free-running horizon empirically rather than assuming it's fine at any
length -- see diagnostics/2026-08-12-tcn-forecaster-generative/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from features.returns import build_feature_frame
from features.windows import make_windows
from generators.path import GeneratedPath
from ingest.storage import read_bars
from models.forecaster_train import ForecasterTrainConfig, train
from models.tcn_forecaster import TCNForecaster


@torch.no_grad()
def ancestral_sample(
    model: TCNForecaster,
    seed_window: np.ndarray,
    n_steps: int,
    vol_window: int,
    seed: int | None = None,
) -> np.ndarray:
    """`seed_window`: (window_len, 2) real [log_return, realized_vol]
    values used only to prime the model's causal context -- never
    extended or mutated beyond providing the initial buffer. Rolls forward
    `n_steps` beyond it: at each step, samples
    next_return ~ N(mean, exp(logvar)) from the model's prediction at the
    buffer's last position, recomputes realized_vol via a rolling std over
    the trailing `vol_window` *generated* returns (falling back to the
    seed's own last real vol estimate until enough generated returns
    exist), then slides the window forward. Returns only the `n_steps`
    newly generated log_returns (not the seed).
    """
    if seed_window.shape[1] != 2:
        raise ValueError(f"seed_window must have 2 features [log_return, realized_vol], got shape {seed_window.shape}")
    model.eval()
    rng = np.random.default_rng(seed)
    buffer = seed_window.astype(np.float64).copy()
    all_returns = list(buffer[:, 0])
    generated = np.empty(n_steps)

    for t in range(n_steps):
        x = torch.from_numpy(buffer).float().unsqueeze(0)
        mean, logvar = model(x)
        m = mean[0, -1].item()
        std = float(np.exp(0.5 * logvar[0, -1].item()))
        next_return = float(rng.normal(m, std))
        generated[t] = next_return
        all_returns.append(next_return)

        recent = all_returns[-vol_window:]
        next_vol = float(np.std(recent, ddof=1)) if len(recent) >= 2 else float(buffer[-1, 1])

        buffer = np.concatenate([buffer[1:], np.array([[next_return, next_vol]])], axis=0)

    return generated


def generate_forecaster_paths(
    data_dir: Path,
    ticker: str = "SPY",
    window_len: int = 45,
    hidden_dim: int = 16,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 5,
    n_sims: int = 25,
    n_steps: int = 1950,
    vol_window: int = 15,
    seed: int = 0,
) -> list[GeneratedPath]:
    """End-to-end: reads real bars, trains a fresh TCNForecaster (2
    features: log_return, realized_vol -- deliberately dropping
    volume_zscore, see models/tcn_forecaster.py's docstring), then
    generates `n_sims` independent ancestral-sampling realizations from a
    real seed window. Mirrors
    generators/hawkes_jump_diffusion.py::generate_ablation_paths's
    end-to-end shape so scripts/run_generator_comparison.py can call every
    generator arm the same way.
    """
    bars = read_bars(data_dir, tickers=[ticker])
    if bars.empty:
        raise ValueError(f"no real bar data for {ticker} in {data_dir}")
    frame = build_feature_frame(bars, vol_window=vol_window, volume_window=vol_window)
    two_feature_frame = frame[["log_return", "realized_vol"]]

    windows, _ = make_windows(two_feature_frame, window_len=window_len, stride=5)
    if len(windows) < 20:
        raise ValueError(f"only {len(windows)} training windows for {ticker} -- not enough to train on")

    n_val = max(1, int(len(windows) * 0.15))
    train_windows, val_windows = windows[:-n_val], windows[-n_val:]

    config = ForecasterTrainConfig(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed)
    model_config = TCNForecaster(n_features=2, hidden_dim=hidden_dim, dilations=dilations)
    result = train(train_windows, val_windows=val_windows, config=config, model=model_config)
    model = result.model
    model.eval()

    seed_window = windows[len(windows) // 2]  # a representative real window, not the very last (avoid edge effects)
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_sims):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        generated = ancestral_sample(model, seed_window, n_steps=n_steps, vol_window=vol_window, seed=sim_seed)
        paths.append(
            GeneratedPath(
                generator_id="tcn_forecaster",
                log_returns=generated,
                seed=sim_seed,
                params={"window_len": window_len, "n_steps": n_steps, "best_epoch": result.best_epoch},
            )
        )
    return paths
