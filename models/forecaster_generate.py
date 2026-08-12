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

import numpy as np
import torch

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
