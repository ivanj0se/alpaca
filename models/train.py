"""CPU-tuned training loop for the TCN-VAE. No GPU assumed anywhere --
`torch.set_num_threads` is called explicitly rather than relying on
PyTorch's default thread heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from models.tcn_vae import TCNVAE


def vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard VAE loss: reconstruction (MSE, summed over features and
    time, averaged over the batch) + beta * KL(q(z|x) || N(0, I)). Returns
    (total_loss, recon_loss, kl_loss) so callers can log both terms.
    """
    recon_loss = F.mse_loss(recon, x, reduction="none").sum(dim=(1, 2)).mean()
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    beta: float = 1.0
    patience: int = 5
    num_threads: int = 1
    seed: int = 0


@dataclass
class TrainResult:
    model: TCNVAE
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def _iterate_batches(windows: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = len(windows)
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield torch.from_numpy(windows[idx]).float()


def train_epoch(model: TCNVAE, windows: np.ndarray, config: TrainConfig, optimizer, rng: np.random.Generator) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in _iterate_batches(windows, config.batch_size, rng):
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=config.beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: TCNVAE, windows: np.ndarray, config: TrainConfig) -> float:
    if len(windows) == 0:
        return float("nan")
    model.eval()
    x = torch.from_numpy(windows).float()
    recon, mu, logvar = model(x)
    loss, _, _ = vae_loss(recon, x, mu, logvar, beta=config.beta)
    return loss.item() / len(windows) * config.batch_size  # roughly comparable scale to per-batch train loss


def train(
    train_windows: np.ndarray,
    val_windows: np.ndarray | None = None,
    config: TrainConfig | None = None,
    model: TCNVAE | None = None,
) -> TrainResult:
    """Trains a TCNVAE with early stopping on val_windows (if provided;
    otherwise runs the full `config.epochs` with no early stopping).
    `model` can be passed in pre-constructed (e.g. with a specific
    n_features/window_len matching the data); otherwise a default TCNVAE is
    built from the windows' own shape.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    rng = np.random.default_rng(config.seed)

    if model is None:
        n_features = train_windows.shape[2]
        window_len = train_windows.shape[1]
        model = TCNVAE(n_features=n_features, window_len=window_len)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    result = TrainResult(model=model)
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_windows, config, optimizer, rng)
        result.train_losses.append(train_loss)

        if val_windows is not None and len(val_windows) > 0:
            val_loss = evaluate(model, val_windows, config)
            result.val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                result.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    result.stopped_early = True
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return result
