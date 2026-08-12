"""CPU-tuned training loop for TCNForecaster -- mirrors models/train.py's
pattern (seeded, single-threaded by default, early stopping on a
validation set), but the loss is a Gaussian NLL computed at EVERY offset
in parallel (teacher forcing), not a window-reconstruction MSE+KL.

The shift-by-one alignment here is exactly the class of off-by-one bug
this codebase's diagnostics history is full of (GARCH walk-forward,
NLL scale-invariance) -- get it wrong and the model trivially "predicts"
the current step's own value instead of the next one, and training loss
looks great for the wrong reason. See
tests/unit/test_forecaster_train.py::test_recovers_true_conditional_mean_on_a_known_ar1_process
for the regression guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from models.tcn_forecaster import TCNForecaster


def forecaster_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, target_next: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL, averaged over batch and time. `target_next[:, t]`
    must already be the value being predicted (i.e. shifted by the caller
    -- this function does no shifting itself).
    """
    var = torch.exp(logvar)
    return (0.5 * (torch.log(2 * torch.pi * var) + (target_next - mean) ** 2 / var)).mean()


@dataclass
class ForecasterTrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    patience: int = 5
    num_threads: int = 1
    seed: int = 0


@dataclass
class ForecasterTrainResult:
    model: TCNForecaster
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


def train_epoch(
    model: TCNForecaster, windows: np.ndarray, config: ForecasterTrainConfig, optimizer, rng: np.random.Generator
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in _iterate_batches(windows, config.batch_size, rng):
        optimizer.zero_grad()
        mean, logvar = model(batch)
        # position t's prediction targets log_return at t+1 (channel 0);
        # the last position has nothing to predict against (no t+1 in this
        # window) so it's dropped from the loss, not from the forward pass.
        loss = forecaster_nll_loss(mean[:, :-1], logvar[:, :-1], batch[:, 1:, 0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model: TCNForecaster, windows: np.ndarray, config: ForecasterTrainConfig) -> float:
    if len(windows) == 0:
        return float("nan")
    model.eval()
    x = torch.from_numpy(windows).float()
    mean, logvar = model(x)
    return forecaster_nll_loss(mean[:, :-1], logvar[:, :-1], x[:, 1:, 0]).item()


def train(
    train_windows: np.ndarray,
    val_windows: np.ndarray | None = None,
    config: ForecasterTrainConfig | None = None,
    model: TCNForecaster | None = None,
) -> ForecasterTrainResult:
    config = config or ForecasterTrainConfig()
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    rng = np.random.default_rng(config.seed)

    if model is None:
        n_features = train_windows.shape[2]
        model = TCNForecaster(n_features=n_features)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    result = ForecasterTrainResult(model=model)
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
