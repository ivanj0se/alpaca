import numpy as np
import pytest
import torch

from models.tcn_vae import TCNVAE
from models.train import TrainConfig, evaluate, train, train_epoch, vae_loss


class TestVaeLoss:
    def test_kl_divergence_zero_for_standard_normal(self):
        # KL(N(0,I) || N(0,I)) == 0.
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)  # logvar=0 -> var=1
        x = torch.zeros(4, 10, 3)
        recon = torch.zeros(4, 10, 3)
        total, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
        assert kl_loss.item() == pytest.approx(0.0, abs=1e-6)
        assert recon_loss.item() == pytest.approx(0.0, abs=1e-6)
        assert total.item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_matches_closed_form(self):
        mu = torch.tensor([[1.0, 2.0]])
        logvar = torch.tensor([[0.5, -0.5]])
        expected_kl = -0.5 * (1 + 0.5 - 1.0**2 - np.exp(0.5) + 1 + (-0.5) - 2.0**2 - np.exp(-0.5))
        x = torch.zeros(1, 5, 2)
        recon = torch.zeros(1, 5, 2)
        _, _, kl_loss = vae_loss(recon, x, mu, logvar)
        assert kl_loss.item() == pytest.approx(expected_kl, rel=1e-4)

    def test_reconstruction_loss_matches_manual_sse(self):
        x = torch.zeros(2, 3, 2)
        recon = torch.ones(2, 3, 2)  # every element off by 1
        mu = torch.zeros(2, 4)
        logvar = torch.zeros(2, 4)
        _, recon_loss, _ = vae_loss(recon, x, mu, logvar)
        # sum of squared errors per sample = 3*2*1^2 = 6, averaged over batch = 6
        assert recon_loss.item() == pytest.approx(6.0)

    def test_beta_scales_kl_term_only(self):
        mu = torch.tensor([[1.0, 2.0]])
        logvar = torch.tensor([[0.5, -0.5]])
        x = torch.ones(1, 3, 2)
        recon = torch.zeros(1, 3, 2)
        total_b1, recon_loss_b1, kl_b1 = vae_loss(recon, x, mu, logvar, beta=1.0)
        total_b2, recon_loss_b2, kl_b2 = vae_loss(recon, x, mu, logvar, beta=2.0)
        assert recon_loss_b1.item() == pytest.approx(recon_loss_b2.item())
        assert total_b2.item() == pytest.approx(recon_loss_b2.item() + 2 * kl_b2.item())


class TestTrainEpoch:
    def test_reduces_loss_over_epochs(self):
        rng = np.random.default_rng(0)
        windows = rng.normal(0, 0.1, (100, 15, 3)).astype(np.float32)
        torch.manual_seed(0)
        model = TCNVAE(n_features=3, window_len=15, hidden_dim=8, latent_dim=4)
        config = TrainConfig(epochs=1, batch_size=16, lr=1e-3, beta=0.1, num_threads=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        torch_rng = np.random.default_rng(0)

        first_loss = train_epoch(model, windows, config, optimizer, torch_rng)
        for _ in range(10):
            last_loss = train_epoch(model, windows, config, optimizer, torch_rng)

        assert last_loss < first_loss


class TestEvaluate:
    def test_empty_windows_returns_nan(self):
        model = TCNVAE(n_features=3, window_len=15, hidden_dim=8, latent_dim=4)
        config = TrainConfig()
        result = evaluate(model, np.empty((0, 15, 3), dtype=np.float32), config)
        assert np.isnan(result)


class TestTrain:
    def test_builds_default_model_from_window_shape(self):
        rng = np.random.default_rng(0)
        windows = rng.normal(0, 0.1, (60, 12, 4)).astype(np.float32)
        config = TrainConfig(epochs=2, batch_size=16, num_threads=1)
        result = train(windows, config=config)
        assert result.model.window_len == 12

    def test_early_stopping_triggers_with_low_patience(self):
        rng = np.random.default_rng(0)
        train_windows = rng.normal(0, 0.1, (80, 10, 3)).astype(np.float32)
        val_windows = rng.normal(0, 0.1, (20, 10, 3)).astype(np.float32)
        config = TrainConfig(epochs=200, batch_size=16, patience=1, num_threads=1)
        result = train(train_windows, val_windows, config=config)
        assert result.stopped_early
        assert len(result.train_losses) < 200

    def test_restores_best_val_epoch_weights(self):
        rng = np.random.default_rng(0)
        train_windows = rng.normal(0, 0.1, (80, 10, 3)).astype(np.float32)
        val_windows = rng.normal(0, 0.1, (20, 10, 3)).astype(np.float32)
        config = TrainConfig(epochs=20, batch_size=16, patience=3, num_threads=1)
        result = train(train_windows, val_windows, config=config)
        final_val_loss = evaluate(result.model, val_windows, config)
        # The restored model's val loss should match (approximately) the
        # best recorded val loss, not necessarily the last epoch's.
        assert final_val_loss == pytest.approx(min(result.val_losses), rel=0.05)
