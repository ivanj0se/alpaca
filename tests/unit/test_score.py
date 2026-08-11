import numpy as np
import pandas as pd
import pytest

from benchmark.cv import make_t1
from benchmark.ladder import evaluate_rung
from models.score import (
    make_fold_scorer,
    reconstruction_residual,
    reconstruction_residuals,
    standardize_windows,
)
from models.tcn_vae import TCNVAE
from models.train import TrainConfig


class TestReconstructionResiduals:
    def test_shape_matches_input(self):
        model = TCNVAE(n_features=3, window_len=15, hidden_dim=8, latent_dim=4)
        windows = np.random.default_rng(0).normal(0, 0.1, (10, 15, 3)).astype(np.float32)
        residuals = reconstruction_residuals(model, windows)
        assert residuals.shape == windows.shape

    def test_zero_for_untrained_model_reconstructing_its_own_bias(self):
        # Not a meaningful assertion about untrained-model behavior beyond
        # "doesn't crash and returns finite values."
        model = TCNVAE(n_features=2, window_len=10, hidden_dim=4, latent_dim=2)
        windows = np.zeros((3, 10, 2), dtype=np.float32)
        residuals = reconstruction_residuals(model, windows)
        assert np.isfinite(residuals).all()


class TestReconstructionResidual:
    def test_anomalous_window_scores_higher_than_normal(self):
        rng = np.random.default_rng(0)
        train_windows = rng.normal(0, 0.01, (200, 15, 3)).astype(np.float32)
        from models.train import train

        result = train(train_windows, config=TrainConfig(epochs=20, batch_size=32, beta=0.1, num_threads=1))

        normal = rng.normal(0, 0.01, (10, 15, 3)).astype(np.float32)
        anomalous = rng.normal(0, 0.5, (10, 15, 3)).astype(np.float32)
        normal_scores = reconstruction_residual(result.model, normal)
        anomalous_scores = reconstruction_residual(result.model, anomalous)
        assert anomalous_scores.mean() > normal_scores.mean() * 10

    def test_output_shape_is_one_per_window(self):
        model = TCNVAE(n_features=3, window_len=10, hidden_dim=8, latent_dim=4)
        windows = np.random.default_rng(0).normal(0, 0.1, (7, 10, 3)).astype(np.float32)
        scores = reconstruction_residual(model, windows)
        assert scores.shape == (7,)
        assert (scores >= 0).all()


class TestStandardizeWindows:
    def test_train_output_has_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        train_windows = rng.normal(5, 2, (100, 10, 3)).astype(np.float32)
        test_windows = rng.normal(5, 2, (20, 10, 3)).astype(np.float32)
        train_std, _ = standardize_windows(train_windows, test_windows)
        assert train_std.mean() == pytest.approx(0.0, abs=0.05)
        assert train_std.std() == pytest.approx(1.0, abs=0.05)

    def test_test_windows_use_train_statistics_not_their_own(self):
        train_windows = np.zeros((10, 5, 2), dtype=np.float32)
        train_windows[..., 0] = 10.0  # channel 0 mean=10, std=0 in train
        test_windows = np.zeros((3, 5, 2), dtype=np.float32)
        test_windows[..., 0] = 20.0  # very different in test

        train_std, test_std = standardize_windows(train_windows, test_windows)
        # channel 0's train std is ~0, so eps floors it -- test values,
        # centered on TRAIN's mean (10), should come out large, not
        # normalized against their own (different) statistics.
        assert np.all(test_std[..., 0] > 1e6)  # (20-10)/eps is huge

    def test_zero_variance_channel_does_not_produce_nan(self):
        train_windows = np.ones((10, 5, 2), dtype=np.float32) * 3.0
        test_windows = np.ones((3, 5, 2), dtype=np.float32) * 3.0
        train_std, test_std = standardize_windows(train_windows, test_windows)
        assert not np.isnan(train_std).any()
        assert not np.isnan(test_std).any()


class TestMakeFoldScorer:
    def _windows(self, n=300, window_len=15, seed=0):
        rng = np.random.default_rng(seed)
        # 3 channels at deliberately different scales, matching the real
        # log_return/realized_vol/volume_zscore mismatch found on real data.
        log_return = rng.normal(0, 0.001, (n, window_len))
        realized_vol = rng.normal(0.0007, 0.0002, (n, window_len))
        volume_zscore = rng.normal(0, 1.0, (n, window_len))
        return np.stack([log_return, realized_vol, volume_zscore], axis=-1).astype(np.float32)

    def test_returns_none_for_too_short_training_fold(self):
        windows = self._windows(n=50)
        scorer = make_fold_scorer(windows, config=TrainConfig(epochs=2, num_threads=1))
        assert scorer(np.arange(0, 5), np.arange(5, 10)) is None

    def test_returns_finite_score_on_a_reasonable_fold(self):
        windows = self._windows(n=200)
        scorer = make_fold_scorer(windows, config=TrainConfig(epochs=3, batch_size=32, num_threads=1))
        n = len(windows)
        score = scorer(np.arange(0, n - 40), np.arange(n - 40, n))
        assert score is not None
        assert np.isfinite(score)

    def test_score_is_on_primary_channel_raw_scale_not_standardized(self):
        # A sanity magnitude check: NLL computed on the raw tiny-scale
        # log_return channel should land somewhere in the same rough
        # ballpark as GARCH's (very negative, since log(2*pi*tiny_var) is
        # large negative) -- not near 0 or positive, which would indicate
        # the pooled/standardized-scale bug this design specifically fixes.
        windows = self._windows(n=250)
        scorer = make_fold_scorer(windows, config=TrainConfig(epochs=5, batch_size=32, num_threads=1))
        n = len(windows)
        score = scorer(np.arange(0, n - 40), np.arange(n - 40, n))
        assert score < -3.0

    def test_integrates_with_evaluate_rung(self):
        windows = self._windows(n=300)
        idx = pd.date_range("2026-01-02 09:30", periods=len(windows), freq="1min", tz="UTC")
        t1 = make_t1(idx, pd.Timedelta(minutes=1))
        result = evaluate_rung(
            "tcn_vae",
            make_fold_scorer(windows, config=TrainConfig(epochs=3, batch_size=32, num_threads=1)),
            t1,
            n_splits=4,
            embargo_td=pd.Timedelta(minutes=5),
        )
        assert result.n_folds > 0
        assert np.isfinite(result.mean_nll)
