import numpy as np
import pytest
import torch

from models.forecaster_train import ForecasterTrainConfig, evaluate, forecaster_nll_loss, train, train_epoch
from models.tcn_forecaster import TCNForecaster


class TestForecasterNllLoss:
    def test_matches_closed_form_for_known_values(self):
        mean = torch.tensor([[0.0]])
        logvar = torch.tensor([[0.0]])  # var=1
        target = torch.tensor([[1.0]])
        # 0.5 * (log(2*pi*1) + (1-0)^2/1)
        expected = 0.5 * (np.log(2 * np.pi) + 1.0)
        assert forecaster_nll_loss(mean, logvar, target).item() == pytest.approx(expected, abs=1e-5)

    def test_lower_loss_for_correct_mean(self):
        target = torch.tensor([[2.0]])
        logvar = torch.tensor([[0.0]])
        loss_correct = forecaster_nll_loss(torch.tensor([[2.0]]), logvar, target)
        loss_wrong = forecaster_nll_loss(torch.tensor([[0.0]]), logvar, target)
        assert loss_correct.item() < loss_wrong.item()


class TestTrainEpochAndEvaluate:
    def test_train_epoch_reduces_loss_over_several_epochs(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        windows = rng.normal(0, 0.01, size=(200, 20, 2)).astype(np.float32)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        config = ForecasterTrainConfig(epochs=1, batch_size=16, lr=1e-2, seed=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        losses = [train_epoch(model, windows, config, optimizer, np.random.default_rng(i)) for i in range(5)]
        assert losses[-1] < losses[0]

    def test_evaluate_returns_nan_for_empty_windows(self):
        model = TCNForecaster(n_features=2, hidden_dim=4, dilations=(1,))
        result = evaluate(model, np.empty((0, 10, 2), dtype=np.float32), ForecasterTrainConfig())
        assert np.isnan(result)


class TestTrainRecoversKnownProcess:
    def test_recovers_true_conditional_mean_on_a_known_ar1_process(self):
        # y_{t+1} = phi*y_t + noise. A model that correctly learns the
        # shift-by-one alignment should predict mean(t) ~= phi*y_t. This
        # is the regression guard against an off-by-one in the shift,
        # which would make training loss look fine for the wrong reason
        # (predicting the CURRENT step instead of the next one) -- mirrors
        # TestSimulateRefitRecover in test_hawkes.py.
        rng = np.random.default_rng(0)
        phi, sigma = 0.5, 0.01
        n = 6000
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.normal(0, sigma)
        vol = np.full(n, sigma)  # uninformative second channel
        data = np.stack([y, vol], axis=1).astype(np.float32)

        window_len = 30
        windows = np.stack([data[i : i + window_len] for i in range(0, n - window_len, 5)])

        config = ForecasterTrainConfig(epochs=40, batch_size=32, lr=5e-3, patience=15, seed=0)
        result = train(windows, val_windows=windows[-50:], config=config)
        model = result.model
        model.eval()

        test_windows = windows[-30:]
        x = torch.from_numpy(test_windows).float()
        mean, _ = model(x)
        predicted_mean = mean[:, -2].detach().numpy()
        actual_prev = test_windows[:, -2, 0]
        expected_mean = phi * actual_prev

        corr = np.corrcoef(predicted_mean, expected_mean)[0, 1]
        assert corr > 0.7
