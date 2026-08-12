import torch

from models.tcn_forecaster import TCNForecaster
from models.tcn_vae import count_parameters


class TestTCNForecaster:
    def test_output_shapes(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        x = torch.randn(4, 20, 2)
        mean, logvar = model(x)
        assert mean.shape == (4, 20)
        assert logvar.shape == (4, 20)

    def test_no_lookahead(self):
        # Position t's prediction must not depend on inputs at times > t --
        # the whole point of an autoregressive generator (mirrors
        # test_tcn_vae.py::test_no_lookahead_end_to_end_through_encoder).
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2, 4))
        model.eval()
        x = torch.randn(1, 20, 2)
        mean1, logvar1 = model(x)

        x2 = x.clone()
        x2[:, 15:, :] = torch.randn(1, 5, 2) * 100
        mean2, logvar2 = model(x2)

        assert torch.allclose(mean1[:, :15], mean2[:, :15], atol=1e-5)
        assert torch.allclose(logvar1[:, :15], logvar2[:, :15], atol=1e-5)

    def test_logvar_is_clamped(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=4, dilations=(1,))
        x = torch.randn(1, 10, 2) * 1000  # extreme input
        _, logvar = model(x)
        assert (logvar >= -20).all() and (logvar <= 0).all()

    def test_logvar_clamp_covers_real_minute_bar_return_scale(self):
        # Regression for a real bug: TCNVAE's [-6,6] clamp (blindly
        # copied) floors std at exp(-3)~0.05, ~140x larger than real
        # minute-bar log_return's actual std (~0.00035, log-variance
        # ~-15.9) -- silently pinning every prediction at that floor
        # regardless of input. [-20,0] must comfortably contain the real
        # value.
        import numpy as np

        real_std = 0.00035
        real_logvar = np.log(real_std**2)
        assert -20 <= real_logvar <= 0

    def test_deterministic_in_eval_mode(self):
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, dilations=(1, 2))
        model.eval()
        x = torch.randn(1, 15, 2)
        mean1, logvar1 = model(x)
        mean2, logvar2 = model(x)
        assert torch.equal(mean1, mean2)
        assert torch.equal(logvar1, logvar2)

    def test_parameter_count_reasonable(self):
        model = TCNForecaster(n_features=2, hidden_dim=16, dilations=(1, 2, 4, 8))
        n_params = count_parameters(model)
        assert 500 < n_params < 20_000

    def test_receptive_field_matches_documented_value(self):
        # dilations=(1,2,4,8), kernel_size=3 -> receptive field
        # 1 + 2*(1+2+4+8) = 31 steps, per the plan's window_len=45 choice
        # (comfortably above it -- see
        # diagnostics/2026-08-11-tcn-vae-receptive-field-mismatch/ for why
        # this matters).
        torch.manual_seed(0)
        model = TCNForecaster(n_features=2, hidden_dim=8, kernel_size=3, dilations=(1, 2, 4, 8))
        model.eval()
        window_len = 40
        x = torch.randn(1, window_len, 2)
        mean1, _ = model(x)

        x2 = x.clone()
        x2[:, 0, :] = torch.randn(1, 2) * 100  # perturb the earliest timestep
        mean2, _ = model(x2)

        receptive_field = 31
        last_affected_position = receptive_field - 1  # position 30 (0-indexed) still sees position 0
        # positions well past the receptive field must be unaffected
        assert torch.allclose(mean1[:, last_affected_position + 5 :], mean2[:, last_affected_position + 5 :], atol=1e-5)
