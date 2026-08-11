import torch

from models.tcn_vae import CausalConv1d, TCNVAE, count_parameters


class TestCausalConv1d:
    def test_no_lookahead_leakage(self):
        # Output at time t must be unaffected by changes to inputs at
        # times > t -- the whole point of causal padding.
        torch.manual_seed(0)
        conv = CausalConv1d(3, 5, kernel_size=3, dilation=2)
        conv.eval()
        x = torch.randn(1, 3, 20)
        out1 = conv(x)

        x2 = x.clone()
        x2[:, :, 15:] = torch.randn(1, 3, 5) * 100
        out2 = conv(x2)

        assert torch.allclose(out1[:, :, :15], out2[:, :, :15])
        assert not torch.allclose(out1[:, :, 15:], out2[:, :, 15:])

    def test_output_length_matches_input_length(self):
        conv = CausalConv1d(4, 4, kernel_size=5, dilation=3)
        x = torch.randn(2, 4, 30)
        assert conv(x).shape == (2, 4, 30)


class TestTCNVAE:
    def test_forward_pass_shapes(self):
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        x = torch.randn(5, 20, 3)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape
        assert mu.shape == (5, 4)
        assert logvar.shape == (5, 4)

    def test_logvar_is_clamped(self):
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        # Push logvar's linear layer to extreme output values.
        with torch.no_grad():
            model.fc_logvar.weight.fill_(100.0)
            model.fc_logvar.bias.fill_(100.0)
        x = torch.randn(3, 20, 3)
        _, _, logvar = model(x)
        assert (logvar <= 6).all()
        assert (logvar >= -6).all()

    def test_eval_mode_is_deterministic(self):
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        model.eval()
        x = torch.randn(2, 20, 3)
        with torch.no_grad():
            recon1, mu1, logvar1 = model(x)
            recon2, mu2, logvar2 = model(x)
        assert torch.equal(recon1, recon2)  # no sampling noise in eval mode

    def test_no_lookahead_end_to_end_through_encoder(self):
        # The encoder's per-timestep hidden states must not depend on
        # future inputs either -- end-to-end causality, not just one layer.
        torch.manual_seed(0)
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        model.eval()
        x = torch.randn(1, 20, 3)
        h1 = model.encoder(x)

        x2 = x.clone()
        x2[:, 15:, :] = torch.randn(1, 5, 3) * 100
        h2 = model.encoder(x2)

        assert torch.allclose(h1[:, :15, :], h2[:, :15, :], atol=1e-5)

    def test_parameter_count_reasonable_for_input_size(self):
        model = TCNVAE(n_features=3, window_len=90, hidden_dim=16, latent_dim=8)
        n_params = count_parameters(model)
        # Small by design: 3 input channels (vs. the SSA reference model's
        # 29 PCA components) warrants a proportionally smaller model, not
        # the SSA model's ~83K-param budget. Sanity bounds, not a strict
        # target.
        assert 500 < n_params < 20_000
