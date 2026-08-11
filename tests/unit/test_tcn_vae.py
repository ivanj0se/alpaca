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


class TestSkipConnection:
    """Regression coverage for the flat-reconstruction bug: pooling the
    encoder's per-timestep output into a single latent, then broadcasting
    it identically to every position before a translation-equivariant
    decoder, structurally limits reconstructions to near-constant output
    regardless of encoder tuning -- confirmed on real AAPL data
    (reconstructed log_return std ~0.01-0.015 vs. actual ~0.5-0.72, a
    50-70x gap). Fixed with a skip connection carrying the encoder's own
    per-timestep hidden states into the decoder alongside the broadcast
    global latent. See diagnostics/2026-08-11-tcn-vae-flat-reconstruction/.
    """

    def test_encode_returns_per_timestep_hidden_states(self):
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        x = torch.randn(2, 20, 3)
        mu, logvar, h_seq = model.encode(x)
        assert h_seq.shape == (2, 20, 8)

    def test_decode_requires_skip_signal(self):
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        z = torch.randn(2, 4)
        h_seq = torch.randn(2, 20, 8)
        recon = model.decode(z, h_seq)
        assert recon.shape == (2, 20, 3)

    def test_skip_path_is_causal_per_position(self):
        # The skip signal (h_seq) must retain the encoder's own causality:
        # unaffected by changes to later positions in the same window.
        torch.manual_seed(0)
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        model.eval()
        x = torch.randn(1, 20, 3)
        with torch.no_grad():
            _, _, h_seq1 = model.encode(x)
        x2 = x.clone()
        x2[:, 15:, :] = torch.randn(1, 5, 3) * 100
        with torch.no_grad():
            _, _, h_seq2 = model.encode(x2)
        assert torch.allclose(h_seq1[:, :15, :], h_seq2[:, :15, :], atol=1e-5)

    def test_global_latent_is_not_position_causal(self):
        # Documented honestly, not silently assumed: mu pools over the
        # *whole* window (mean over all timesteps), so it depends on
        # "future" positions too -- this is pre-existing pooling behavior,
        # not something the skip connection introduces or worsens. The
        # window is the unit of analysis for reconstruction-based anomaly
        # detection (the whole window is given, not forecast step by step),
        # but this makes the temporal-lane NLL comparison against GARCH
        # (which is genuinely one-step causal) an approximation, not a
        # strictly like-for-like comparison -- see
        # diagnostics/2026-08-11-tcn-vae-flat-reconstruction/.
        torch.manual_seed(0)
        model = TCNVAE(n_features=3, window_len=20, hidden_dim=8, latent_dim=4)
        model.eval()
        x = torch.randn(1, 20, 3)
        with torch.no_grad():
            mu1, _, _ = model.encode(x)
        x2 = x.clone()
        x2[:, 15:, :] = torch.randn(1, 5, 3) * 100
        with torch.no_grad():
            mu2, _, _ = model.encode(x2)
        assert not torch.allclose(mu1, mu2, atol=1e-5)

    def test_model_can_produce_genuinely_time_varying_reconstructions(self):
        # The actual regression test for the bug: an untrained model with
        # random weights should still be *capable* of non-constant output
        # across the window (capability, not learned quality -- training
        # quality is validated separately against real data).
        torch.manual_seed(1)
        model = TCNVAE(n_features=3, window_len=30, hidden_dim=8, latent_dim=4)
        model.eval()
        x = torch.randn(1, 30, 3)
        with torch.no_grad():
            recon, _, _ = model(x)
        # Std across time (not batch) for a single sample/channel -- a
        # flat-reconstruction bug would make this collapse near zero.
        assert recon[0, :, 0].std().item() > 1e-3
