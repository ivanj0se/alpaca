"""Small TCN-VAE for the temporal lane (Rung 4): causal dilated-convolution
encoder/decoder with a standard Gaussian variational bottleneck. Same
architectural family as the SSA project's TCN-VAE, but considerably
smaller (3 input channels here -- log_return, realized_vol, volume_zscore
-- vs. 29 PCA components there) and without that project's dual-decoder
USAD/normalizing-flow additions, which aren't needed for this simpler
single-instrument reconstruction baseline. Reconstruction residual is the
anomaly score (models/score.py); this is the temporal lane's nonlinear
upgrade over Rung 2a's GARCH (Rung 0 -> GARCH -> TCN-VAE, each must beat
the previous within the lane -- see benchmark/ladder.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D convolution padded only on the left, so output at time t depends
    only on inputs at time <= t (no lookahead).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[..., : -self.padding]
        return out


class TemporalBlock(nn.Module):
    """Dilated causal conv + residual connection (a "TCN block")."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.activation(self.conv(x)))
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 16, kernel_size: int = 3, dilations=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_ch = n_features
        for d in dilations:
            blocks.append(TemporalBlock(in_ch, hidden_dim, kernel_size, dilation=d, dropout=dropout))
            in_ch = hidden_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> conv wants (B, C, T)
        h = self.blocks(x.transpose(1, 2))
        return h.transpose(1, 2)  # (B, T, hidden_dim)


class TCNDecoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 16, kernel_size: int = 3, dilations=(1, 2), dropout: float = 0.1):
        super().__init__()
        blocks = []
        in_ch = hidden_dim
        for d in dilations:
            blocks.append(TemporalBlock(in_ch, hidden_dim, kernel_size, dilation=d, dropout=dropout))
            in_ch = hidden_dim
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(hidden_dim, n_features, 1)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B, T, hidden_dim)
        h = self.blocks(z_seq.transpose(1, 2))
        out = self.proj(h)
        return out.transpose(1, 2)  # (B, T, n_features)


class TCNVAE(nn.Module):
    """Encoder produces a per-timestep hidden representation, mean-pooled
    over time into a single latent distribution per window (mu, logvar).
    The decoder combines the sampled latent (broadcast across the window)
    with a skip connection carrying the encoder's own per-timestep hidden
    states. logvar is clamped to [-6, 6] for numerical stability (matches
    the SSA project's convention).

    The skip connection fixes a real, measured problem with the
    pool-then-broadcast-only design: reconstructing a 90-step window from
    a single global latent, broadcast identically to every timestep before
    a translation-equivariant causal-conv decoder, structurally limits the
    decoder to near-constant output across the window regardless of how
    the encoder is tuned. Confirmed on real AAPL data: reconstructed
    log_return had std ~0.01-0.015 across a window vs. the actual data's
    ~0.5-0.72 -- a 50-70x gap, i.e. an almost flat line. This is the
    well-known "VAE reconstructions are blurry/oversmoothed" failure mode
    (pure-bottleneck VAEs discard local detail), and skip connections from
    encoder to decoder are the standard fix in reconstruction-focused (as
    opposed to generation-focused) VAEs. The encoder's own limited
    receptive field (~15 timesteps per position, see
    diagnostics/2026-08-11-tcn-vae-receptive-field-mismatch/) means the
    skip path still only carries *local* per-position context, not the
    whole window -- it isn't a trivial identity shortcut, and the global
    latent still has to capture the window's overall regime.
    """

    def __init__(
        self,
        n_features: int = 3,
        window_len: int = 90,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        encoder_dilations: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.window_len = window_len
        self.encoder = TCNEncoder(n_features, hidden_dim, kernel_size, dilations=encoder_dilations, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.skip_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = TCNDecoder(n_features, hidden_dim, kernel_size, dilations=(1, 2), dropout=dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_seq = self.encoder(x)  # (B, T, hidden_dim) -- also reused as the skip-connection signal
        pooled = h_seq.mean(dim=1)  # (B, hidden_dim)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled).clamp(-6, 6)
        return mu, logvar, h_seq

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, h_seq: torch.Tensor) -> torch.Tensor:
        h_global = self.fc_latent_to_hidden(z)  # (B, hidden_dim)
        h_global_seq = h_global.unsqueeze(1).expand(-1, self.window_len, -1)  # broadcast across time
        fused = self.skip_fusion(torch.cat([h_global_seq, h_seq], dim=-1))  # (B, T, hidden_dim)
        return self.decoder(fused)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, h_seq = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, h_seq)
        return recon, mu, logvar


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
