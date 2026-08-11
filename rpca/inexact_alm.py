"""From-scratch Robust PCA via Principal Component Pursuit (Candes, Li, Ma
& Wright 2011), solved by the Inexact Augmented Lagrange Multiplier (IALM)
method (Lin, Chen & Ma 2010). Decomposes M = L + S: L low-rank
(common/systematic movement across the basket), S sparse (per-ticker
idiosyncratic residual) -- see docs/architecture.md for why this replaces
plain PCA (PCA can't separate common from idiosyncratic; RPCA is built to).

    minimize  ||L||_* + lambda * ||S||_1   s.t.  L + S = M

Matrices here are small (T x N with N ~ 20-30 tickers), so plain
`numpy.linalg.svd` every iteration is fine -- no randomized SVD needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def shrink(x: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding operator: sign(x) * max(|x| - tau, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svd_threshold(M: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """Singular value thresholding operator D_tau(M) = U shrink(Sigma, tau) V^T.
    Returns the thresholded matrix and its resulting rank.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_shrunk = shrink(s, tau)
    rank = int(np.sum(s_shrunk > 0))
    return (U * s_shrunk) @ Vt, rank


@dataclass
class RobustPcaResult:
    L: np.ndarray
    S: np.ndarray
    rank: int
    n_iter: int
    converged: bool
    lam: float


def robust_pca(
    M: np.ndarray,
    lam: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 1000,
    rho: float = 1.5,
) -> RobustPcaResult:
    """Inexact ALM solver for Principal Component Pursuit.

    lam: sparsity weight; defaults to 1/sqrt(max(n, d)), the value with
    exact-recovery guarantees under Candes et al.'s (2011) random model.
    tol: convergence threshold on ||M - L - S||_F / ||M||_F.
    rho: multiplicative growth rate of the penalty parameter mu per
    iteration -- a fixed-rho simplification of IALM (vs. an adaptive
    schedule), standard in widely-used reference implementations and
    sufficient at this matrix scale.

    Note on scale: exact-recovery guarantees are asymptotic/probabilistic
    and don't tightly apply to very small matrices. A dense random
    low-rank matrix with *zero* true sparse corruption at ~50x10 scale gets
    a genuine, stable ~15% of its Frobenius norm attributed to S anyway
    (confirmed by tracing mu/rank/||S|| across iterations) -- not a bug,
    just the relaxation gap at small scale. At this project's actual scale
    (~100+ rows x ~20-30 tickers) the same test recovers near-zero S and
    exact rank, matching theory.
    """
    M = np.asarray(M, dtype=float)
    n, d = M.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))

    norm_M = np.linalg.norm(M, ord="fro")
    if norm_M == 0:
        return RobustPcaResult(L=np.zeros_like(M), S=np.zeros_like(M), rank=0, n_iter=0, converged=True, lam=lam)

    spectral_norm = np.linalg.norm(M, ord=2)
    inf_norm = np.max(np.abs(M))
    mu = 1.25 / spectral_norm
    mu_max = mu * 1e7

    Y = M / max(spectral_norm, inf_norm / lam)
    L = np.zeros_like(M)
    S = np.zeros_like(M)

    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        L, _ = svd_threshold(M - S + Y / mu, 1.0 / mu)
        S = shrink(M - L + Y / mu, lam / mu)
        residual = M - L - S
        Y = Y + mu * residual
        mu = min(mu * rho, mu_max)

        err = np.linalg.norm(residual, ord="fro") / norm_M
        if err < tol:
            converged = True
            break

    # Report rank using a fixed relative threshold on L's own singular
    # values, not the raw 1/mu SVT threshold used during optimization: mu
    # grows geometrically and can reach ~1e7x its starting value by
    # convergence, making 1/mu numerically meaningless as a rank cutoff by
    # the end (confirmed during development -- floating-point noise in L's
    # smallest singular values was crossing that vanishing threshold and
    # inflating the reported rank on inputs with no true sparse corruption
    # at all, even though ||S||/||M|| itself had already stabilized).
    singular_values = np.linalg.svd(L, compute_uv=False)
    rank = int(np.sum(singular_values > 1e-3 * singular_values[0])) if singular_values[0] > 0 else 0

    return RobustPcaResult(L=L, S=S, rank=rank, n_iter=n_iter, converged=converged, lam=lam)
