"""Applies robust_pca to rolling windows of a cross-sectional (time x
ticker) return matrix, producing a time-varying low-rank/sparse
decomposition and a per-ticker anomaly score from the sparse component.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpca.inexact_alm import robust_pca


def rolling_rpca_decompose(
    feature_matrix: pd.DataFrame,
    window_len: int,
    step: int = 1,
    **rpca_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """feature_matrix: index=timestamp, columns=tickers (e.g. log returns).

    For each window [start, start+window_len), runs robust_pca and records
    only the *last* row's L/S values as the decomposition estimate for that
    window's end timestamp -- one decomposition per output timestamp, lined
    up 1:1 with input rows from the end of the first window onward (earlier
    rows have no full window yet and are omitted, not NaN-filled).

    Re-running RPCA on an overlapping window for every step is the
    standard, correct way to do this (same idea as a rolling z-score, just
    a matrix decomposition instead of a scalar). At this project's matrix
    scale (T x ~20-30 tickers) each fit is fast, but a full dataset at
    step=1 still means one fit per row -- raise `step` if that's too slow
    for a given use case.

    IMPORTANT for anomaly scoring specifically: with step > 1, only every
    step-th row gets its own score (the ones landing exactly on a window's
    last row) -- an anomaly at a non-strided timestamp is scored as part of
    a *later* window's last-row value, not at its own timestamp, and won't
    show up if you look up its own index directly (confirmed the hard way
    during development: with step=5 an injected single-ticker anomaly
    looked like noise because the score at its own timestamp was never
    computed; step=1 correctly ranked it #1 of 20 by a wide margin at its
    exact timestamp). Use step=1 (the default) whenever you need a score
    at every timestamp, e.g. for real anomaly detection; only raise it for
    a downstream consumer that genuinely wants a strided/subsampled series.
    """
    if window_len > len(feature_matrix):
        raise ValueError("window_len cannot exceed the number of rows in feature_matrix")
    if window_len < 2:
        raise ValueError("window_len must be at least 2")

    tickers = feature_matrix.columns
    values = feature_matrix.to_numpy()
    n_rows = len(feature_matrix)

    l_rows, s_rows, out_index = [], [], []
    for start in range(0, n_rows - window_len + 1, step):
        end = start + window_len
        window = values[start:end]
        result = robust_pca(window, **rpca_kwargs)
        l_rows.append(result.L[-1])
        s_rows.append(result.S[-1])
        out_index.append(feature_matrix.index[end - 1])

    L_df = pd.DataFrame(l_rows, index=out_index, columns=tickers)
    S_df = pd.DataFrame(s_rows, index=out_index, columns=tickers)
    return L_df, S_df


def make_fold_scorer(feature_matrix: pd.DataFrame, rank_threshold: float = 1e-3, **rpca_kwargs):
    """Adapter for benchmark/ladder.py's evaluate_rung -- Rung 3's entry in
    the cross-sectional lane (must beat Rung 2b's factor model). Fits
    Robust PCA on the training fold to learn a low-rank subspace from
    L_train's right singular vectors (robust in the sense that
    idiosyncratic/outlier rows were excluded from contaminating the
    subspace estimate via the sparse term), then scores held-out rows by
    projecting them onto that subspace and computing the Gaussian NLL of
    the residual against the in-sample per-ticker residual variance (from
    S_train) -- the same NLL form and homoskedastic-per-column assumption
    as the factor model's scorer, so the two are comparable within the
    cross-sectional lane.
    """
    values = feature_matrix.to_numpy()

    def score(train_idx: np.ndarray, test_idx: np.ndarray) -> float | None:
        if len(train_idx) < 20 or len(test_idx) == 0:
            return None
        train = values[train_idx]
        test = values[test_idx]

        result = robust_pca(train, **rpca_kwargs)
        if not result.converged:
            return None

        U, s, Vt = np.linalg.svd(result.L, full_matrices=False)
        keep = s > rank_threshold * s[0] if s[0] > 0 else np.zeros_like(s, dtype=bool)
        V = Vt[keep].T  # (n_tickers, effective_rank)

        residual_var = result.S.var(axis=0, ddof=1)
        residual_var = np.maximum(residual_var, 1e-12)

        projected_test = test @ V @ V.T if V.shape[1] > 0 else np.zeros_like(test)
        test_residuals = test - projected_test

        nlls = 0.5 * (np.log(2 * np.pi * residual_var) + test_residuals**2 / residual_var)
        return float(np.mean(nlls))

    return score


def sparse_anomaly_score(S_df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker, per-timestamp anomaly score from the sparse component:
    the absolute sparse residual, z-scored per ticker across the S_df
    history so scores are comparable across tickers with different scales.
    A ticker with zero variance in S (never had a nonzero sparse residual)
    scores 0 throughout rather than dividing by zero.
    """
    std = S_df.std(ddof=1)
    std_safe = std.replace(0, np.nan)
    return S_df.abs().div(std_safe, axis=1).fillna(0.0)
