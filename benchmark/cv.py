"""Purged, embargoed cross-validation for time-series data with overlapping
labels, per Lopez de Prado, *Advances in Financial Machine Learning* (2018),
ch. 7 and ch. 12.

Every rung of the benchmark ladder (random walk, GARCH, Robust PCA, TCN-VAE)
evaluates through this module. A naive train/test split leaks information
whenever a training sample's label window overlaps the test period -- that
training sample was effectively computed using data that "sees the future"
relative to the test set. Purging drops those overlapping training samples;
embargo additionally drops a buffer immediately after each test fold, since
serial correlation lets information leak forward too.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def _purge_and_embargo_mask(
    sample_t0: np.ndarray,
    sample_t1: np.ndarray,
    test_idx: np.ndarray,
    test_start_time,
    test_end_time,
    embargo_td: pd.Timedelta,
) -> np.ndarray:
    """Boolean train-eligibility mask over all samples, given one test fold."""
    n = len(sample_t0)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    # Purge: drop any training sample whose [t0, t1] label window overlaps
    # [test_start_time, test_end_time].
    overlaps = (sample_t0 <= test_end_time) & (sample_t1 >= test_start_time)
    train_mask &= ~overlaps

    # Embargo: drop training samples starting within embargo_td after the
    # test fold's last label resolves.
    embargo_end_time = test_end_time + embargo_td
    in_embargo = (sample_t0 > test_end_time) & (sample_t0 <= embargo_end_time)
    train_mask &= ~in_embargo

    return train_mask


class PurgedKFold:
    """K-fold CV where each sample carries a label window [t0, t1]; training
    samples whose window overlaps a test fold, or fall within the embargo
    period right after it, are excluded from that fold's training set.

    Parameters
    ----------
    t1 : pd.Series
        Index = sample start time (t0), values = label end time (t1) for
        that sample. Must be sorted ascending by index. For a point-in-time
        label (t1 == t0), purging reduces to a hard time boundary.
    n_splits : int
    embargo_td : pd.Timedelta
        Extra buffer dropped from training immediately after each test fold.
    """

    def __init__(
        self,
        t1: pd.Series,
        n_splits: int = 6,
        embargo_td: pd.Timedelta = pd.Timedelta(0),
    ):
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 index (sample start times) must be sorted ascending")
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.t1 = t1
        self.n_splits = n_splits
        self.embargo_td = embargo_td

    def split(self):
        n = len(self.t1)
        indices = np.arange(n)
        fold_groups = np.array_split(indices, self.n_splits)
        sample_t0 = self.t1.index.to_numpy()
        sample_t1 = self.t1.to_numpy()

        for test_idx in fold_groups:
            if len(test_idx) == 0:
                continue
            test_start_time = sample_t0[test_idx[0]]
            test_end_time = sample_t1[test_idx].max()

            train_mask = _purge_and_embargo_mask(
                sample_t0, sample_t1, test_idx, test_start_time, test_end_time, self.embargo_td
            )
            yield Fold(train_idx=indices[train_mask], test_idx=test_idx)


class CombinatorialPurgedKFold:
    """Combinatorial Purged CV (CPCV), Lopez de Prado ch. 12: split into
    n_groups groups, form every C(n_groups, n_test_groups) combination of
    groups as a combined test set, purge/embargo around it. Produces more
    train/test splits than plain k-fold from the same data, at the cost of
    overlapping test sets across splits -- by design, since the point is to
    estimate a distribution of backtest outcomes rather than a single path.

    This covers the purge/embargo/combination-generation core, which is what
    the benchmark ladder needs (multiple leakage-free train/test estimates
    per rung). Full backtest-path reconstruction (assembling per-path
    out-of-sample series from overlapping splits, ch. 12) is not
    implemented -- each split's test metrics are aggregated independently,
    which is sufficient for "does rung N beat rung N-1."
    """

    def __init__(
        self,
        t1: pd.Series,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_td: pd.Timedelta = pd.Timedelta(0),
    ):
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 index (sample start times) must be sorted ascending")
        if not (0 < n_test_groups < n_groups):
            raise ValueError("require 0 < n_test_groups < n_groups")
        self.t1 = t1
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_td = embargo_td

    def split(self):
        n = len(self.t1)
        indices = np.arange(n)
        groups = np.array_split(indices, self.n_groups)
        sample_t0 = self.t1.index.to_numpy()
        sample_t1 = self.t1.to_numpy()

        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_idx = np.concatenate([groups[g] for g in sorted(combo)])
            test_idx.sort()

            test_start_time = sample_t0[test_idx[0]]
            test_end_time = sample_t1[test_idx].max()

            train_mask = _purge_and_embargo_mask(
                sample_t0, sample_t1, test_idx, test_start_time, test_end_time, self.embargo_td
            )
            yield Fold(train_idx=indices[train_mask], test_idx=test_idx)

    @property
    def n_splits(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)


def make_t1(sample_times: pd.DatetimeIndex, label_horizon: pd.Timedelta) -> pd.Series:
    """Convenience for point-in-time samples with a fixed forward label
    horizon (e.g. a window that reconstructs/predicts `label_horizon` into
    the future): t1 = t0 + label_horizon for every sample.
    """
    return pd.Series(sample_times + label_horizon, index=sample_times)
