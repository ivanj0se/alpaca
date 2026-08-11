"""Hand-built toy timestamp examples for benchmark/cv.py. This is the
highest-value test in the repo: every rung's evaluation depends on purging
and embargo being correct, so bugs here silently invalidate every downstream
result rather than raising an error.
"""

import numpy as np
import pandas as pd
import pytest

from benchmark.cv import CombinatorialPurgedKFold, PurgedKFold, make_t1


def _toy_t1(n=40, freq_minutes=1, label_horizon_minutes=5):
    start = pd.Timestamp("2026-01-02 09:30", tz="UTC")
    times = pd.DatetimeIndex([start + pd.Timedelta(minutes=i * freq_minutes) for i in range(n)])
    return make_t1(times, pd.Timedelta(minutes=label_horizon_minutes))


class TestMakeT1:
    def test_t1_offset_by_horizon(self):
        t1 = _toy_t1(n=5, label_horizon_minutes=5)
        assert (t1.to_numpy() == (t1.index + pd.Timedelta(minutes=5)).to_numpy()).all()


class TestPurgedKFold:
    def test_train_test_disjoint_every_fold(self):
        t1 = _toy_t1()
        cv = PurgedKFold(t1, n_splits=4, embargo_td=pd.Timedelta(0))
        for fold in cv.split():
            assert set(fold.train_idx).isdisjoint(set(fold.test_idx))

    def test_no_purge_with_point_in_time_labels(self):
        # Point-in-time labels (t1 == t0, horizon 0): a sample's window is a
        # single instant, so it only overlaps the test fold if it falls
        # inside [test_start, test_end] -- i.e. only test-set samples
        # themselves. Nothing else should be purged.
        t1 = _toy_t1(n=40, freq_minutes=1, label_horizon_minutes=0)
        cv = PurgedKFold(t1, n_splits=4, embargo_td=pd.Timedelta(0))
        n = len(t1)
        for fold in cv.split():
            expected_train = set(range(n)) - set(fold.test_idx)
            assert set(fold.train_idx) == expected_train

    def test_boundary_sample_touching_test_end_is_conservatively_purged(self):
        # With a 1-min horizon and 1-min sample spacing, the sample
        # immediately after a test fold has t0 exactly equal to the fold's
        # last label end time. The inclusive overlap check purges it --
        # documenting this as intentional (conservative) rather than a gap.
        t1 = _toy_t1(n=40, freq_minutes=1, label_horizon_minutes=1)
        cv = PurgedKFold(t1, n_splits=4, embargo_td=pd.Timedelta(0))
        first_fold = next(cv.split())
        boundary_sample = first_fold.test_idx[-1] + 1
        assert boundary_sample not in first_fold.train_idx

    def test_purge_drops_overlapping_training_samples(self):
        # Label horizon (10 min) is long relative to sample spacing (1 min):
        # samples immediately before a test fold should be purged because
        # their label window [t0, t0+10min] overlaps the test fold's start.
        t1 = _toy_t1(n=40, freq_minutes=1, label_horizon_minutes=10)
        cv = PurgedKFold(t1, n_splits=4, embargo_td=pd.Timedelta(0))
        folds = list(cv.split())
        # Second fold's test set starts right after the first fold ends;
        # some of fold 1's own samples (near its end) have labels that
        # extend past fold 2's start and must not appear in fold 2's train.
        fold2 = folds[1]
        test_start = t1.index[fold2.test_idx[0]]
        for i in fold2.train_idx:
            t0, t1_i = t1.index[i], t1.iloc[i]
            test_end = t1.iloc[fold2.test_idx].max()
            overlaps = (t0 <= test_end) and (t1_i >= test_start)
            assert not overlaps, f"sample {i} should have been purged"

    def test_embargo_drops_samples_immediately_after_test_fold(self):
        t1 = _toy_t1(n=40, freq_minutes=1, label_horizon_minutes=1)
        no_embargo = PurgedKFold(t1, n_splits=4, embargo_td=pd.Timedelta(0))
        with_embargo = PurgedKFold(t1, n_splits=4, embargo_td=pd.Timedelta(minutes=5))

        no_embargo_folds = list(no_embargo.split())
        with_embargo_folds = list(with_embargo.split())

        # Embargo should never add training samples, only remove them.
        for f_ne, f_we in zip(no_embargo_folds, with_embargo_folds):
            assert set(f_we.train_idx) <= set(f_ne.train_idx)

        # At least one fold should actually lose samples to the embargo
        # (the fold whose test set is followed by more data).
        removed_any = any(
            len(f_we.train_idx) < len(f_ne.train_idx)
            for f_ne, f_we in zip(no_embargo_folds, with_embargo_folds)
        )
        assert removed_any

    def test_rejects_unsorted_index(self):
        t1 = _toy_t1()
        shuffled = t1.iloc[np.random.RandomState(0).permutation(len(t1))]
        with pytest.raises(ValueError):
            PurgedKFold(shuffled, n_splits=4)

    def test_all_samples_appear_in_exactly_one_test_fold(self):
        t1 = _toy_t1(n=40)
        cv = PurgedKFold(t1, n_splits=4)
        all_test = np.concatenate([f.test_idx for f in cv.split()])
        assert sorted(all_test) == list(range(len(t1)))


class TestCombinatorialPurgedKFold:
    def test_number_of_splits_matches_combinatorics(self):
        t1 = _toy_t1(n=60)
        cv = CombinatorialPurgedKFold(t1, n_groups=6, n_test_groups=2)
        folds = list(cv.split())
        assert len(folds) == cv.n_splits == 15  # C(6, 2)

    def test_train_test_disjoint_every_split(self):
        t1 = _toy_t1(n=60)
        cv = CombinatorialPurgedKFold(t1, n_groups=6, n_test_groups=2, embargo_td=pd.Timedelta(minutes=3))
        for fold in cv.split():
            assert set(fold.train_idx).isdisjoint(set(fold.test_idx))

    def test_rejects_invalid_group_counts(self):
        t1 = _toy_t1(n=10)
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(t1, n_groups=4, n_test_groups=4)
        with pytest.raises(ValueError):
            CombinatorialPurgedKFold(t1, n_groups=4, n_test_groups=0)
