import numpy as np
import pytest

from benchmark.conformal import (
    ConformalBand,
    _order_statistic_threshold,
    calibrate_band,
    coverage_rate,
    covered,
    moving_block_bootstrap,
)


class TestMovingBlockBootstrap:
    def test_output_length_matches_n_out(self):
        x = np.arange(100.0)
        out = moving_block_bootstrap(x, block_size=10, n_out=57, seed=0)
        assert len(out) == 57

    def test_values_are_a_subset_of_the_original(self):
        x = np.arange(100.0)
        out = moving_block_bootstrap(x, block_size=10, n_out=100, seed=0)
        assert set(out.tolist()) <= set(x.tolist())

    def test_reproducible_with_same_seed(self):
        x = np.arange(200.0)
        a = moving_block_bootstrap(x, block_size=15, n_out=100, seed=42)
        b = moving_block_bootstrap(x, block_size=15, n_out=100, seed=42)
        assert np.array_equal(a, b)

    def test_different_seeds_give_different_resamples(self):
        x = np.arange(200.0)
        a = moving_block_bootstrap(x, block_size=15, n_out=100, seed=1)
        b = moving_block_bootstrap(x, block_size=15, n_out=100, seed=2)
        assert not np.array_equal(a, b)

    def test_preserves_contiguous_runs_within_a_block(self):
        # A block-bootstrap resample should contain runs of consecutive
        # original values (e.g. [5,6,7]) -- an iid resample essentially
        # never would at this array size. This is the whole point of using
        # blocks instead of single-point resampling.
        x = np.arange(500.0)
        out = moving_block_bootstrap(x, block_size=20, n_out=500, seed=3)
        diffs = np.diff(out)
        assert np.any(diffs[:19] == 1.0)  # first block is a contiguous run

    def test_raises_for_block_size_out_of_range(self):
        x = np.arange(10.0)
        with pytest.raises(ValueError):
            moving_block_bootstrap(x, block_size=0, n_out=5, seed=0)
        with pytest.raises(ValueError):
            moving_block_bootstrap(x, block_size=11, n_out=5, seed=0)


class TestOrderStatisticThreshold:
    def test_matches_expected_order_statistic(self):
        # n=9, alpha=0.1 -> k = ceil(10*0.9) = 9 -> the max.
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        assert _order_statistic_threshold(distances, alpha=0.1) == 9.0

    def test_clamps_when_target_exceeds_n(self):
        # n=5, alpha=0.01 -> k = ceil(6*0.99) = 6 > 5 -> clamp to n=5 (the max).
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _order_statistic_threshold(distances, alpha=0.01) == 5.0


class TestCalibrateBand:
    def test_returns_conformal_band_with_reference_stat(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, 1000)
        band = calibrate_band(
            reference, stat_fn=np.mean, distance_fn=lambda a, b: abs(a - b), alpha=0.1, n_bootstrap=50, block_size=20, seed=1
        )
        assert isinstance(band, ConformalBand)
        assert band.reference_stat == pytest.approx(np.mean(reference))
        assert band.threshold >= 0
        assert len(band.null_distances) == 50

    def test_threshold_is_small_for_a_stable_statistic(self):
        # The mean of a long stationary series barely moves under
        # resampling -- the calibrated threshold should be a small number,
        # not something wildly large.
        rng = np.random.default_rng(2)
        reference = rng.normal(0, 1, 5000)
        band = calibrate_band(
            reference, stat_fn=np.mean, distance_fn=lambda a, b: abs(a - b), alpha=0.1, n_bootstrap=100, block_size=25, seed=3
        )
        assert band.threshold < 0.5

    def test_raises_for_invalid_alpha(self):
        reference = np.arange(100.0)
        with pytest.raises(ValueError):
            calibrate_band(reference, stat_fn=np.mean, distance_fn=lambda a, b: abs(a - b), alpha=0.0)
        with pytest.raises(ValueError):
            calibrate_band(reference, stat_fn=np.mean, distance_fn=lambda a, b: abs(a - b), alpha=1.0)

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(4)
        reference = rng.normal(0, 1, 500)
        a = calibrate_band(
            reference, stat_fn=np.mean, distance_fn=lambda x, y: abs(x - y), alpha=0.1, n_bootstrap=30, block_size=10, seed=7
        )
        b = calibrate_band(
            reference, stat_fn=np.mean, distance_fn=lambda x, y: abs(x - y), alpha=0.1, n_bootstrap=30, block_size=10, seed=7
        )
        assert np.array_equal(a.null_distances, b.null_distances)
        assert a.threshold == b.threshold


class TestCoveredAndCoverageRate:
    def test_covered_true_within_threshold(self):
        band = ConformalBand(reference_stat=0.0, null_distances=np.array([0.1, 0.2]), threshold=0.5, alpha=0.1)
        assert covered(0.3, band)
        assert covered(0.5, band)
        assert not covered(0.6, band)

    def test_coverage_rate_computes_fraction_within_threshold(self):
        band = ConformalBand(reference_stat=0.0, null_distances=np.array([]), threshold=1.0, alpha=0.1)
        observed = np.array([0.5, 0.9, 1.0, 1.1, 2.0])
        assert coverage_rate(observed, band) == pytest.approx(3 / 5)
