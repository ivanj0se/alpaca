import numpy as np
import pytest

from benchmark.stylized_facts import (
    FACT_NAMES,
    StylizedFactsSummary,
    acf,
    aggregational_gaussianity_curve,
    compute_stylized_facts,
    excess_kurtosis,
    fact_distance,
    leverage_effect_curve,
    raw_return_acf_curve,
    volatility_clustering_curve,
)


class TestAcf:
    def test_near_zero_for_iid_white_noise(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 20000)
        result = acf(x, max_lag=10)
        assert np.all(np.abs(result) < 0.03)

    def test_constant_series_returns_zeros_not_nan(self):
        x = np.full(50, 3.0)
        result = acf(x, max_lag=5)
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))

    def test_raises_when_not_enough_data(self):
        with pytest.raises(ValueError):
            acf(np.arange(5.0), max_lag=10)

    def test_returns_max_lag_elements(self):
        rng = np.random.default_rng(1)
        result = acf(rng.normal(0, 1, 500), max_lag=7)
        assert len(result) == 7


class TestVolatilityClusteringCurve:
    def test_higher_than_iid_for_a_regime_switching_process(self):
        # Construct a synthetic process with real volatility clustering
        # (alternating high/low-vol regimes in contiguous blocks) and
        # confirm the |return| ACF picks it up, vs. plain iid noise.
        rng = np.random.default_rng(2)
        block = 200
        n_blocks = 40
        vols = np.repeat(rng.choice([0.5, 3.0], size=n_blocks), block)
        clustered = rng.normal(0, 1, len(vols)) * vols
        iid = rng.normal(0, vols.std(), len(vols))

        clustered_acf = volatility_clustering_curve(clustered, max_lag=20)
        iid_acf = volatility_clustering_curve(iid, max_lag=20)
        assert clustered_acf.mean() > iid_acf.mean()
        assert clustered_acf.mean() > 0.1  # a real, not just marginal, effect


class TestExcessKurtosis:
    def test_near_zero_for_large_gaussian_sample(self):
        rng = np.random.default_rng(3)
        assert abs(excess_kurtosis(rng.normal(0, 1, 50000))) < 0.1

    def test_positive_for_fat_tailed_distribution(self):
        rng = np.random.default_rng(4)
        t_dist = rng.standard_t(df=3, size=20000)  # heavy tails
        assert excess_kurtosis(t_dist) > 1.0

    def test_raises_with_too_few_observations(self):
        with pytest.raises(ValueError):
            excess_kurtosis(np.array([1.0, 2.0, 3.0]))


class TestLeverageEffectCurve:
    def test_negative_for_a_constructed_leverage_process(self):
        # sigma_t scales up right after a negative return -- the defining
        # property of the leverage effect. Confirm the curve detects it.
        rng = np.random.default_rng(5)
        n = 20000
        r = np.empty(n)
        r[0] = rng.normal(0, 1)
        for t in range(1, n):
            sigma = 1.0 + 2.0 * max(-r[t - 1], 0)
            r[t] = rng.normal(0, sigma)
        curve = leverage_effect_curve(r, max_lag=5)
        assert curve[0] < -0.05

    def test_returns_expected_length(self):
        rng = np.random.default_rng(6)
        assert len(leverage_effect_curve(rng.normal(0, 1, 500), max_lag=8)) == 8

    def test_raises_when_not_enough_data(self):
        with pytest.raises(ValueError):
            leverage_effect_curve(np.arange(5.0), max_lag=10)


class TestAggregationalGaussianityCurve:
    def test_kurtosis_declines_at_coarser_scales(self):
        rng = np.random.default_rng(7)
        fat_tailed = rng.standard_t(df=3, size=30000)
        curve = aggregational_gaussianity_curve(fat_tailed, scales=(1, 30))
        assert curve[0] > curve[1]  # CLT: aggregation reduces excess kurtosis

    def test_raises_if_scale_leaves_too_few_points(self):
        with pytest.raises(ValueError):
            aggregational_gaussianity_curve(np.arange(10.0), scales=(1, 5))


class TestComputeStylizedFacts:
    def test_returns_summary_with_expected_shapes(self):
        rng = np.random.default_rng(8)
        returns = rng.normal(0, 1, 5000)
        summary = compute_stylized_facts(returns, max_lag=10, leverage_max_lag=6, agg_scales=(1, 5, 10))
        assert isinstance(summary, StylizedFactsSummary)
        assert len(summary.raw_return_acf) == 10
        assert len(summary.volatility_clustering_acf) == 10
        assert len(summary.leverage_curve) == 6
        assert len(summary.aggregational_kurtosis) == 3
        assert isinstance(summary.excess_kurtosis, float)

    def test_leverage_max_lag_defaults_to_max_lag(self):
        rng = np.random.default_rng(9)
        summary = compute_stylized_facts(rng.normal(0, 1, 5000), max_lag=12, agg_scales=(1, 5))
        assert len(summary.leverage_curve) == 12


class TestFactDistance:
    def _summary(self, **overrides):
        base = dict(
            raw_return_acf=np.zeros(5),
            volatility_clustering_acf=np.full(5, 0.2),
            excess_kurtosis=1.0,
            leverage_curve=np.full(5, -0.1),
            aggregational_kurtosis=np.array([1.0, 0.5]),
        )
        base.update(overrides)
        return StylizedFactsSummary(**base)

    def test_zero_for_identical_summaries(self):
        a = self._summary()
        b = self._summary()
        for fact in FACT_NAMES:
            assert fact_distance(a, b, fact) == pytest.approx(0.0)

    def test_scalar_fact_is_plain_abs_diff(self):
        a = self._summary(excess_kurtosis=1.0)
        b = self._summary(excess_kurtosis=1.7)
        assert fact_distance(a, b, "excess_kurtosis") == pytest.approx(0.7)

    def test_curve_fact_is_mean_abs_diff(self):
        a = self._summary(volatility_clustering_acf=np.array([0.1, 0.2, 0.3]))
        b = self._summary(volatility_clustering_acf=np.array([0.2, 0.2, 0.1]))
        assert fact_distance(a, b, "volatility_clustering_acf") == pytest.approx((0.1 + 0.0 + 0.2) / 3)

    def test_raises_on_unknown_fact_name(self):
        a, b = self._summary(), self._summary()
        with pytest.raises(ValueError):
            fact_distance(a, b, "not_a_real_fact")
