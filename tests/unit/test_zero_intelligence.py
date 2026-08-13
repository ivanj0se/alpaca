import numpy as np
import pytest

from benchmark.stylized_facts import volatility_clustering_curve
from generators.zero_intelligence import (
    ZeroIntelligenceParams,
    calibrate_zero_intelligence_params,
    generate_zero_intelligence_paths,
    simulate_zero_intelligence,
)


class TestCalibrateZeroIntelligenceParams:
    def test_matches_target_std_approximately(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 0.001, 20000)
        params = calibrate_zero_intelligence_params(reference, n_agents=200, order_prob=0.05)
        simulated = simulate_zero_intelligence(params, n_steps=50000, seed=1)
        assert simulated.std(ddof=1) == pytest.approx(reference.std(ddof=1), rel=0.1)

    def test_raises_for_invalid_n_agents(self):
        with pytest.raises(ValueError, match="n_agents"):
            calibrate_zero_intelligence_params(np.random.default_rng(0).normal(0, 1, 100), n_agents=0)

    def test_raises_for_invalid_order_prob(self):
        ref = np.random.default_rng(0).normal(0, 1, 100)
        with pytest.raises(ValueError, match="order_prob"):
            calibrate_zero_intelligence_params(ref, order_prob=0.0)
        with pytest.raises(ValueError, match="order_prob"):
            calibrate_zero_intelligence_params(ref, order_prob=1.0)


class TestSimulateZeroIntelligence:
    def test_output_length_matches_n_steps(self):
        params = ZeroIntelligenceParams(n_agents=100, order_prob=0.05, impact_lambda=0.001, noise_std=0.0005)
        result = simulate_zero_intelligence(params, n_steps=500, seed=0)
        assert len(result) == 500

    def test_reproducible_with_seed(self):
        params = ZeroIntelligenceParams(n_agents=100, order_prob=0.05, impact_lambda=0.001, noise_std=0.0005)
        a = simulate_zero_intelligence(params, n_steps=200, seed=5)
        b = simulate_zero_intelligence(params, n_steps=200, seed=5)
        assert np.array_equal(a, b)

    def test_no_linear_autocorrelation_by_construction(self):
        # i.i.d. steps by construction -- lag-1 autocorrelation should be
        # near zero, same as real data but for a structurally different
        # reason (no mechanism at all here, vs. real markets' genuinely
        # near-zero but non-trivial linear structure).
        params = ZeroIntelligenceParams(n_agents=200, order_prob=0.05, impact_lambda=0.001, noise_std=0.0005)
        result = simulate_zero_intelligence(params, n_steps=50000, seed=0)
        r = result - result.mean()
        lag1_acf = np.sum(r[:-1] * r[1:]) / np.sum(r**2)
        assert abs(lag1_acf) < 0.02

    def test_no_volatility_clustering_by_construction(self):
        # The expected, useful negative result this generator exists to
        # demonstrate: with no persistence/herding mechanism, |returns|
        # ACF should stay near zero -- unlike real markets (see
        # diagnostics/2026-08-12-stylized-facts-module-validation/, where
        # real SPY's volatility-clustering ACF averages ~0.21).
        params = ZeroIntelligenceParams(n_agents=200, order_prob=0.05, impact_lambda=0.001, noise_std=0.0005)
        result = simulate_zero_intelligence(params, n_steps=50000, seed=0)
        curve = volatility_clustering_curve(result, max_lag=20)
        assert abs(curve.mean()) < 0.03


class TestGenerateZeroIntelligencePaths:
    def test_returns_expected_number_of_paths(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 0.001, 5000)
        paths = generate_zero_intelligence_paths(reference, n_steps=100, n_sims=7, seed=0)
        assert len(paths) == 7

    def test_paths_have_expected_generator_id_and_length(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 0.001, 5000)
        paths = generate_zero_intelligence_paths(reference, n_steps=50, n_sims=3, seed=0)
        for p in paths:
            assert p.generator_id == "zero_intelligence"
            assert len(p.log_returns) == 50
