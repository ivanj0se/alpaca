import numpy as np
import pytest

from benchmark.generator_ladder import (
    GeneratorResult,
    calibrate_reference_bands,
    evaluate_generator,
    generator_gate_check,
    rank_generators,
    save_generator_report,
)
from benchmark.stylized_facts import FACT_NAMES
from generators.path import GeneratedPath


def _reference():
    rng = np.random.default_rng(0)
    return rng.normal(0, 0.001, 3000)


class TestCalibrateReferenceBands:
    def test_returns_a_band_per_fact_and_a_reference_summary(self):
        bands, reference_facts = calibrate_reference_bands(_reference(), n_bootstrap=20, block_size=50, seed=0)
        assert set(bands.keys()) == set(FACT_NAMES)
        assert all(b.threshold >= 0 for b in bands.values())
        assert reference_facts.excess_kurtosis is not None


class TestEvaluateGenerator:
    def test_raises_on_empty_paths(self):
        bands, reference_facts = calibrate_reference_bands(_reference(), n_bootstrap=10, block_size=50, seed=0)
        with pytest.raises(ValueError, match="no paths"):
            evaluate_generator("empty", [], reference_facts, bands)

    def test_paths_matching_reference_distribution_score_well(self):
        reference = _reference()
        bands, reference_facts = calibrate_reference_bands(reference, n_bootstrap=100, block_size=50, seed=0)

        rng = np.random.default_rng(1)
        same_dist_paths = [
            GeneratedPath(generator_id="same", log_returns=rng.normal(0, 0.001, 3000), seed=i, params={})
            for i in range(10)
        ]
        result = evaluate_generator("same", same_dist_paths, reference_facts, bands)
        assert isinstance(result, GeneratorResult)
        assert result.overall_score > 0.3  # should pass a meaningful fraction, drawn from the same process

    def test_paths_from_a_very_different_distribution_score_poorly(self):
        reference = _reference()
        bands, reference_facts = calibrate_reference_bands(reference, n_bootstrap=100, block_size=50, seed=0)

        # Strong positive lag-1 autocorrelation in raw returns -- real
        # returns have near-zero raw ACF, so this should clearly fail
        # raw_return_acf (unlike a degenerate constant-zero series, which
        # trivially satisfies several near-zero checks by construction --
        # not a useful "obviously wrong" fixture for this metric suite).
        rng = np.random.default_rng(2)

        def ar1_path(seed):
            r = np.random.default_rng(seed).normal(0, 0.001, 3000)
            for t in range(1, len(r)):
                r[t] = 0.8 * r[t - 1] + r[t]
            return r

        different_paths = [GeneratedPath(generator_id="different", log_returns=ar1_path(i), seed=i, params={}) for i in range(5)]
        result = evaluate_generator("different", different_paths, reference_facts, bands)
        assert result.per_fact_coverage["raw_return_acf"] == 0.0
        assert result.overall_score < 0.5

    def test_n_paths_recorded_correctly(self):
        reference = _reference()
        bands, reference_facts = calibrate_reference_bands(reference, n_bootstrap=10, block_size=50, seed=0)
        rng = np.random.default_rng(3)
        paths = [GeneratedPath(generator_id="x", log_returns=rng.normal(0, 0.001, 3000), seed=i, params={}) for i in range(7)]
        result = evaluate_generator("x", paths, reference_facts, bands)
        assert result.n_paths == 7


class TestRankGenerators:
    def test_sorted_by_overall_score_descending(self):
        results = [
            GeneratorResult("low", {"a": 0.1}, {"a": 1.0}, overall_score=0.1, n_paths=5),
            GeneratorResult("high", {"a": 0.9}, {"a": 0.1}, overall_score=0.9, n_paths=5),
            GeneratorResult("mid", {"a": 0.5}, {"a": 0.5}, overall_score=0.5, n_paths=5),
        ]
        ranking = rank_generators(results)
        assert list(ranking["generator_id"]) == ["high", "mid", "low"]


class TestGeneratorGateCheck:
    def test_higher_score_passes(self):
        candidate = GeneratorResult("c", {}, {}, overall_score=0.5, n_paths=5)
        baseline = GeneratorResult("b", {}, {}, overall_score=0.3, n_paths=5)
        assert generator_gate_check(candidate, baseline)

    def test_lower_score_fails(self):
        candidate = GeneratorResult("c", {}, {}, overall_score=0.2, n_paths=5)
        baseline = GeneratorResult("b", {}, {}, overall_score=0.3, n_paths=5)
        assert not generator_gate_check(candidate, baseline)

    def test_respects_min_improvement(self):
        candidate = GeneratorResult("c", {}, {}, overall_score=0.31, n_paths=5)
        baseline = GeneratorResult("b", {}, {}, overall_score=0.3, n_paths=5)
        assert not generator_gate_check(candidate, baseline, min_improvement=0.05)
        assert generator_gate_check(candidate, baseline, min_improvement=0.005)


class TestSaveGeneratorReport:
    def test_writes_report_with_expected_content(self, tmp_path):
        reference = _reference()
        bands, reference_facts = calibrate_reference_bands(reference, n_bootstrap=10, block_size=50, seed=0)
        rng = np.random.default_rng(4)
        paths = [GeneratedPath(generator_id="x", log_returns=rng.normal(0, 0.001, 3000), seed=i, params={}) for i in range(3)]
        result = evaluate_generator("x", paths, reference_facts, bands)

        report_path = save_generator_report([result], bands, tmp_path, topic="test-comparison")
        assert report_path.exists()
        content = report_path.read_text()
        assert "x" in content
        assert "overall_score" in content
        assert "Calibrated band thresholds" in content
        assert report_path.name == "report.md"
        assert "test-comparison" in str(report_path.parent)
