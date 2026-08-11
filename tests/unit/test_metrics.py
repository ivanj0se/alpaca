import pytest

from benchmark.metrics import branching_ratio_error


def test_zero_error_when_estimate_matches_published():
    assert branching_ratio_error(0.81, published=0.81) == 0.0


def test_relative_error_computed_correctly():
    assert branching_ratio_error(0.405, published=0.81) == pytest.approx(0.5)


def test_default_published_value_is_filimonov_sornette():
    assert branching_ratio_error(0.0) == pytest.approx(1.0)
