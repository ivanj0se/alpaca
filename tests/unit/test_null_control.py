import numpy as np
import pandas as pd
import pytest

from attribution.null_control import (
    PermutationTestResult,
    permutation_test,
    shuffle_news_timestamps,
    sidak_correction,
    summarize_attribution,
)

WINDOW_START = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW_END = WINDOW_START + pd.Timedelta(days=10)


def _sparse_news(n=20, seed=1):
    rng = np.random.default_rng(seed)
    span_s = (WINDOW_END - WINDOW_START).total_seconds()
    times = WINDOW_START + pd.to_timedelta(rng.uniform(0, span_s, n), unit="s")
    return pd.DataFrame({"timestamp": pd.DatetimeIndex(times).sort_values()})


class TestShuffleNewsTimestamps:
    def test_preserves_row_count(self):
        news = _sparse_news(n=15)
        shuffled = shuffle_news_timestamps(news, WINDOW_START, WINDOW_END, seed=0)
        assert len(shuffled) == len(news)

    def test_timestamps_fall_within_window(self):
        news = _sparse_news(n=15)
        shuffled = shuffle_news_timestamps(news, WINDOW_START, WINDOW_END, seed=0)
        assert (shuffled["timestamp"] >= WINDOW_START).all()
        assert (shuffled["timestamp"] <= WINDOW_END).all()

    def test_is_not_a_permutation_of_the_same_values(self):
        # Regression test for the exact bug found on real data: an earlier
        # implementation permuted the existing timestamp values among
        # rows, which is a no-op for anything that only cares about the
        # *set* of timestamps (as match_to_news does) -- every "shuffle"
        # was silently identical to the original.
        news = _sparse_news(n=15)
        shuffled = shuffle_news_timestamps(news, WINDOW_START, WINDOW_END, seed=0)
        original_set = set(news["timestamp"])
        shuffled_set = set(shuffled["timestamp"])
        assert original_set != shuffled_set

    def test_different_seeds_give_different_draws(self):
        news = _sparse_news(n=15)
        a = shuffle_news_timestamps(news, WINDOW_START, WINDOW_END, seed=1)
        b = shuffle_news_timestamps(news, WINDOW_START, WINDOW_END, seed=2)
        assert not a["timestamp"].equals(b["timestamp"])

    def test_raises_on_invalid_window(self):
        news = _sparse_news(n=5)
        with pytest.raises(ValueError):
            shuffle_news_timestamps(news, WINDOW_END, WINDOW_START, seed=0)


class TestPermutationTest:
    def test_null_distribution_has_genuine_variance(self):
        # Same regression as above, at the permutation_test level: if
        # shuffling were a no-op, null_std would come out exactly 0.
        news = _sparse_news(n=20)
        anomalies = pd.DatetimeIndex(news["timestamp"].iloc[:10]) + pd.Timedelta(minutes=5)
        result = permutation_test(anomalies, news, pd.Timedelta(minutes=15), n_permutations=300, seed=0)
        assert result.null_std > 0.01

    def test_detects_genuine_correlation(self):
        news = _sparse_news(n=20)
        correlated_anomalies = pd.DatetimeIndex(news["timestamp"].iloc[:15]) + pd.Timedelta(minutes=5)
        result = permutation_test(
            correlated_anomalies, news, pd.Timedelta(minutes=15), n_permutations=500, seed=10
        )
        assert result.observed_rate > result.null_mean
        assert result.p_value < 0.05

    def test_does_not_falsely_flag_unrelated_anomalies(self):
        news = _sparse_news(n=20)
        rng = np.random.default_rng(2)
        span_s = (WINDOW_END - WINDOW_START).total_seconds()
        random_anomalies = pd.DatetimeIndex(
            WINDOW_START + pd.to_timedelta(rng.uniform(0, span_s, 15), unit="s")
        )
        result = permutation_test(random_anomalies, news, pd.Timedelta(minutes=15), n_permutations=500, seed=11)
        assert result.p_value > 0.05

    def test_empty_news_returns_zero_rate_and_p_one(self):
        anomalies = pd.DatetimeIndex(pd.to_datetime(["2026-01-02 09:30"], utc=True))
        result = permutation_test(anomalies, pd.DataFrame(columns=["timestamp"]), pd.Timedelta(minutes=15))
        assert result.observed_rate == 0.0
        assert result.p_value == 1.0

    def test_default_window_derived_from_news_range(self):
        news = _sparse_news(n=10)
        anomalies = pd.DatetimeIndex(news["timestamp"].iloc[:3]) + pd.Timedelta(minutes=2)
        # Should not raise, and should use news's own min/max as the window.
        result = permutation_test(anomalies, news, pd.Timedelta(minutes=10), n_permutations=50, seed=0)
        assert 0.0 <= result.observed_rate <= 1.0


class TestSidakCorrection:
    def test_matches_closed_form(self):
        p = np.array([0.01, 0.05])
        corrected = sidak_correction(p, n_tests=2)
        expected = 1 - (1 - p) ** 2
        assert np.allclose(corrected, expected)

    def test_uses_array_length_when_n_tests_not_given(self):
        p = np.array([0.1, 0.1, 0.1])
        corrected = sidak_correction(p)
        assert np.allclose(corrected, 1 - (1 - p) ** 3)

    def test_correction_increases_p_values(self):
        p = np.array([0.01, 0.02, 0.03])
        corrected = sidak_correction(p, n_tests=10)
        assert (corrected >= p).all()


class TestSummarizeAttribution:
    def test_sorted_by_corrected_p_value(self):
        results = {
            "AAPL": PermutationTestResult(0.5, 0.1, 0.05, np.array([0.1]), p_value=0.3),
            "MSFT": PermutationTestResult(0.8, 0.1, 0.05, np.array([0.1]), p_value=0.001),
        }
        summary = summarize_attribution(results)
        assert summary.iloc[0]["ticker"] == "MSFT"
        assert summary.iloc[0]["p_value_sidak"] <= summary.iloc[1]["p_value_sidak"]

    def test_significance_flag_respects_alpha(self):
        results = {
            "AAPL": PermutationTestResult(0.5, 0.1, 0.05, np.array([0.1]), p_value=0.001),
            "MSFT": PermutationTestResult(0.5, 0.1, 0.05, np.array([0.1]), p_value=0.9),
        }
        summary = summarize_attribution(results, alpha=0.05)
        row_aapl = summary[summary["ticker"] == "AAPL"].iloc[0]
        row_msft = summary[summary["ticker"] == "MSFT"].iloc[0]
        assert row_aapl["significant"]
        assert not row_msft["significant"]

    def test_expected_columns(self):
        results = {"AAPL": PermutationTestResult(0.5, 0.1, 0.05, np.array([0.1]), p_value=0.5)}
        summary = summarize_attribution(results)
        assert set(summary.columns) == {
            "ticker",
            "observed_rate",
            "null_mean",
            "null_std",
            "p_value",
            "p_value_sidak",
            "significant",
        }
