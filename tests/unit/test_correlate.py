import numpy as np
import pandas as pd
import pytest

from attribution.correlate import correlation_rate, high_residual_windows, match_to_news


class TestHighResidualWindows:
    def test_returns_timestamps_above_threshold(self):
        idx = pd.date_range("2026-01-02 09:30", periods=5, freq="1min", tz="UTC")
        scores = pd.Series([0.1, 5.0, 0.2, 8.0, 0.3], index=idx)
        result = high_residual_windows(scores, threshold=1.0)
        assert list(result) == [idx[1], idx[3]]

    def test_empty_when_nothing_exceeds_threshold(self):
        idx = pd.date_range("2026-01-02 09:30", periods=3, freq="1min", tz="UTC")
        scores = pd.Series([0.1, 0.2, 0.3], index=idx)
        assert len(high_residual_windows(scores, threshold=1.0)) == 0


class TestMatchToNews:
    def test_empty_anomaly_times_returns_empty(self):
        news = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-02 09:30"], utc=True)})
        result = match_to_news(pd.DatetimeIndex([]), news, pd.Timedelta(minutes=5))
        assert result.empty

    def test_empty_news_marks_everything_unmatched(self):
        anomalies = pd.DatetimeIndex(pd.to_datetime(["2026-01-02 09:30", "2026-01-02 10:00"], utc=True))
        result = match_to_news(anomalies, pd.DataFrame(columns=["timestamp"]), pd.Timedelta(minutes=5))
        assert (~result["matched"]).all()

    def test_match_within_tolerance(self):
        anomalies = pd.DatetimeIndex(pd.to_datetime(["2026-01-02 09:30"], utc=True))
        news = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-02 09:32"], utc=True)})
        result = match_to_news(anomalies, news, pd.Timedelta(minutes=5))
        assert result.iloc[0]["matched"]
        assert result.iloc[0]["gap_seconds"] == pytest.approx(120.0)

    def test_no_match_outside_tolerance(self):
        anomalies = pd.DatetimeIndex(pd.to_datetime(["2026-01-02 09:30"], utc=True))
        news = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-02 11:00"], utc=True)})
        result = match_to_news(anomalies, news, pd.Timedelta(minutes=5))
        assert not result.iloc[0]["matched"]

    def test_one_news_event_can_match_multiple_anomalies(self):
        anomalies = pd.DatetimeIndex(pd.to_datetime(["2026-01-02 09:29", "2026-01-02 09:31"], utc=True))
        news = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-02 09:30"], utc=True)})
        result = match_to_news(anomalies, news, pd.Timedelta(minutes=5))
        assert result["matched"].all()

    def test_handles_mismatched_datetime64_resolutions(self):
        # Confirmed the hard way on real GDELT data: anomaly_times at one
        # resolution (e.g. ns) and news_df at another (e.g. us) raised a
        # pandas MergeError inside merge_asof without explicit coercion.
        anomalies = pd.DatetimeIndex(pd.to_datetime(["2026-01-02 09:30"], utc=True))
        news = pd.DataFrame(
            {"timestamp": pd.DatetimeIndex(["2026-01-02 09:31"], tz="UTC").as_unit("us")}
        )
        result = match_to_news(anomalies, news, pd.Timedelta(minutes=5))
        assert result.iloc[0]["matched"]


class TestCorrelationRate:
    def test_empty_returns_zero(self):
        assert correlation_rate(pd.DataFrame()) == 0.0

    def test_computes_fraction_matched(self):
        df = pd.DataFrame({"matched": [True, True, False, False]})
        assert correlation_rate(df) == pytest.approx(0.5)

    def test_all_matched(self):
        df = pd.DataFrame({"matched": [True, True]})
        assert correlation_rate(df) == pytest.approx(1.0)
