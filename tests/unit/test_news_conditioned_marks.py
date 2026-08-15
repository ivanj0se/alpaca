import numpy as np
import pandas as pd
import pytest

from research.news_conditioned_marks import compare_magnitude_distributions, label_events_by_news_proximity

WINDOW_START = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW_END = WINDOW_START + pd.Timedelta(days=10)


def _news(n=30, seed=0):
    rng = np.random.default_rng(seed)
    span_s = (WINDOW_END - WINDOW_START).total_seconds()
    times = WINDOW_START + pd.to_timedelta(rng.uniform(0, span_s, n), unit="s")
    return pd.DataFrame({"timestamp": pd.DatetimeIndex(times).sort_values()})


class TestLabelEventsByNewsProximity:
    def test_events_near_news_are_labeled_true(self):
        news = _news(n=10)
        near_news = pd.DatetimeIndex(news["timestamp"]) + pd.Timedelta(minutes=2)
        mask = label_events_by_news_proximity(near_news, news, match_window=pd.Timedelta(minutes=10))
        assert mask.all()

    def test_events_far_from_news_are_labeled_false(self):
        news = _news(n=5)
        far_events = pd.DatetimeIndex([WINDOW_START + pd.Timedelta(hours=100)])
        mask = label_events_by_news_proximity(far_events, news, match_window=pd.Timedelta(minutes=10))
        assert not mask.any()

    def test_duplicate_event_timestamps_are_labeled_independently_by_position(self):
        # Real tick data has genuine duplicate timestamps (multiple trades
        # sharing a microsecond during a burst -- confirmed on real SPY
        # data, see diagnostics/2026-08-13-cox-hawkes-rpca-baseline/).
        # label_events_by_news_proximity restores order via argsort + a
        # fancy-index scatter (result[order] = matched_sorted), which
        # operates on POSITIONS, not on the timestamp VALUES -- unlike the
        # reindex/union chain that broke in the Cox-Hawkes real-data
        # script (a genuinely different failure mode: that bug came from
        # pandas' Index.union() not fully deduplicating a duplicate-valued
        # DatetimeIndex used as a merge key). Confirms directly rather
        # than only reasoning about it: three input events share the
        # SAME timestamp, one near news and two far from any news, and
        # each should still get its own independent, correct label.
        news = _news(n=3, seed=5)
        news_times = pd.DatetimeIndex(news["timestamp"])
        near_time = news_times[0] + pd.Timedelta(minutes=1)
        far_time = WINDOW_START + pd.Timedelta(hours=100)

        event_times = pd.DatetimeIndex([near_time, far_time, far_time])
        mask = label_events_by_news_proximity(event_times, news, match_window=pd.Timedelta(minutes=10))
        assert len(mask) == 3
        assert mask[0]  # near-news event, real timestamp value shared with nothing else here
        assert not mask[1] and not mask[2]  # both far-from-news events, despite sharing an identical timestamp


class TestCompareMagnitudeDistributions:
    def test_detects_a_planted_magnitude_difference(self):
        rng = np.random.default_rng(1)
        # Sparse news (1/day over 10 days) so randomly-placed "unmatched"
        # candidates have negligible chance of accidentally colliding with
        # a news timestamp within the match window -- with dense news
        # (e.g. 40 events/10 days) a first version of this test found
        # ~25% of intended-unmatched events landed within the window by
        # pure chance, contaminating both groups and diluting the signal.
        news = _news(n=10, seed=1)
        news_times = pd.DatetimeIndex(news["timestamp"])

        # Half of events are news-adjacent with SYSTEMATICALLY LARGER
        # magnitude (planted); half are explicitly placed at the
        # midpoints between consecutive news events (guaranteed far from
        # any news) with smaller magnitude.
        matched_times = news_times + pd.Timedelta(minutes=1)
        matched_mags = np.abs(rng.normal(0.05, 0.01, len(news_times)))

        sorted_news = news_times.sort_values()
        midpoints = sorted_news[:-1] + (sorted_news[1:] - sorted_news[:-1]) / 2
        unmatched_times = pd.DatetimeIndex(midpoints)
        unmatched_mags = np.abs(rng.normal(0.01, 0.005, len(midpoints)))

        event_times = matched_times.append(unmatched_times)
        magnitudes = np.concatenate([matched_mags, unmatched_mags])

        result = compare_magnitude_distributions(
            magnitudes, event_times, news, match_window=pd.Timedelta(minutes=10), n_permutations=200, seed=2
        )
        assert result.n_matched == len(news_times)
        assert result.n_unmatched == len(midpoints)
        assert result.matched_mean > result.unmatched_mean
        assert result.mannwhitney_p < 0.01
        assert result.null_control_p < 0.05

    def test_no_false_positive_when_magnitude_is_independent_of_news(self):
        rng = np.random.default_rng(3)
        news = _news(n=40, seed=3)
        span_s = (WINDOW_END - WINDOW_START).total_seconds()
        event_times = pd.DatetimeIndex(WINDOW_START + pd.to_timedelta(rng.uniform(0, span_s, 60), unit="s"))
        magnitudes = np.abs(rng.normal(0.02, 0.01, 60))  # same distribution regardless of news proximity

        result = compare_magnitude_distributions(
            magnitudes, event_times, news, match_window=pd.Timedelta(hours=2), n_permutations=200, seed=4
        )
        assert result.null_control_p > 0.05

    def test_raises_with_too_few_events_in_one_group(self):
        news = _news(n=5)
        news_times = pd.DatetimeIndex(news["timestamp"])
        event_times = news_times[:2] + pd.Timedelta(minutes=1)  # all matched, zero unmatched
        magnitudes = np.array([0.01, 0.02])
        with pytest.raises(ValueError, match="too few"):
            compare_magnitude_distributions(magnitudes, event_times, news, match_window=pd.Timedelta(minutes=10))
