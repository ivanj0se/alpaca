import numpy as np
import pandas as pd
import pytest

from benchmark.cv import make_t1
from benchmark.ladder import evaluate_rung
from rpca.rolling_rpca import make_fold_scorer, rolling_rpca_decompose, sparse_anomaly_score


def _basket(n_rows=60, n_tickers=10, seed=3):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.001, n_rows)
    data = {
        f"T{i:02d}": common * rng.uniform(0.3, 1.3) + rng.normal(0, 0.0002, n_rows) for i in range(n_tickers)
    }
    idx = pd.date_range("2026-01-02 09:30", periods=n_rows, freq="1min", tz="UTC")
    return pd.DataFrame(data, index=idx)


class TestRollingRpcaDecompose:
    def test_raises_when_window_exceeds_data_length(self):
        base = _basket(n_rows=20)
        with pytest.raises(ValueError, match="cannot exceed"):
            rolling_rpca_decompose(base, window_len=25)

    def test_raises_when_window_too_small(self):
        base = _basket(n_rows=20)
        with pytest.raises(ValueError, match="at least 2"):
            rolling_rpca_decompose(base, window_len=1)

    def test_output_shape_and_columns_with_step_one(self):
        base = _basket(n_rows=60, n_tickers=8)
        L_df, S_df = rolling_rpca_decompose(base, window_len=20, step=1)
        assert list(L_df.columns) == list(base.columns)
        assert list(S_df.columns) == list(base.columns)
        assert len(L_df) == 60 - 20 + 1  # one output row per window
        assert len(S_df) == len(L_df)

    def test_output_index_starts_at_end_of_first_window(self):
        base = _basket(n_rows=60, n_tickers=8)
        L_df, _ = rolling_rpca_decompose(base, window_len=20, step=1)
        assert L_df.index[0] == base.index[19]  # window_len - 1

    def test_step_greater_than_one_produces_strided_output(self):
        base = _basket(n_rows=60, n_tickers=8)
        L_df, _ = rolling_rpca_decompose(base, window_len=20, step=5)
        # windows starting at 0, 5, 10, ..., up to 60-20=40 -> starts 0..40 step 5 -> 9 windows
        assert len(L_df) == 9
        assert L_df.index[1] == base.index[19 + 5]

    def test_reconstruction_l_plus_s_approximates_input(self):
        base = _basket(n_rows=60, n_tickers=8)
        L_df, S_df = rolling_rpca_decompose(base, window_len=20, step=1)
        window_last_rows = base.iloc[19:]  # rows corresponding to L_df/S_df's index
        reconstructed = L_df.to_numpy() + S_df.to_numpy()
        assert np.allclose(reconstructed, window_last_rows.to_numpy(), atol=1e-4)


class TestSparseAnomalyScore:
    def test_nonnegative(self):
        base = _basket(n_rows=60, n_tickers=8)
        _, S_df = rolling_rpca_decompose(base, window_len=20, step=1)
        score = sparse_anomaly_score(S_df)
        assert (score >= 0).all().all()

    def test_zero_variance_column_scores_zero_not_nan(self):
        S_df = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.1, -0.2, 0.3]})
        score = sparse_anomaly_score(S_df)
        assert (score["A"] == 0).all()
        assert not score["B"].isna().any()

    def test_injected_single_ticker_anomaly_ranks_highest_at_its_own_timestamp(self):
        base = _basket(n_rows=60, n_tickers=10)
        anomaly_pos, target = 40, "T03"
        base = base.copy()
        base.iloc[anomaly_pos, base.columns.get_loc(target)] += 0.02

        _, S_df = rolling_rpca_decompose(base, window_len=20, step=1)
        score = sparse_anomaly_score(S_df)
        anomaly_ts = base.index[anomaly_pos]
        assert anomaly_ts in score.index

        row = score.loc[anomaly_ts].sort_values(ascending=False)
        assert row.index[0] == target
        assert row[target] > row.drop(index=target).median() * 5


class TestMakeFoldScorer:
    def test_returns_finite_score_on_a_reasonable_fold(self):
        base = _basket(n_rows=200, n_tickers=15)
        scorer = make_fold_scorer(base)
        n = len(base)
        score = scorer(np.arange(0, n - 50), np.arange(n - 50, n))
        assert score is not None
        assert np.isfinite(score)

    def test_returns_none_for_too_short_training_fold(self):
        base = _basket(n_rows=100, n_tickers=10)
        scorer = make_fold_scorer(base)
        assert scorer(np.arange(0, 5), np.arange(5, 10)) is None

    def test_integrates_with_evaluate_rung(self):
        base = _basket(n_rows=300, n_tickers=15)
        t1 = make_t1(base.index, pd.Timedelta(minutes=1))
        result = evaluate_rung("rpca", make_fold_scorer(base), t1, n_splits=5, embargo_td=pd.Timedelta(minutes=10))
        assert result.n_folds > 0
        assert np.isfinite(result.mean_nll)
