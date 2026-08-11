import numpy as np
import pandas as pd
import pytest

from benchmark.cv import make_t1
from benchmark.ladder import RungResult, evaluate_rung, gate_check, save_ladder_report


def _t1(n=60, freq_min=1):
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq=f"{freq_min}min", tz="UTC")
    return make_t1(idx, pd.Timedelta(minutes=1))


class TestEvaluateRung:
    def test_aggregates_constant_fold_scores(self):
        t1 = _t1()
        scorer = lambda train_idx, test_idx: 2.5
        result = evaluate_rung("constant", scorer, t1, n_splits=5, min_train_size=1)
        assert result.rung_id == "constant"
        assert result.mean_nll == pytest.approx(2.5)
        assert result.std_nll == pytest.approx(0.0)
        assert result.n_folds == 5

    def test_skips_none_folds(self):
        calls = {"n": 0}

        def scorer(train_idx, test_idx):
            calls["n"] += 1
            return None if calls["n"] == 1 else 1.0

        t1 = _t1()
        result = evaluate_rung("some_none", scorer, t1, n_splits=5, min_train_size=1)
        assert result.n_folds == 4  # one fold skipped

    def test_skips_folds_with_too_few_training_samples(self):
        # min_train_size larger than any fold's train set -> every fold skipped.
        t1 = _t1(n=20)
        with pytest.raises(ValueError, match="no fold produced"):
            evaluate_rung("too_strict", lambda a, b: 1.0, t1, n_splits=5, min_train_size=10_000)

    def test_skips_nonfinite_scores(self):
        t1 = _t1()
        scores = iter([np.inf, np.nan, 1.0, 2.0, 3.0])
        result = evaluate_rung("mixed", lambda a, b: next(scores), t1, n_splits=5, min_train_size=1)
        assert result.n_folds == 3
        assert result.mean_nll == pytest.approx(2.0)

    def test_raises_when_no_valid_folds(self):
        t1 = _t1()
        with pytest.raises(ValueError, match="no fold produced"):
            evaluate_rung("all_none", lambda a, b: None, t1, n_splits=5, min_train_size=1)


class TestGateCheck:
    def test_true_when_candidate_strictly_better(self):
        baseline = RungResult("baseline", mean_nll=5.0, std_nll=0.1, n_folds=5)
        candidate = RungResult("candidate", mean_nll=3.0, std_nll=0.1, n_folds=5)
        assert gate_check(candidate, baseline) is True

    def test_false_when_candidate_worse(self):
        baseline = RungResult("baseline", mean_nll=3.0, std_nll=0.1, n_folds=5)
        candidate = RungResult("candidate", mean_nll=5.0, std_nll=0.1, n_folds=5)
        assert gate_check(candidate, baseline) is False

    def test_respects_min_improvement_margin(self):
        baseline = RungResult("baseline", mean_nll=5.0, std_nll=0.1, n_folds=5)
        candidate = RungResult("candidate", mean_nll=4.95, std_nll=0.1, n_folds=5)
        assert gate_check(candidate, baseline, min_improvement=0.0) is True
        assert gate_check(candidate, baseline, min_improvement=0.5) is False


class TestSaveLadderReport:
    def test_writes_markdown_sorted_by_mean_nll(self, tmp_path):
        results = [
            RungResult("worse", mean_nll=5.0, std_nll=0.2, n_folds=5),
            RungResult("better", mean_nll=1.0, std_nll=0.1, n_folds=5),
        ]
        path = save_ladder_report(results, tmp_path, "test-topic")
        assert path.exists()
        content = path.read_text()
        assert content.index("better") < content.index("worse")
        assert "test-topic" in content

    def test_creates_dated_subdirectory(self, tmp_path):
        results = [RungResult("r", mean_nll=1.0, std_nll=0.0, n_folds=3)]
        path = save_ladder_report(results, tmp_path, "my-topic")
        assert "my-topic" in path.parent.name
        assert path.parent.parent == tmp_path
