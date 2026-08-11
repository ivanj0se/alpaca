"""Shared evaluation harness comparing rungs of the benchmark ladder on
common footing.

Unlike the SSA anomaly-detection project this pattern is adapted from,
there's no BELY-style confirmed ground truth for market anomalies -- that's
what this project is trying to measure, not something with pre-existing
labels. So "beat the previous rung" can't mean higher precision/recall
against labeled events. Instead it means: lower average out-of-sample
Gaussian negative log-likelihood on held-out data, via purged walk-forward
CV (benchmark/cv.py). This is a standard proper scoring rule that's
comparable across structurally different models -- GARCH's time-varying
variance forecast, a factor model's homoskedastic residual variance, RPCA's
and the TCN-VAE's reconstruction-residual variance all reduce to the same
NLL form, evaluated only on data each fold's model never saw.

The ladder is two parallel lanes, not one ranked list -- confirmed necessary
after Rung 2's GARCH and factor-model baselines turned out not to be
comparable on the same NLL axis (the factor model scores against a
contemporaneous cross-sectional factor, GARCH forecasts purely from an
instrument's own past; see
diagnostics/2026-08-11-garch-vs-factor-model-not-comparable/). This mirrors
the SSA project's reconstruction-track/population-track split:

- Temporal lane (one instrument, forecasts its own future): Rung 0 (random
  walk) -> Rung 2a (GARCH) -> Rung 4 (TCN-VAE reconstruction).
- Cross-sectional lane (contemporaneous, explains one instrument via
  others): Rung 2b (factor model) -> Rung 3 (Robust PCA).

`evaluate_rung`/`gate_check`/`save_ladder_report` are lane-agnostic --
gate_check should only be called between two rungs in the *same* lane.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from benchmark.cv import PurgedKFold

FoldScorer = Callable[[np.ndarray, np.ndarray], float | None]


@dataclass
class RungResult:
    rung_id: str
    mean_nll: float
    std_nll: float
    n_folds: int
    fold_nlls: list[float] = field(default_factory=list)


def evaluate_rung(
    rung_id: str,
    fit_and_score_fold: FoldScorer,
    t1: pd.Series,
    n_splits: int = 6,
    embargo_td: pd.Timedelta = pd.Timedelta(0),
    min_train_size: int = 20,
) -> RungResult:
    """Generic walk-forward evaluation. `fit_and_score_fold(train_idx,
    test_idx)` is rung-specific (fits a model on the training fold, returns
    the mean out-of-sample NLL on the test fold, or None if that fold can't
    be scored); this function only owns the CV splitting and aggregation,
    identical for every rung.
    """
    cv = PurgedKFold(t1, n_splits=n_splits, embargo_td=embargo_td)
    fold_nlls = []
    for fold in cv.split():
        if len(fold.train_idx) < min_train_size or len(fold.test_idx) == 0:
            continue
        nll = fit_and_score_fold(fold.train_idx, fold.test_idx)
        if nll is not None and np.isfinite(nll):
            fold_nlls.append(nll)

    if not fold_nlls:
        raise ValueError(f"rung '{rung_id}': no fold produced a valid score")

    return RungResult(
        rung_id=rung_id,
        mean_nll=float(np.mean(fold_nlls)),
        std_nll=float(np.std(fold_nlls)),
        n_folds=len(fold_nlls),
        fold_nlls=fold_nlls,
    )


def gate_check(candidate: RungResult, baseline: RungResult, min_improvement: float = 0.0) -> bool:
    """True if `candidate` beats `baseline` (strictly lower mean NLL) by at
    least `min_improvement` absolute NLL units -- the "must beat the
    previous rung" gate.
    """
    return (baseline.mean_nll - candidate.mean_nll) >= min_improvement


def save_ladder_report(results: list[RungResult], diagnostics_dir: Path, topic: str) -> Path:
    """Write a dated markdown report comparing all rungs run so far,
    matching the diagnostics/ investigation culture established for Rung 1.
    """
    date_str = dt.date.today().isoformat()
    report_dir = Path(diagnostics_dir) / f"{date_str}-{topic}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "findings.md"

    lines = [f"# Benchmark ladder: {topic}", "", f"Date: {date_str}", "", "| Rung | Mean NLL | Std NLL | Folds |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r.mean_nll):
        lines.append(f"| {r.rung_id} | {r.mean_nll:.4f} | {r.std_nll:.4f} | {r.n_folds} |")
    lines.append("")
    lines.append("Lower mean NLL is better (higher out-of-sample likelihood of the held-out data under the fitted model).")

    report_path.write_text("\n".join(lines))
    return report_path
