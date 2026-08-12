"""Shared evaluation harness for the market-generator comparison suite --
the generative analogue of benchmark/ladder.py's evaluate_rung/gate_check
for the predictive rungs. Every generator arm (GBM null, Hawkes
control/treatment, TCN-forecaster, zero-intelligence) gets scored through
this one interface so the final comparison report is apples-to-apples:
one shared reference dataset, one shared set of calibrated bands
(benchmark/conformal.py), one shared stylized-facts definition
(benchmark/stylized_facts.py).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.conformal import ConformalBand, calibrate_band, coverage_rate
from benchmark.stylized_facts import FACT_NAMES, StylizedFactsSummary, compute_stylized_facts, fact_distance
from generators.path import GeneratedPath


@dataclass
class GeneratorResult:
    generator_id: str
    per_fact_coverage: dict[str, float]
    per_fact_mean_distance: dict[str, float]
    overall_score: float
    n_paths: int


def calibrate_reference_bands(
    reference_returns: np.ndarray,
    alpha: float = 0.10,
    n_bootstrap: int = 300,
    block_size: int = 90,
    max_lag: int = 50,
    leverage_max_lag: int | None = None,
    agg_scales: tuple[int, ...] = (1, 5, 15, 30),
    seed: int | None = None,
) -> tuple[dict[str, ConformalBand], StylizedFactsSummary]:
    """One-time, real-data-only calibration step: for each stylized fact,
    learn how much it naturally varies across block-bootstrap resamples of
    `reference_returns` against itself (no generator involved). Returns
    the per-fact bands plus the reference's own StylizedFactsSummary
    (reused by evaluate_generator rather than recomputed per generator).
    """

    def stat_fn(r: np.ndarray) -> StylizedFactsSummary:
        return compute_stylized_facts(r, max_lag=max_lag, leverage_max_lag=leverage_max_lag, agg_scales=agg_scales)

    reference_facts = stat_fn(reference_returns)
    bands = {
        fact: calibrate_band(
            reference_returns,
            stat_fn=stat_fn,
            distance_fn=lambda a, b, f=fact: fact_distance(a, b, f),
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=seed,
        )
        for fact in FACT_NAMES
    }
    return bands, reference_facts


def evaluate_generator(
    generator_id: str,
    paths: list[GeneratedPath],
    reference_facts: StylizedFactsSummary,
    bands: dict[str, ConformalBand],
    max_lag: int = 50,
    leverage_max_lag: int | None = None,
    agg_scales: tuple[int, ...] = (1, 5, 15, 30),
) -> GeneratorResult:
    """Scores one generator's independent synthetic realizations against
    the (already-calibrated) reference bands. `overall_score` = mean
    coverage rate across the 5 stylized facts -- higher is better, the
    opposite direction from benchmark.ladder's lower-NLL-is-better.
    """
    if not paths:
        raise ValueError(f"no paths provided for generator {generator_id!r}")

    def stat_fn(r: np.ndarray) -> StylizedFactsSummary:
        return compute_stylized_facts(r, max_lag=max_lag, leverage_max_lag=leverage_max_lag, agg_scales=agg_scales)

    per_fact_coverage: dict[str, float] = {}
    per_fact_mean_distance: dict[str, float] = {}
    for fact in FACT_NAMES:
        distances = np.array([fact_distance(stat_fn(p.log_returns), reference_facts, fact) for p in paths])
        per_fact_coverage[fact] = coverage_rate(distances, bands[fact])
        per_fact_mean_distance[fact] = float(distances.mean())

    overall_score = float(np.mean(list(per_fact_coverage.values())))
    return GeneratorResult(
        generator_id=generator_id,
        per_fact_coverage=per_fact_coverage,
        per_fact_mean_distance=per_fact_mean_distance,
        overall_score=overall_score,
        n_paths=len(paths),
    )


def rank_generators(results: list[GeneratorResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"generator_id": r.generator_id, "overall_score": r.overall_score, "n_paths": r.n_paths}
        row.update({f"coverage_{k}": v for k, v in r.per_fact_coverage.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("overall_score", ascending=False).reset_index(drop=True)


def generator_gate_check(candidate: GeneratorResult, baseline: GeneratorResult, min_improvement: float = 0.0) -> bool:
    """Higher overall_score is better -- opposite direction from
    benchmark.ladder.gate_check's lower-NLL-is-better convention.
    """
    return candidate.overall_score >= baseline.overall_score + min_improvement


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join(['---'] * len(headers))}|"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return lines


def save_generator_report(
    results: list[GeneratorResult],
    bands: dict[str, ConformalBand],
    diagnostics_dir: Path,
    topic: str = "generator-comparison",
) -> Path:
    """Writes a dated report.md (matching scripts/run_ladder.py's
    convention, not benchmark/ladder.py::save_ladder_report's findings.md
    -- the project's actual full-run reports have always been report.md).
    """
    date_str = dt.date.today().isoformat()
    out_dir = Path(diagnostics_dir) / f"{date_str}-{topic}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking = rank_generators(results)
    lines = [f"# Generator comparison: {date_str}", ""]

    lines.append("## Calibrated band thresholds (per fact, real-data-only)")
    lines.append("")
    lines.extend(_markdown_table(["Fact", "Threshold", "Alpha"], [[f, f"{b.threshold:.5f}", f"{b.alpha:.2f}"] for f, b in bands.items()]))
    lines.append("")

    lines.append("## Ranking (higher overall_score is better)")
    lines.append("")
    ranking_headers = list(ranking.columns)
    ranking_rows = [[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row] for row in ranking.itertuples(index=False)]
    lines.extend(_markdown_table(ranking_headers, ranking_rows))
    lines.append("")

    lines.append("## Per-fact mean distances")
    lines.append("")
    fact_list = list(FACT_NAMES)
    dist_rows = [[r.generator_id] + [f"{r.per_fact_mean_distance[f]:.5f}" for f in fact_list] for r in results]
    lines.extend(_markdown_table(["generator_id"] + fact_list, dist_rows))

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path
