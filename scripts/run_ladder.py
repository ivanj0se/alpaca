"""Runs the full benchmark ladder end-to-end on real backfilled data and
produces a comprehensive report. This is the repeatable version of the
ad-hoc validation scripts used during development -- see diagnostics/ for
what those found (GARCH-vs-factor-model incomparability, the NLL
scale-invariance issue, the permutation-test no-op bug, etc.). All of that
is already fixed in the library code this script calls; this just wires it
together and persists a report instead of printing to a throwaway script.

Usage:
    python -m scripts.run_ladder
    python -m scripts.run_ladder --tickers AAPL MSFT --epochs 10
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attribution.correlate import high_residual_windows, match_to_news
from attribution.null_control import calibrate_match_window, permutation_test, summarize_attribution
from baselines.factor_model import make_fold_scorer as factor_scorer
from baselines.garch import make_fold_scorer as garch_scorer
from baselines.random_walk import make_fold_scorer as random_walk_scorer
from benchmark.cv import make_t1
from benchmark.ladder import evaluate_rung, gate_check
from events.hawkes import branching_ratio, fit_hawkes_exponential
from events.price_events import bar_threshold_events, event_times_array, tick_events_from_recorder
from features.returns import build_feature_frame, log_returns, session_boundary_mask
from features.windows import make_windows
from ingest.gdelt_client import download_gdelt_range, filter_events_for_universe, parse_gdelt_range
from ingest.storage import read_bars, read_ticks
from models.score import make_fold_scorer as tcn_scorer, reconstruction_residual
from models.train import TrainConfig, train
from rpca.rolling_rpca import make_fold_scorer as rpca_scorer

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(universe_path: Path, settings_path: Path) -> tuple[dict, dict]:
    with open(universe_path) as f:
        universe_cfg = yaml.safe_load(f)
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    return universe_cfg, settings


def log_returns_from_bars(bars: pd.DataFrame) -> pd.Series:
    """Session-boundary returns are excluded (NaN'd), not computed as a
    normal 1-minute return -- see
    features.returns.session_boundary_mask and
    diagnostics/2026-08-11-session-boundary-returns/findings.md. Matters
    here too: an un-excluded overnight/weekend jump shared across many
    tickers on the same calendar boundary would show up as a spurious
    "common" (low-rank) move in Rung 3's cross-sectional RPCA decomposition.
    """
    g = bars.sort_values("timestamp")
    close = g.set_index("timestamp")["close"]
    r = log_returns(close)
    boundary = session_boundary_mask(close.index).loc[r.index]
    return r.mask(boundary)


# --------------------------------------------------------------------------
# Rung 1
# --------------------------------------------------------------------------


def run_rung1_hawkes(data_dir: Path, settings: dict) -> dict:
    """Bar-proxy (always runs) + real-tick check (gated on accumulated
    data -- see diagnostics/2026-08-11-hawkes-bar-proxy-underdispersion/
    for why the bar-proxy result should be near zero, not ~0.81).
    """
    result: dict = {"bar_proxy": None, "tick": None, "tick_skipped_reason": None}

    spy_bars = read_bars(data_dir, tickers=["SPY"])
    if not spy_bars.empty:
        events = bar_threshold_events(spy_bars, sigma_threshold=settings["hawkes"]["bar_event_sigma_threshold"])
        times = event_times_array(events, ticker="SPY")
        if len(times) > 10:
            fit = fit_hawkes_exponential(times)
            result["bar_proxy"] = {
                "n_events": len(times),
                "branching_ratio": branching_ratio(fit),
                "converged": fit.converged,
                "expected_max": settings["hawkes"]["bar_proxy_max_branching_ratio"],
            }

    ticks = read_ticks(data_dir, tickers=["SPY"])
    if ticks.empty:
        result["tick_skipped_reason"] = "no real tick data recorded yet"
        return result

    span_days = (ticks["timestamp"].max() - ticks["timestamp"].min()).total_seconds() / 86400
    if len(ticks) < 5000 or span_days < 5:
        result["tick_skipped_reason"] = f"{len(ticks)} ticks over {span_days:.1f} days (need >=5000 over >=5 days)"
        return result

    tick_events = tick_events_from_recorder(ticks, sigma_threshold=settings["hawkes"]["bar_event_sigma_threshold"])
    tick_times = event_times_array(tick_events, ticker="SPY")
    fit = fit_hawkes_exponential(tick_times)
    result["tick"] = {
        "n_events": len(tick_times),
        "branching_ratio": branching_ratio(fit),
        "converged": fit.converged,
        "plausible_band": settings["hawkes"]["plausible_band"],
    }
    return result


# --------------------------------------------------------------------------
# Temporal lane: Rung 0 -> Rung 2a -> Rung 4
# --------------------------------------------------------------------------


def run_temporal_lane(ticker: str, bars: pd.DataFrame, settings: dict, train_config: TrainConfig) -> dict | None:
    feature_frame = build_feature_frame(bars, vol_window=15, volume_window=15)
    window_len = settings["windows"]["window_minutes"]
    if len(feature_frame) < window_len * 3:
        return None

    returns_series = feature_frame["log_return"]
    embargo = pd.Timedelta(minutes=settings["cv"]["embargo_minutes"])
    n_splits = settings["cv"]["n_splits"]

    t1_returns = make_t1(returns_series.index, pd.Timedelta(minutes=1))
    try:
        rung0 = evaluate_rung("rung0", random_walk_scorer(returns_series), t1_returns, n_splits=n_splits, embargo_td=embargo)
        rung2a = evaluate_rung(
            "rung2a_garch", garch_scorer(returns_series), t1_returns, n_splits=n_splits, embargo_td=embargo
        )
    except ValueError:
        return None

    stride = settings["windows"]["stride_minutes"]
    windows, end_times = make_windows(feature_frame, window_len=window_len, stride=stride)
    t1_windows = make_t1(end_times, pd.Timedelta(minutes=1))
    try:
        rung4 = evaluate_rung(
            "rung4_tcn_vae", tcn_scorer(windows, config=train_config), t1_windows, n_splits=n_splits, embargo_td=embargo
        )
    except ValueError:
        rung4 = None

    return {
        "ticker": ticker,
        "rung0": rung0,
        "rung2a": rung2a,
        "rung4": rung4,
        "rung2a_beats_rung0": gate_check(rung2a, rung0),
        "rung4_beats_rung2a": gate_check(rung4, rung2a) if rung4 is not None else None,
        "windows": windows,
        "end_times": end_times,
    }


# --------------------------------------------------------------------------
# Cross-sectional lane: Rung 2b -> Rung 3
# --------------------------------------------------------------------------


def run_cross_sectional_lane(tickers: list[str], data_dir: Path, settings: dict) -> dict | None:
    returns_by_ticker = {}
    for t in tickers:
        bars = read_bars(data_dir, tickers=[t])
        if not bars.empty:
            returns_by_ticker[t] = log_returns_from_bars(bars)

    spy_bars = read_bars(data_dir, tickers=["SPY"])
    if spy_bars.empty or not returns_by_ticker:
        return None
    spy_returns = log_returns_from_bars(spy_bars)

    common_idx = spy_returns.index
    for r in returns_by_ticker.values():
        common_idx = common_idx.intersection(r.index)
    common_idx = common_idx.sort_values()

    spy_aligned = spy_returns.loc[common_idx]
    feature_matrix = pd.DataFrame({t: r.loc[common_idx] for t, r in returns_by_ticker.items()})
    # log_returns_from_bars NaNs out session-boundary returns (see its
    # docstring) -- drop those rows rather than feed NaN into RPCA's SVD.
    # A boundary shared across the whole market (e.g. Monday's open) would
    # otherwise NaN nearly every column on the same row.
    valid = feature_matrix.notna().all(axis=1) & spy_aligned.notna()
    common_idx = common_idx[valid.to_numpy()]
    spy_aligned = spy_aligned.loc[common_idx]
    feature_matrix = feature_matrix.loc[common_idx]
    if len(common_idx) < 100:
        return None

    embargo = pd.Timedelta(minutes=settings["cv"]["embargo_minutes"])
    n_splits = settings["cv"]["n_splits"]
    t1 = make_t1(common_idx, pd.Timedelta(minutes=1))

    factor_results = {}
    for t in feature_matrix.columns:
        try:
            factor_results[t] = evaluate_rung(
                f"rung2b_{t}", factor_scorer(feature_matrix[t], spy_aligned), t1, n_splits=n_splits, embargo_td=embargo
            )
        except ValueError:
            continue
    if not factor_results:
        return None
    factor_mean_nll = float(np.mean([r.mean_nll for r in factor_results.values()]))

    rpca_result = evaluate_rung("rung3_rpca", rpca_scorer(feature_matrix), t1, n_splits=n_splits, embargo_td=embargo)

    return {
        "factor_results": factor_results,
        "factor_mean_nll": factor_mean_nll,
        "rpca_result": rpca_result,
        "rpca_beats_factor": rpca_result.mean_nll < factor_mean_nll,
    }


# --------------------------------------------------------------------------
# Rung 5: news attribution
# --------------------------------------------------------------------------


def run_rung5_attribution(
    temporal_results: dict[str, dict],
    universe_cfg: dict,
    settings: dict,
    news_days: int,
    data_dir: Path,
    train_config: TrainConfig,
    anomaly_percentile: float = 98.0,
) -> pd.DataFrame | None:
    """Trains a final TCN-VAE per ticker on its full window set (the CV
    folds in run_temporal_lane only produce aggregate NLLs, not a usable
    per-window score series), scores every window, and flags the top
    `100 - anomaly_percentile`% as anomalies (self-calibrated per ticker,
    not a shared absolute threshold). Restricted to the most recent
    `news_days` of the analysis period -- GDELT's 15-min-interval files
    make a full-backfill-length pull impractical (see docs/data_sources.md).
    """
    usable = {t: r for t, r in temporal_results.items() if r is not None and len(r["windows"]) > 50}
    if not usable:
        return None

    news_end = max(r["end_times"].max() for r in usable.values())
    news_start = news_end - pd.Timedelta(days=news_days)

    news_cache_dir = data_dir / "gdelt_cache"
    paths = download_gdelt_range(news_start, news_end, news_cache_dir)
    gkg = parse_gdelt_range(paths)
    if gkg.empty:
        return None
    matches = filter_events_for_universe(gkg, universe_cfg["universe"])

    fallback_window = pd.Timedelta(minutes=settings["attribution"]["match_window_minutes"])
    target_null_rate = settings["attribution"]["target_null_rate"]
    observation_minutes = (news_end - news_start).total_seconds() / 60
    n_permutations = settings["attribution"]["n_permutations"]
    alpha = settings["attribution"]["sidak_alpha"]

    permutation_results = {}
    calibrated_windows = {}
    for ticker, result in usable.items():
        fit_result = train(result["windows"], config=train_config)
        scores = reconstruction_residual(fit_result.model, result["windows"])
        score_series = pd.Series(scores, index=result["end_times"])

        recent_mask = (score_series.index >= news_start) & (score_series.index <= news_end)
        recent_scores = score_series[recent_mask]
        if len(recent_scores) < 20:
            continue

        threshold = np.percentile(recent_scores, anomaly_percentile)
        anomaly_times = high_residual_windows(recent_scores, threshold)
        if len(anomaly_times) == 0:
            continue

        ticker_news = matches[matches["ticker"] == ticker]
        # Per-ticker calibrated window (see calibrate_match_window) instead
        # of one fixed window for every ticker -- a flat window saturates
        # high-coverage names (real and null match rates both ~100%, no
        # power to distinguish signal from noise) while under-using sparse
        # ones. Falls back to the flat setting only when a ticker has zero
        # news events to calibrate against (nothing to compute from).
        window = calibrate_match_window(len(ticker_news), observation_minutes, target_null_rate=target_null_rate)
        window = window if window is not None else fallback_window
        calibrated_windows[ticker] = window

        permutation_results[ticker] = permutation_test(
            anomaly_times, ticker_news, window, n_permutations=n_permutations, seed=hash(ticker) % (2**31)
        )

    if not permutation_results:
        return None
    summary = summarize_attribution(permutation_results, alpha=alpha)
    summary["match_window_minutes"] = summary["ticker"].map(
        lambda t: calibrated_windows[t].total_seconds() / 60
    )
    return summary


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def write_report(
    output_dir: Path,
    rung1: dict,
    temporal_results: dict[str, dict],
    cross_sectional: dict | None,
    attribution_summary: pd.DataFrame | None,
) -> Path:
    date_str = dt.date.today().isoformat()
    report_dir = output_dir / f"{date_str}-full-ladder-run"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    lines = [f"# Full ladder run: {date_str}", ""]

    lines.append("## Rung 1: Hawkes branching ratio (trust gate)")
    if rung1["bar_proxy"]:
        bp = rung1["bar_proxy"]
        status = "PASS" if bp["branching_ratio"] < bp["expected_max"] else "FAIL"
        lines.append(
            f"- Bar-proxy (SPY): branching_ratio={bp['branching_ratio']:.4f} "
            f"(expect < {bp['expected_max']}) [{status}], {bp['n_events']} events"
        )
    else:
        lines.append("- Bar-proxy: not enough events to fit")
    if rung1["tick"]:
        tk = rung1["tick"]
        lo, hi = tk["plausible_band"]
        status = "PASS" if lo <= tk["branching_ratio"] <= hi else "FAIL"
        lines.append(
            f"- Real-tick replication (SPY): branching_ratio={tk['branching_ratio']:.4f} "
            f"(expect in [{lo}, {hi}]) [{status}], {tk['n_events']} events"
        )
    else:
        lines.append(f"- Real-tick replication: SKIPPED -- {rung1['tick_skipped_reason']}")
    lines.append("")

    lines.append("## Temporal lane (Rung 0 -> Rung 2a GARCH -> Rung 4 TCN-VAE)")
    lines.append("| Ticker | Rung0 NLL | Rung2a NLL | 2a beats 0 | Rung4 NLL | 4 beats 2a |")
    lines.append("|---|---|---|---|---|---|")
    for ticker, r in sorted(temporal_results.items()):
        if r is None:
            lines.append(f"| {ticker} | insufficient data | | | | |")
            continue
        r4_nll = f"{r['rung4'].mean_nll:.4f}" if r["rung4"] else "N/A"
        r4_beats = str(r["rung4_beats_rung2a"]) if r["rung4"] else "N/A"
        lines.append(
            f"| {ticker} | {r['rung0'].mean_nll:.4f} | {r['rung2a'].mean_nll:.4f} | "
            f"{r['rung2a_beats_rung0']} | {r4_nll} | {r4_beats} |"
        )
    lines.append("")

    lines.append("## Cross-sectional lane (Rung 2b factor model -> Rung 3 RPCA)")
    if cross_sectional:
        lines.append(f"- Factor model (avg across {len(cross_sectional['factor_results'])} tickers): "
                     f"mean_nll={cross_sectional['factor_mean_nll']:.4f}")
        lines.append(f"- RPCA (whole basket): mean_nll={cross_sectional['rpca_result'].mean_nll:.4f}")
        lines.append(f"- RPCA beats factor model: {cross_sectional['rpca_beats_factor']}")
    else:
        lines.append("- SKIPPED -- insufficient aligned cross-sectional data")
    lines.append("")

    lines.append("## Rung 5: News-correlation attribution")
    if attribution_summary is not None and not attribution_summary.empty:
        lines.append(attribution_summary.to_markdown(index=False))
    else:
        lines.append("- SKIPPED -- insufficient news data or anomaly windows in the analysis period")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full benchmark ladder end-to-end on real data.")
    parser.add_argument("--universe", type=Path, default=PROJECT_ROOT / "config" / "universe.yaml")
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "config" / "settings.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    parser.add_argument("--tickers", nargs="*", default=None, help="Subset of tickers (default: full universe)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--news-days", type=int, default=3)
    args = parser.parse_args()

    universe_cfg, settings = load_config(args.universe, args.settings)
    tickers = args.tickers or [row["ticker"] for row in universe_cfg["universe"]]
    train_config = TrainConfig(epochs=args.epochs, batch_size=32, beta=0.1, patience=3, num_threads=1)

    print(f"Running ladder for {len(tickers)} tickers: {tickers}")

    print("\n=== Rung 1: Hawkes ===")
    rung1 = run_rung1_hawkes(args.data_dir, settings)
    print(rung1)

    print("\n=== Temporal lane ===")
    temporal_results = {}
    for ticker in tickers:
        bars = read_bars(args.data_dir, tickers=[ticker])
        if bars.empty:
            print(f"{ticker}: no data, skipping")
            temporal_results[ticker] = None
            continue
        print(f"{ticker}: running temporal lane...")
        temporal_results[ticker] = run_temporal_lane(ticker, bars, settings, train_config)

    print("\n=== Cross-sectional lane ===")
    cross_sectional = run_cross_sectional_lane(tickers, args.data_dir, settings)
    print(cross_sectional)

    print("\n=== Rung 5: Attribution ===")
    attribution_summary = run_rung5_attribution(
        temporal_results, universe_cfg, settings, args.news_days, args.data_dir, train_config
    )
    print(attribution_summary)

    report_path = write_report(args.output_dir, rung1, temporal_results, cross_sectional, attribution_summary)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
