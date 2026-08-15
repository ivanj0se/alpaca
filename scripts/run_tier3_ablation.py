"""Tier 3: controlled ablation of the two self-excitation EXTENSIONS
(multi-kernel Hawkes, Cox-Hawkes with a real RPCA baseline) against the
already-scored single-exponential Hawkes control/treatment arms, all
scored through the SAME shared stylized-facts/conformal harness used by
the original market-generator comparison suite
(scripts/run_generator_comparison.py -- see CLAUDE.md's "Market-generator
comparison suite" section for that suite's already-reported result,
hawkes_treatment overall_score=0.896).

Deliberately reuses `data/` (the IEX-fed always-on-recorder store) for
every arm's real tick data, NOT `data_sip_diagnostic/` (the higher-
quality SIP consolidated tape used for this session's earlier real-data
diagnostics runs) -- the single-exponential control/treatment arms'
already-published 0.896 baseline was fit on `data/`, and a genuinely
controlled ablation requires every arm compared here to share the exact
same underlying tick source. The multi-kernel and Cox-Hawkes real fitted
parameters reported by THIS script will therefore differ somewhat from
the SIP-based numbers in diagnostics/2026-08-13-multi-timescale-hawkes/
and diagnostics/2026-08-13-cox-hawkes-rpca-baseline/ -- expected, not a
bug, and noted in this run's own diagnostics writeup.

Usage:
    python -m scripts.run_tier3_ablation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmark.generator_ladder import calibrate_reference_bands, evaluate_generator, rank_generators, save_generator_report
from features.returns import build_feature_frame
from generators.hawkes_extensions_generator import generate_cox_hawkes_paths, generate_multi_kernel_paths
from generators.hawkes_jump_diffusion import generate_ablation_paths
from ingest.storage import read_bars

PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    with open(PROJECT_ROOT / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    gen_cfg = settings["generators"]
    sf_cfg = settings["stylized_facts"]
    cf_cfg = settings["conformal"]
    hawkes_cfg = dict(gen_cfg["hawkes_jump_diffusion"])
    ticker = hawkes_cfg["ticker"]
    data_dir = PROJECT_ROOT / "data"
    agg_scales = tuple(sf_cfg["agg_scales"])

    print(f"=== Loading real reference data for {ticker} (from {data_dir}) ===")
    bars = read_bars(data_dir, tickers=[ticker])
    if bars.empty:
        raise SystemExit(f"no real bar data for {ticker} in {data_dir}")
    frame = build_feature_frame(bars, vol_window=15, volume_window=15)
    reference_returns = frame["log_return"].to_numpy()
    print(f"{len(reference_returns)} real reference returns")

    n_bars = int(np.ceil(hawkes_cfg["T_days"] * 6.5 * 3600 / hawkes_cfg["bar_seconds"]))

    print("\n=== Calibrating reference bands (real data only, same convention as the original comparison) ===")
    bands, reference_facts = calibrate_reference_bands(
        reference_returns,
        path_length=n_bars,
        alpha=cf_cfg["alpha"],
        n_bootstrap=cf_cfg["n_bootstrap"],
        block_size=cf_cfg["block_size"],
        max_lag=sf_cfg["max_lag"],
        leverage_max_lag=sf_cfg["leverage_max_lag"],
        agg_scales=agg_scales,
        seed=cf_cfg["seed"],
    )
    for fact, band in bands.items():
        print(f"  {fact}: threshold={band.threshold:.5f}")

    print("\n=== Single-exponential Hawkes control/treatment (baseline, same as the original comparison) ===")
    hawkes_results = generate_ablation_paths(
        data_dir,
        ticker=ticker,
        sigma_threshold=hawkes_cfg["sigma_threshold"],
        T_days=hawkes_cfg["T_days"],
        bar_seconds=hawkes_cfg["bar_seconds"],
        n_sims=hawkes_cfg["n_sims"],
        max_events=hawkes_cfg["max_events"],
        seed=hawkes_cfg["seed"],
    )

    print("\n=== Multi-kernel Hawkes (Tier 1 extension) ===")
    multi_kernel_paths = generate_multi_kernel_paths(
        data_dir,
        ticker=ticker,
        sigma_threshold=hawkes_cfg["sigma_threshold"],
        T_days=hawkes_cfg["T_days"],
        bar_seconds=hawkes_cfg["bar_seconds"],
        n_sims=hawkes_cfg["n_sims"],
        max_events=hawkes_cfg["max_events"],
        seed=hawkes_cfg["seed"],
    )
    print(
        f"  fitted: mu={multi_kernel_paths[0].params['mu']:.6g}, "
        f"alphas={[round(a, 4) for a in multi_kernel_paths[0].params['alphas']]}, "
        f"betas={[round(b, 4) for b in multi_kernel_paths[0].params['betas']]}"
    )

    print("\n=== Cox-Hawkes RPCA baseline (Tier 4 extension) ===")
    common_factor = pd.read_pickle("/tmp/rpca_common_factor.pkl")
    cox_hawkes_paths = generate_cox_hawkes_paths(
        data_dir,
        common_factor,
        ticker=ticker,
        sigma_threshold=hawkes_cfg["sigma_threshold"],
        T_days=hawkes_cfg["T_days"],
        bar_seconds=hawkes_cfg["bar_seconds"],
        n_sims=hawkes_cfg["n_sims"],
        max_events=hawkes_cfg["max_events"],
        seed=hawkes_cfg["seed"],
    )
    print(
        f"  fitted: mu0={cox_hawkes_paths[0].params['mu0']:.6g}, gamma={cox_hawkes_paths[0].params['gamma']:.4f}, "
        f"alpha={cox_hawkes_paths[0].params['alpha']:.6g}, beta={cox_hawkes_paths[0].params['beta']:.6g}"
    )

    print("\n=== Evaluating all arms against the same reference bands ===")
    all_arms = {
        "hawkes_control": hawkes_results["control"],
        "hawkes_treatment": hawkes_results["treatment"],
        "hawkes_multi_kernel": multi_kernel_paths,
        "cox_hawkes_rpca": cox_hawkes_paths,
    }
    results = []
    for generator_id, paths in all_arms.items():
        result = evaluate_generator(
            generator_id,
            paths,
            reference_facts,
            bands,
            max_lag=sf_cfg["max_lag"],
            leverage_max_lag=sf_cfg["leverage_max_lag"],
            agg_scales=agg_scales,
        )
        results.append(result)
        print(f"{generator_id}: overall_score={result.overall_score:.3f}")

    ranking = rank_generators(results)
    print("\n=== Final ranking ===")
    print(ranking.to_string(index=False))

    report_path = save_generator_report(results, bands, PROJECT_ROOT / "diagnostics", topic="tier3-hawkes-extensions-ablation")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
