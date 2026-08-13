"""Runs the market-generator comparison suite end-to-end on real data and
produces a dated report -- the generative analogue of scripts/run_ladder.py.
Wires: real reference returns -> one shared set of calibrated stylized-fact
bands -> each generator arm (GBM null, Hawkes control/treatment,
TCN-forecaster) -> evaluate_generator -> ranked report.

Usage:
    python -m scripts.run_generator_comparison
    python -m scripts.run_generator_comparison --ticker SPY --n-sims 25 --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from baselines.random_walk import estimate_gbm_params, simulate_gbm
from benchmark.generator_ladder import calibrate_reference_bands, evaluate_generator, rank_generators, save_generator_report
from features.returns import build_feature_frame
from generators.hawkes_jump_diffusion import generate_ablation_paths
from generators.path import GeneratedPath
from generators.zero_intelligence import generate_zero_intelligence_paths
from ingest.storage import read_bars
from models.forecaster_generate import generate_forecaster_paths

PROJECT_ROOT = Path(__file__).parent.parent


def load_settings(settings_path: Path) -> dict:
    with open(settings_path) as f:
        return yaml.safe_load(f)


def build_gbm_paths(reference_prices, n_steps: int, n_sims: int, seed: int) -> list[GeneratedPath]:
    """Rung G-1: vol-matched GBM null, reusing baselines/random_walk.py's
    already-tested simulate_gbm as-is (the same function used as Rung 0 in
    the predictive ladder).
    """
    params = estimate_gbm_params(reference_prices)
    price_paths = simulate_gbm(params, n_steps=n_steps, n_sims=n_sims, seed=seed)
    return [
        GeneratedPath(generator_id="gbm_null", log_returns=np.diff(np.log(p)), seed=seed, params={"n_steps": n_steps})
        for p in price_paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the market-generator comparison suite end-to-end on real data.")
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "config" / "settings.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics")
    parser.add_argument("--ticker", type=str, default=None, help="Overrides settings.yaml's per-generator ticker")
    parser.add_argument("--n-sims", type=int, default=None, help="Overrides settings.yaml's per-generator n_sims")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides settings.yaml's tcn_forecaster epochs")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    gen_cfg = settings["generators"]
    sf_cfg = settings["stylized_facts"]
    cf_cfg = settings["conformal"]

    hawkes_cfg = dict(gen_cfg["hawkes_jump_diffusion"])
    tcn_cfg = dict(gen_cfg["tcn_forecaster"])
    zi_cfg = dict(gen_cfg["zero_intelligence"])
    ticker = args.ticker or hawkes_cfg["ticker"]
    if args.n_sims is not None:
        hawkes_cfg["n_sims"] = args.n_sims
        tcn_cfg["n_sims"] = args.n_sims
        zi_cfg["n_sims"] = args.n_sims
    if args.epochs is not None:
        tcn_cfg["epochs"] = args.epochs

    agg_scales = tuple(sf_cfg["agg_scales"])

    print(f"=== Loading real reference data for {ticker} ===")
    bars = read_bars(args.data_dir, tickers=[ticker])
    if bars.empty:
        raise SystemExit(f"no real bar data for {ticker} in {args.data_dir}")
    frame = build_feature_frame(bars, vol_window=tcn_cfg["vol_window"], volume_window=tcn_cfg["vol_window"])
    reference_returns = frame["log_return"].to_numpy()
    print(f"{len(reference_returns)} real reference returns")

    print("\n=== Calibrating reference bands (real data only) ===")
    bands, reference_facts = calibrate_reference_bands(
        reference_returns,
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

    # Shared bar count so every generator arm covers the same horizon.
    n_bars = int(np.ceil(hawkes_cfg["T_days"] * 6.5 * 3600 / hawkes_cfg["bar_seconds"]))

    print("\n=== Rung G-1: GBM null ===")
    closes = bars.sort_values("timestamp")["close"]
    gbm_paths = build_gbm_paths(closes, n_steps=n_bars, n_sims=hawkes_cfg["n_sims"], seed=hawkes_cfg["seed"])

    print("\n=== Rung G2: Hawkes jump-diffusion ablation ===")
    hawkes_results = generate_ablation_paths(
        args.data_dir,
        ticker=ticker,
        sigma_threshold=hawkes_cfg["sigma_threshold"],
        T_days=hawkes_cfg["T_days"],
        bar_seconds=hawkes_cfg["bar_seconds"],
        n_sims=hawkes_cfg["n_sims"],
        max_events=hawkes_cfg["max_events"],
        seed=hawkes_cfg["seed"],
    )

    print("\n=== Rung G3: TCN-forecaster autoregressive ===")
    tcn_paths = generate_forecaster_paths(
        args.data_dir,
        ticker=ticker,
        window_len=tcn_cfg["window_len"],
        hidden_dim=tcn_cfg["hidden_dim"],
        dilations=tuple(tcn_cfg["dilations"]),
        epochs=tcn_cfg["epochs"],
        batch_size=tcn_cfg["batch_size"],
        lr=tcn_cfg["lr"],
        patience=tcn_cfg["patience"],
        n_sims=tcn_cfg["n_sims"],
        n_steps=n_bars,
        vol_window=tcn_cfg["vol_window"],
        seed=tcn_cfg["seed"],
    )

    print("\n=== Rung G1: Zero-intelligence agent-based baseline ===")
    zi_paths = generate_zero_intelligence_paths(
        reference_returns,
        n_steps=n_bars,
        n_sims=zi_cfg["n_sims"],
        n_agents=zi_cfg["n_agents"],
        order_prob=zi_cfg["order_prob"],
        seed=zi_cfg["seed"],
    )

    print("\n=== Evaluating all arms ===")
    all_arms = {
        "gbm_null": gbm_paths,
        "zero_intelligence": zi_paths,
        "hawkes_control": hawkes_results["control"],
        "hawkes_treatment": hawkes_results["treatment"],
        "tcn_forecaster": tcn_paths,
    }
    results = []
    for generator_id, paths in all_arms.items():
        result = evaluate_generator(
            generator_id, paths, reference_facts, bands, max_lag=sf_cfg["max_lag"], leverage_max_lag=sf_cfg["leverage_max_lag"], agg_scales=agg_scales
        )
        results.append(result)
        print(f"{generator_id}: overall_score={result.overall_score:.3f}")

    ranking = rank_generators(results)
    print("\n=== Final ranking ===")
    print(ranking.to_string(index=False))

    report_path = save_generator_report(results, bands, args.output_dir)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
