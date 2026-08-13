"""Zero-intelligence agent-based baseline (Rung G1): the simplest
possible generative mechanism, in the spirit of Gode & Sunder (1993).
`n_agents` independently emit buy(+1)/sell(-1)/no-order(0) signals each
step with probability `order_prob`; log_return_t is a linear function of
the net order imbalance plus i.i.d. noise. No limit order book, no
matching engine, no bid/ask spread model, and deliberately no
herding/volatility-feedback rule (every step is i.i.d. by construction) --
this isolates "does a plausible mechanistic rule alone manufacture
realistic stylized facts" as a clean foil to the Hawkes arm's genuine
point-process self-excitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generators.path import GeneratedPath


@dataclass
class ZeroIntelligenceParams:
    n_agents: int
    order_prob: float
    impact_lambda: float
    noise_std: float


def calibrate_zero_intelligence_params(
    reference_returns: np.ndarray,
    n_agents: int = 200,
    order_prob: float = 0.05,
) -> ZeroIntelligenceParams:
    """Calibrates `impact_lambda` so the model's UNCONDITIONAL return std
    matches real data's -- the same vol-matching spirit as
    baselines/random_walk.py::estimate_gbm_params, since there's no real
    order-imbalance series to regress impact against directly.

    Each agent's per-step signal is nonzero with probability `order_prob`
    (sign uniform), so Var(one agent's signal) = order_prob (mean zero by
    symmetry, E[signal^2] = order_prob). For `n_agents` independent
    agents, Var(net_imbalance) = n_agents * order_prob. Target variance is
    split evenly between the imbalance-driven component and a pure noise
    term -- an explicit, documented 50/50 split, not a claim that this is
    the "correct" decomposition (there's no real data to fit that split
    against), just a defensible and clearly-stated one.
    """
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    if not 0 < order_prob < 1:
        raise ValueError("order_prob must be in (0, 1)")
    target_std = float(np.std(np.asarray(reference_returns, dtype=float), ddof=1))
    imbalance_std = np.sqrt(n_agents * order_prob)
    noise_std = target_std / np.sqrt(2)
    impact_lambda = (target_std / np.sqrt(2)) / imbalance_std
    return ZeroIntelligenceParams(n_agents=n_agents, order_prob=order_prob, impact_lambda=impact_lambda, noise_std=noise_std)


def simulate_zero_intelligence(params: ZeroIntelligenceParams, n_steps: int, seed: int | None = None) -> np.ndarray:
    """Discrete-time linear-impact simulation: at each step, `n_agents`
    independently emit a buy(+1)/sell(-1)/no-order(0) signal
    (P(nonzero)=order_prob, sign uniform), net_imbalance = sum of
    signals, log_return_t = impact_lambda * net_imbalance + noise. Steps
    are i.i.d. by construction -- no persistence or herding mechanism,
    deliberately, per this module's docstring.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        [-1.0, 0.0, 1.0],
        size=(n_steps, params.n_agents),
        p=[params.order_prob / 2, 1 - params.order_prob, params.order_prob / 2],
    )
    net_imbalance = signs.sum(axis=1)
    noise = rng.normal(0, params.noise_std, n_steps)
    return params.impact_lambda * net_imbalance + noise


def generate_zero_intelligence_paths(
    reference_returns: np.ndarray,
    n_steps: int,
    n_sims: int = 25,
    n_agents: int = 200,
    order_prob: float = 0.05,
    seed: int = 0,
) -> list[GeneratedPath]:
    params = calibrate_zero_intelligence_params(reference_returns, n_agents=n_agents, order_prob=order_prob)
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_sims):
        sim_seed = int(rng.integers(0, 2**32 - 1))
        returns = simulate_zero_intelligence(params, n_steps=n_steps, seed=sim_seed)
        paths.append(
            GeneratedPath(
                generator_id="zero_intelligence",
                log_returns=returns,
                seed=sim_seed,
                params={"n_agents": n_agents, "order_prob": order_prob, "impact_lambda": params.impact_lambda},
            )
        )
    return paths
