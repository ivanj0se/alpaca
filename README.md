# Market Endogeneity

Research project (not a trading bot): how much of a stock's price behavior is
**endogenous** (the market reacting to itself) versus **exogenous**
(triggered by identifiable real-world news)? A model is trained on price/volume
data only -- it never sees news -- and its anomaly/residual signal is
correlated against a real news feed after the fact to measure the split.

This is a personal project, independent of any employer's infrastructure,
accounts, or credentials.

## Method

Benchmark ladder, each rung gated on beating the previous one (see
`docs/architecture.md`):

0. Random-walk null (sanity floor)
1. Classical Hawkes-process branching ratio, replicated against the published
   Filimonov & Sornette (2012) result (~0.81 on E-mini S&P 500) -- the trust
   gate. Nothing below is believed until this reproduces.
2. GARCH / linear factor model residuals (the "boring baseline")
3. Robust PCA (low-rank + sparse) cross-sectional decomposition
4. TCN-VAE nonlinear reconstruction track
5. News-correlation attribution with a permutation/null-control significance
   test (GDELT)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in ALPACA_API_KEY / ALPACA_SECRET_KEY
pytest                 # runs unit + integration (mocked); live tests skipped by default
```

Alpaca keys: free at https://alpaca.markets/ (a data/paper account, no funding
needed -- this project never places an order).

## Layout

See `docs/architecture.md` for the full design and `docs/data_sources.md` for
data provenance and known caveats (IEX-sourced bars, heuristic GDELT
ticker matching, etc.). Each investigation/benchmark run produces a dated
report under `diagnostics/`.
