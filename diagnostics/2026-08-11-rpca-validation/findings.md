# Rung 3 (Robust PCA) validation summary

Date: 2026-08-11

## Synthetic recovery

Standard low-rank + sparse recovery test (n=200, d=20, true rank=2, 5%
sparse corruption at magnitude ~10): converged in 22 iterations, exact rank
recovered, relative L error 4.1%, relative S error 4.9%, 100% sparse
support recall. Matches published PCP recovery behavior.

## Scale-dependence of the relaxation gap

A dense random Gaussian low-rank matrix with **zero** true sparse
corruption should ideally decompose to (L=M, S=0), but PCP is a convex
relaxation and its exact-recovery guarantees are asymptotic/probabilistic.
At small scale (50 rows x 10 cols) the algorithm genuinely attributes a
stable ~15% of the matrix's Frobenius norm to S even with no corruption at
all -- confirmed by tracing mu/rank/||S|| across iterations (not a
runaway/divergence, ||S||/||M|| settles and stays flat). At this project's
actual scale (100+ rows x 20-30 tickers) the same test gives near-zero S
(0-2%) and exact rank recovery. Test suite uses the realistic scale.

Separately: the *reported* rank was inflated (7 instead of 3) at small
scale due to using the raw IALM optimization threshold (1/mu) for rank
reporting -- mu grows geometrically (~1.5^n) and can reach ~1e7x its
starting value by convergence, making 1/mu numerically meaningless as a
rank cutoff late in optimization (floating-point noise in the smallest
singular values crosses an ever-shrinking threshold). Fixed by reporting
rank via a fixed *relative* threshold (1e-3 x largest singular value) on
the converged L, independent of the optimization path.

## Real-data validation: single-ticker anomaly detection

Injected a single obvious return spike into one ticker of a synthetic
20-ticker basket built from real SPY minute returns. With `step=1` (every
row scored), the injected ticker ranked #1 of 20 by a wide margin (score
16.75 vs. median 0.67) at its exact timestamp. An earlier attempt with
`step=5` appeared to fail (the injected ticker didn't rank highly) --
traced to a usage bug in the validation script, not the algorithm:
`rolling_rpca_decompose` only reports a score for a window's *last* row,
so with a stride most individual timestamps (including the injected
anomaly's own timestamp) never get their own score at all. Documented
prominently in the function's docstring to prevent repeating this mistake.

## Real-data validation: cross-sectional lane (Rung 3 vs. Rung 2b)

Using real AAPL, MSFT, NVDA, JPM, GS, UNH, JNJ, XOM, AMZN, GOOGL (vs. SPY),
30 days of minute bars, purged 5-fold CV, 60-min embargo:

| Rung | Mean out-of-sample NLL |
|---|---|
| Factor model (avg across 10 tickers, each vs. SPY alone) | -5.352 |
| Robust PCA (whole 10-ticker basket) | -5.958 |

RPCA beats the factor model average (lower is better) -- unlike the
GARCH-vs-factor-model comparison (see
diagnostics/2026-08-11-garch-vs-factor-model-not-comparable/), this
comparison **is** fair: both scorers use contemporaneous cross-sectional
information and the same NLL form with homoskedastic per-ticker residual
variance. The result matches expectation -- RPCA has access to the whole
basket's cross-sectional structure instead of a single market factor, so a
richer low-rank representation should out-predict a single-factor model,
exactly as the two-lane design in docs/architecture.md anticipated.
