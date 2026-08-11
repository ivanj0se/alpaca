# Benchmark ladder: rung2-garch-vs-factor-aapl

Date: 2026-08-10

**Do not read the table below as "factor_model beats garch."** These two
rungs are not comparable on the same NLL axis -- the factor model scores
against a contemporaneous cross-sectional factor (SPY's return at the same
timestamp), which GARCH's purely-past-information forecast doesn't get
access to. Full explanation:
diagnostics/2026-08-11-garch-vs-factor-model-not-comparable/findings.md.
This file is kept as the raw output that prompted that investigation, not
as a standalone conclusion.

| Rung | Mean NLL | Std NLL | Folds |
|---|---|---|---|
| factor_model | -5.7508 | 0.1667 | 6 |
| garch | -5.0375 | 0.0287 | 6 |

Lower mean NLL is better (higher out-of-sample likelihood of the held-out data under the fitted model).