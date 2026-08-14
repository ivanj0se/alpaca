# Cox-Hawkes with an observed RPCA common-factor baseline: real, small, negative effect

Date: 2026-08-14

Tier 4 (RPCA half) of the private self-excitation research extension.
Question: does replacing the constant baseline `mu` in a standard Hawkes
fit with a covariate-driven baseline `mu(t) = mu0 * exp(gamma * x(t))`,
where `x(t)` is this project's own real, already-validated RPCA
cross-sectional common factor (`rpca/rolling_rpca.py`, standardized),
genuinely improve the fit to real SPY Hawkes-flagged events -- and if so,
in which direction?

## Setup

- Events: SPY tick-level anomaly events from `data_sip_diagnostic/` via
  `events/price_events.py::tick_events_from_recorder`
  (`sigma_threshold=2.0`, `config/settings.yaml`), 59,916 raw events,
  58,850 falling within the RPCA common factor's coverage window
  (2026-05-14 to 2026-08-07).
- Covariate: rolling-RPCA common factor, 307 points over the same span,
  standardized (mean~0, std=1, right-skewed -- min -1.58, max +7.03,
  consistent with a commonality-magnitude-like quantity rather than a
  signed return factor).
- Alignment: `pd.merge_asof(direction="backward")` between event
  timestamps and the covariate's own (much coarser) timestamps -- see
  Errors below for why this replaced an earlier manual
  reindex/union/`.loc[]` chain.
- `beta` fixed at the standard (constant-`mu`) fit's own converged value
  (0.367152), not refit jointly -- isolates "does adding the
  covariate-driven baseline help" from "did the excitation timescale
  itself change," same discipline as the unit tests for this module.

## Result

| Fit | mu / mu0 | gamma | alpha | branching ratio | loglik |
|---|---|---|---|---|---|
| Standard (constant mu) | 0.001820 | -- | 0.342162 | 0.9319 | -90723.56 |
| Cox-Hawkes (RPCA baseline) | 0.001888 | **-0.1396** | 0.342168 | 0.9320 | -90627.95 |

Log-likelihood improvement: **+95.62** for one extra parameter (gamma).
Likelihood-ratio statistic = 2 x 95.62 = 191.2 on 1 degree of freedom --
astronomically significant by the usual chi-squared reference, far past
any reasonable multiple-comparisons discount for having tried this at
all. This is a real, robust effect at this sample size, not a marginal
one.

**gamma is negative.** Holding self-excitation fixed (alpha and the
branching ratio are essentially unchanged between the two fits, 0.9319
vs 0.9320), a higher RPCA common factor is associated with a *lower*
baseline event-arrival rate for SPY's own Hawkes-flagged anomalies.

## Reading the sign (two candidate mechanisms, neither confirmed)

1. **Commonality crowds out idiosyncratic flagging.** The events being
   modeled here are threshold-crossing anomalies in SPY's own price
   series. If periods of high common-factor magnitude correspond to
   broad, synchronized market-wide moves (smooth, trending, "everything
   moves together"), that kind of move may cross the anomaly threshold
   *less* often than choppier, more idiosyncratic SPY-specific action
   does -- even though the common factor is itself derived partly from
   SPY's own returns. Consistent with the found sign, not proven by it.
2. **Regime confound.** The common factor's construction (rolling RPCA
   over a ~20-30 ticker cross-section) folds in dynamics from the whole
   universe, not just SPY. A negative gamma could reflect some other
   real but unmodeled regime variable that happens to correlate
   negatively with both "common factor elevated" and "SPY anomaly rate,"
   rather than a direct causal link between the two.

No attempt was made here to adjudicate between these -- would need
either a cross-ticker replication (does the same sign show up for other
names, not just SPY) or a lead-lag/Granger-style check, neither of which
this fit does. Reporting the honest, unresolved read rather than picking
the more satisfying story.

## What this does NOT show

This is an *observed*-covariate Hawkes model, not a latent Cox process --
`x(t)` is real, already-computed data, not inferred. The improvement
says the RPCA common factor carries real information about SPY's
baseline anomaly rate beyond what self-excitation alone explains; it
does not establish which of the two mechanisms above (or a third,
unconsidered one) is responsible, and does not imply the common factor
*causes* the anomaly rate to change in either direction.

## Real bug found and fixed along the way

The first version of this real-data script built `event_covariate` via a
manual `common_factor.reindex(common_factor.index.union(event_timestamps)).sort_index().ffill().loc[event_timestamps]`
chain. This raised `ValueError: event_covariate must have one value per
event` inside `fit_cox_hawkes` -- `event_timestamps` has 58,850 real
values but only 58,816 unique ones (34 duplicate microsecond-level
timestamps, consistent with the already-established real phenomenon of
multiple trades sharing a timestamp during bursts). `.union()` against a
`DatetimeIndex` containing internal duplicates did not fully deduplicate
the reindexed result, silently producing a length mismatch. Root cause
not fully chased down inside pandas; instead switched to
`pd.merge_asof(direction="backward")`, the same "as-of" join mechanism
already validated in `attribution/correlate.py::match_to_news`, which
handles duplicate keys correctly by construction (each left row gets
exactly one matched right row, always). `research/cox_hawkes.py` itself
needed no change -- the bug was confined to the throwaway analysis
script's alignment logic, not the module.

## Not resolved

- Whether the negative sign replicates on other tickers.
- Whether it's driven by a specific sub-period (e.g. one volatile week)
  rather than a stable relationship across the full 3-month window --
  not checked here.
- The GDELT/news half of Tier 4 (an alternative real exogenous covariate)
  was treated as optional per the original tier framing ("RPCA and/or
  GDELT") and is not part of this result.
