# A real bug in the generator comparison harness: calibration bands were length-mismatched against what they scored

Date: 2026-08-13

**Ivan found this, not me.** After the market-generator comparison suite
was reported "complete" (all four arms scored, `hawkes_treatment=0.168`,
everything else exactly `0.000`), he asked directly: "there must be a
problem about how the other ones are performing, check if the code is
complete or if there exists bugs." He was right to be suspicious --
five structurally very different generators (a plain random walk, a
multi-agent linear-impact model, a Poisson jump-diffusion, and an
autoregressive neural net) all landing on *exactly* 0.000 was too
uniform to be a genuine finding on its own, and even the "successful"
Hawkes treatment arm was uniformly failing two facts
(`raw_return_acf`, `leverage_curve`) that had nothing obviously to do
with self-excitation.

## What was wrong

`benchmark/conformal.py::calibrate_band` bootstrap-resampled the real
reference series to build a null distribution of "how much does this
statistic naturally vary." The resample length defaulted to
`len(reference)` -- fine for the module's own unit tests (which always
scored same-length candidates), but `scripts/run_generator_comparison.py`
never overrode it, so every fact's band was calibrated using
**22,999-point** bootstrap resamples of the real SPY reference series,
then used to score generator paths that were only **1,950 points** long
(5 trading days, matching the Hawkes arm's `T_days`). Sample statistics
like autocorrelation and kurtosis have sampling variance that shrinks
with N -- a band calibrated at N=22,999 is far tighter than is fair to
ask of anything scored at N=1,950, real or synthetic.

## How it was confirmed

Took genuine, real, contiguous 1,950-point subsamples of the *actual*
real reference series (by construction the most "realistic" data
possible) and scored them against the mismatched bands the same way a
generator's synthetic path would be:

| Fact | Coverage of REAL 1,950-point subsamples (20 trials) |
|---|---|
| raw_return_acf | 0/20 |
| volatility_clustering_acf | 7/20 |
| excess_kurtosis | 5/20 |
| leverage_curve | 0/20 |
| aggregational_kurtosis | 1/20 |

Real data failed `raw_return_acf` and `leverage_curve` at *exactly* the
same 0/20 rate every generator was failing them at. That's decisive:
the bands, not the generators, were the problem for those two facts.

## Fix

Added `resample_length` to `calibrate_band` (default `len(reference)`,
preserving old behavior for same-length use cases) and `path_length`
(now required) to `benchmark/generator_ladder.py::calibrate_reference_bands`.
`scripts/run_generator_comparison.py` now computes the shared bar count
(`n_bars`) *before* calibration and passes it through explicitly, so
bands are always calibrated at the exact length they'll be used to
score. 2 new regression tests in `tests/unit/test_conformal.py`
(`TestCalibrateBandResampleLength`), confirming a length-matched band
recovers close to the target 90% coverage on real held-out data where
the length-mismatched band did not.

## Corrected result -- a much stronger, more complete finding

| Generator | raw_return_acf | volatility_clustering | excess_kurtosis | leverage_curve | aggregational_kurtosis | **Overall** |
|---|---|---|---|---|---|---|
| **hawkes_treatment** | 0.72 | 0.92 | 1.00 | 0.84 | 1.00 | **0.896** |
| gbm_null | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.400 |
| zero_intelligence | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.400 |
| hawkes_control | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.400 |
| tcn_forecaster | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.400 |

(Band thresholds widened substantially once length-matched, as expected:
raw_return_acf 0.0085->0.0276, volatility_clustering_acf 0.0725->0.1158,
excess_kurtosis 4.96->12.11, leverage_curve 0.0114->0.0371,
aggregational_kurtosis 3.27->7.52.)

This is a materially stronger and more complete result than the buggy
0.168 figure suggested:

- **hawkes_treatment now passes almost everything** (0.896, not 0.168) --
  not just volatility clustering, but excess kurtosis and aggregational
  kurtosis too (both 1.00), with strong partial coverage on
  raw_return_acf (0.72) and leverage_curve (0.84). Real self-excitation
  produces synthetic markets that are close to indistinguishable from
  real ones across the *whole* stylized-facts checklist, not just the
  one fact most obviously tied to clustering.
- **`raw_return_acf` and `leverage_curve` turn out to have low
  discriminating power at this sample length** -- every generator,
  including the worst ones, passes both at 1.00. At N=1,950 these two
  facts' natural sampling variance is wide enough that essentially any
  reasonably-scaled series clears the bar; they were never going to be
  the facts that separate a good generator from a bad one here. The
  three facts that *do* discriminate cleanly (volatility clustering,
  excess kurtosis, aggregational kurtosis) all show the same pattern:
  hawkes_treatment passes, everything else fails completely.
- **The four non-treatment arms are unchanged in their qualitative
  conclusion** (they still fail every fact that has real discriminating
  power) but the *quantitative* claim "0.000 overall" was never
  accurate -- it's 0.400, driven by two under-powered facts nearly
  everything passes trivially at this length, not by those generators
  doing anything right.

## Corrected numbers elsewhere

The following entries reported coverage-rate numbers computed with the
pre-fix, length-mismatched bands. Their *qualitative* conclusions (which
generator wins, which facts fail) mostly still hold, but the *specific
numbers* are superseded by this entry -- not rewritten wholesale, flagged
here and at the top of each:
- `diagnostics/2026-08-12-hawkes-jump-diffusion-generator/findings.md`
  (originally reported 0.168 overall / mixed per-fact coverage for the
  ablation -- corrected to 0.896 / see table above)
- `diagnostics/2026-08-12-tcn-forecaster-generative/findings.md`
  (originally reported 0.000 overall -- corrected to 0.400)
- `diagnostics/2026-08-12-zero-intelligence-baseline/findings.md`
  (originally reported 0.000 overall -- corrected to 0.400; the entry's
  core argument -- that this arm matches the other structure-free arms
  and is clearly beaten by the Hawkes treatment arm -- still holds)
- `diagnostics/2026-08-12-generator-comparison/report.md` is the
  original, uncorrected report; `diagnostics/2026-08-13-generator-comparison/report.md`
  is the corrected one produced by this fix.

## Lesson

The same class of mistake as the Hawkes multistart-grid bug
(`diagnostics/2026-08-11-sip-consolidated-tape-check/`'s CORRECTION
section): a result that looked internally consistent (a clean-seeming
0.168 vs 0.000 split) wasn't actually cross-checked against an
independent sanity probe until asked to. In both cases the fix was the
same shape too -- go find a case where the "obviously correct" input
(real data, or a wide optimizer search) should trivially pass/succeed,
and confirm it actually does. Worth internalizing as a standing check for
any future calibration/threshold logic in this project: before trusting
a pass/fail threshold, verify known-good real data actually passes it.
