# Private self-excitation research extension: wrap-up

Date: 2026-08-15

Closing entry for the six-tier research extension built on top of the
original market-generator comparison suite (`diagnostics/2026-08-13-generator-comparison/`
and CLAUDE.md's "Market-generator comparison suite" section). This is a
private, unpublished research thread -- each tier was chosen to try
something genuinely new to this project rather than replicate published
work, per the explicit standing constraint this whole extension was
built under.

## What was built and found, tier by tier

1. **Multi-timescale Hawkes kernel** (`research/multi_kernel_hawkes.py`,
   `diagnostics/2026-08-13-multi-timescale-hawkes/`). Real finding: IEX
   and SIP self-excitation operate at genuinely different clock speeds
   (IEX dominated by a single ~5-min timescale; SIP spread across three
   faster ones down to milliseconds), not just different overall
   branching ratios.
2. **News-conditioned marks** (`research/news_conditioned_marks.py`,
   `diagnostics/2026-08-14-news-conditioned-marks/`). SPY blocked by a
   real GDELT coverage gap (index/ETF names aren't captured by GDELT's
   entity extraction). AMZN retry gave a real, clean null: no magnitude
   difference between news-adjacent and self-triggered jumps.
3. **Controlled ablation** (`generators/hawkes_extensions_generator.py`,
   `diagnostics/2026-08-14-tier3-hawkes-extensions-ablation/`). Real
   result: multi-kernel Hawkes generatively BEATS the original
   single-exponential baseline (0.904 vs 0.864 overall_score, same
   harness). Cox-Hawkes/RPCA clears the no-excitation control (0.520 vs
   0.400) but underperforms plain self-excitation. Caught and fixed a
   real length-mismatch bug before trusting the first (wrong) number.
4. **Cox-Hawkes with a real RPCA baseline** (`research/cox_hawkes.py`,
   `diagnostics/2026-08-13-cox-hawkes-rpca-baseline/`). Real, highly
   significant detection-side finding: gamma=-0.1396, loglik +95.62 --
   the RPCA common factor carries real information about SPY's baseline
   anomaly rate beyond self-excitation alone. Sign not mechanistically
   resolved.
5. **Rough volatility / Hurst exponent**
   (`diagnostics/2026-08-13-rough-volatility-check/`). Real, striking
   confirmation of rough-volatility theory: SIP H=0.1085 (R²=0.995), IEX
   H=0.0773 (R²=0.876) -- both near the H≈0.1 literature value, and
   theoretically connected to this project's own near-critical branching
   ratio via El Euch/Fukasawa/Rosenbaum's nearly-unstable-Hawkes scaling
   limit.
6. **Quadratic and sign-asymmetric Hawkes**
   (`research/quadratic_hawkes.py`, `research/asymmetric_quadratic_hawkes.py`,
   `diagnostics/2026-08-14-quadratic-hawkes-real-fit/`,
   `diagnostics/2026-08-15-asymmetric-quadratic-hawkes-real-fit/`). Real,
   strongly significant squared-feedback (Zumbach-style) effect
   (kappa=4.4e-05, loglik +154.74). Caught a real modeling gap on
   reflection -- a symmetric squared term can't represent genuine sign
   asymmetry -- and built the fix: kappa_minus=5.69e-05 >
   kappa_plus=3.56e-05, the classic leverage-effect direction,
   significant beyond the symmetric model alone (p=0.00034).

## The trading-viability question, asked directly and tested honestly

Prompted by a direct question about whether any of this could generate
real trading gains. Tested empirically rather than argued abstractly,
across five real signals on real data:

| Signal | Best gross effect | Net of assumed costs | Diagnostics |
|---|---|---|---|
| Leverage-effect timing | 0.068 bps/trade | -1.93 bps | `2026-08-15-leverage-signal-tradability-check/` |
| News tone | ~0 (p=0.93) | -- | `2026-08-15-alternative-signal-sources/` |
| Order-flow imbalance | 0.11 bps/trade (top decile) | -1.89 bps | `2026-08-15-alternative-signal-sources/` |
| Cross-asset lead-lag | <0.02 corr, incoherent | -- | `2026-08-15-cross-asset-lead-lag/` |

Consistent finding across all five: real, statistically genuine
structure exists in this market data, but every measured effect falls
short of realistic trading costs by roughly an order of magnitude or
more. This isn't "efficient markets, nothing here" -- it's "real
structure, wrong scale for a retail-accessible edge." Confirms
empirically what this project's own framing assumed from the start
(CLAUDE.md: "not a trading bot").

## Overall assessment

This project can now say, with real evidence rather than a prior belief:
its models describe real SPY market dynamics well -- multiple genuine
self-excitation timescales, a real exogenous baseline effect, a real
Zumbach-style trend-sensitivity, and a real sign-asymmetric leverage
effect, each found independently and each holding up under its own
trust-gate (simulate/refit/recover) and real-data validation. None of
that, individually or combined, produces a tradable edge at realistic
costs. Both halves of that statement are real findings from this
extension, not assumptions carried in from outside it.

## Status: closed

No further tiers planned under this initiative as of this entry. The
codebase, tests (all passing), and diagnostics entries listed above are
the permanent record of what was tried and found. Future work on this
project, if any, should start from a fresh question rather than
continuing to extend this specific line -- the self-excitation
structure question this extension set out to answer has real, if now
fairly complete, answers.
