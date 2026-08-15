# market-endogeneity

Personal research project (not a trading bot) -- see README.md and
docs/architecture.md for what this is and why.

## Status (2026-08-11)

Full 6-rung benchmark ladder is built and tested (236+ passing tests):
Rung 0 (random walk null) -> Rung 2a (GARCH) -> Rung 4 (TCN-VAE) in the
temporal lane; Rung 2b (factor model) -> Rung 3 (Robust PCA) in the
cross-sectional lane; Rung 1 (Hawkes branching ratio) is the trust gate;
Rung 5 (news attribution) is the actual deliverable. See
docs/architecture.md for the two-lane design and why it's split that way.

- `data/` has ~90 days of real minute bars backfilled for the full
  universe + SPY (`python -m ingest.historical_bars --start ... --end ...`
  to extend/refresh).
- `ingest/tick_recorder.py` runs continuously via launchd (see below),
  recording real live ticks. Real tick-level history can *also* be
  backfilled directly via `python -m ingest.historical_trades --tickers
  SPY --start ... --end ...` (Alpaca's free IEX feed has at least a year
  of real historical trade-level data, not just bars -- confirmed
  2026-08-11, see diagnostics/2026-08-11-real-tick-hawkes-replication/).
  Both write into the same `data/ticks` store with an identical schema.
  The Rung 1 real-tick gate (`>=5000 ticks over >=5 days`) is unblocked as
  of 2026-08-11 (489,575 real SPY ticks over 29.3 days backfilled) --
  first real result: branching_ratio=0.9969, outside the configured
  plausible_band, an open methodological question, not yet resolved (see
  that diagnostics entry).
- `scripts/run_ladder.py` runs the full ladder end-to-end on real data and
  writes a dated report to `diagnostics/<date>-full-ladder-run/report.md`.
  Check that report for the latest real numbers before assuming anything
  here is current -- rerun periodically as more data accumulates.

**Read `diagnostics/` before touching the math-heavy modules** (events/hawkes.py,
baselines/garch.py, rpca/inexact_alm.py, models/score.py,
attribution/null_control.py, features/returns.py, events/price_events.py,
ingest/storage.py). Every one of them had at least one real, non-obvious bug caught only by
running against real data, not by unit tests alone -- e.g. an MLE
optimizer that reported `converged=True` after making zero real progress
(bad initial-guess scaling), a walk-forward evaluator that made a good
model look worse than a naive baseline (static vs. one-step-ahead
forecasting), a "2000 random permutations" significance test that was
silently a no-op (permuting a fixed set among itself changes nothing), an
NLL comparison that wasn't apples-to-apples across differently-scaled
targets, and every return-series computation in the codebase diffing
straight across session boundaries (overnight/weekend/holiday gaps
computed as ordinary 1-minute returns, confirmed 4.4x larger on average
than genuine intraday moves on real data -- see
diagnostics/2026-08-11-session-boundary-returns/). The fixes are in the
code; the diagnostics entries explain *why*, which matters if you're
adding a new rung or metric and want to avoid repeating the same class of
mistake -- in particular, any new return computation should go through
`features/returns.py::session_boundary_mask` rather than reimplementing a
naive `np.diff(np.log(close))`.

## Market-generator comparison suite (2026-08-12)

A second, separate initiative from the detection ladder above -- see
`/Users/ivanpaiewonsky/.claude/plans/fuzzy-prancing-meteor.md` for the
full plan. Answers a different question: not "how anomalous is this
window" but "can we generate synthetic price paths that statistically
match real markets, with calibrated confidence." Four generator arms
(`generators/hawkes_jump_diffusion.py`, `generators/zero_intelligence.py`,
`baselines/random_walk.py::simulate_gbm` reused as the null,
`models/tcn_forecaster.py` + `forecaster_generate.py`) are scored through
one shared harness (`benchmark/stylized_facts.py` -- Cont 2001's five
checks; `benchmark/conformal.py` -- block-bootstrap calibrated bands,
deliberately not called "conformal prediction" since real returns aren't
exchangeable; `benchmark/generator_ladder.py` -- the generative analogue
of `benchmark/ladder.py`). Run via
`python -m scripts.run_generator_comparison --ticker SPY`, writes
`diagnostics/<date>-generator-comparison/report.md`.

**Result as of 2026-08-13** (corrected -- see below):
Hawkes self-excitation (real, live-refit branching_ratio) is the only
mechanism among everything tested that produces measurably realistic
market dynamics -- overall_score=0.896 vs. 0.400 for the
zero-self-excitation control, GBM null, zero-intelligence agents, and the
TCN-forecaster alike (all four fail every fact with real discriminating
power at this sample length -- volatility clustering, excess kurtosis,
aggregational kurtosis -- while trivially passing two facts,
raw_return_acf and leverage_curve, that turn out to have low
discriminating power for anyone at this length).

Three real bugs were caught building this, all worth reading before
touching the relevant module:
1. The Hawkes optimizer's multistart grid was too narrow (silently
   returned a wrong branching ratio that looked plausible -- see
   `diagnostics/2026-08-11-sip-consolidated-tape-check/`'s CORRECTION
   section) -- read before touching
   `events/hawkes.py::fit_hawkes_exponential_multistart`.
2. `TCNForecaster`'s logvar clamp was blindly copied from `TCNVAE` at the
   wrong scale (~140x too large -- see
   `diagnostics/2026-08-12-tcn-forecaster-generative/`) -- read before
   touching `models/tcn_forecaster.py`.
3. The comparison harness itself calibrated confidence bands against the
   reference data's full length (~23,000 points) but scored much shorter
   generator paths (~1,950 points) against them -- silently rejecting
   everything, real or synthetic, from the sample-size mismatch alone,
   not genuine unrealism. Found because Ivan asked "check if the code is
   complete or if there exists bugs" after noticing suspiciously uniform
   0.000 scores across structurally unrelated generators -- see
   `diagnostics/2026-08-13-conformal-band-length-mismatch/`. Read before
   touching `benchmark/conformal.py::calibrate_band` or
   `benchmark/generator_ladder.py::calibrate_reference_bands` --
   `resample_length`/`path_length` must always match whatever will
   actually be scored against the calibrated band.

## Self-excitation research extension -- CLOSED (2026-08-15)

A private, unpublished six-tier follow-on to the generator suite above,
each tier chosen to try something genuinely new rather than replicate
published work. See `diagnostics/2026-08-15-self-excitation-research-wrap-up/`
for the full closing summary. Short version: real, statistically
significant structure was found and validated in several places --
multi-timescale self-excitation (`research/multi_kernel_hawkes.py`,
IEX/SIP operate at genuinely different clock speeds and the multi-kernel
arm generatively beats the original single-exponential baseline, 0.904
vs 0.864), a real RPCA exogenous-baseline effect
(`research/cox_hawkes.py`, gamma=-0.1396), rough-volatility-consistent
Hurst exponents (H≈0.08-0.11, matching El Euch/Fukasawa/Rosenbaum's
theory), and a real sign-asymmetric leverage effect
(`research/asymmetric_quadratic_hawkes.py`, kappa_minus > kappa_plus,
p=0.00034).

**None of it is a tradable edge.** Tested directly on real data across
five signal types (the above plus news tone and order-flow imbalance) --
every real effect found falls short of realistic trading costs by
roughly an order of magnitude or more (best case: order-flow imbalance's
top-decile subset, still ~18x short). This is a real, evidence-based
finding from this extension, not an assumption -- see
`diagnostics/2026-08-15-leverage-signal-tradability-check/` and
`diagnostics/2026-08-15-alternative-signal-sources/`. Consistent with
this project's standing "not a trading bot" framing.

Closed as of this entry -- no further tiers planned. If this line of
work resumes, start from a fresh question rather than extending these
specific models further.

## Git

Remote: https://github.com/ivanj0se/alpaca (branch `main`, pushed over
HTTPS via the `gh` CLI's stored credentials -- SSH is not configured for
this account on this machine).

**Commit and push after making a meaningful change** (a working module +
its tests, a completed build-order phase, etc.) rather than batching many
sessions' worth of work into one commit. Small, working commits, each with
passing tests at HEAD.

Never commit `.env` or anything under `data/` -- both are gitignored;
double-check `git status` before `git add` if either ever shows up
unexpectedly.

## Tick recorder (always-on)

`ingest/tick_recorder.py` runs as a launchd LaunchAgent, not a foreground
process -- it needs to survive terminal close, logout, and reboot, since
real historical tick data doesn't exist any other way (see
docs/architecture.md).

- Plist: `~/Library/LaunchAgents/com.ivan.market-endogeneity.tickrecorder.plist`
- `KeepAlive`: true (auto-restarts on crash) -- `RunAtLoad`: true (starts on login)
- Logs: `logs/tick_recorder.log` / `logs/tick_recorder.error.log` (gitignored)
- Status: `launchctl list | grep market-endogeneity`
- Stop: `launchctl bootout gui/$(id -u)/com.ivan.market-endogeneity.tickrecorder`

**Restart procedure (read before touching this live process):** bootout,
then run `ps aux | grep tick_recorder` and confirm **zero** processes are
running -- if bootout doesn't respond within a few seconds, `kill -KILL`
the PID rather than leaving it ambiguous. Wait a beat, then one single
`launchctl bootstrap gui/$(id -u) <plist path>`. Never start a manual
fallback (nohup, etc.) "just in case" while unsure the old instance is
fully gone -- two instances holding live connections at once exhausts
Alpaca's per-account connection limit and puts the SDK's own reconnect
loop into a backoff-free hammering loop that's hard to distinguish from a
real outage (see
diagnostics/2026-08-11-tick-recorder-connection-limit-incident/, a real
incident from exactly this). `run_forever` now refuses to start a second
instance itself (`AlreadyRunningError`, via a PID-file lock at
`logs/tick_recorder.lock`) as a backstop, but the manual discipline above
still matters -- the lock only protects against literally-simultaneous
starts, not a `launchctl bootstrap` that itself transiently fails and
tempts a manual workaround.

## Before committing

Run the test suite:
```bash
source .venv/bin/activate
pytest
```
`pytest -m live tests/integration` hits the real Alpaca API and requires
`.env` -- not run by default, run it explicitly when checking live
connectivity.
