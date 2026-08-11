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
- `ingest/tick_recorder.py` runs continuously via launchd (see below) --
  real tick data accumulates automatically now, needed for the Rung 1
  real-tick replication test, which is gated/skipped until enough exists.
- `scripts/run_ladder.py` runs the full ladder end-to-end on real data and
  writes a dated report to `diagnostics/<date>-full-ladder-run/report.md`.
  Last full run (all ~20 tickers, 15 epochs, 3-day news window) was kicked
  off 2026-08-11; check that report for the latest real numbers before
  assuming anything here is current -- rerun periodically as more data
  (especially real ticks) accumulates.

**Read `diagnostics/` before touching the math-heavy modules** (events/hawkes.py,
baselines/garch.py, rpca/inexact_alm.py, models/score.py,
attribution/null_control.py). Every one of them had at least one real,
non-obvious bug caught only by running against real data, not by unit
tests alone -- e.g. an MLE optimizer that reported `converged=True` after
making zero real progress (bad initial-guess scaling), a walk-forward
evaluator that made a good model look worse than a naive baseline (static
vs. one-step-ahead forecasting), a "2000 random permutations" significance
test that was silently a no-op (permuting a fixed set among itself changes
nothing), and an NLL comparison that wasn't apples-to-apples across
differently-scaled targets. The fixes are in the code; the diagnostics
entries explain *why*, which matters if you're adding a new rung or metric
and want to avoid repeating the same class of mistake.

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
