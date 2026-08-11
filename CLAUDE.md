# market-endogeneity

Personal research project (not a trading bot) -- see README.md and
docs/architecture.md for what this is and why.

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
- Restart after code changes: bootout, then `launchctl bootstrap gui/$(id -u) <plist path>`

## Before committing

Run the test suite:
```bash
source .venv/bin/activate
pytest
```
`pytest -m live tests/integration` hits the real Alpaca API and requires
`.env` -- not run by default, run it explicitly when checking live
connectivity.
