# Operational incident: overlapping restarts exhausted Alpaca's connection limit

Date: 2026-08-11, ~10:52-10:58 EDT, live market hours.

## What happened

While deploying the blocking-flush fix (see
diagnostics/2026-08-11-tick-recorder-blocking-flush/), the restart
sequence was: `launchctl bootout` -> `launchctl bootstrap` (failed with
"Bootstrap failed: 5: Input/output error") -> started a `nohup` fallback
to restore data collection immediately -> retried `launchctl bootstrap`
(succeeded) -> now two instances running simultaneously (the nohup
fallback and the new launchd job) -> killed the nohup instance, but it
didn't respond to SIGTERM and needed SIGKILL.

Within about a minute, the error log went from 61 lines (accumulated over
several hours) to 3,445 lines, almost all `ValueError: connection limit
exceeded` during `_auth()` inside alpaca-py's `_start_ws()`.

## Root cause

Multiple simultaneous connection attempts (old launchd instance not fully
torn down, the nohup fallback, and the new launchd instance all
overlapping within a short window) exceeded Alpaca's per-account
websocket connection limit. Once that happens, alpaca-py's own
`_run_forever` reconnect loop hits this specific failure through a
`ValueError` code path that -- unlike its `websockets.WebSocketException`
handling, which does `await self.close(); self._running = False` -- falls
through to a bare `except Exception: log.exception(...)` with **no**
backoff. The `finally: await asyncio.sleep(0)` yields for one event-loop
tick, not a real delay, so the loop retries essentially instantly,
hammering the auth endpoint hundreds of times a minute until killed. This
is a gap in the SDK's own error handling, not something fixable from this
codebase without patching it.

Separately: SIGTERM did not cleanly stop either the nohup fallback or (on
the next bootout) the launchd instance within a few seconds, requiring
SIGKILL both times. Data integrity was checked afterward and confirmed
clean (no corrupted files, no duplicate timestamps) despite the brief
window of concurrent writes to the same partition files from two
processes -- worth understanding *why* shutdown didn't respond promptly
if this recurs, but not chased further tonight since it happened
specifically during the connection-limit failure storm (the event loop
was busy in a tight retry loop) rather than in normal operation.

## Fix

`ingest.tick_recorder.acquire_singleton_lock` / `release_singleton_lock`:
a PID-file lock at `logs/tick_recorder.lock` (gitignored). `run_forever`
acquires it before ever touching the network and raises
`AlreadyRunningError` immediately if another live process already holds
it (checked via `os.kill(pid, 0)`, so a stale lock left by a crash is
correctly recovered from rather than blocking forever). This doesn't fix
the SDK's missing backoff, but it prevents the actual root cause here:
two of *our own* instances ever running at once.

## Recovery

Stopped everything (bootout + confirmed no processes via `ps`), waited
~4 minutes for any in-flight connection attempts to fully drain, then did
one clean `launchctl bootstrap`. Verified: error log did not grow over the
following 43+ seconds of run time, and the most recent recorded tick
timestamp was 25 seconds old at check time -- real-time data flowing
normally on a single instance.

## Takeaway for future restarts

Always verify zero processes running (`ps aux | grep tick_recorder`) and
wait a beat before starting a new one -- never start a fallback instance
"just in case" while another might still be tearing down. The new
singleton lock makes this a hard error instead of a silent multi-connection
pile-up if that discipline slips again.
