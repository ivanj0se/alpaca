# Fix: automatic flush blocked the websocket event loop

Date: 2026-08-11
Found while: checking on the live recorder the morning after deployment,
market open, real trading in progress.

## What was observed

`logs/tick_recorder.error.log` had 61 "data websocket error, restarting
connection: no close frame received or sent" entries by mid-morning, and
real data coverage had large gaps (5-20 minute stretches with zero trades
recorded) during active market hours across a 20-liquid-large-cap
universe -- far too sparse to be real trading inactivity.

## Root cause

`TickRecorder.on_trade` is an `async def` running directly on the
websocket connection's event loop (it's the handler alpaca-py's
`StockDataStream` invokes per message). The original implementation
called `self.flush_buffer()` -- synchronous parquet read-merge-write, one
disk round-trip per distinct ticker present in the buffer -- directly from
inside that handler whenever the buffer hit its threshold (500 trades or
30s). During a real trade burst (exactly when the buffer fills fastest),
that blocking write stalls the event loop for however long it takes,
including the low-level frames the websocket connection needs serviced to
stay alive. Classic asyncio anti-pattern: blocking I/O inside an
event-loop callback.

## Fix

Automatic flushes triggered from `on_trade` now hand off to a dedicated
single-worker `ThreadPoolExecutor` (`self._executor.submit(...)`) instead
of writing inline. The buffer swap itself (`to_flush, self._buffer =
self._buffer, []`) stays synchronous and fast (pure in-memory, protected
by a `threading.Lock`), so `on_trade` returns to the event loop almost
immediately regardless of how long the actual disk write takes.
`max_workers=1` keeps writes to the same partition file serialized (no
concurrent-write races). `flush_buffer()` itself stays synchronous and is
still used for shutdown (where blocking briefly is correct -- we want the
process to stay alive until the final write lands) and manual calls.

New `wait_for_pending_flushes()` method lets tests/tools deterministically
wait for background writes instead of sleeping.

## Validation

Regression tests added in `tests/unit/test_tick_recorder.py`
(`TestAsyncFlushBehavior`): confirms the buffer clears and `on_trade`
returns before a deliberately-blocked background write completes, confirms
rapid back-to-back flushes with no manual waiting don't corrupt or drop
data, confirms pending-flush bookkeeping.

**Caveat on effect size**: deploying this fix required restarting the live
recorder, which triggered a separate, unrelated incident (overlapping
connections exhausting Alpaca's connection limit -- see
diagnostics/2026-08-11-tick-recorder-connection-limit-incident/) that
generated a much larger, more obviously diagnosable error flood than the
original 61. That means there wasn't a clean "before vs. after" comparison
window to directly measure whether this fix reduced the *original*
disconnect rate. Blocking I/O in an asyncio callback is objectively wrong
regardless, and worth keeping fixed, but treat "did it fix the original 61
disconnects" as not fully confirmed -- watch the error log over the next
few real trading sessions to see if the baseline disconnect rate is lower
than this morning's pre-fix 61-in-a-few-hours.
