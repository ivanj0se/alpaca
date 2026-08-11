# Rung 5 result notes: full-universe run, 2026-08-11

No ticker's news correlation survived Sidak correction across the 20
simultaneous tests (best: UNH, raw p=0.071, corrected p=0.77). Before
reading that as "no endogenous/exogenous signal found," three real
caveats from this specific run, checked directly against the underlying
GDELT match data (data/gdelt_cache, 290 files, 3-day window):

1. **5 of 20 tickers had zero GDELT matches at all** (PG, AMZN, JNJ, BAC,
   LIN don't appear in the match table) -- their observed_rate=0 /
   null_mean=0 rows mean "no news data existed to test against," not "no
   correlation despite available news." **Fixed**:
   diagnostics/2026-08-11-gdelt-alias-matching/ -- all 5 were formal/legal
   name strings ("Amazon.com," "Linde plc," etc.) that essentially never
   appear verbatim in news text; a multi-alias match now recovers real
   coverage for all 5 (e.g. AMZN 0 -> 483 matches).

2. **Match counts vary by ~65x across tickers** (WMT 1179, GS 916, JPM 677
   vs. UNH 18, AAPL 42, NVDA 53) over the same 3-day window. **Checked,
   confirmed real, not a bug**: spot-checked 8 sample matches each for WMT
   and GS directly -- genuine mentions in retail/financial news (e.g. "blackrock;jp
   morgan;morgan stanley;goldman sachs;reuters;citigroup" is a real
   markets article naming several banks together). A major retailer and a
   major investment bank draw far more routine financial-news coverage
   than a single health insurer -- the variance reflects real coverage
   intensity, not false-positive inflation.

3. **Several tickers are saturated** (BA, WMT, GS, JPM, CVX: both real and
   null rates near/at 1.0) -- the 30-min match_window is too generous
   relative to how often these specific names appear in the news, so
   almost any timestamp matches something regardless of whether it's a
   real anomaly. Same effect documented in
   diagnostics/2026-08-11-permutation-test-noop-bug/'s real-GDELT smoke
   test; still not recalibrated per-ticker.

Combined with the TCN-VAE receptive-field issue
(diagnostics/2026-08-11-tcn-vae-receptive-field-mismatch/) -- the anomaly
windows feeding Rung 5 come from a model that isn't yet clearly better
than GARCH at this window size -- this run's Rung 5 numbers are a
legitimate first pass, not a final answer. Next full run should follow
the TCN-VAE architecture fix and ideally a per-ticker or coverage-aware
match_window before drawing conclusions from the attribution table.
