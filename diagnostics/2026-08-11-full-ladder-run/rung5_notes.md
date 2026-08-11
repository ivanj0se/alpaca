# Rung 5 result notes: full-universe run, 2026-08-11

No ticker's news correlation survived Sidak correction across the 20
simultaneous tests (best: UNH, raw p=0.071, corrected p=0.77). Before
reading that as "no endogenous/exogenous signal found," three real
caveats from this specific run, checked directly against the underlying
GDELT match data (data/gdelt_cache, 290 files, 3-day window):

1. **5 of 20 tickers had zero GDELT matches at all** (PG, AMZN, JNJ, BAC,
   LIN don't appear in the match table) -- their observed_rate=0 /
   null_mean=0 rows mean "no news data existed to test against," not "no
   correlation despite available news." Not a finding either way.

2. **Match counts vary by ~65x across tickers with no obvious reason**
   (WMT 1179, GS 916, JPM 677 vs. UNH 18, AAPL 42, NVDA 53) over the same
   3-day window. Plausibly real (banks/industrials draw more macro-news
   coverage than a single healthcare name), but not verified against
   actual matched headlines -- worth spot-checking before trusting
   per-ticker rates, especially for the high-count names, given GDELT's
   confirmed heuristic-matching false-positive risk
   (attribution/correlate.py's docstring; the "Alphabet Bar Grill" case).

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
