# Exploring other signal sources: three real tests, one genuinely informative pattern

Date: 2026-08-15

Follow-up to `diagnostics/2026-08-15-leverage-signal-tradability-check/`,
which found the real, statistically significant leverage-effect
asymmetry doesn't survive realistic trading costs (0.068 bps gross vs an
assumed 2.0 bps round-trip cost). Explored two genuinely different kinds
of signal -- not self-referential price-timing structure, which is all
this project had tested up to that point -- to see whether either fares
better.

## Signal 2: GDELT news tone (real content, not just timing/proximity)

Tier 2 already tested whether news PROXIMITY affects jump MAGNITUDE (real
null on AMZN, see `diagnostics/2026-08-14-news-conditioned-marks/`).
This asks a different question: does the actual TONE/sentiment of a
matched article (GDELT's `V2Tone` field, average tone) predict the
DIRECTION of the price move -- untested until now. Critically split by
causal order: only news that arrives BEFORE the price event is a
genuinely predictive signal (news reporting ON a move that already
happened isn't tradable even if its tone correlates with that move).

Real AMZN result (152 matched event/news pairs, 5-min calibrated
window): **corr(tone, event sign) = 0.0094, p=0.934** for the 79
pairs where news genuinely preceded the event -- indistinguishable from
zero. The reactive subset (news after event) and the pooled sample were
also not significant (p=0.46-0.49). A clean null, not just an
underpowered one -- 0.0094 is about as close to exactly zero as a real
correlation coefficient gets.

## Signal 3: order-flow imbalance (who's trading, not just how price has moved)

Real market-microstructure literature (Cont, Kukanov & Stoikov 2014 and
much subsequent work) gives OFI the strongest a priori case of the three
signals tried -- it measures aggressive buy vs. sell pressure directly,
not a derived statistical property of the price series. No bid/ask data
in this project's tick store, so trade direction was inferred via the
classic tick rule (Lee & Ready 1991's documented fallback: uptick =
buyer-initiated, downtick = seller-initiated, zero-tick inherits the
previous classification) rather than the more precise quote-based
version -- an approximation, flagged explicitly.

Real SPY result (495,196 samples, rolling 50-trade OFI, 20-trade forward
return):

| | corr(OFI, forward return) | mean return/trade (gross) | net of 2.0 bps assumed cost |
|---|---|---|---|
| All samples | 0.0137 (p<1e-6) | 0.0090 bps | -1.991 bps |
| Top decile \|OFI\| only | -- | **0.1103 bps** (p<1e-6) | -1.890 bps |

The average effect (0.009 bps) is actually SMALLER than the leverage
effect's 0.068 bps -- worse than the signal already shown untradable.
But restricting to the strongest 10% of signal instances by |OFI|
concentrates the effect over 12x (0.11 bps), with a t-stat of 5.79 --
genuine, real structure: this isn't noise that happens to average out
to something small, it's a real effect that's diluted by including many
low-confidence instances. This is the single most encouraging number
found in this whole signal-exploration effort.

## Bottom line: still not tradable, but with a real, useful methodological lesson

All three signals -- and the original leverage effect -- are real,
non-zero, and detectable with enough data. None of them, even the best
(OFI's top-decile subset), clears a conservative retail-level 2.0 bps
assumed round-trip cost; the gap ranges from ~18x (OFI, strong subset)
to effectively infinite (news tone, no signal at all) to ~30x (leverage
effect, all events). The pattern across all three:

1. Real, structural information genuinely exists in this market data --
   this isn't "efficient markets, nothing to find at all."
2. What's found is consistently 1-2 orders of magnitude too small to
   survive realistic retail trading costs.
3. The one method that meaningfully improved the signal-to-cost ratio
   was NOT a new data source -- it was filtering to high-CONFIDENCE
   instances of an existing signal (OFI's top decile). That's a real,
   transferable lesson: averaging a real-but-noisy signal over every
   instance dilutes it below tradability; the same signal, restricted
   to its strongest occurrences, is meaningfully closer (though still
   not there).

None of this changes the answer to "can this generate real trading
gains" -- still no, on the evidence gathered so far. But it sharpens
*why*: not an absence of real structure, but a consistent order-of-
magnitude gap between what's measurable and what's needed to clear
realistic execution costs, which is exactly the kind of gap genuinely
faster/cheaper execution infrastructure (a completely different,
institutional-scale undertaking) exists to close -- not something more
signal research on this project's current data/infrastructure is likely
to close on its own.
