# Fix: 5 tickers had zero GDELT matches due to overly formal name strings

Date: 2026-08-11
Follow-up to diagnostics/2026-08-11-full-ladder-run/rung5_notes.md, which
flagged (but hadn't investigated) that PG, AMZN, JNJ, BAC, and LIN had
literally zero GDELT matches in the first full-universe run.

## Investigation

Tested simpler name variants directly against the real cached GDELT data
(290 files, 346,575 rows) rather than guessing:

| Ticker | Formal name (0 matches) | Simpler alternative | Matches |
|---|---|---|---|
| PG | "Procter & Gamble" | "Procter" | 53 |
| AMZN | "Amazon.com" | "Amazon" | 483 |
| BAC | "Bank of America" | "BofA" | 21 |
| LIN | "Linde plc" | "Linde" | 71 |
| JNJ | "Johnson & Johnson" | "Johnson" (too broad) | 765 (rejected) |
| JNJ | -- | "Janssen" (J&J's pharma brand) | 6 (used) |

Pattern: legal-entity suffixes (".com", "plc") and ampersands essentially
never appear in how news actually refers to a company -- an NLP-extracted
Organizations field naturally produces "Amazon," not "Amazon.com." "Johnson"
alone was checked and rejected: 765 matches is almost certainly dominated
by unrelated people/companies sharing a common surname (Johnson Controls,
S.C. Johnson, politicians, etc.), which would dilute rather than improve
the signal even with the null-control permutation test as a backstop.
"Janssen" is a legitimate, more specific alias with real but modest
coverage.

## Fix

`config/universe.yaml` entries can now carry an optional `aliases: [...]`
list; `ingest/gdelt_client.py::filter_events_for_universe` matches on ANY
alias (OR logic) instead of just the single `name` field, falling back to
`name` alone when `aliases` isn't given (backward compatible -- the other
15 tickers weren't touched, not because they're confirmed fine, just not
audited this pass).

## Validation

Real match counts, same 290 cached GDELT files, before -> after:
AMZN 0->483, LIN 0->71, PG 0->53, BAC 0->21, JNJ 0->6. Total universe
matches: 4,673 -> 5,307 (+634, exactly the sum of the new tickers' counts).

## Not done this pass

Only the 5 confirmed-zero tickers were investigated. The other 15 (whose
match counts ranged 18-1,179) weren't audited -- the earlier full-ladder
run's rung5_notes.md also flagged that ~65x variance across tickers isn't
yet explained (could be genuine coverage differences, could be some of
those "matches" are themselves false positives inflating certain tickers'
counts). Worth a similar spot-check pass later, lower priority than the
zero-match cases fixed here.
