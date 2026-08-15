# Quadratic Hawkes on real SPY data: a real, strongly significant squared-feedback effect -- but a needed correction on what it actually measures

Date: 2026-08-14

Tier 6 of the private self-excitation research extension. Question: does
adding a squared, signed-return feedback term (kappa*L1(t)^2) to the
Hawkes intensity, on top of the ordinary sign-blind clustering term
(alpha*L2(t)) every earlier model in this project already has, genuinely
improve the fit to real SPY tick data?

## Setup

Real IEX SPY ticks from `data/` (23,515 events, `sigma_threshold=2.0`,
same event source Tier 3's other arms use). Fit a standard
(sign-blind, kappa=0) single-exponential Hawkes first via
`fit_hawkes_exponential_multistart` (unconstrained beta), then used
**that fit's own converged beta** as the fixed `beta_leverage`/`beta` for
the quadratic model -- a true nested, apples-to-apples comparison.

## A real bug caught first: the exact same beta-mismatch mistake this project already made twice

The first attempt used an independent data-adaptive default
(1/median inter-event gap = 0.110) for the quadratic model's beta,
instead of the standard fit's own converged beta (0.00710) -- a ~15x
mismatch. Result: the quadratic model's loglik came out **5,442 units
WORSE** than the standard fit's, which looked like "adding a leverage
term makes things worse" but was actually the SAME beta-mismatch bug
already caught once building the Cox-Hawkes trust-gate test this
session, and structurally the same class of false start as Tier 1's
first (wrong) shared-grid IEX/SIP comparison. Refit with the standard
fit's own beta -- the picture completely reversed (see below). Worth
noting for future extensions in this project: pin the fixed decay
rate(s) to an ALREADY-CONVERGED comparison fit's own value, never to an
independently-computed heuristic, whenever the two are meant to be
compared head-to-head.

## Real, corrected result

| Fit | mu/lambda0 | alpha | beta | kappa | branching ratio | loglik |
|---|---|---|---|---|---|---|
| Standard (kappa=0) | 0.001414 | 0.006811 | 0.007097 | -- | 0.9598 | -88216.56 |
| Quadratic | 0.001451 | 0.006454 | 0.007097 (fixed) | **4.396e-05** | 0.9095 | -88061.83 |

Log-likelihood improvement: **+154.74** for one extra parameter (kappa).
Likelihood-ratio statistic = 2 x 154.74 = 309.5 on 1 degree of freedom --
even more strongly significant than the Cox-Hawkes/RPCA result from
2026-08-13 (+95.62 there). This is a real, robust effect, not a fitting
artifact -- confirmed via the corrected apples-to-apples comparison
above, not just accepted from the first (buggy) run.

`kappa=4.4e-05` looks tiny in isolation, but it multiplies a SQUARED
quantity built from real signed z-scores (marks std=3.82) over a slowly-
decaying kernel (beta_leverage=0.0071, implying a memory timescale of
~140s) -- `E[mark^2]/(2*beta_leverage) ~ 1030`, so this small kappa
still contributes comparably to the linear alpha term at realistic L1
magnitudes. `stability_heuristic=0.955` -- real, uncomfortably close to
the (already-flagged-as-approximate) danger zone; simulating from these
exact fitted values would need the module's `max_events` safety cap
active, not attempted in this entry.

## Important correction: this measures a Zumbach-style effect, not necessarily the classic (sign-asymmetric) leverage effect

Caught this thinking it through carefully rather than just reporting the
positive kappa as "found the leverage effect": **kappa*L1(t)^2 is
symmetric in sign by construction** -- squaring erases the difference
between a strong up-trend and a strong down-trend of the same magnitude.
So this specific mechanism predicts MORE future activity after either
kind of trend equally. The classic empirical leverage effect (negative
past returns raise future volatility MORE than positive past returns of
the same size) is a SIGN-asymmetric relationship; a purely-squared term
structurally cannot represent that asymmetry on its own, regardless of
how large or significant kappa turns out to be.

What this result DOES robustly show: real SPY jump activity has a
genuine "trend triggers more activity" structure beyond simple
event-count clustering -- closer to the Zumbach effect (past squared/
absolute returns forecast future volatility better than the reverse) than
to sign-asymmetric leverage specifically. That's still a real, useful,
previously-undemonstrated-in-this-project finding, and likely the more
direct explanation for why Tier 3 found `volatility_clustering_acf`
weak (84%) for every earlier model -- none of them had ANY mechanism
sensitive to the SIZE of a recent trend, only to raw event counts.

Whether this also helps `leverage_curve` specifically (also 84% in
Tier 3) is a separate, unresolved question -- would need a genuinely
sign-asymmetric extension (e.g. a linear-in-L1 term in addition to the
quadratic one, or letting kappa differ for upward vs. downward L1) that
this build does not attempt. Flagged as the concrete next step, not
assumed to already be solved by this result.

## Not resolved

- Real leverage-effect (sign-asymmetric) mechanism not yet built --
  see above.
- Whether beta_leverage should genuinely differ from beta (this fit
  used the same value for both, an assumption of convenience/no strong
  prior, not evidence they're actually equal) -- untested; a real
  diagnostics follow-up could grid-search beta_leverage independently.
- Not yet wired into the Tier 3 generator-comparison harness to check
  whether the real, statistically strong detection-side finding here
  translates into a generative-realism improvement, the same open
  question already found for Cox-Hawkes/RPCA (real detection value,
  weaker generative value, in `diagnostics/2026-08-14-tier3-hawkes-extensions-ablation/`).
