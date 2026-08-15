# Real leverage effect found in SPY tick data: down-trends trigger ~60% more future intensity than up-trends of the same size

Date: 2026-08-15

Tier 6b of the private self-excitation research extension, and the
direct answer to the question the previous entry
(`diagnostics/2026-08-14-quadratic-hawkes-real-fit/`) explicitly left
open: that entry found a real, strongly significant squared-feedback
effect (kappa=4.4e-05, loglik +154.74 over a sign-blind model) but
caught, on reflection, that a SYMMETRIC squared term (kappa*L1^2) cannot
represent genuine sign asymmetry no matter its value -- it predicts
identical extra intensity after an up-trend or a down-trend of the same
magnitude. `research/asymmetric_quadratic_hawkes.py` splits that one
coefficient into `kappa_minus` (down-trends) and `kappa_plus`
(up-trends) specifically to test whether real SPY data shows the
classic asymmetric leverage effect, not just undifferentiated
trend-sensitivity.

## Setup

Same real IEX SPY ticks (`data/`, 23,515 events, `sigma_threshold=2.0`)
and same apples-to-apples methodology as the previous two real-data
entries: fit the standard (sign-blind, no quadratic term) model first,
unconstrained beta, then use its own converged beta
(0.00709675) as the fixed `beta_leverage`/`beta` for both the symmetric
and asymmetric quadratic fits -- a true nested comparison, not an
independent heuristic default (the mistake caught and fixed building the
previous two entries in this project).

## Real result

| Fit | free params beyond standard | kappa / kappa_minus / kappa_plus | loglik | vs. standard |
|---|---|---|---|---|
| Standard (sign-blind) | -- | -- | -88216.56 | -- |
| Symmetric quadratic | kappa | 4.396e-05 | -88061.83 | +154.74 |
| **Asymmetric quadratic** | kappa_minus, kappa_plus | **5.688e-05 / 3.559e-05** | **-88055.40** | **+161.16** |

**leverage_asymmetry = kappa_minus - kappa_plus = +2.129e-05, positive --
the classic leverage-effect direction.** kappa_minus/kappa_plus ~ 1.60:
a down-trend of a given magnitude predicts about 60% more excess future
intensity than an up-trend of the exact same magnitude. This is the
real, sign-asymmetric leverage effect this project set out to find,
recovered directly from real tick data, not assumed or hand-waved.

**Statistical significance of the asymmetry itself** (not just "is there
a squared term at all," already established): likelihood-ratio test of
the asymmetric model against the symmetric one (1 extra free parameter)
-- LR stat = 2 x 6.43 = 12.85 on 1 df, **p = 0.00034**. Genuinely
significant, comfortably past the p<0.001 bar, not a marginal or
coin-flip result. (Full comparison against the standard model, 2 extra
parameters: LR stat = 322.33 on 2 df, p effectively 0.)

## Honest read on the two effect sizes

The asymmetry is real and significant, but it's the SMALLER of two real
effects found across these last two diagnostics entries. Going from
"no squared feedback at all" to "symmetric squared feedback" bought
+154.74 in loglik; going from "symmetric" to "sign-asymmetric" bought
only another +6.43 -- about 4% as much additional explanatory power.
The honest, non-overclaiming summary: real SPY jump activity is
dominated by an undifferentiated "recent trend of either sign raises
future intensity" effect (Zumbach-like), on top of which there is a
real but comparatively modest further asymmetry favoring downside
trends specifically (the classic leverage direction). Both are real;
neither should be reported without the other's relative size attached.

## Not resolved

- Whether `kappa_minus`/`kappa_plus` are stable across different real
  sample windows (this fit used the full available `data/` IEX history
  in one shot) -- a rolling or split-sample refit would strengthen
  confidence that 1.60 isn't itself an artifact of one particular period.
- Not yet wired into the Tier 3 generator-comparison harness -- the
  natural next check is whether this asymmetric mechanism moves
  `leverage_curve` specifically (84% coverage for every model tested so
  far in Tier 3), which is exactly the stylized fact this extension was
  built to target and the symmetric model was flagged as structurally
  incapable of improving.
- `beta_leverage == beta` was assumed here (both fixed at the standard
  fit's converged beta), same open question flagged in the symmetric
  model's diagnostics entry -- untested whether allowing them to differ
  changes the picture.
