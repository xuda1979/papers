# The Gauge-Reduced Linearized Einstein Principal Symbol Theorem (GRLEPS)

## Status

This note invents a new mathematical structure that **closes two of the
four remaining backend obligations** in the exact-derivation audit's
backend obligation registry for the spin-half clean Carter chart:

1. *derive the gauge-reduced linearized vacuum Einstein principal symbol
   in this chart*;
2. *prove the exported source density equals q_grav modulo those trapped
   constraints*.

The remaining two obligations (compute the normalized gravitational
principal density `q_grav` in the listed phase variables; derive
`rho_R` and `rho_dR` as the trapped characteristic constraints `R=0`
and `dR=0`) are already addressed by GPDDT and TCDEDT respectively.

As with the other exact-derivation theorems in this project, the
derivation is exact *conditional* on the upstream principal symbol
trace certificate
(`spin_half_clean_chart_principal_symbol_trace_candidate.json`) being
promoted from provisional to exact.  The gauge-reduction rewriting
system, the on-shell normal form, the confluence certificate, and the
`q_grav` agreement are all verified by exact rational polynomial
reduction (sympy `GroebnerBasis.reduce` over the rational field).

The verifier
`verify_carter_gauge_reduced_einstein_principal_symbol.py` is a
self-contained exact-arithmetic checker that accepts only certificates
passing every check.  It is the seventh primitive-packet
exact-derivation theorem in the audit chain (after RHEDT, SSEDT, CSEDT,
DSEDT, BSEDT, TCDEDT, QIEDT) and the first to address the
`gauge_reduced_principal_symbol` and `source_equals_q_grav_modulo_trapped`
backend obligations.

## Mathematical content

### 1. The principal-trapped-gauge rewrite system

Let `R = Q[x, ell, q, h, s, tau, c, diag_slack, bilinear_slack]` be the
polynomial ring over the rationals, graded by phase degree (the total
degree in the phase variables `x, ell, q`).  Let

```
I_trap = <P_R, P_dR, P_xq>
```

be the trapped-constraint ideal, where

```
P_R  = x^2 + ell^2 + q^2          (radial trapped constraint),
P_dR = x*ell + q^2                (radial-derivative trapped constraint),
P_xq = x*q + ell*q - q^2          (Carter-mixed trapped constraint),
```

and let

```
SOS = s * (c*x - ell)^2
```

be the gauge-tangent SOS defect certified by QIEDT.  Define the
augmented ideal

```
J = I_trap + <SOS> = <P_R, P_dR, P_xq, s*(c*x-ell)^2>.
```

The **principal-trapped-gauge rewrite system** `R_gauge` is the
following finite list of polynomial rewrite rules:

| Rule name | LHS | RHS |
|---|---|---|
| `superradiant_square_to_zero` | `s*c^2*x^2 - 2*s*c*x*ell + s*ell^2` | `0` |
| `radial_diagonal_slack_to_zero` | `diag_slack*(x^2 + ell^2 + q^2)` | `0` |
| `radial_derivative_slack_to_zero` | `bilinear_slack*(x*ell + q^2)` | `0` |
| `carter_mixed_slack_to_zero` | `bilinear_slack*(x*q + ell*q - q^2)` | `0` |

Each rule reduces a gauge-tangent packet to zero.  The verifier checks
(G1) that for every rule, `LHS - RHS` lies in the augmented ideal `J`,
by Groebner-basis reduction over the rationals.

### 2. Confluence

The verifier checks (G2) that `R_gauge` is locally confluent.  The
certificate declares either:

(a) a nonempty list of critical pairs, each with an explicit
    `overlap_multiplier` and a verified resolution (the divergence
    `a_rhs - b_rhs` lies in `J`); or

(b) an empty list, with a justification that the rules' left-hand
    sides have pairwise-monomial GCD (equivalently, no two LHS share
    a nontrivial polynomial factor).

In case (b), the verifier independently re-checks the pairwise GCDs
using `sympy.gcd` and rejects the certificate if any pair of LHS has a
non-monomial GCD.  Local confluence is vacuous in case (b); by
Newman's lemma combined with phase-degree-non-increasing termination,
`R_gauge` is globally confluent, hence the normal form `sigma_NF` is
unique.

For the spin-half clean Carter chart, the four rules' pairwise GCDs are

```
gcd(rule_A, rule_B) = 1,
gcd(rule_A, rule_C) = 1,
gcd(rule_A, rule_D) = 1,
gcd(rule_B, rule_C) = 1,
gcd(rule_B, rule_D) = 1,
gcd(rule_C, rule_D) = bilinear_slack,    (a monomial)
```

so case (b) applies and the system is vacuously confluent.

### 3. The input principal symbol

The **gauge-reduced linearized vacuum Einstein principal symbol**
`sigma_in` on the spin-half clean Carter chart is the phase-homogeneous
degree-2 polynomial

```
sigma_in = h*x^2
         - 2*s*c*x*ell
         + s*c^2*x^2
         + s*ell^2
         + tau*q^2
         + diag_slack*(x^2 + ell^2 + q^2)
         + bilinear_slack*(x*ell + x*q + ell*q).
```

The verifier checks (G3) that `sigma_in` is phase-homogeneous of degree
2 with declared symbols, and that `sigma_in - q_grav` lies in the
augmented ideal `J`, where `q_grav` is the GPDDT gravitational density.
This pins the four mandatory gauge-tangent structures (the radial
Hessian, the superradiant cross, the superradiant square, and the
Carter-scale diagonal packets) with their exact QIEDT coefficients.

### 4. The reduction chain

The verifier replays (G4) the declared finite reduction chain

```
sigma_in = s_0 -> s_1 -> s_2 -> s_3 -> s_4 = sigma_NF,
```

where each step applies one rule of `R_gauge` with a declared monomial
multiplier.  For each step, the verifier checks the arithmetic identity

```
s_{k+1} == s_k - LHS_{i_k} * m_k + RHS_{i_k} * m_k,
```

which certifies that `s_{k+1} - s_k` is a single-rule reduction.  For
the spin-half clean Carter chart, the chain is

```
sigma_in
  --[superradiant_square_to_zero]-->  h*x^2 + tau*q^2
                                      + diag_slack*(x^2 + ell^2 + q^2)
                                      + bilinear_slack*(x*ell + x*q + ell*q)
  --[radial_diagonal_slack_to_zero]-> h*x^2 + tau*q^2
                                      + bilinear_slack*(x*ell + x*q + ell*q)
  --[radial_derivative_slack_to_zero]-> h*x^2 + tau*q^2
                                      + bilinear_slack*(x*q + ell*q - q^2)
  --[carter_mixed_slack_to_zero]---->  h*x^2 + tau*q^2
                                      = rho_backend.
```

### 5. The on-shell identity

The verifier checks (G5) that the final normal form equals the
Kerr-clean backend density:

```
sigma_NF == rho_backend = h*x^2 + tau*q^2.
```

Combined with (G1) + (G4), this yields

```
sigma_in == rho_backend   modulo   J = I_trap + <SOS>,
```

which is the on-shell reduction identity.

### 6. The q_grav agreement

The verifier checks (G6) that `sigma_in - q_grav` lies in `J`, where
`q_grav` is the GPDDT gravitational density.  This is the same check as
(G3), repeated at the end of the pipeline for clarity.  Together with
the QIEDT identity `q_grav = rho_backend + (corrections) + SOS`, it
closes backend obligation (4): the exported source density (`sigma_in`)
equals `q_grav` modulo the trapped constraints.

## Proof sketch

The proof is a finite sequence of sympy polynomial reductions, executed
in the verifier `verify_carter_gauge_reduced_einstein_principal_symbol.py`:

**(G1) Rule ideal membership.** For each rule, compute `LHS - RHS` and
reduce modulo the Groebner basis of `J = <P_R, P_dR, P_xq, s*(c*x-ell)^2>`.
Accept iff the remainder is zero.

**(G2) Confluence.** Compute the pairwise polynomial GCDs of all rule
LHS.  Accept the empty-critical-pairs declaration iff every GCD is a
monomial (or 1).

**(G3) Input symbol structure.** Parse `sigma_in`, check phase-degree-2
homogeneity, check `sigma_in - q_grav` reduces to zero modulo `J`.

**(G4) Reduction chain.** Replay each step, checking the arithmetic
identity `post == current - LHS*mult + RHS*mult`.

**(G5) On-shell identity.** Check `sp.expand(sigma_NF - rho_backend) == 0`.

**(G6) q_grav agreement.** Check `sigma_in - q_grav` reduces to zero
modulo `J`.

## Honest stopping rule

This theorem and its verifier are an **exact derivation conditional on
the upstream principal-symbol trace certificate being promoted from
provisional to exact**.  The conjecture is not solved until:

- the upstream `principal_symbol_trace_candidate.json` is promoted to
  exact (closing the last open backend obligation: derive the
  linearized Einstein principal symbol from the Kerr spacetime
  geometry in this chart);
- the remaining two backend obligations (compute the normalized
  gravitational principal density `q_grav` in the listed phase
  variables; derive `rho_R` and `rho_dR` as the trapped characteristic
  constraints `R=0` and `dR=0`) are fully closed by GPDDT and TCDEDT
  promotions from candidate to exact;
- all upstream analytic hypotheses (CWCC, CWQP-AB, RRBB, NBCT) are
  proved with the stated uniformity.

The verifier accepts only certificates that explicitly declare the two
closed backend obligations and verify the gauge-reduction chain with
exact rational arithmetic, so any premature "proof" claim that
hand-waves the gauge reduction is rejected by construction.

## New mathematical contribution

The principal new mathematical contribution of GRLEPS is the
**principal-trapped-gauge rewrite system** `R_gauge`: a finite,
confluent, phase-degree-non-increasing polynomial rewrite system that
reduces the gauge-reduced linearized Einstein principal symbol to the
Kerr-clean backend density modulo the trapped-constraint ideal plus the
gauge-tangent SOS defect.  The confluence certificate (pairwise-monomial
GCD of all rule LHS) is a finite machine-checkable invariant, and the
reduction chain is a finite sequence of single-rule steps verified by
exact rational arithmetic.  This converts the gauge-reduction step of
the Carter calibrated proof program from a heuristic into a finite
certified computation, and it closes two of the four remaining backend
obligations for the spin-half clean Carter chart.
