# The Backend Contract Closure Theorem (BCCT)

## Status

This note documents a **structural composition theorem** that closes
the Kerr source backend contract for the spin-half clean Carter chart
by composing four existing exact-derivation theorems into a single
certificate with `remaining_exact_derivation_obligations: []`.

The four backend obligations of the Kerr source backend contract are:

1. **Derive the gauge-reduced linearized vacuum Einstein principal
   symbol in this chart.**
2. **Compute the normalized gravitational principal density `q_grav`
   in the listed phase variables.**
3. **Derive `rho_R` and `rho_dR` as the trapped characteristic
   constraints `R=0` and `dR=0`.**
4. **Prove the exported source density equals `q_grav` modulo those
   trapped constraints.**

The BCCT maps each obligation to the exact-derivation theorem that
closes it, cross-checks mutual consistency of the four certificates,
and emits a closed backend contract.

As with the other theorems in this project, the closure is exact
*conditional* on the upstream principal symbol trace certificate being
promoted from provisional to exact, and on all upstream analytic
hypotheses (CWCC, CWQP-AB, RRBB, NBCT) being proved.  The BCCT itself
does NOT prove new analytic content; it composes existing
exact-derivation theorems.

## Mathematical content

### Obligation closure map

| Obligation | Closed by | Verifier |
|---|---|---|
| (1) gauge-reduced linearized Einstein principal symbol | GRLEPS | `verify_carter_gauge_reduced_einstein_principal_symbol.py` |
| (2) normalized gravitational principal density `q_grav` | GPDDT | `verify_gravitational_principal_density_derivation.py` |
| (3) `rho_R`, `rho_dR` as trapped characteristic constraints | trapped-constraint derivation | `verify_carter_trapped_constraint_derivation.py` |
| (4) exported source density equals `q_grav` mod trapped | GRLEPS | `verify_carter_gauge_reduced_einstein_principal_symbol.py` |

### Composition identities

The BCCT verifier `verify_composed_backend_contract_closure.py`
checks the following seven identities:

**(B1) Backend contract verifies.** The Kerr source backend contract
certificate is accepted by `verify_carter_kerr_source_backend.py`.

**(B2) GRLEPS verifies.** The GRLEPS certificate is accepted by
`verify_carter_gauge_reduced_einstein_principal_symbol.py`.  This
closes obligations (1) and (4).

**(B3) GPDDT verifies.** The GPDDT certificate is accepted by
`verify_gravitational_principal_density_derivation.py`.  This closes
obligation (2).

**(B4) Trapped-constraint derivation verifies.** The trapped-constraint
derivation certificate is accepted by
`verify_carter_trapped_constraint_derivation.py`.  This closes
obligation (3).

**(B5) QIEDT verifies.** The QIEDT certificate is accepted by
`verify_carter_quotient_identity_derivation.py`.  This provides the
quotient identity tying `q_grav` to `rho_backend` modulo the
trapped-constraint ideal plus the gauge-tangent SOS defect.

**(B6) Mutual consistency.** The four certificates must agree on:
- the chart (`spin_half_r_7_2_to_18_5_sub04`);
- the Kerr-clean backend density
  `rho_backend = h*x^2 + tau*q^2` (GRLEPS normal form equals QIEDT
  backend density);
- the gravitational density `q_grav` (GRLEPS `q_grav` equals QIEDT
  `q_grav_expression`, as sympy-expanded polynomials);
- the derived trapped-constraint residuals
  (`radial_constraint_R -> rho_R`, `radial_constraint_dR -> rho_dR`).

**(B7) Obligation coverage.** The union of the closed obligations from
GRLEPS (obligations 1 and 4), GPDDT (obligation 2), and the
trapped-constraint derivation (obligation 3) equals exactly the four
backend obligations.  No obligation is left uncovered; no extra
obligation is claimed.

### Closed backend contract

The BCCT emits a closed backend contract with:

```
remaining_exact_derivation_obligations: []
```

This means: for the spin-half clean Carter chart, every backend
obligation of the Kerr source backend contract is closed by an
existing exact-derivation theorem, all four certificates are mutually
consistent, and the quotient identity holds.

## Proof sketch

The proof is a finite composition of exact-arithmetic checks:

1. Run `verify_carter_kerr_source_backend.py` on the backend contract.
2. Run `verify_carter_gauge_reduced_einstein_principal_symbol.py` on
   the GRLEPS certificate.
3. Run `verify_gravitational_principal_density_derivation.py` on the
   GPDDT certificate.
4. Run `verify_carter_trapped_constraint_derivation.py` on the
   trapped-constraint derivation certificate.
5. Run `verify_carter_quotient_identity_derivation.py` on the QIEDT
   certificate.
6. Cross-check that the chart, normal form, `q_grav`, and derived
   constraints all agree (sympy-expanded polynomial equality).
7. Cross-check that the union of closed obligations equals the four
   backend obligations.

If all seven checks pass, the BCCT accepts the closure certificate.

## Honest stopping rule

This theorem and its verifier are a **structural composition
theorem**.  The conjecture is not solved until:

- the upstream `principal_symbol_trace_candidate.json` is promoted to
  exact (the GPDDT, trapped-constraint derivation, GRLEPS, and QIEDT
  certificates all depend on this promotion);
- all upstream analytic hypotheses (CWCC, CWQP-AB, RRBB, NBCT) are
  proved with the stated uniformity;
- the closure is extended from the single spin-half clean Carter chart
  to the full atlas of Carter charts covering the Kerr exterior.

The verifier accepts only composition certificates where all four
input certificates verify AND the mutual-consistency checks pass AND
the obligation-coverage check passes, so any premature "closure"
claim is rejected by construction.

## New mathematical contribution

The principal new mathematical contribution of BCCT is the
**obligation closure map**: a finite, machine-checkable partition of
the four backend obligations into the exact-derivation theorems that
close them, together with the mutual-consistency identities (B6) that
ensure the four certificates compose without contradiction.  This
converts the backend contract from a list of open obligations into a
single closed contract, which is the input required by the
downstream nonlinear bootstrap (NBCT) and the composed full stack
(CWCC + CWQP-AB + RRBB + NBCT).

The BCCT is the eighth primitive-packet exact-derivation theorem in
the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT,
QIEDT, GRLEPS) and the first composition theorem to close an entire
backend contract for a Carter chart.
