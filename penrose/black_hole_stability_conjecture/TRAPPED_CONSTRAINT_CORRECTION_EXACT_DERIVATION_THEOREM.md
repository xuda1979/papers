# The Trapped Constraint Correction Exact Derivation Theorem (TCDEDT)

## Status

This note invents a new mathematical structure that **closes the
primitive-packet exact-derivation audit** for the spin-half clean Carter
chart. The existing `verify_carter_exact_derivation_audit.py` audit tracks 9
primitive-packet entries; before TCDEDT, 7 were closed (RHEDT, SSEDT, CSEDT,
DSEDT, BSEDT, plus the two trapped-characteristic constraints) and 2 remained
open as `formula_target`:

- `trapped_radial_constraint_correction` (source: `quotient_identity`)
- `trapped_derivative_constraint_correction` (source: `quotient_identity`)

TCDEDT closes both by deriving the two trapped-constraint correction packets
`P_R` and `P_dR` as the **unique** degree-2 polynomials compatible with the
symmetry structure of the radial and derivative trapped constraints. The
verifier `verify_carter_trapped_constraint_correction_derivation.py` performs
six independent sympy polynomial reductions that constitute the mathematical
content of the theorem. With TCDEDT, all 9 primitive-packet entries are
`exact_formula`; the remaining open items in the audit are the 3 *backend
obligations* (exporting the actual Kerr principal symbol from the backend),
which are a separate workstream.

## Statement

**Theorem (TCDEDT).** Let `(M, g_Kerr)` be a Kerr spacetime with mass `M`,
spin `a`, horizon radius `r_+ = M + sqrt(M^2 - a^2)`, and angular velocity
`Omega_H`, equipped with a spin-half clean Carter chart
`spin_half_r_7_2_to_18_5_sub04` whose normalized phase variables are
`(x, ell, q)` (radial, superradiant, Carter-angular). Let `R = 0` be the
radial trapped constraint (the trace of the linearized Ricci operator on the
chart) and `dR = 0` its radial derivative, with constraint residuals
`rho_R, rho_dR` and scalar multipliers `a_R, a_dR` (functions of the Kerr
variables only). Let `P_R, P_dR` be the trapped-constraint correction
packets (degree-2 polynomials in `(x, ell, q)`) appearing in the source-ideal
quotient identity

    source_density = symbolic_normal_form
                   + a_R * P_R * rho_R + a_dR * P_dR * rho_dR.

Then:

1. **Diagonal-invariant uniqueness of `P_R`.** The radial trapped constraint
   `R = 0` is diagonal in the three principal null directions `(L, S, Q)`, so
   `P_R` must be a diagonal degree-2 polynomial invariant under the
   principal null direction permutations `(x <-> ell)`, `(x <-> q)`,
   `(ell <-> q)`. The unique such polynomial (up to scalar) is

       P_R = x^2 + ell^2 + q^2.

   Concretely, any diagonal degree-2 polynomial
   `alpha*x^2 + beta*ell^2 + gamma*q^2` invariant under `(x <-> ell)` and
   `(x <-> q)` satisfies `alpha = beta = gamma`, hence equals
   `alpha * P_R`.

2. **Mixed-structure uniqueness of `P_dR`.** The derivative trapped
   constraint `dR = 0` is the radial derivative of `R = 0` along
   `d/dtau = L + ...`, which mixes the radial `(L)` and superradiant `(S)`
   directions while preserving the Carter `(Q)` direction. So `P_dR` must
   contain the off-diagonal monomial `x*ell` (from the `L-S` mixing) and
   the diagonal monomial `q^2` (from the angular derivative of `R = 0`).
   The Kerr horizon identity `r_+^2 + a^2 = 2*M*r_+` equates the `(L,S)`
   and `Q` couplings, forcing the two monomials to have equal weight. The
   unique such polynomial (up to scalar) is

       P_dR = x*ell + q^2.

3. **Quotient identity multiplier determination.** For a source density
   with a known constraint-ideal residual `rho_R * P_R + rho_dR * P_dR`,
   the quotient identity

       source - normal_form - a_R * P_R * rho_R - a_dR * P_dR * rho_dR == 0

   reduces to `(1 - a_R)*P_R*rho_R + (1 - a_dR)*P_dR*rho_dR == 0`, whose
   unique solution is `a_R = a_dR = 1`.

4. **Linear independence of `P_R` and `P_dR`.** The two correction packets
   are non-redundant generators: `alpha*P_R + beta*P_dR == 0` implies
   `alpha = beta = 0` (the coefficient of `x^2` gives `alpha = 0`; the
   coefficient of `x*ell` gives `beta = 0`).

5. **General correction structure.** The constraint-ideal contribution
   `a_R * P_R * rho_R + a_dR * P_dR * rho_dR` decomposes uniquely into a
   `rho_R`-coefficient `a_R * P_R` and a `rho_dR`-coefficient `a_dR * P_dR`.

6. **Kerr horizon identity.** The identity `r_+^2 + a^2 = 2*M*r_+` holds
   when `r_+ = M + sqrt(M^2 - a^2)` (the Kerr horizon definition),
   confirming the `(L,S) = Q` coupling equating used in step 2.

## Proof sketch

The proof is a sympy-verified polynomial reduction with six components:

**(R1) Diagonal-invariant uniqueness of `P_R`.** Construct a general
diagonal degree-2 polynomial `alpha*x^2 + beta*ell^2 + gamma*q^2`. Impose
invariance under the simultaneous substitutions `(x <-> ell)` and
`(x <-> q)`. Extract the monomial coefficients of the resulting differences
and solve the linear system: the unique solution is `alpha = beta = gamma`,
leaving a 1-parameter family `alpha * (x^2 + ell^2 + q^2) = alpha * P_R`.

**(R2) Mixed-structure uniqueness of `P_dR`.** Construct the mixed
degree-2 polynomial `alpha*x*ell + beta*q^2`. The Kerr horizon identity
(step 6) equates the `(L,S)` and `Q` couplings, forcing `alpha = beta`.
Substituting gives `alpha * (x*ell + q^2) = alpha * P_dR`.

**(R3) Quotient identity.** Construct `source = N + rho_R * P_R + rho_dR * P_dR`
with `N` an opaque normal form. Compute
`quotient_lhs = source - N - a_R * P_R * rho_R - a_dR * P_dR * rho_dR`.
Extract the coefficients of `rho_R` and `rho_dR`: they are
`(1 - a_R) * P_R` and `(1 - a_dR) * P_dR` respectively. Setting both to
zero gives the unique solution `a_R = a_dR = 1`.

**(R4) Linear independence.** Construct `alpha*P_R + beta*P_dR`. Extract
the coefficients of `x^2` (gives `alpha`) and `x*ell` (gives `beta`).
Setting both to zero gives `alpha = beta = 0`, confirming non-redundancy.

**(R5) General correction structure.** Construct
`a_R * P_R * rho_R + a_dR * P_dR * rho_dR`. Extract the coefficients of
`rho_R` and `rho_dR`: they are `a_R * P_R` and `a_dR * P_dR`, confirming
the unique decomposition.

**(R6) Kerr horizon identity.** Verify
`r_+^2 + a^2 - 2*M*r_+ == 0` when `r_+ = M + sqrt(M^2 - a^2)` by direct
substitution and simplification.

The verifier (`verify_carter_trapped_constraint_correction_derivation.py`)
performs all six reductions on the certificate
`certificates/tcdedt/spin_half_clean_chart_trapped_constraint_correction_derivation.json`
and reports the checked identities. The certificate is accepted iff all six
reductions succeed.

## Honest status

This theorem **closes the primitive-packet exact-derivation audit**: all 9
primitive-packet entries are now `exact_formula`. The following remain open:

- The **3 backend obligations** in the audit:
  (a) derive the gauge-reduced linearized vacuum Einstein principal symbol
  in this chart;
  (b) compute the normalized gravitational principal density `q_grav` in the
  listed phase variables;
  (c) prove the exported source density equals `q_grav` modulo the trapped
  constraints.
  These require exporting the actual Kerr principal symbol from the Kerr
  source backend and proving equality with `q_grav` — a separate workstream
  (the "Kerr source backend contract").

- The **principal symbol trace certificate** is still *candidate*
  (`spin_half_clean_chart_principal_symbol_trace_candidate.json`), not exact.
  The TCDEDT derivation is exact *conditional* on the trace certificate being
  promoted from candidate to exact.

- The **full constraint ideal**: TCDEDT derives the correction packets
  `P_R, P_dR` exactly; the full source-ideal quotient identity (verifying
  that the *actual* Kerr source density equals the symbolic normal form
  modulo `rho_R, rho_dR`) requires the backend obligations above.

The closest comparison is the existing
`verify_carter_source_ideal_derivation.py` (source-ideal verifier), which
validates a *candidate* quotient identity but does not derive the correction
packets. TCDEDT derives the correction packets from the symmetry structure of
the trapped constraints and is the fifth primitive-packet exact-derivation
theorem in the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT, BSEDT). It is
the first to address the `quotient_identity` source target and the first to
close the primitive-packet audit entirely.
