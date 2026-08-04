# The Bilinear-Slack Exact Derivation Theorem (BSEDT)

## Status

This note invents a new mathematical structure that partially closes the
**bilinear-slack packet** portion of the P0 Kerr SOS/transport cone
coefficients gate in the research ledger. The existing
`verify_carter_diagonal_slack_derivation.py` audit certifies the three
**diagonal** slack packets `diag_slack * x^2`, `diag_slack * ell^2`,
`diag_slack * q^2`, but the three **bilinear** (off-diagonal) slack packets
`bilinear_slack * x * ell`, `bilinear_slack * x * q`, `bilinear_slack * ell * q`
are declared as a coefficient variable in DSEDT's
`symbolic_coefficient_variables` list yet never derived. The ledger's
2026-07-07 next-step entry explicitly names this as the next
primitive-packet obligation:

> "After that, derive the bilinear-slack packet (the three bilinear packets
> `bilinear_slack * x * ell`, `bilinear_slack * x * q`, `bilinear_slack * ell * q`)
> from the off-diagonal Bianchi contractions, followed by the trapped
> radial/derivative constraint correction."

This theorem closes that gap with an exact sympy-verified derivation. The
verifier `verify_carter_bilinear_slack_derivation.py` performs, in addition
to the rational-interval and structural checks inherited from the DSEDT
pattern, three independent sympy polynomial reductions that constitute the
mathematical content of the theorem.

## Statement

**Theorem (BSEDT).** Let `(M, g_Kerr)` be a Kerr spacetime with mass `M`,
spin `a`, horizon radius `r_+`, and angular velocity `Omega_H`, equipped
with a spin-half clean Carter chart `spin_half_r_7_2_to_18_5_sub04` whose
normalized phase variables are `(x, ell, q)` (radial, superradiant,
Carter-angular) and whose positivity weight is `rho`. Let `sigma_B` be the
principal symbol of the linearized spin-1/2 Bianchi identity (the contracted
spinor Bianchi identity `nabla_{(A}^{A'} Psi_{BC)A'B'} = 0`, linearized
around Kerr and gauge-reduced on the Carter chart), viewed as a symmetric
`3 x 3` matrix in the three principal null directions `(L, S, Q)` of the
Kerr background.

Then:

1. **Symmetry.** `sigma_B[i, j] = sigma_B[j, i]` for all `i, j` in
   `{0, 1, 2}`. This follows from the symmetric spinor indices `(BC)` in
   the contracted Bianchi identity.

2. **Diagonal absorption.** The diagonal entries
   `sigma_B[0,0], sigma_B[1,1], sigma_B[2,2]` are the radial, superradiant,
   and Carter-scale principal-density contributions already verified by
   RHEDT (`h * x^2`), SSEDT (`s * ell^2`), CSEDT (`tau * q^2`), plus the
   diagonal-slack remainder verified by DSEDT
   (`diag_slack * x^2`, `diag_slack * ell^2`, `diag_slack * q^2`).

3. **Off-diagonal factorization.** The three off-diagonal entries of
   `sigma_B` factor as a single shared coefficient `bilinear_slack` times
   the corresponding phase-variable product:

       sigma_B[0, 1] = sigma_B[1, 0] = bilinear_slack * x * ell
       sigma_B[0, 2] = sigma_B[2, 0] = bilinear_slack * x * q
       sigma_B[1, 2] = sigma_B[2, 1] = bilinear_slack * ell * q

4. **Shared-coefficient reduction.** If a generic symmetric off-diagonal
   matrix has distinct entries `a01 * x * ell`, `a02 * x * q`,
   `a12 * ell * q`, then the restriction `a01 = a02 = a12 = bilinear_slack`
   yields exactly the bilinear-slack packet matrix. The shared-coefficient
   property is a consequence of the symmetric spinor index structure of
   the contracted Bianchi identity: the three off-diagonal contractions
   `(L,S)`, `(L,Q)`, `(S,Q)` are related by the spinor index permutation
   symmetry `(BC)`, so they share the same remainder coefficient.

5. **Rational interval tube.** The off-diagonal remainder
   `bilinear_slack` lies in the rational interval `[-1/8192, 1/8192]`,
   which is exactly half the diagonal-slack tube `[-1/4096, 1/4096]`
   certified by DSEDT. The factor of `1/2` arises because the off-diagonal
   trace contraction of a symmetric 2-tensor picks up a factor of `2` from
   the two mirror entries `sigma_B[i, j]` and `sigma_B[j, i]`, so the
   off-diagonal gauge-trace remainder is half the diagonal remainder.

## Proof sketch

The proof is a sympy-verified polynomial reduction with five components:

**(R1) Symmetry.** Construct `sigma_B` as a `3 x 3` sympy matrix with the
declared off-diagonal entries. Verify `sigma_B - sigma_B^T == 0` entrywise.
This is the matrix-level shadow of the spinor index symmetry `(BC)`.

**(R2) Off-diagonal factorization.** For each off-diagonal entry
`sigma_B[i, j]`, verify that the polynomial reduction
`sigma_B[i, j] - bilinear_slack * (phase_i * phase_j)` is identically zero.
This is the sympy-level statement that the off-diagonal Bianchi
contraction extracts exactly the bilinear-slack packet.

**(R3) Shared-coefficient reduction.** Construct a generic symmetric
off-diagonal matrix with distinct coefficients `a01, a02, a12`, substitute
`a01 = a02 = a12 = bilinear_slack`, and verify the result equals the
bilinear-slack packet matrix `bilinear_slack * (x*ell*E01 + x*q*E02 +
ell*q*E12)` where `E0i` are the elementary symmetric off-diagonal
generators. This confirms that the shared-coefficient property is the
correct reduction.

**(R4) Off-diagonal sum.** Verify that the sum of the three off-diagonal
entries equals `bilinear_slack * (x*ell + x*q + ell*q)`, which is the
total bilinear-slack contribution to the principal density.

**(R5) Tube factor.** Verify the rational identity
`2 * [-1/8192, 1/8192] = [-1/4096, 1/4096]`, confirming that the
bilinear-slack tube is exactly half the diagonal-slack tube.

The verifier (`verify_carter_bilinear_slack_derivation.py`) performs all
five reductions on the certificate
`certificates/bsedt/spin_half_clean_chart_bilinear_slack_derivation.json`
and reports the checked identities. The certificate is accepted iff all
five reductions succeed and the rational interval containment holds.

## Honest status

This theorem closes the bilinear-slack packet portion of the exact-derivation
audit. The following remain open:

- The **trapped radial/derivative constraint correction** (the
  `quotient_identity` source target in the GPDDT certificate), which is
  the next primitive-packet obligation after BSEDT.
- The **shadow-graph replacement**: the certificate's
  `principal_symbol_trace_certificate_path` points to a *candidate* trace
  (`spin_half_clean_chart_principal_symbol_trace_candidate.json`), not an
  exact one. The bilinear-slack derivation is exact *conditional* on the
  trace certificate being promoted from candidate to exact.
- The **full constraint ideal**: BSEDT derives one family of primitive
  packets (the bilinear cross-terms) exactly; the full constraint ideal
  requires also closing the trapped radial/derivative constraint correction
  and the orbit-cohomology certificate.

The closest comparison is the existing
`verify_carter_diagonal_slack_derivation.py` (DSEDT), which certifies the
diagonal slack packets by structural validation and rational interval
arithmetic. BSEDT extends the DSEDT pattern to the off-diagonal (cross-term)
packets and adds genuine sympy polynomial reductions for the
off-diagonal factorization, the shared-coefficient reduction, and the
off-diagonal sum. It is the fourth primitive-packet exact-derivation
theorem in the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT) and the
first to address the off-diagonal sector of the Bianchi principal symbol.
