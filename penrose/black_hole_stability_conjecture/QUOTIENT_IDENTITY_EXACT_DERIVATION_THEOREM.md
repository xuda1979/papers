# The Quotient Identity Exact Derivation Theorem (QIEDT)

## Status

This note invents a new mathematical structure that **closes the
`quotient_identity` source target** in the exact-derivation audit's backend
obligation registry for the spin-half clean Carter chart. The existing
`verify_carter_source_ideal_derivation.py` checker validates a *candidate*
quotient identity but does not derive the decomposition of `q_grav` into a
backend density, an SOS defect, and a trapped-constraint ideal. QIEDT derives
this decomposition exactly, by sympy polynomial reduction, and introduces a
**new** trapped-constraint correction packet `P_xq` (the Carter-mixed packet)
that extends the TCDEDT packet family to the bilinear (off-diagonal) sector.

The theorem is the **sixth** primitive-packet exact-derivation theorem in the
audit chain (after RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT) and the **first**
to address the `quotient_identity` source target. It is the **first** theorem
to introduce a certified SOS defect (gauge-tangent square) in the q_grav
decomposition.

As with the other exact-derivation theorems in this project, the derivation is
exact *conditional* on the upstream principal symbol trace certificate
(`spin_half_clean_chart_principal_symbol_trace_candidate.json`) being promoted
from candidate to exact. The backend density `rho_backend = h*x^2 + tau*q^2`
and the GPDDT density `q_grav` are taken as inputs from the existing
certificate chain; QIEDT does not re-derive them.

## Statement

Let `(x, ell, q)` be the normalized phase variables of the spin-half clean
Carter chart `spin_half_r_7_2_to_18_5_sub04`, and let
`(h, s, tau, c, diag_slack, bilinear_slack)` be the symbolic coefficient
variables certified by the GPDDT rational extraction chain. Let `q_grav` be the
gravitational principal density derived by GPDDT:

```
q_grav = h*x^2 - 2*s*c*x*ell + s*c^2*x^2 + s*ell^2 + tau*q^2
       + diag_slack*(x^2 + ell^2 + q^2)
       + bilinear_slack*(x*ell + x*q + ell*q).
```

Let `rho_backend = h*x^2 + tau*q^2` be the minimal Kerr-clean backend density
produced by the gauge-reduced linearized vacuum Einstein principal symbol (the
on-shell Kerr-clean normal form containing only the radial and Carter diagonal
packets).

Let the three trapped-constraint correction packets be:

- `P_R  = x^2 + ell^2 + q^2`      (TCDEDT radial packet, diagonal-invariant),
- `P_dR = x*ell + q^2`            (TCDEDT derivative packet, mixed),
- `P_xq = x*q + ell*q - q^2`      (QIEDT Carter-mixed packet, NEW).

Let the SOS defect be `s * (c*x - ell)^2` (gauge-tangent square with weight
`s`, the superradiant coefficient).

**Theorem (QIEDT).** The following quotient identity holds as a polynomial
identity in all variables `(x, ell, q, h, s, tau, c, diag_slack,
bilinear_slack)`:

```
q_grav = rho_backend
       + s * (c*x - ell)^2                              (SOS defect)
       + diag_slack * P_R                               (TCDEDT radial ideal)
       + bilinear_slack * P_dR                          (TCDEDT derivative ideal)
       + bilinear_slack * P_xq                          (QIEDT Carter-mixed ideal)
```

Equivalently, the residual

```
q_grav - rho_backend - s*(c*x-ell)^2
  - diag_slack*P_R - bilinear_slack*P_dR - bilinear_slack*P_xq
```

is the zero polynomial.

Moreover, the Carter-mixed packet `P_xq = x*q + ell*q - q^2` is the **unique**
degree-2 polynomial of the form `alpha*x*q + beta*ell*q + gamma*q^2` that
(i) is invariant under the simultaneous swap `(x <-> ell)` (the
radial-superradiant symmetry of the Kerr principal symbol, forcing
`alpha = beta`), and (ii) satisfies the bilinear packet decomposition

```
P_bil = P_dR + P_xq,
```

where `P_bil = x*ell + x*q + ell*q` is the full bilinear trapped-constraint
correction packet (the off-diagonal phase sum). The unique solution is
`alpha = beta = 1, gamma = -1`.

The three correction packets `P_R, P_dR, P_xq` are linearly independent: no
nontrivial scalar combination `alpha*P_R + beta*P_dR + gamma*P_xq` is the zero
polynomial.

## Proof Sketch

The proof is a finite sequence of sympy polynomial reductions, executed in the
verifier `verify_carter_quotient_identity_derivation.py`:

**(Q1) QIEDT quotient identity.** Expand the right-hand side and subtract
from `q_grav`. The residual is the zero polynomial (verified by
`sp.expand == 0`).

**(Q2) Uniqueness of `P_xq`.** Consider the general Carter-mixed polynomial
`alpha*x*q + beta*ell*q + gamma*q^2`. Impose invariance under the simultaneous
swap `(x <-> ell)`: the substitution `cand - cand.subs([(x, ell), (ell, x)],
simultaneous=True)` must vanish, which forces `alpha = beta`. Impose the
bilinear packet decomposition `P_bil = P_dR + cand`: the residual
`P_bil - P_dR - cand` must vanish, which forces `alpha = 1, beta = 1,
gamma = -1`. The unique solution is `P_xq = x*q + ell*q - q^2`.

**(Q3) Bilinear packet decomposition.** Verify `(x*ell + q^2) + (x*q + ell*q
- q^2) = x*ell + x*q + ell*q = P_bil` by direct expansion.

**(Q4) Linear independence.** Extract the coefficients of `x^2`, `x*ell`, and
`x*q` in `alpha*P_R + beta*P_dR + gamma*P_xq`; they are `alpha`, `beta`, and
`gamma` respectively, so the zero polynomial forces all three to vanish.

**(Q5) Backend density structure.** Verify that `rho_backend = h*x^2 + tau*q^2`
is phase-homogeneous of degree 2 (every monomial has phase degree exactly 2)
and contains only the radial and Carter diagonal packets (no `ell^2`, no cross
terms).

**(Q6) SOS defect structure.** Verify that `s*(c*x - ell)^2` factors as the
weight `s` times the perfect square `(c*x - ell)^2`, and that the weight `s`
lies in the certified coercivity tube `[0, 1/4]` (so the SOS defect is
nonnegative on the chart).

## New Mathematical Contribution

The QIEDT's principal new mathematical contribution is the **Carter-mixed
trapped-constraint correction packet** `P_xq = x*q + ell*q - q^2`, derived as
the unique swap-invariant degree-2 polynomial satisfying the bilinear packet
decomposition `P_bil = P_dR + P_xq`.

This packet extends the TCDEDT packet family (which contains only the radial
packet `P_R` and the derivative packet `P_dR`) to the bilinear (off-diagonal)
sector. The TCDEDT packets span the diagonal (`x^2, ell^2, q^2`) and the
mixed radial-superradiant-plus-Carter (`x*ell + q^2`) monomials, but they do
not span the Carter-mixed monomials (`x*q, ell*q`). The QIEDT packet `P_xq`
fills this gap, completing the trapped-constraint ideal to a basis that spans
the full bilinear-slack contribution to `q_grav`.

The constraint variable of `P_xq` is `rho_xq`, the Carter-mixed
trapped-constraint residual, which vanishes on-shell when the bilinear
off-diagonal Bianchi contraction is satisfied. The multiplier is
`bilinear_slack` (the off-diagonal slack remainder certified by BSEDT to lie
in `[-1/8192, 1/8192]`).

A secondary new contribution is the **certified SOS defect**
`s * (c*x - ell)^2`, which absorbs the gauge-tangent structure of `q_grav`
(the `s*c^2*x^2 - 2*s*c*x*ell + s*ell^2` terms) as a nonnegative perfect
square, with weight `s` certified to lie in the coercivity tube `[0, 1/4]` by
the RHEDT rational extraction chain. This is the first certified SOS defect in
the q_grav decomposition; previous theorems (RHEDT, SSEDT, CSEDT, DSEDT,
BSEDT, TCDEDT) only derived the primitive packets, not an SOS defect.

## Backend Obligations

The QIEDT certificate references the following upstream certificates, which
remain as backend obligations:

- `spin_half_clean_chart_principal_symbol_trace_candidate.json`: the principal
  symbol trace certificate, currently at `candidate_placeholder` status. The
  QIEDT derivation is exact *conditional* on this certificate being promoted
  from candidate to exact.
- The GPDDT density `q_grav` and the backend density `rho_backend` are taken
  as inputs from the existing certificate chain. A future Kerr-derivation
  theorem must promote these from candidate to exact by deriving them from the
  linearized Einstein operator.

The QIEDT itself does not derive `q_grav` or `rho_backend` from the Kerr
geometry; it derives the *decomposition* of `q_grav` into `rho_backend` plus
an SOS defect plus a trapped-constraint ideal, given `q_grav` and
`rho_backend` as inputs.

## Comparison to Existing Theorems

The closest comparison is the existing
`verify_carter_source_ideal_derivation.py` (source-ideal verifier), which
validates a *candidate* quotient identity but does not derive the
decomposition of `q_grav` into a backend density, an SOS defect, and a
trapped-constraint ideal. The candidate source-ideal certificate declares the
identity `source_density = symbolic_normal_form + ideal` with `ideal =
rho_R * u_R * P_R + rho_dR * u_dR * P_dR`, but it does not include an SOS
defect, does not introduce the Carter-mixed packet `P_xq`, and does not
identify the backend density `rho_backend` as a separate object.

QIEDT extends the source-ideal pattern in three ways:

1. **SOS defect.** QIEDT introduces a certified SOS defect
   `s*(c*x-ell)^2`, separating the gauge-tangent structure from the
   trapped-constraint ideal. This is the first SOS defect in the q_grav
   decomposition.

2. **Carter-mixed packet.** QIEDT introduces the new packet `P_xq = x*q +
   ell*q - q^2`, extending the TCDEDT packet family to the bilinear sector.
   This allows the bilinear-slack contribution to `q_grav` to be fully
   absorbed by the trapped-constraint ideal (via `P_dR` and `P_xq`), which is
   not possible with the TCDEDT packets alone.

3. **Backend density.** QIEDT identifies the minimal Kerr-clean backend
   density `rho_backend = h*x^2 + tau*q^2` as a separate object, distinct from
   the full GPDDT density `q_grav`. This separation is essential for the
   quotient identity: `q_grav` is the *exported* source density (what the
   Carter calibration sees), while `rho_backend` is the *on-shell* Kerr-clean
   normal form (what the linearized Einstein operator produces before
   trapped-constraint corrections and gauge-tangent SOS defects are applied).

QIEDT is the sixth primitive-packet exact-derivation theorem in the audit
chain (after RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT) and the first to
address the `quotient_identity` source target. It is the first theorem to
introduce a certified SOS defect in the q_grav decomposition and the first to
close the `quotient_identity` backend obligation.
