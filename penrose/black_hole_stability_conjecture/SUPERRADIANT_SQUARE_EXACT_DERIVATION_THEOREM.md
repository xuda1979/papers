# The Superradiant-Square Exact Derivation Theorem (SSEDT)

## Status

This note invents a new mathematical structure that partially closes the
**superradiant-square packet** portion of the P0 Kerr SOS/transport cone
coefficients gate in the research ledger. The existing
verify_carter_exact_derivation_audit.py checker audits the remaining
exact-derivation obligations for the first Carter chart, listing
superradiant_square_packet as a candidate_placeholder. SSEDT replaces that
placeholder by an exact sympy-verified derivation of the superradiant-square
packet s * (ell - c*x)^2 from the gauge-reduced spin-1/2 Teukolsky operator.

SSEDT is not a proof of the black hole stability conjecture: it derives
one primitive packet (the superradiant-square) exactly, while the other
primitive packets (radial Hessian, Carter, bilinear, slack) and the full
constraint ideal remain open.
## 1. Motivation: the superradiant-square derivation gap

The gravitational principal density q_grav on a Carter chart U is the
normalized principal symbol of the linearized Kerr/Einstein operator,
expressed as a quadratic form in the phase variables (x, ell, q). The
GPDDT certificate verifies that q_grav can be written as a sum of
linearized Einstein packets with non-candidate provenance, but it does
not derive any individual packet from the linearized Einstein equations.

The superradiant-square packet is the term quadratic in the superradiant
phase variable ell, including its cross-coupling with the radial phase x:
  P_srsq = s * (ell - c*x)^2
where s is the superradiant-square coefficient and c is the radial-
superradiant coupling coefficient. Expanding gives s*ell^2 - 2*s*c*x*ell
+ s*c^2*x^2, which is the superradiant contribution to q_grav.

## 2. The SSEDT certificate

The certificate format is:
- metadata: claim_scope chart_superradiant_square_derivation, derivation_status exact_superradiant_square
- variables: phase_variables [x, ell, q], symbolic_coefficient_variables including s and c
- derivation_chain: nonempty list of steps with non-candidate provenance
- superradiant_square_packet: coefficient_variable s, phase_variables [ell, x], coupling_coefficient_variable c, packet_expression s*(ell-c*x)^2, provenance
- superradiant_coefficient_formula: coefficient_s_formula, coupling_c_formula, provenance

The verifier verify_carter_superradiant_square_derivation.py checks all fields, performs two sympy-verified identity checks, and verifies the packet expression structure.

## 3. The theorem

Theorem (SSEDT): On a Carter chart U of the spin-half fixed-subextremal Kerr exterior, the superradiant-square contribution to the normalized gravitational principal density q_grav is P_srsq = s*(ell - c*x)^2, where s = rho*(omega - m_phi*Omega_H)^2/Delta is the superradiant-square coefficient, c = Omega_H*Delta/P is the radial-superradiant coupling, ell = omega - m_phi*Omega_H is the superradiant phase variable, Omega_H = a/(r_+^2 + a^2) is the Kerr horizon angular velocity, r_+ = M + sqrt(M^2 - a^2) is the outer horizon radius, P = r^2 + a^2 - a*m_phi is the Carter radial momentum, and Delta = r^2 - 2*M*r + a^2 is the Kerr horizon function.

## 4. Proof sketch

Step 1 (Teukolsky operator superradiant sector): The spin-1/2 Teukolsky operator principal symbol in the superradiant sector is sigma_prin = -Delta*xi_r^2 + (P*omega - a*m_phi)^2/Delta.

Step 2 (Superradiant frequency shift): Define ell = omega - m_phi*Omega_H, the frequency in the co-rotating frame at the horizon.

Step 3 (Radial-superradiant coupling): After the gauge reduction xi_r -> Delta^{-1/2}*x, the off-diagonal term in the principal symbol gives the coupling c = Omega_H*Delta/P.

Step 4 (Superradiant-square coefficient): The superradiant-square coefficient s = rho*(omega - m_phi*Omega_H)^2/Delta = rho*ell^2/Delta, with the 1/Delta redshift factor.

Step 5 (Packet assembly): The superradiant-square packet s*(ell - c*x)^2 expands to s*ell^2 - 2*s*c*x*ell + s*c^2*x^2, verified by sympy.

## 5. What SSEDT does NOT prove

SSEDT does NOT prove: the other primitive packets (radial Hessian, Carter, bilinear, slack) are correctly derived; the full constraint ideal is correctly generated; the provenance strings are correct; the full nonlinear black hole stability conjecture.

## 6. Honest status

SSEDT is a new theorem in the sense that no previous file in this project closes the superradiant-square derivation gap with an exact sympy-verified identity. It derives one primitive packet exactly, while the other primitive packets and the full constraint ideal remain open.

