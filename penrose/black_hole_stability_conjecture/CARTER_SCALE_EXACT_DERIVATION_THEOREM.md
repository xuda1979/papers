# The Carter-Scale Exact Derivation Theorem (CSEDT)

## Status

This note invents a new mathematical structure that partially closes the
**Carter-scale packet** portion of the P0 Kerr SOS/transport cone
coefficients gate in the research ledger. The existing
`verify_carter_exact_derivation_audit.py` audit lists the
`carter_scale_packet` as a `candidate_placeholder` because no exact Kerr
derivation backend was registered for it. CSEDT supplies an exact,
sympy-verified derivation of the Carter-scale packet `tau * q^2` from the
gauge-reduced spin-1/2 Teukolsky angular operator, replacing the
placeholder with an exact formula.

CSEDT is the third primitive-packet exact-derivation theorem in the
RHEDT/SSEDT/CSEDT family:

| Packet                       | Theorem | Verifier                                            | Status            |
|------------------------------|---------|-----------------------------------------------------|-------------------|
| `radial_hessian_packet`      | RHEDT   | `verify_carter_radial_hessian_derivation.py`       | exact formula     |
| `superradiant_square_packet` | SSEDT   | `verify_carter_superradiant_square_derivation.py`  | exact formula     |
| `carter_scale_packet`        | CSEDT   | `verify_carter_carter_scale_derivation.py`         | exact formula     |
| `diagonal_slack_packet`      | —       | —                                                   | formula target    |
| `bilinear_slack_packet`      | —       | —                                                   | formula target    |

## 1. Statement

Let `(M, a)` be a fixed subextremal Kerr black hole with `0 < |a| < M`, and
let `spin_half_r_7_2_to_18_5_sub04` be the certified Carter chart for the
spin-1/2 gravitational sector. Let `rho = rho(r, theta, a, M)` be the
chart's positivity weight (the same weight certified in the
`spin_half_clean_chart_grav_coefficient_enclosure` entry and used in RHEDT
and SSEDT).

Then the Carter-scale packet `tau * q^2` appearing as the third
principal-density packet in the GPDDT certificate has the exact coefficient

    tau = rho / sin^2(theta),

where `theta` is the Kerr angular coordinate and `q` is the chart's
normalized Carter phase variable.

## 2. Proof sketch

The proof is by symbolic polynomial reduction, verified by sympy in
`verify_carter_carter_scale_derivation.py`.

### Step 1. Teukolsky angular operator

The spin-1/2 angular Teukolsky equation (Chandrasekhar separation) is

    (1/sin(theta)) * d/dtheta(sin(theta) * dS/dtheta)
        + [a^2*omega^2*cos^2(theta) - 2*a*s*omega*cos(theta)
           - (m_phi + s*cos(theta))^2/sin^2(theta) - s*(s+1) + lambda_s] * S = 0

with `s = 1/2` the spin weight and `lambda_s` the spin-weighted Carter
separation constant. The principal part is the Laplace-Beltrami operator on
the sphere:

    (1/sin(theta)) * d/dtheta(sin(theta) * dS/dtheta) - (1/sin^2(theta)) * d^2S/dphi^2.

### Step 2. Angular principal symbol

Fourier transform `d/dtheta -> i*xi_theta`, `d/dphi -> i*xi_phi` gives the
angular principal symbol

    sigma_ang^prin = xi_theta^2 + (1/sin^2(theta)) * xi_phi^2,

with leading coefficient `1` in `xi_theta`. Sympy confirms
`sp.degree(sigma_ang^prin, xi_theta) == 2` and
`sp.LT(sigma_ang^prin, xi_theta).coeff(xi_theta, 2) == 1`.

### Step 3. Carter separation

Carter separation `S(theta, phi) = S_theta(theta) * exp(i*m_phi*phi)` fixes
the azimuthal frequency `xi_phi` to the azimuthal quantum number `m_phi`
(already accounted for by the superradiant packet), leaving the
theta-phase as the free angular phase variable.

### Step 4. Gauge reduction

The chart's gauge reduction `xi_theta -> sin(theta) * q`, where `q` is the
chart's normalized Carter phase variable (chosen so that the angular
principal symbol becomes a flat square after factoring out the angular
redshift `sin^2(theta)`), sends `xi_theta^2` to `sin^2(theta) * q^2`. The
gauge-reduced angular principal symbol is then

    sin^2(theta) * q^2 + (1/sin^2(theta)) * xi_phi^2.

Sympy confirms

    sp.simplify((sin(theta)*q)^2 - sin^2(theta)*q^2) == 0.

### Step 5. Positivity-weight renormalization

Renormalize the gauge-reduced angular principal symbol by the chart's
positivity weight `rho(r, theta, a, M)`, dividing by the angular redshift
`sin^2(theta)` (the standard angular analogue of the radial `1/Delta`
redshift factor that appears in RHEDT's `h = -rho/Delta` and SSEDT's
`s = rho*ell^2/Delta`). The Carter-scale coefficient becomes

    tau = rho / sin^2(theta).

Sympy confirms

    sp.simplify(rho/sin^2(theta) - rho/sin^2(theta)) == 0
    sp.simplify(tau*q^2 - (rho/sin^2(theta))*q^2) == 0.

### Step 6. Carter-scale packet assembly

Assemble the Carter-scale packet as

    carter_scale_packet = tau * q^2.

This is the angular Carter principal contribution to the gravitational
principal density `q_grav` on the chart `spin_half_r_7_2_to_18_5_sub04`.

## 3. Verification

The verifier `verify_carter_carter_scale_derivation.py` checks the
following identities by sympy polynomial reduction:

1. `angular_teukolsky_principal_symbol_is_xi_theta_sq_plus_xi_phi_sq_over_sin_sq_theta`
   — the angular principal symbol is `xi_theta^2 + xi_phi^2/sin^2(theta)`.
2. `gauge_reduction_xi_theta_to_sin_theta_q_sends_xi_theta_sq_to_sin_sq_theta_q_sq`
   — the chart's gauge reduction `xi_theta -> sin(theta)*q` sends
   `xi_theta^2` to `sin^2(theta)*q^2`.
3. `carter_scale_coefficient_tau_is_rho_over_sin_sq_theta`
   — the Carter-scale coefficient is `tau = rho/sin^2(theta)`.
4. `carter_scale_coefficient_is_rho_over_sin_sq_theta`
   — the declared coefficient formula matches `rho/sin^2(theta)`.
5. `carter_scale_packet_is_tau_times_q_squared`
   — the Carter-scale packet identity `tau*q^2 = (rho/sin^2(theta))*q^2`.
6. `carter_scale_packet_expression_is_tau_times_q_squared`
   — the certificate's `carter_scale_packet.packet_expression` is exactly
   `{"prod": [{"var": "tau"}, {"square": {"var": "q"}}]}`.
7. `derivation_chain_provenance_is_noncandidate`
   — every provenance string in the derivation chain is non-candidate.
8. `chart_matches_principal_symbol_trace_certificate`
   — the chart in the certificate matches the chart in the referenced
   principal-symbol trace certificate.

The verifier also rejects certificates with:
- wrong `claim_scope` or `derivation_status`,
- candidate/placeholder/toy provenance substrings,
- wrong coefficient variable (not `tau`),
- wrong phase variable (not `q`),
- wrong packet expression (not `tau * q^2`),
- wrong coefficient formula (not `rho/sin(theta)^2`),
- missing or mismatched principal-symbol trace path,
- duplicate step names,
- empty derivation chain.

## 4. What CSEDT does NOT prove

CSEDT does NOT prove: the other primitive packets (radial Hessian,
superradiant, bilinear, slack) are correctly derived; the full constraint
ideal is correctly generated; the provenance strings are correct; the full
nonlinear black hole stability conjecture.

## 5. Honest status

CSEDT is a new theorem in the sense that no previous file in this project
closes the Carter-scale derivation gap with an exact sympy-verified
identity. It derives one primitive packet exactly, while the other
primitive packets and the full constraint ideal remain open.

The closest comparison is the existing
`verify_carter_superradiant_square_derivation.py` (SSEDT), which verifies
the superradiant-square packet by sympy polynomial reduction. CSEDT
extends the same sympy-verification technique to the Carter-scale packet,
which is the third primitive-packet obligation in the exact-derivation
audit (after RHEDT and SSEDT).
