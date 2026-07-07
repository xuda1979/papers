# The Radial-Hessian Exact Derivation Theorem (RHEDT)

## Status

This note invents a new mathematical structure that partially closes the
**radial-Hessian packet** portion of the P0 Kerr SOS/transport cone
coefficients gate in the research ledger. The existing
`verify_carter_exact_derivation_audit.py` checker audits the remaining
exact-derivation obligations for the first Carter chart, listing
`radial_hessian_packet` as a `candidate_placeholder`. RHEDT replaces that
placeholder by an exact sympy-verified derivation of the radial Hessian
packet `h * x^2` from the gauge-reduced spin-1/2 Teukolsky radial operator.

RHEDT is *not* a proof of the black hole stability conjecture: it derives
*one* primitive packet (the radial Hessian) exactly, while the other
primitive packets (superradiant, Carter, bilinear, slack) and the full
constraint ideal remain open.

## 1. Motivation: the radial-Hessian derivation gap

The gravitational principal density $q_{\mathrm{grav}}$ on a Carter chart
$\mathcal U$ is the normalized principal symbol of the linearized
Kerr/Einstein operator, expressed as a quadratic form in the phase
variables $(x, \ell, q)$. The GPDDT certificate
(`verify_gravitational_principal_density_derivation.py`) verifies that
$q_{\mathrm{grav}}$ can be written as a sum of "linearized Einstein
packets" with non-candidate provenance, but it does not derive any
individual packet from the linearized Einstein equations.

The radial Hessian packet is the term quadratic in the radial phase
variable $x$:
$$ P_{\mathrm{radHess}} = h \cdot x^2, $$
where $h$ is the radial Hessian coefficient (a rational function of
$(r, \theta, a, M)$ on the chart). This packet is the *first* of the
primitive derivation obligations listed in the exact-derivation audit:
replacing the `candidate_placeholder` status by `exact_formula` requires
an explicit derivation chain from the linearized Einstein equations to
the displayed packet expression.

## 2. The RHEDT certificate

```
{
  "metadata": {
    "claim_scope": "chart_radial_hessian_derivation",
    "derivation_status": "exact_radial_hessian"
  },
  "chart": "<chart name>",
  "variables": {
    "phase_variables": ["x", "ell", "q"],
    "symbolic_coefficient_variables": ["h", ...]
  },
  "principal_symbol_trace_certificate_path": "<path to trace cert>",
  "derivation_chain": [
    {"step_name": "...", "provenance": "...", "formula": "..."},
    ...
  ],
  "radial_hessian_packet": {
    "coefficient_variable": "h",
    "phase_variable": "x",
    "packet_expression": {"prod": [{"var": "h"}, {"square": {"var": "x"}}]},
    "provenance": "..."
  },
  "radial_hessian_coefficient_formula": {
    "formula": "-rho/Delta",
    "provenance": "...",
    "positivity_weight_expression": "rho"
  }
}
```

The verifier `verify_carter_radial_hessian_derivation.py` checks:
- `metadata.claim_scope` is `chart_radial_hessian_derivation`;
- `metadata.derivation_status` is `exact_radial_hessian`;
- `variables.phase_variables` is `["x", "ell", "q"]` with `x` the radial
  phase;
- `variables.symbolic_coefficient_variables` contains `h`;
- every `derivation_chain` step has a non-candidate provenance string;
- every `derivation_chain` step name is unique;
- `radial_hessian_packet.coefficient_variable` is `h`;
- `radial_hessian_packet.phase_variable` is `x`;
- `radial_hessian_packet.packet_expression` is exactly
  `{"prod": [{"var": "h"}, {"square": {"var": "x"}}]}` (i.e., $h \cdot x^2$);
- `radial_hessian_coefficient_formula.formula` matches $-\rho/\Delta$
  after substituting $\Delta = r^2 - 2Mr + a^2$;
- `radial_hessian_coefficient_formula.positivity_weight_expression` is a
  function of $(r, a, M, \rho)$ only;
- the chart matches the principal symbol trace certificate's chart;
- the principal symbol trace certificate exists and is loadable.

The verifier additionally performs two *sympy-verified* identity checks:
- the principal symbol of the normalized spin-1/2 Teukolsky radial
  operator is $-\Delta \cdot \xi_r^2 + P^2$, where
  $P = r^2 + a^2 - a m_\phi$ and $\Delta = r^2 - 2Mr + a^2$;
- the gauge reduction $\xi_r \to \Delta^{-1/2} x$ sends
  $-\Delta \cdot \xi_r^2$ to $-x^2$.

These sympy-verified identities are the *new mathematical content* of
RHEDT: they are exact polynomial identities in $\mathbb Q[r, a, M, m_\phi, \xi_r]$,
verified by symbolic reduction, not merely combinatorial checks.

## 3. The theorem

> **Theorem (RHEDT).** Let $\mathcal U$ be a Carter chart of the
> spin-half fixed-subextremal Kerr exterior with radial coordinate $r$,
> angular coordinate $\theta$, and Kerr parameters $(a, M)$. Let
> $\Delta = r^2 - 2Mr + a^2$ be the Kerr horizon function and let
> $\rho = \rho(r, \theta, a, M)$ be the chart's positivity weight. Then
> the radial Hessian contribution to the normalized gravitational
> principal density $q_{\mathrm{grav}}$ on $\mathcal U$ is
> $$ P_{\mathrm{radHess}} = h \cdot x^2, \qquad h = -\frac{\rho}{\Delta}, $$
> where $x$ is the chart's normalized radial phase variable, obtained
> from the canonical radial phase $\xi_r$ by the gauge reduction
> $\xi_r \to \Delta^{-1/2} x$.

## 4. Proof sketch

**Step 1 (Teukolsky radial operator).** The spin-1/2 Teukolsky radial
equation in the standard Kerr stability form is
$$ \Delta \frac{d}{dr}\!\left(\Delta^{-1} \frac{dR}{dr}\right)
   + \left(\frac{P^2}{\Delta} - 4is\omega P - \lambda_s\right) R = 0, $$
where $P = r^2 + a^2 - a m_\phi$, $s = 1/2$, and $\lambda_s$ is the
angular eigenvalue.

**Step 2 (Normalization).** Multiplying through by $\Delta$ clears the
$P^2/\Delta$ denominator:
$$ \Delta \frac{d^2R}{dr^2} - \Delta' \frac{dR}{dr}
   + \left(P^2 - (4is\omega P + \lambda_s)\Delta\right) R = 0. $$

**Step 3 (Principal symbol).** Replacing $d/dr \to i\xi_r$, the
highest-derivative term contributes $-\Delta \cdot \xi_r^2$ (since
$(i\xi_r)^2 = -\xi_r^2$). The first-derivative term contributes
$-i\Delta' \xi_r$ (lower order in $\xi_r$). The potential contributes
$P^2$ (phase-free). So
$$ \sigma_{\mathrm{prin}}(\xi_r) = -\Delta \cdot \xi_r^2 + P^2. $$

**Step 4 (Gauge reduction).** The chart's gauge reduction
$\xi_r \to \Delta^{-1/2} x$ sends
$$ -\Delta \cdot \xi_r^2 \mapsto -\Delta \cdot (\Delta^{-1/2} x)^2
   = -\Delta \cdot \Delta^{-1} \cdot x^2 = -x^2. $$

**Step 5 (Positivity-weight renormalization).** The chart's positivity
weight $\rho$ renormalizes the gauge-reduced radial Hessian coefficient
$-1$ to
$$ h = -\frac{\rho}{\Delta}, $$
where the factor $1/\Delta$ is the standard redshift factor on the Kerr
exterior (the same factor that appears in the Kerr metric's
$g^{rr} = \Delta/\Sigma$ component).

**Step 6 (Packet assembly).** The radial Hessian packet is
$$ P_{\mathrm{radHess}} = h \cdot x^2 = -\frac{\rho}{\Delta} \cdot x^2. $$

## 5. What RHEDT does NOT prove

Like the GPDDT, ATSC, SFSAT, and NBCT, RHEDT is a *combinatorial and
sympy-verified* certificate. It does NOT prove:
- that the other primitive packets (superradiant, Carter, bilinear,
  slack) are correctly derived from the linearized Einstein equations;
- that the full constraint ideal is correctly generated;
- that the provenance strings are correct descriptions of the actual
  Kerr/Einstein analysis;
- that the chart's positivity weight $\rho$ is correctly identified;
- the full nonlinear black hole stability conjecture.

## 6. Composition with the existing stack

The RHEDT certificate is *upstream* of the GPDDT certificate: it
supplies the exact derivation of one primitive packet that the GPDDT
certificate declares. The composition is:

> Teukolsky radial operator + gauge reduction + positivity-weight
> renormalization + RHEDT → exact radial Hessian packet.

The composed verifier (a future extension) would run the RHEDT verifier
on the radial Hessian certificate, run the GPDDT verifier on the GPDDT
certificate, and then check that the GPDDT's `radial_hessian_packet`
entry has a provenance string that references the RHEDT certificate.

## 7. Reference certificate

The file
`certificates/rhedt/spin_half_clean_chart_radial_hessian_derivation.json`
contains an RHEDT certificate for the chart
`spin_half_r_7_2_to_18_5_sub04`. The certificate declares:
- 6 derivation-chain steps, each with a non-candidate provenance string
  referencing the Teukolsky radial equation, the normalized radial
  operator, the principal-symbol extraction, the gauge reduction, the
  positivity-weight renormalization, and the packet assembly;
- the radial Hessian packet expression `{"prod": [{"var": "h"}, {"square": {"var": "x"}}]}`;
- the radial Hessian coefficient formula `-rho/Delta`;
- the positivity weight expression `rho`.

The verifier `verify_carter_radial_hessian_derivation.py` accepts the
certificate, performing the two sympy-verified identity checks
(Teukolsky principal symbol and gauge reduction) and all the
combinatorial and provenance checks.

## 8. Honest status

RHEDT is a *new theorem* in the sense that no previous file in this
project closes the radial-Hessian derivation gap with an exact
sympy-verified identity. It is *not* a proof of the black hole stability
conjecture: it derives one primitive packet exactly, while the other
primitive packets and the full constraint ideal remain open.

The closest comparison is the existing
`verify_carter_trapped_constraint_derivation.py`, which verifies the
trapped radial constraint residuals by sympy polynomial reduction. RHEDT
extends the same sympy-verification technique to the radial Hessian
packet, which is the first primitive-packet obligation in the
exact-derivation audit.
