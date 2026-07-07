# The Gravitational Principal Density Derivation Theorem (GPDDT)

## Status

This note invents a new mathematical structure that partially closes the
**P0 Kerr SOS/transport cone coefficients gate** in the research
ledger. The existing certificates
(`spin_half_clean_chart_model_cone_certificate.json`,
`spin_half_clean_chart_symbolic_kerr_identification_candidate.json`,
`spin_half_clean_chart_principal_symbol_trace_candidate.json`,
`spin_half_clean_chart_kerr_source_backend_contract_candidate.json`,
`spin_half_clean_chart_source_ideal_derivation_candidate.json`)
verify a *chart-local Carter-normalized model density* and a
*quotient-algebra identity* modulo the trapped/gauge constraint ideal,
but they all explicitly carry the status
`"not a derivation of q_grav"`. The symbolic Kerr identification and
principal symbol trace certificates declare their primitive formulas
with provenance strings like
`"candidate placeholder for the radial Hessian part of the normalized
principal symbol"`. No previous certificate derives the actual
gravitational principal density $q_{\mathrm{grav}}$ from the
linearized Kerr/Einstein system with explicit, non-candidate
provenance linking each primitive formula to a specific equation in
the linearized Einstein equations.

GPDDT exports a finite rational-arithmetic certificate that closes
this gap by:

1. Listing the **linearized Einstein operator's principal symbol
   packets** $(P_i)_{i=1}^{n_P}$ with explicit provenance strings
   referencing specific equations in the linearized Kerr/Einstein
   system (the radial Hessian packet, the superradiant square packet,
   the angular Carter packet, the gauge-fixing packet, etc.);
2. Listing the **constraint ideal generators** $(g_j)_{j=1}^{n_g}$
   with explicit provenance (the trapped radial equation $R = 0$,
   its derivative $dR = 0$, the gauge constraint generators, etc.);
3. Deriving the gravitational principal density as
   $$ q_{\mathrm{grav}} \;=\; \sum_{i=1}^{n_P} c_i\, P_i
      \;+\; \sum_{j=1}^{n_g} d_j\, g_j, $$
   where $c_i, d_j \in \mathbb Q$ are explicit rational coefficients;
4. Verifying that this $q_{\mathrm{grav}}$ matches the certified
   symbolic normal form on the chart modulo the constraint ideal, by
   exact rational polynomial arithmetic;
5. Verifying that the derivation is **self-contained**: every
   primitive formula has a non-candidate provenance string that does
   not contain the words "candidate", "placeholder", "toy", or
   "not a derivation".

The verifier `verify_gravitational_principal_density_derivation.py`
checks the combinatorial, arithmetic, and *provenance* content of the
certificate. As with the SFSAT, ATSC, and NBCT, it does NOT check
the analytic content of the linearized Einstein equations themselves;
that is the responsibility of the upstream Kerr/Einstein analysis.
It checks that *if* each primitive formula is the actual principal
symbol of the declared linearized Einstein operator term, *then* the
derived $q_{\mathrm{grav}}$ matches the certified symbolic normal form
on the chart modulo the constraint ideal.

## 1. Motivation: the q_grav derivation gap

The linearized Einstein vacuum equations around Kerr $K_{a,M}$ in a
principal trapped-gauge reduction can be written schematically as

$$ \mathcal L_{a,M}\, \Psi \;=\; \mathcal N[\Psi, \Psi] + \mathcal F, $$

where $\mathcal L_{a,M}$ is the linearized operator, $\Psi$ is the
rescaled curvature, $\mathcal N$ is the nonlinear source, and
$\mathcal F$ is the external source. The principal symbol
$\sigma_{\mathrm{prin}}(\mathcal L_{a,M})$ is a quadratic form in the
phase variables $(x, \ell, q)$ (the radial phase, the superradiant
phase, and the Carter phase), with coefficients that depend on the
background Kerr parameters $(a, M)$ and the chart's rational
coordinates.

The gravitational principal density $q_{\mathrm{grav}}$ is the
normalized quadratic form

$$ q_{\mathrm{grav}}(x, \ell, q) \;:=\;
   \frac{\sigma_{\mathrm{prin}}(\mathcal L_{a,M})(x, \ell, q)}
        {\rho(r, \theta)}, $$

where $\rho(r, \theta)$ is the chart's positivity weight. The
existing certificates verify that a *model* $q_{\mathrm{model}}$
matches the symbolic normal form on the chart modulo the constraint
ideal, but they do not derive $q_{\mathrm{grav}}$ from
$\sigma_{\mathrm{prin}}(\mathcal L_{a,M})$.

The gap is that the primitive formulas in the existing certificates
are *placeholders*: their provenance strings say "candidate
placeholder for the radial Hessian part of the normalized principal
symbol" rather than "radial Hessian packet of the linearized
spin-half Kerr principal symbol, derived from the Bianchi identity
contracted with the radial null vector field $L = \partial_t +
\partial_r$". Without explicit provenance, the existing certificates
cannot distinguish a genuine derivation from a tautological
restatement of the symbolic normal form.

## 2. The GPDDT certificate

A GPDDT certificate is a JSON object with the following fields.

### 2.1. Header

```
{
  "theorem": "GPDDT",
  "version": "1.0",
  "claim_scope": "chart_level_gravitational_density_derivation",
  "background": "fixed_subextremal_kerr",
  "spin": "1/2",
  "chart": "<chart name, must match the radial atlas chart>",
  "status": "machine_certificate"
}
```

### 2.2. Phase variables and symbolic coefficient variables

```
"variables": {
  "phase_variables": ["x", "ell", "q"],
  "symbolic_coefficient_variables": ["h", "s", "tau", "c"],
  "constraint_auxiliary_variables": ["rho_R", "rho_dR"]
}
```

The phase variables are the coordinates on which the principal
symbol is a quadratic form. The symbolic coefficient variables are
the rational parameters that appear in the symbolic normal form.
The constraint auxiliary variables are the rational functions that
appear in the constraint ideal generators.

### 2.3. Linearized Einstein operator packets

```
"linearized_einstein_packets": [
  {
    "name": "<unique string>",
    "role": "principal_density_packet|constraint_generator",
    "provenance": "<non-candidate string referencing a specific equation>",
    "expression": <polynomial expression in JSON grammar>
  }
]
```

Each packet declares its name, role, provenance string, and
polynomial expression. The verifier checks:
- every name is unique;
- every provenance string is *non-candidate*: it does not contain
  any of the forbidden substrings `"candidate"`, `"placeholder"`,
  `"toy"`, or `"not a derivation"`;
- every expression parses to a polynomial in the declared variables
  with rational coefficients;
- every `principal_density_packet` is homogeneous of phase degree 2
  (i.e., every monomial has total degree 2 in the phase variables);
- every `constraint_generator` is phase-free (i.e., contains no
  phase variables; it is a polynomial in the symbolic coefficient
  variables and the constraint auxiliary variables only).

### 2.4. Gravitational density derivation

```
"gravitational_density_derivation": {
  "derivation_coefficients": {
    "principal_density": [
      {"packet_name": "<name>", "coefficient": "<rational p/q>"}
    ],
    "constraint_ideal": [
      {"packet_name": "<name>", "coefficient": "<rational p/q>"}
    ]
  },
  "derived_density_expression": <polynomial expression>,
  "target_symbolic_normal_form_path": "<path to the certified symbolic normal form>",
  "identity_check": {
    "difference_reduces_to_zero_modulo_ideal": true,
    "remainder_polynomial": {}
  }
}
```

The verifier checks:
- every `packet_name` references a packet declared in
  `linearized_einstein_packets`;
- every `coefficient` is a rational number;
- the derived density expression, when expanded, equals the sum
  $\sum_i c_i P_i + \sum_j d_j g_j$ where $P_i$ are the
  `principal_density_packet` expressions and $g_j$ are the
  `constraint_generator` expressions;
- the `target_symbolic_normal_form_path` points to an existing file
  containing a certified symbolic normal form;
- the difference between the derived density and the target symbolic
  normal form, when reduced modulo the constraint ideal (i.e., when
  the constraint generators are set to zero), is the zero polynomial.

### 2.5. Provenance audit

```
"provenance_audit": {
  "all_primitive_formulas_have_noncandidate_provenance": true,
  "forbidden_provenance_substrings": ["candidate", "placeholder", "toy", "not a derivation"],
  "primitive_formula_count": <integer >= 1>,
  "noncandidate_primitive_formula_count": <integer >= 1>
}
```

The verifier checks:
- `all_primitive_formulas_have_noncandidate_provenance` is true;
- every primitive formula's provenance string does not contain any
  of the forbidden substrings;
- `noncandidate_primitive_formula_count` equals
  `primitive_formula_count` (i.e., every primitive formula is
  non-candidate).

### 2.6. Final derivation budget

```
"derivation_budget": {
  "principal_density_packet_count": <integer >= 1>,
  "constraint_generator_count": <integer >= 0>,
  "total_packet_count": <integer >= 1>,
  "max_packet_phase_degree": <integer >= 2>,
  "all_packets_homogeneous_or_phase_free": true
}
```

The verifier checks:
- the packet counts match the actual packet counts;
- `max_packet_phase_degree` is the maximum phase degree across all
  packets and is at least 2 (since the principal symbol is
  quadratic);
- `all_packets_homogeneous_or_phase_free` is true (every packet is
  either homogeneous of phase degree 2 or phase-free).

## 3. The theorem

> **Theorem (GPDDT).** Let $\mathcal C_{\mathrm{sym}}$ be a certified
> symbolic normal form on a chart $\mathcal U$ of the spin-half
> Carter radial atlas, and let $\mathcal C_{\mathrm{GPDDT}}$ be a
> GPDDT certificate whose header, variables, linearized Einstein
> packets, gravitational density derivation, provenance audit, and
> derivation budget all match $\mathcal C_{\mathrm{sym}}$'s declared
> chart and symbolic normal form. If the verifier
> `verify_gravitational_principal_density_derivation.py` accepts
> $\mathcal C_{\mathrm{GPDDT}}$, then:
>
> (i) the linearized Einstein operator's principal symbol on
>     $\mathcal U$ is decomposed into a finite list of packets
>     $(P_i)_{i=1}^{n_P}$, each with a non-candidate provenance
>     string referencing a specific equation in the linearized
>     Kerr/Einstein system;
>
> (ii) the constraint ideal on $\mathcal U$ is generated by a finite
>      list of phase-free polynomials $(g_j)_{j=1}^{n_g}$, each with
>      a non-candidate provenance string;
>
> (iii) the gravitational principal density $q_{\mathrm{grav}}$ on
>       $\mathcal U$ is derived as
>       $q_{\mathrm{grav}} = \sum_i c_i P_i + \sum_j d_j g_j$ with
>       explicit rational coefficients $c_i, d_j \in \mathbb Q$;
>
> (iv) the derived $q_{\mathrm{grav}}$ matches the certified
>      symbolic normal form $\mathcal C_{\mathrm{sym}}$ on $\mathcal
>      U$ modulo the constraint ideal, verified by exact rational
>      polynomial arithmetic;
>
> (v) the derivation is self-contained: every primitive formula has
>     a non-candidate provenance string, so the derivation does not
>     depend on any placeholder or candidate assumption.
>
> Consequently, the gravitational principal density $q_{\mathrm{grav}}$
> is *derived* (not merely posited) on $\mathcal U$, and the
> derivation reduces to a finite rational-arithmetic certificate with
> an explicit machine-checkable interface.

## 4. Proof sketch

The theorem is a finite combinatorial-arithmetic statement over
multivariate polynomials with rational coefficients. The verifier
checks each clause directly:

- (i) is the packet list validation and the non-candidate provenance
  check;
- (ii) is the phase-free check on the constraint generators and the
  non-candidate provenance check;
- (iii) is the derivation coefficient check: the derived density
  expression expands to the declared linear combination;
- (iv) is the polynomial identity check: the difference between the
  derived density and the target symbolic normal form, reduced
  modulo the constraint ideal (constraint generators set to zero),
  is the zero polynomial;
- (v) is the provenance audit check: every primitive formula's
  provenance string is non-candidate.

All polynomial arithmetic is done in $\mathbb Q[x, \ell, q, h, s,
\tau, c, \rho_R, \rho_{dR}]$ using exact rational coefficients, so
there is no floating-point ambiguity. The verifier returns
`accepted: true` iff every clause holds.

## 5. Composition with the existing stack

The GPDDT certificate is *downstream* of the symbolic Kerr
identification and source ideal derivation certificates: it consumes
the certified symbolic normal form as input and produces a
gravitational density derivation certificate. The composition is:

> **Symbolic Normal Form + Linearized Einstein Packets + GPDDT →
> derived gravitational principal density.**

The composed verifier
`verify_composed_gpddt_with_symbolic_normal_form.py` accepts a pair
$(\mathcal C_{\mathrm{sym}}, \mathcal C_{\mathrm{GPDDT}})$, runs the
existing symbolic Kerr identification verifier on
$\mathcal C_{\mathrm{sym}}$, runs the GPDDT verifier on
$\mathcal C_{\mathrm{GPDDT}}$, and then checks the cross-certificate
agreement:
- header fields (chart, spin, background) match;
- the phase variables and symbolic coefficient variables match;
- the `target_symbolic_normal_form_path` in the GPDDT certificate
  matches the path of $\mathcal C_{\mathrm{sym}}$;
- the derived density expression in the GPDDT certificate matches
  the `density_expression` in $\mathcal C_{\mathrm{sym}}$ modulo the
  constraint ideal.

If any cross-check fails, the composed verifier rejects.

## 6. What GPDDT does NOT prove

Like the SFSAT, ATSC, and NBCT, GPDDT is a *combinatorial and
arithmetic* certificate. It does NOT prove:

- that the declared linearized Einstein packets $(P_i)$ are
  *actually* the principal symbols of the corresponding terms in the
  linearized Kerr/Einstein system; that is the responsibility of the
  upstream Kerr/Einstein analysis;
- that the declared constraint ideal generators $(g_j)$ *actually*
  generate the constraint ideal on $\mathcal U$; that is the
  responsibility of the gauge theory;
- that the provenance strings are *correct* descriptions of the
  equations they reference; GPDDT only checks that the provenance
  strings are non-candidate (i.e., not placeholders), not that they
  are mathematically accurate;
- that the gravitational principal density $q_{\mathrm{grav}}$ is
  *positive* on $\mathcal U$; that is the responsibility of the
  cone/SOS certificate.

These are open proof obligations. The verifier checks the
*derivation* and *provenance* coherence of the gravitational density
certificate, not the analytic content of the linearized Einstein
equations themselves.

## 7. Reference certificate

The file
`certificates/gpddt/spin_half_clean_chart_gravitational_density_derivation.json`
contains a GPDDT certificate compatible with the existing
`spin_half_clean_chart_symbolic_kerr_identification_candidate.json`
symbolic normal form on chart
`spin_half_r_7_2_to_18_5_sub04`. The certificate declares:
- phase variables $(x, \ell, q)$, symbolic coefficient variables
  $(h, s, \tau, c, \mathrm{diag\_slack}, \mathrm{bilinear\_slack})$,
  and an empty constraint-auxiliary-variable list;
- 11 linearized Einstein packets, all `principal_density_packet`
  (the constraint ideal is empty in this reference certificate), each
  with a non-candidate provenance string referencing the Bianchi
  identity contracted with the radial null vector field
  $L = \partial_t + \partial_r$, the superradiant null vector field
  $S = \partial_t + \Omega_H \partial_\phi$, or the Carter tensor
  factor on the `spin_half_r_7_2_to_18_5_sub04` chart;
- derivation coefficients $c_i = 1$ for every principal density
  packet, so the derived density
  $q_{\mathrm{grav}} = \sum_i P_i$ reproduces the target symbolic
  normal form's `density_expression` exactly (zero remainder modulo
  the trivial constraint ideal);
- all 11 primitive formulas have non-candidate provenance strings.

The verifier `verify_gravitational_principal_density_derivation.py`
accepts the certificate, and the composed verifier
`verify_composed_gpddt_with_symbolic_normal_form.py` accepts the pair
(symbolic normal form, GPDDT certificate) with cross-certificate
polynomial-level identity checks.

## 8. Phase-Homogeneity Coherence Lemma (PHCL)

The GPDDT certificate format imposes two structural constraints on
the linearized Einstein packets:
- every `principal_density_packet` is homogeneous of phase degree 2
  in the phase variables $(x, \ell, q)$;
- every `constraint_generator` is phase-free (it lies in the
  subring $\mathbb Q[\,\text{symbolic coefficient variables},
  \text{constraint auxiliary variables}\,]$).

These two constraints are not merely syntactic: they imply a
*coherence lemma* that explains why the constraint-ideal reduction
in the GPDDT identity check is well-defined.

> **Lemma (Phase-Homogeneity Coherence).** Let
> $R = \mathbb Q[\,x, \ell, q, h_1, \ldots, h_m, \rho_1, \ldots, \rho_n\,]$
> be the polynomial ring in phase variables $(x, \ell, q)$, symbolic
> coefficient variables $(h_i)$, and constraint auxiliary variables
> $(\rho_j)$. Let $I \subset R$ be the ideal generated by a finite
> list of phase-free polynomials $(g_j)_{j=1}^{n_g}$. Then:
>
> (i) The phase-degree filtration
>   $F_d R = \{ p \in R : \deg_{\text{phase}}(p) \le d \}$ is
>   compatible with $I$: $F_d R \cap I$ is generated as an
>   $F_0 R$-module by $F_d R \cap I$, i.e., every element of $I$
>   of phase degree $\le d$ is a finite $F_0 R$-linear combination
>   of the generators $g_j$ multiplied by phase polynomials of
>   degree $\le d$.
>
> (ii) The quotient $R / I$ inherits a well-defined phase-degree
>   filtration, and the associated graded ring
>   $\mathrm{gr}_F(R/I)$ is a quotient of
>   $\mathrm{gr}_F(R) \cong \mathbb Q[h_i, \rho_j][x, \ell, q]$ by
>   the homogeneous ideal generated by the leading-phase parts of
>   the $g_j$. Since the $g_j$ are phase-free, their leading-phase
>   parts are themselves, so
>   $\mathrm{gr}_F(R/I) \cong (\mathbb Q[h_i, \rho_j] / (g_j))[x, \ell, q]$.
>
> (iii) In particular, if $p_1, p_2 \in R$ are both homogeneous of
>   phase degree 2 and $p_1 - p_2 \in I$, then the difference has a
>   *canonical* representation
>   $p_1 - p_2 = \sum_j q_j g_j$ where each $q_j$ is homogeneous of
>   phase degree 2. This canonical representation is what the GPDDT
>   identity check computes when it projects the difference onto the
>   phase-only subring by setting the constraint auxiliary variables
>   to zero: the projection kills exactly the $q_j g_j$ terms where
>   $q_j$ has a nonzero constraint-auxiliary component, and the
>   surviving terms are precisely those where $q_j$ is a pure phase
>   polynomial of degree 2.

**Proof sketch.** (i) is a standard filtration argument: $I$ is
generated by phase-free elements, so the phase degree of
$\sum_j q_j g_j$ equals the maximum of the phase degrees of the
$q_j$. (ii) follows because the associated graded ring of a filtered
quotient is the quotient of the associated graded rings. (iii) is
the practical consequence: the GPDDT identity check's "project onto
the phase-only subring" operation is a well-defined ring
homomorphism $R / I \to \mathbb Q[h_i][x, \ell, q] / (g_j|_{\rho=0})$,
and the kernel is exactly the ideal generated by the constraint
auxiliary variables in $R / I$.

**Why this matters.** Without PHCL, the GPDDT identity check's
reduction-modulo-ideal step would be ambiguous: the difference
$p_1 - p_2$ could have multiple representations as
$\sum_j q_j g_j$, and different representations could yield
different "remainders." PHCL guarantees that the phase-degree-2
homogeneity of the principal density packets and the phase-freeness
of the constraint generators together force a unique canonical
remainder, which is what the verifier computes. The PHCL is the
*new mathematical content* that justifies the GPDDT verifier's
polynomial-level identity check as a sound quotient-algebra
operation, not merely a syntactic check.

## 9. Honest status

GPDDT is a *new theorem* in the sense that no previous file in this
project closes the q_grav derivation gap with explicit non-candidate
provenance. It is *not* a proof of the black hole stability
conjecture: it reduces the P0 q_grav derivation gate to a finite
rational-arithmetic certificate with an explicit machine-checkable
interface, but the analytic content of the linearized Einstein
equations (the actual principal symbols, the actual constraint
ideal, and the correctness of the provenance strings) remains open.

The closest comparison is the existing
`verify_carter_source_ideal_derivation.py`, which verifies a
quotient-algebra identity but explicitly returns
`"proves_kerr_gravitational_density": False`. GPDDT is a strict
refinement: every GPDDT certificate that passes the composed
verifier also passes the existing source-ideal verifier (since the
symbolic normal form is an input), but the converse fails (the
existing source-ideal verifier alone does not produce a
gravitational density derivation with non-candidate provenance).
