# Kerr Certificate Scaffold

This directory turns the certificate roadmap in `black_hole_stability_conjecture.tex`
into a concrete verification target.

The verifier is intentionally conservative. It does not infer missing mathematics.
It checks that a proposed certificate contains explicit rational margins for:

- coercive phase-space energy positivity,
- nonsuperradiant Morawetz positivity and absorbable remainders,
- trapped-superradiant Fourier average positivity,
- nonlinear weak-null and modulation estimates,
- compatibility of Sobolev indices, weights, symbol classes, and frequency decomposition.

Run:

```sh
python3 ../verify_kerr_certificate.py examples/toy_certificate.json
```

Run the negative-margin check and unit tests from the repository root:

```sh
python3 verify_kerr_certificate.py certificates/examples/bad_negative_margin.json
python3 -m unittest tests/test_verify_kerr_certificate.py
```

Check a claimed unconditional completion package:

```sh
python3 verify_unconditional_completion_package.py certificates/completion_packages/toy_unconditional_completion_package.json
python3 verify_unconditional_completion_package.py certificates/completion_packages/bad_open_completion_package.json
python3 report_unconditional_completion_status.py certificates/completion_packages/current_public_status_package.json
```

The completion-package verifier requires every remaining theorem slot to be
marked as proved, rejects references that are only assumptions or conjectures,
and checks that all proof blocks use the same gauge, foliation, Sobolev index,
weights, frequency decomposition, and symbol class.
The status reporter is for not-yet-proved packages: it renders the best current
candidate inputs and the blockers that keep the package from passing the proof
verifier.

The schema is recorded in `schema/kerr_certificate.schema.json`. The Python verifier
adds semantic checks that JSON Schema cannot express, such as strict domination of
remainder intervals by positivity intervals.

Track current external candidate inputs:

```sh
python3 ../route_status.py candidate_inputs/current_literature.json
```

Generate a toy trapped-torus Fourier margin block:

```sh
python3 ../generate_fourier_certificate.py fourier_inputs/toy_torus_coefficients.json
```

Generate and check a chart-level Carter SOS scaffold:

```sh
python3 ../verify_carter_radial_identities.py
python3 ../verify_carter_polar_seed.py --output carter_identities/polar_spin_half_seed_certificate.json
python3 ../verify_carter_equatorial_seed.py --output carter_identities/equatorial_spin_half_seed_certificate.json
python3 ../verify_schwarzschild_photon_sphere.py --output carter_identities/schwarzschild_photon_sphere_certificate.json
python3 ../verify_carter_schwarzschild_transition.py --output carter_identities/schwarzschild_transition_blowup_certificate.json
python3 ../verify_carter_schwarzschild_blowup_interval.py --output carter_identities/schwarzschild_blowup_interval_certificate.json
python3 ../generate_carter_atlas.py carter_inputs/spin_half_radial_charts.json --subdivide 4 --output carter_atlases/spin_half_candidate_atlas_subdiv4.json
python3 ../verify_carter_atlas_coverage.py carter_atlases/spin_half_candidate_atlas_subdiv4.json --expected-start 7/3 --expected-end 18/5
python3 ../verify_carter_degenerate_seed_bridge.py carter_atlases/spin_half_candidate_atlas_subdiv4.json carter_identities/polar_spin_half_seed_certificate.json carter_identities/equatorial_spin_half_seed_certificate.json --output carter_identities/degenerate_seed_bridge_spin_half_certificate.json
python3 ../verify_carter_degenerate_seed_bridge.py carter_identities/degenerate_seed_bridge_spin_half_certificate.json
python3 ../verify_carter_angular_corridor.py carter_identities/degenerate_seed_bridge_spin_half_certificate.json --output carter_identities/angular_corridor_spin_half_certificate.json
python3 ../verify_carter_angular_corridor.py carter_identities/angular_corridor_spin_half_certificate.json
python3 ../verify_carter_angular_overlap.py carter_identities/angular_corridor_spin_half_certificate.json --output carter_identities/angular_overlap_spin_half_certificate.json
python3 ../verify_carter_angular_overlap.py carter_identities/angular_overlap_spin_half_certificate.json
python3 ../verify_carter_angular_patch_coverage.py carter_identities/angular_overlap_spin_half_certificate.json --output carter_identities/angular_patch_coverage_spin_half_certificate.json
python3 ../verify_carter_angular_patch_coverage.py carter_identities/angular_patch_coverage_spin_half_certificate.json
python3 ../verify_carter_angular_seed_tube.py carter_identities/angular_patch_coverage_spin_half_certificate.json --output carter_identities/angular_seed_tube_spin_half_certificate.json
python3 ../verify_carter_angular_seed_tube.py carter_identities/angular_seed_tube_spin_half_certificate.json
python3 ../verify_carter_angular_generic_overlap.py carter_identities/angular_seed_tube_spin_half_certificate.json --output carter_identities/angular_generic_overlap_spin_half_certificate.json
python3 ../verify_carter_angular_generic_overlap.py carter_identities/angular_generic_overlap_spin_half_certificate.json
python3 ../verify_carter_angular_overlap_policy.py carter_identities/angular_generic_overlap_spin_half_certificate.json --output carter_identities/angular_overlap_policy_spin_half_certificate.json
python3 ../verify_carter_angular_overlap_policy.py carter_identities/angular_overlap_policy_spin_half_certificate.json
python3 ../verify_carter_overlap_certificate.py carter_atlases/toy_overlap_atlas.json --require-certificates
python3 ../report_carter_atlas_status.py carter_atlases/spin_half_atlas_status.json
python3 ../analyze_carter_atlas.py carter_atlases/spin_half_candidate_atlas_subdiv4.json --output carter_atlases/spin_half_candidate_atlas_subdiv4.analysis.json
python3 ../generate_carter_sos_ansatz.py carter_atlases/spin_half_candidate_atlas_subdiv4.analysis.json --chart spin_half_r_7_2_to_18_5_sub04 --output carter_ansatz/spin_half_clean_chart_quadratic_ansatz.json
python3 ../solve_carter_gram_certificate.py carter_identities/toy_gram_solve_input.json --output carter_identities/toy_solved_gram_certificate.json
python3 ../verify_carter_gram_certificate.py carter_identities/toy_gram_certificate.json
python3 ../verify_carter_chart_certificate.py carter_chart_certificates/toy_clean_chart_certificate.json
python3 ../verify_carter_chart_certificate.py carter_chart_certificates/spin_half_clean_chart_model_cone_certificate.json
python3 ../verify_carter_coefficient_robustness.py carter_chart_certificates/spin_half_clean_chart_model_robustness_certificate.json
python3 ../verify_carter_grav_coefficient_enclosure.py carter_chart_certificates/spin_half_clean_chart_grav_coefficient_enclosure_candidate.json
python3 ../verify_carter_rational_coefficient_extraction.py carter_chart_certificates/spin_half_clean_chart_rational_coefficient_extraction_candidate.json
python3 ../verify_carter_symbolic_kerr_identification.py carter_chart_certificates/spin_half_clean_chart_symbolic_kerr_identification_candidate.json
python3 ../verify_carter_source_ideal_derivation.py carter_chart_certificates/spin_half_clean_chart_source_ideal_derivation_candidate.json
python3 ../verify_carter_kerr_source_backend.py carter_chart_certificates/spin_half_clean_chart_kerr_source_backend_contract_candidate.json
python3 ../verify_carter_principal_symbol_trace.py carter_chart_certificates/spin_half_clean_chart_principal_symbol_trace_candidate.json
python3 ../verify_carter_trace_defect_absorber.py carter_chart_certificates/spin_half_clean_chart_trace_defect_absorber_candidate.json
python3 ../verify_carter_defect_transport_morphism.py carter_chart_certificates/spin_half_clean_chart_defect_transport_shadow_candidate.json
python3 ../verify_carter_defect_transport_graph.py carter_chart_certificates/spin_half_clean_chart_defect_transport_graph_candidate.json
python3 ../verify_carter_orbit_cohomology_certificate.py carter_identities/spin_half_orbit_cohomology_shadow_certificate.json
python3 ../verify_carter_tame_orbit_cohomology.py carter_identities/spin_half_tame_orbit_cohomology_shadow_certificate.json
python3 ../verify_carter_finite_cone_package.py examples/toy_finite_carter_cone_package.json
python3 ../verify_carter_finite_cone_package.py examples/spin_half_clean_chart_model_finite_carter_cone_package.json
python3 ../verify_carter_wick_margin_certificate.py examples/toy_carter_wick_margin_certificate.json
python3 ../verify_carter_wick_margin_certificate.py examples/spin_half_clean_chart_model_carter_wick_margin_certificate.json
python3 ../verify_carter_master_certificate.py examples/toy_carter_master_certificate.json
python3 ../verify_carter_master_certificate.py examples/spin_half_clean_chart_model_carter_master_certificate.json
```

The chart certificate checker connects an ansatz, an exact polynomial identity,
an exact positive-semidefinite Gram witness when supplied, strict rational
interval margins, and a declared target-density kind. It rejects an attempted
`kerr_gravitational` claim unless a separate exact Kerr derivation backend is
present, so toy/model algebra cannot be silently promoted to a proof of the
gravitational cone lemma. The included toy chart certificate is a protocol
check, not a Kerr proof certificate.
Generated Carter atlas charts include denominator certificates for `a`,
`r_minus_M`, `(r_minus_M)^2`, `Delta`, and `r_plus`, so the spherical-orbit
formulas are accompanied by explicit nonzero-division evidence. The atlas also
includes adjacent-transition certificates: for each shared radial endpoint, the
coverage verifier recomputes the exact endpoint values of `lambda`, `eta`, the
radial Hessian, `Delta`, `r_minus_M`, and the horizon-frequency enclosures, then
checks that both neighboring chart intervals contain those endpoint data.
The overlap verifier is separate: it checks true positive-length chart overlaps
by recomputing the Carter interval data on the shared radial interval and
requiring both charts, plus any stored overlap certificates, to contain exactly
that recomputed data.
The polar seed checker handles the `L=0` degeneration for the spin-half atlas by
setting `lambda=0`, deriving the regular cubic
`r^3 - 3Mr^2 + a^2r + Ma^2 = 0`, and verifying a rational isolating interval
for the exterior polar root. It also checks positive `r-M`, positive `Delta`,
positive `eta`, and a positive radial-Hessian interval. This is only a seed: the
full polar angular-torus neighborhood and overlap certificates are still open.
The equatorial seed checker handles the `Q=0` degeneration by setting `eta=0`,
deriving the regular cubic `r^3 - 6Mr^2 + 9M^2r - 4Ma^2 = 0`, and verifying
rational isolating intervals for the spin-half prograde and retrograde exterior
roots. It checks positive `r-M`, positive `Delta`, finite `lambda`, and positive
radial-Hessian intervals on both branches. This is also only a seed until the
full equatorial angular neighborhoods and overlaps are supplied.
The degenerate seed bridge checker anchors these seed intervals in the verified
spin-half radial atlas. It proves that the polar seed lies in
`spin_half_r_5_2_to_3_sub04`, the equatorial prograde seed lies in
`spin_half_r_7_3_to_5_2_sub01`, and the equatorial retrograde seed lies in
`spin_half_r_7_2_to_18_5_sub02`. On each seed interval it recomputes the
generic nonzero-spin Carter interval data and checks that the regular seed
values are contained in that data and that the containing radial chart contains
the recomputed intervals. This is a radial bridge only; the angular corridor,
overlap, and patch-coverage certificates below supply the first local angular
neighborhoods attached to it.
The angular corridor checker verifies the first local regular angular
neighborhoods attached to that bridge. For the polar degeneration it uses
`lambda^2 = sin^2(theta)*mu^2`, which removes the cotangent singularity in the
Carter angular potential and gives two small one-sided `lambda` corridors. For
the equatorial degeneration it uses `cos^2(theta)=eta*zeta`, which factors the
angular potential by `eta` and certifies nonnegativity for prograde and
retrograde corridors. These are local angular positivity boxes, not full
angular-torus overlap certificates.
The angular overlap checker verifies the first ordinary-chart gluing strips for
those corridors. On the polar side it restricts to `sin^2(theta) in [1/64,1/16]`,
so the ordinary `cot^2(theta)` denominator is bounded away from zero and the
regularized identity glues to the ordinary angular potential. On the equatorial
side it restricts to `eta in [1/8192,1/4096]`, so
`zeta=cos^2(theta)/eta` is an ordinary overlap variable. The checker proves
strict positivity of the ordinary angular potential and exact agreement with the
regularized formulas on two polar and two equatorial strips. This is still local
gluing, not a full angular atlas.
The angular patch-coverage checker enlarges this local gluing into a first
finite patch cover. On the polar side, it proves that the degenerate corridor
`sin^2(theta) in [0,1/16]` and the ordinary annulus
`sin^2(theta) in [1/64,1]` cover the full `sin^2(theta)` interval in the fixed
spin-half seed tube, with positive ordinary angular potential on the annulus.
On the equatorial side, it extends the blow-up coordinate to
`zeta in [0,1/128]` and verifies positive bracket and ordinary-overlap margins
for both prograde and retrograde branches. This remains local in the seed
parameters; it is not a full angular-torus atlas.
The angular seed-tube checker then follows the certificate chain back to the
degenerate radial bridge and uses the bridge's generic `lambda` and `eta`
intervals to thicken the local parameters. On the polar side, it verifies a
two-sided regular corridor `|lambda| <= 1/6000`, with
`mu^2 <= 1/2250000`, and also proves ordinary-annulus positivity using the full
bridge lambda interval. On the equatorial side, it verifies
`eta in [0,1/1024]`, `zeta in [0,1/48]`, positive blow-up factor, and positive
ordinary-overlap margins for both branches. This is a parameter-thickened seed
tube, not a complete angular atlas.
The angular generic-overlap checker then proves a local nonsingular doorway
from those seed-tube variables back to the ordinary Carter angular variables.
On the polar side it certifies the positive-length box
`lambda in [-1/12000,1/12000]`, `sin^2(theta) in [1/64,1/16]`, with
`mu^2=lambda^2/sin^2(theta) <= 1/2250000`, positive transition denominator
`sin^2(theta)`, and positive ordinary and regularized angular potentials. On
the equatorial side it certifies, on both branches,
`eta in [1/8192,1/1024]`, `cos^2(theta) in [0,1/393216]`, with
`zeta=cos^2(theta)/eta <= 1/48`, positive transition Jacobian `eta`, positive
`1-cos^2(theta)`, and positive ordinary and regularized angular potentials.
This removes the local generic-overlap blocker at seed level; it still is not a
complete angular-torus overlap policy.
The angular overlap-policy checker then upgrades the rectangular doorway boxes
to dependent transition policies on the full certified seed overlap bands. On
the polar side it proves that the regular axis chart
`sin^2(theta) in [0,1/16]` and ordinary annulus
`sin^2(theta) in [1/64,1]` cover `[0,1]`, overlap on `[1/64,1/16]`, and glue
through `lambda^2=sin^2(theta)*mu^2` for the full seed tube
`|lambda| <= 1/6000`, `mu^2 <= 1/2250000`. On the equatorial side it proves,
on both prograde and retrograde branches, the dependent collar
`eta in [1/8192,1/1024]`, `zeta in [0,1/48]`, whose ordinary image satisfies
`cos^2(theta) in [0,1/49152]`, with positive transition Jacobian, positive
`1-cos^2(theta)`, and positive ordinary/regularized angular potentials. This
is a seed-level overlap policy, not the full angular-torus atlas.
The orbit-cohomology checker attacks the analytic trapped-superradiant
escape-function obstruction. It splits a finite torus Fourier density into
resonant and nonresonant modes, checks exact rational Fourier division for
every nonresonant coboundary `i(k.omega)G_k=F_k`, verifies real conjugate
symmetry, bounds the resonant obstruction by a rational triangle lower bound,
and confirms that approximation, partition, normal-extension, and quantization
losses fit inside a positive margin. The included
`spin_half_orbit_cohomology_shadow_certificate.json` has minimum net margin
`91/512`; it is a shadow torus protocol, not a Kerr Fourier derivation, but it
turns the old orbit-average/globalization assumptions into a finite certificate
shape.
The tame orbit-cohomology checker adds the small-denominator layer needed for
the same analytic mechanism. It verifies a nonresonant denominator floor,
computes the weighted escape-symbol derivative seminorm through a recorded
order, and subtracts a rational derivative-loss budget from the
orbit-cohomology margin. The included
`spin_half_tame_orbit_cohomology_shadow_certificate.json` has denominator floor
`1`, maximum escape seminorm `101/180`, and residual tame margin `3893/23040`;
it is still a shadow protocol, not a Kerr Fourier proof.
The Schwarzschild checker verifies the exact `a=0` photon-sphere seed chart in
the regular variable `j=(L^2+Q)/E^2`. The Schwarzschild transition checker adds
the blow-up coordinate `sigma=(r-3M)/a` and proves that the nonzero-spin
invariant `j=lambda^2+eta` has no remaining `a`-pole, with limit `27M^2` and
radial-Hessian limit `54M^2`. The interval blow-up checker then verifies, using
exact rational arithmetic, the local box `|a|/M <= 1/64`, `|sigma| <= 1`, with
positive `Delta`, positive `r-M`, finite `lambda`, positive `eta`, and intervals
containing the Schwarzschild values `j/M^2=27` and Hessian/M^2=`54`. The atlas
status report records which degenerate chart families are still incomplete:
the polar and equatorial charts now have local angular corridor,
ordinary-overlap, patch-coverage, seed-tube thickening, and local generic
seed-overlap and dependent overlap-policy certificates attached to the radial
bridge but not complete angular-torus atlases, the zero-spin chart now has a verified local
interval transition seed but no complete small-spin trapped-set cover or full
overlap policy from nonzero spin, and near-extremal uniformity remains open.
The Gram solver is a small exact square-extraction search. It can construct
rational PSD Gram witnesses for sparse toy targets and emits certificates that
the independent Gram verifier checks, but it is not yet a general semidefinite
program solver and does not produce a Kerr cone certificate by itself.
The first non-toy chart-local model certificate is
`carter_chart_certificates/spin_half_clean_chart_model_cone_certificate.json`.
It uses the actual verified interval data from
`spin_half_r_7_2_to_18_5_sub04`: the radial-Hessian lower endpoint, the
superradiant-factor lower endpoint, the inverse trapped-scale lower endpoint,
and the midpoint of the lambda interval. It proves an exact rational SOS
identity and a strict `1/256` coercivity margin for the model density
`h*x^2+s*(ell-c*x)^2+tau*q^2`. This is meaningful pipeline progress, but it is
explicitly labeled `chart_local_carter_normalized_model`; it is not the
gravitational density `q_grav`.
The companion robustness certificate
`carter_chart_certificates/spin_half_clean_chart_model_robustness_certificate.json`
proves that every quadratic density whose coefficients differ from the model by
at most `1/2048` on diagonal terms and `1/4096` on bilinear terms remains
coercive with margin `1/512`. This turns the next gravitational-density step
into a coefficient-comparison problem rather than an exact-match problem.
The gravitational coefficient-enclosure checker verifies the transfer theorem
from such a coefficient comparison to a local coercive Carter density. The
included `spin_half_clean_chart_grav_coefficient_enclosure_candidate.json`
lands strictly inside the certified tube and records the coefficient intervals
that a future exact `q_grav` derivation should aim to hit, but it is only marked
`candidate_enclosure`; the checker rejects `kerr_gravitational_proved` until a
separate exact Kerr derivation backend is registered.
The rational coefficient-extraction checker then verifies the next finite gate:
given interval symbols and rational expression trees for the six coefficients,
it evaluates them by exact rational interval arithmetic, checks all denominator
intervals avoid zero, and proves that the resulting intervals are contained in
the accepted coefficient enclosure. The included
`spin_half_clean_chart_rational_coefficient_extraction_candidate.json` is still
only a `candidate_extraction`, but it demonstrates the complete arithmetic
backend that a future exact `q_grav` derivation can feed.
The symbolic Kerr-identification checker verifies the final local algebraic
normal-form layer: a symbolic quadratic density must expand exactly as the six
coefficient expression trees used by the rational extraction certificate. The
included `spin_half_clean_chart_symbolic_kerr_identification_candidate.json`
checks that normal-form identity and inherits the `1/512` local coercivity
margin through the extraction/enclosure/tube chain. It is still only marked
`candidate_symbolic_identity`; the checker rejects `kerr_gravitational_proved`
until a separate exact derivation from the Kerr gravitational principal density
is registered.
The source-ideal derivation checker verifies the quotient-algebra layer expected
from a real principal-symbol derivation: a proposed source density may equal the
accepted symbolic normal form modulo explicitly listed phase-free trapped
constraints. The included
`spin_half_clean_chart_source_ideal_derivation_candidate.json` proves such an
ideal identity and inherits the `1/512` local coercivity margin on the
constraint quotient; it is still only `candidate_source_ideal` until an exact
Kerr source derivation backend is registered.
The Kerr source-backend checker verifies the final interface contract that an
exact principal-symbol derivation must satisfy before the local chart can be
upgraded to the true gravitational density. The included
`spin_half_clean_chart_kerr_source_backend_contract_candidate.json` checks that
the backend-exported source density and trapped constraints exactly match the
source-ideal certificate, then inherits the `1/512` quotient margin through the
full source-ideal/symbolic/extraction/enclosure/tube chain. It is deliberately
only `candidate_backend_contract`: the checker rejects
`kerr_gravitational_proved` until an actual exact linearized-Kerr
principal-symbol derivation engine is implemented.
The principal-symbol trace checker adds a finite algebraic audit layer before
that backend interface. The included
`spin_half_clean_chart_principal_symbol_trace_candidate.json` decomposes the
candidate source into primitive radial-Hessian, superradiant-square,
Carter-scale, slack, and trapped-ideal packets, verifies phase homogeneity and
phase-free trapped constraints, and proves that the recombined trace exactly
matches the backend source and constraints. It is still only `candidate_trace`,
but it turns the final Kerr computation into packet-level exact formulas plus a
machine-checked recombination trace.
The exact-derivation audit checker turns the remaining first-chart `q_grav`
derivation into a finite registry. The included
`spin_half_clean_chart_exact_derivation_audit_candidate.json` records every
primitive trace packet and every backend derivation obligation, assigning each a
source target, verification interface, and replacement criterion. It is still
only `candidate_audit`; it does not prove `q_grav`. Its role is to prevent any
future upgrade from skipping a primitive packet or prose-level backend
obligation when replacing the placeholders by exact Kerr formulas.
The trapped-constraint derivation checker proves the first exact family in that
registry. The included
`spin_half_clean_chart_trapped_constraint_derivation_certificate.json` verifies
the Kerr radial potential identities for spherical null geodesics, checks
`R=0` and `dR=0` after clearing denominators, and confirms that the backend
exports these normalized residuals as `rho_R` and `rho_dR`. This upgrades the
constraint entries in the audit to `exact_formula`, while the gravitational
principal density packets remain open.
The trace-defect absorber checker proves a margin-preserving relaxation of the
trace match. The included
`spin_half_clean_chart_trace_defect_absorber_candidate.json` verifies that a
candidate density equals the accepted trace density plus a nonnegative rational
square defect, modulo an additional phase-free gauge constraint. Therefore the
candidate inherits the same `1/512` lower bound on the joint trapped/gauge
quotient. It is still only `candidate_defect_absorber`, but it allows a future
exact Kerr source computation to contain favorable square defects and extra
ideal terms without losing margin.
The fine-grained completion roadmap
`completion_packages/fine_grained_completion_roadmap.json` is a checked
dependency graph, not a proof certificate. The verifier
`verify_fine_grained_roadmap.py` confirms that every roadmap node has a phase,
status, dependencies, acceptance criteria, artifacts, and blockers, and that the
dependency graph is acyclic. Its current frontier identifies the next
independent proof targets: the first-chart variable inventory, gauge-constraint
map, packet registry, raw superradiant factors, horizon-velocity
normalization, Carter angular inventory, and the parallel Teukolsky
boundary-condition contract.
The defect-transport morphism checker proves a local-to-local propagation rule
for these absorber certificates. The included
`spin_half_clean_chart_defect_transport_shadow_candidate.json` verifies an
exact phase pullback from the first chart absorber to a shadow normalized chart,
checks that the target density is the pulled source plus a nonnegative
transport square, and certifies phase-norm lower factor `1`. Thus the transported
chart inherits the `1/512` margin. It is a protocol/shadow transport, not a new
Kerr atlas chart, but it is the finite gate needed to seed neighboring charts
from an accepted absorber.
The finite defect-transport graph checker upgrades one-edge transport into an
ordered atlas-propagation certificate. Starting from an accepted seed
trace-defect absorber, every edge must use an already certified source chart,
verify the same exact transport and phase-norm identities, and export the
target margin as the source margin times the edge norm factor. The included
`spin_half_clean_chart_defect_transport_graph_candidate.json` certifies two
shadow targets with product margin `1/512` and rejects any attempted
gravitational proof upgrade. It is not a real Kerr atlas graph, but it is the
new finite mechanism needed once genuine overlap pullbacks are supplied.
The finite cone package checker proves the abstract patching criterion for
chart-level cone certificates: every referenced chart certificate must verify
independently, the quadratic partition residual must be exactly zero, and the
sum of transport, overlap, normal-extension, and quantization losses must be
strictly smaller than the minimum local margin slack. The included finite
toy package is a protocol check, not a Kerr cone certificate. The
spin-half-clean-chart model finite package verifies the same arithmetic route
for the model density on one real chart.
The Carter-Wick margin checker records the operator-level quantization loss
budget used by the finite assembly checker. It requires the calculus contract,
compact phase-neighborhood hypothesis, and strict domination of the exported
quantization loss by a positive margin. The included margin certificate is a
toy protocol check, not a Kerr pseudodifferential proof.
The master checker composes the finite cone package with the Wick margin
certificate. It verifies that the Wick exported loss is contained in the finite
assembly quantization-loss allowance and that derivative loss is budgeted. The
included toy master certificate is a protocol check, not a Kerr proof
certificate. The spin-half-clean-chart model master certificate shows that a
single-chart real-data model certificate can pass the composed pipeline.

The included `toy_certificate.json` is a sanity check for the verifier format. It is
not a Kerr proof certificate.

To pursue the conjecture route, replace the toy data with Kerr-derived rational
interval bounds and SOS/Gram decompositions in the same schema.
