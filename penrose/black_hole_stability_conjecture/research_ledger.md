# Research Ledger

## Goal
Develop a certificate-driven roadmap toward fixed-subextremal exterior Kerr
stability, while keeping the global theorem conditional until every analytic
and machine-checkable input is actually proved.

## Current Proof Status
- Main claim: Reduction / Gap. Full nonlinear fixed-subextremal Kerr stability
  is not proved in this repository.
- Strongest theorem currently established: the abstract Carter finite-assembly,
  finite-margin Wick, and master-certificate reductions are proved; in addition,
  one real spin-half Carter radial chart now carries an exact chart-local
  Carter-normalized model cone certificate with strict rational coercivity
  margin, a robust coefficient tube, and a coefficient-enclosure transfer
  theorem plus a rational coefficient-extraction transfer theorem reducing the
  local gravitational cone step to a symbolic quadratic normal-form identity,
  six expression trees, interval denominator checks, a source-ideal quotient
  identity, a checked Kerr source-backend interface contract, and a
  principal-symbol trace decomposition with a positive/ideal defect absorber
  and a finite defect-transport graph protocol. This is not yet the
  gravitational cone certificate.
- Verification last run: `/usr/bin/python3 -m unittest discover tests`; `pdflatex
  -interaction=nonstopmode black_hole_stability_conjecture.tex` twice; LaTeX
  unresolved-reference scan clean.

## Dependency Chain
1. Conditional nonlinear stability follows from a compatible full-subextremal
   linear/Morawetz package, curvature-to-metric reconstruction, global gauge and
   modulation, weak-null nonlinear estimates, fixed-spin parameter control, and
   compatibility assembly.
2. The trapped-superradiant Morawetz gate is reduced to a Renormalized Carter
   cone certificate plus Carter-Wick quantization.
3. The finite Carter certificate route currently has seed-level spin-half atlas
   certificates but lacks a complete angular-torus atlas and actual Kerr
   SOS/transport cone coefficients.

## Open Gates
| Priority | Gate | Status | Needed next |
|---|---|---|---|
| P0 | Complete spin-half angular-torus atlas | Gap | Extend the seed-level polar/equatorial overlap policies to every angular chart with positive-length adjacent overlaps. |
| P0 | Kerr SOS/transport cone coefficients | Gap | Derive the normalized gravitational principal density from the linearized Kerr/Einstein system and identify it with the certified symbolic normal form modulo the trapped/gauge constraint ideal on one verified chart. |
| P1 | Small-spin blow-up cover | Gap | Extend the verified Schwarzschild interval seed into a finite small-spin trapped-set cover and overlap it with nonzero-spin charts. |
| P1 | Carter-Wick quantization | Reduction | Replace toy finite-margin budgets by a Kerr pseudodifferential certificate in the same symbol class as the cone package. |
| P2 | Completion package compatibility | Gap | Put all analytic blocks in one gauge, foliation, Sobolev hierarchy, frequency decomposition, and symbol class. |

## Fine-Grained Completion Roadmap
Status: Reduction. The coarse gates above are now refined by the machine-checked
dependency graph
`certificates/completion_packages/fine_grained_completion_roadmap.json`, checked
by `verify_fine_grained_roadmap.py`. This graph assumes the current certificate
architecture is viable and splits the remaining proof into 37 nodes:
architecture is viable and splits the remaining proof into 46 nodes:

| Block | Nodes | Current frontier |
|---|---|---|
| First-chart `q_grav` | FG-01--FG-07 | FG-02a1a variable inventory, FG-03a1a raw superradiant factors, FG-04a1a angular variable inventory |
| Atlas propagation | FG-08--FG-09 | blocked until first-chart `q_grav` is exact |
| Kerr orbit cohomology | FG-10--FG-12 | blocked until genuine atlas propagation |
| Wick quantization | FG-13 | blocked until atlas and tame orbit certificates |
| Linear package | FG-14--FG-16 | FG-15a Teukolsky boundary contract can proceed in parallel; Carleman waits for Wick/atlas constants |
| Near-extremal bridge | FG-17 | blocked until compatible linear package |
| Nonlinear closure | FG-18--FG-20 | blocked until linear/near-extremal constants exist |

The graph is a work queue, not a proof of the final theorem. A node may be
upgraded to `proved` only when its named checker/certificate or standard cited
theorem satisfies the listed acceptance criteria. The immediate local frontier
is therefore: write the variable inventory, gauge-constraint map, packet
registry, raw superradiant factor list, horizon-velocity normalization, and
Carter angular inventory. Those are the next places where prose must become
exact machine-readable structure.

## Roadmap Audit: Five-Phase Innovation Plan
Status: Speculative / Reduction. The five-phase plan is a useful organizing
map, but none of the named innovations currently closes a manuscript assumption
unless it is converted into a precise theorem with checkable hypotheses and a
finite verification interface. The main risk is analytic debt migration:
phrases such as exact Stokes functor, entropy current, or fractional
interpolation can rename the original spectral/Morawetz/nonlinear obstruction
unless they export numerical constants, symbol classes, compatibility maps, and
regularity thresholds that plug into the conditional theorem.

Decisive obligations:
- Spectral phase: replace the Holomorphic Spectral Stokes Functor slogan by a
  meromorphic Fredholm determinant construction for the separated Teukolsky
  problem, with boundary condition normalization, analyticity domains in
  `omega`, mode-uniform estimates, and an explicit lower half-plane exclusion
  strip. A resurgence description is not enough unless it yields rigorous
  interval bounds for low modes and uniform high-frequency estimates near
  extremality.
- Superradiant trapped phase: treat tame orbit cohomology as the most promising
  local route, but require actual Kerr Carter-torus Fourier coefficients,
  denominator floors, resonant average inequalities, and symbol-seminorm bounds
  in the same Carter-Wick calculus. The existing certificates are shadow
  protocol checks, not Kerr orbit certificates.
- Ergosphere coercivity phase: any unified energy functional must specify a
  gauge, foliation, pseudodifferential class, commutator remainder budget, and
  trapped-set patching mechanism. Tuning multipliers to Lyapunov exponents is
  only a proof step after the relevant pseudoconvexity inequalities are stated
  as verifiable interval inequalities on the Kerr phase atlas.
- Near-extremal phase: a fractional interpolation space must prove uniform
  constants as `kappa -> 0`, compatibility with horizon regularity, branch-cut
  spectral contributions, and nonlinear product estimates. A blended decay
  rate does not by itself control the Aretakis-type limiting obstruction.
- Nonlinear phase: the bootstrap can close only after the linear estimates,
  reconstruction, modulation, gauge, and weak-null estimates share one Sobolev
  hierarchy and one final-state parameter map. The differential inequality is
  currently a conditional schematic, not an exported theorem.

Priority adjustment: the closest real proof-producing target remains the
Carter/superradiant certificate branch, because it already has finite checkers,
explicit margins, and a ledgered interface. The spectral Stokes and
near-extremal interpolation phases should be recorded as parallel speculative
routes until they produce a theorem reducing an infinite analytic statement to
finite interval data or a standard cited theorem.

## Recent Progress
- 2026-05-28: Added
  `verify_carter_angular_generic_overlap.py`,
  `certificates/carter_identities/angular_generic_overlap_spin_half_certificate.json`,
  and Lemma `angular-generic-overlap`. This proves local nonsingular inverse
  coordinate doorways from spin-half seed tubes to ordinary Carter angular
  variables.
- 2026-05-28: Updated `certificates/carter_atlases/spin_half_atlas_status.json`
  to `seed_angular_generic_overlap_verified` for the polar and equatorial
  degenerate families.
- 2026-05-28: Updated
  `certificates/completion_packages/current_public_status_package.json` so the
  global completion status records the new Carter atlas evidence without
  upgrading the global theorem.
- 2026-05-31: Added `verify_carter_angular_overlap_policy.py`,
  `certificates/carter_identities/angular_overlap_policy_spin_half_certificate.json`,
  and Lemma `angular-overlap-policy`. This proves a seed-level dependent
  overlap policy on the polar band and both equatorial branches.
- 2026-06-01: Added the local Carter atlas certification audit to
  `report_carter_atlas_status.py` and Lemma `local-carter-atlas-audit`. This
  proves that every recorded non-extremal spin-half chart family has accepted
  local certified evidence; near-extremal uniformity is the only recorded family
  with no local seed/transition certificate.
- 2026-06-01: Proved the finite Carter cone assembly theorem
  (`finite-carter-cone-assembly`) and added
  `verify_carter_finite_cone_package.py`. This closes the abstract finite
  patching step for the trapped-superradiant cone route: once chart-level
  SOS-plus-transport certificates, an exact quadratic partition, and strict
  assembly-loss margins are supplied, the local certificates glue to the
  Carter calibration commutator on the covered trapped-superradiant phase
  neighborhood. The included JSON packages are toy protocol checks, not Kerr
  cone certificates.
- 2026-06-01: Proved the finite-margin Carter-Wick quantization theorem
  (`finite-margin-carter-wick`) and added
  `verify_carter_wick_margin_certificate.py`. This turns the quantization step
  from an opaque assumption into a finite rational loss budget: square Garding
  loss, commutator remainder, subprincipal loss, and smoothing loss must fit
  inside an exported quantization bound strictly dominated by the recorded
  positive margin. The included certificate is a toy protocol check, not a Kerr
  pseudodifferential certificate.
- 2026-06-01: Proved the master Carter certificate theorem
  (`master-carter-certificate`) and added
  `verify_carter_master_certificate.py`. This composes the finite cone assembly
  package and the finite-margin Carter-Wick package: if the exported Wick loss
  lies inside the assembly quantization allowance and the derivative loss is
  budgeted, the trapped-superradiant Carter calibration commutator follows on
  the covered phase neighborhood. This makes the remaining central task a
  finite Kerr coefficient/atlas artifact, not another abstract patching or
  quantization theorem.
- 2026-06-01: Proved the first certified chart-local Carter cone model theorem
  (`first-chart-local-carter-model`). The certificate
  `certificates/carter_chart_certificates/spin_half_clean_chart_model_cone_certificate.json`
  uses the actual interval data of `spin_half_r_7_2_to_18_5_sub04` to build the
  model density
  `h*x^2+s*(ell-c*x)^2+tau*q^2`, proves an exact rational SOS identity, proves a
  strict `1/256` coercivity margin by a residual PSD Gram witness, and passes a
  paired finite-cone/Wick/master model package. The chart checker now records
  target-density kind and rejects an unbacked `kerr_gravitational` claim. This
  is not a proof of the Renormalized Carter cone lemma because the model
  density is not yet derived as the gravitational principal density.
- 2026-06-01: Strengthened the same chart-local theorem from coercivity margin
  `1/2048` to `1/256`. Added a human-checkable Young-inequality proof of the
  margin in the manuscript, updated the residual Gram certificate, and taught
  `verify_carter_finite_cone_package.py` to reject any finite package whose
  declared `assembly.global_delta` exceeds the minimum local coercivity delta.
  The model master certificate now records global lower density `1/256` while
  keeping the same assembly loss `5/4096`.
- 2026-06-02: Proved the robust coefficient tube theorem
  (`first-chart-coefficient-tube`) and added
  `verify_carter_coefficient_robustness.py`. Around the first spin-half
  chart-local model density, every quadratic perturbation with diagonal
  coefficient errors at most `1/2048` and bilinear coefficient errors at most
  `1/4096` remains coercive with margin `1/512`. This is still not a derivation
  of `q_grav`, but it means the future gravitational-density computation only
  has to land inside an explicit rational coefficient ball rather than match
  the model exactly.
- 2026-06-02: Proved the coefficient-enclosure transfer theorem
  (`first-chart-grav-enclosure-transfer`) and added
  `verify_carter_grav_coefficient_enclosure.py`. The checker validates a
  candidate six-coefficient interval enclosure against the certified tube and
  exports the inherited `1/512` local coercivity margin. It also rejects an
  attempted `kerr_gravitational_proved` status because no exact Kerr derivation
  backend is registered. This closes the transfer layer: once an exact
  derivation supplies coefficient intervals for the true normalized
  `q_grav` on `spin_half_r_7_2_to_18_5_sub04`, local gravitational Carter
  coercivity follows immediately from the proved tube theorem.
- 2026-06-02: Proved the rational coefficient-extraction transfer theorem
  (`first-chart-rational-extraction-transfer`) and added
  `verify_carter_rational_coefficient_extraction.py`. This verifies exact
  rational interval arithmetic for six coefficient expression trees, checks
  that denominator intervals avoid zero, and proves that the evaluated
  intervals are contained in the accepted coefficient enclosure. The included
  candidate is not a `q_grav` derivation, but it closes the arithmetic backend
  needed after a symbolic derivation supplies the six coefficient formulas.
- 2026-06-03: Proved the symbolic normal-form transfer theorem
  (`first-chart-symbolic-normal-form-transfer`) and added
  `verify_carter_symbolic_kerr_identification.py`. The checker expands a
  symbolic quadratic density over
  `Q[x,ell,q,z_1,...,z_N]`, proves it is exactly the six-coefficient normal form
  used by the rational extraction certificate, runs the extraction/enclosure
  chain, and inherits the `1/512` local coercivity margin. It still rejects
  `kerr_gravitational_proved` because no exact Kerr derivation backend is
  registered, but the remaining local input is now just the source derivation of
  the symbolic density from the gravitational principal system.
- 2026-06-03: Proved the source-ideal derivation transfer theorem
  (`first-chart-source-ideal-transfer`) and added
  `verify_carter_source_ideal_derivation.py`. This is a stronger local
  reduction: the source expression only has to equal the certified symbolic
  normal form modulo explicitly listed phase-free trapped/gauge constraints.
  The checker verifies exact ideal membership, runs the symbolic/extraction
  chain, and inherits the `1/512` local coercivity margin on the constraint
  quotient. It remains a candidate until a real Kerr principal-symbol derivation
  supplies the source density and constraints.
- 2026-06-03: Proved the Kerr source-backend closure theorem
  (`first-chart-kerr-source-backend`) and added
  `verify_carter_kerr_source_backend.py`. The new checker verifies the exact
  interface contract between a future linearized-Kerr principal-symbol
  derivation and the source-ideal certificate: backend-exported density and
  trapped constraints must match the source-ideal inputs exactly, after which
  the already verified chain gives the `1/512` local quotient margin. The
  included contract candidate is not a `q_grav` proof; it makes the final
  first-chart derivation obligation machine-detectable.
- 2026-06-03: Proved the traceable principal-symbol closure theorem
  (`first-chart-principal-symbol-trace`) and added
  `verify_carter_principal_symbol_trace.py`. This introduces a new finite trace
  layer: the candidate source is decomposed into primitive packets
  (radial-Hessian, superradiant-square, Carter-scale, interval-slack, and
  trapped-ideal packets), each checked for the right phase behavior, and the
  recombined trace is matched exactly to the backend interface. This does not
  derive `q_grav`, but it turns the remaining derivation into packet-level
  exact principal-symbol formulas plus a machine-checked recombination trace.
- 2026-06-10: Proved the first-chart exact-derivation audit theorem
  (`first-chart-exact-derivation-audit`) and added
  `verify_carter_exact_derivation_audit.py`. The new checker composes the
  principal-symbol trace certificate with the backend obligation list and
  requires every primitive packet and every remaining exact-derivation
  obligation to have a source target, verification interface, and replacement
  criterion. The included audit certificate is only a `candidate_audit`; it
  does not derive `q_grav`, but it turns the next source-backend step into a
  finite registry of exact Kerr formulas that must replace the placeholders.
- 2026-06-10: Proved the exact first-chart trapped-constraint theorem
  (`first-chart-trapped-constraints`) and added
  `verify_carter_trapped_constraint_derivation.py`. The checker proves the
  Kerr radial potential identities for spherical null geodesics:
  `R=0`, `dR=0`, the normal forms for `P` and `B`, and the formal derivative
  identity after clearing denominators. It also verifies that the first-chart
  backend exports exactly `rho_R` and `rho_dR` for those two residuals. The
  corresponding audit primitive entries and backend obligation are upgraded to
  `exact_formula`. This still does not derive `q_grav`; the radial-Hessian,
  superradiant-square, Carter-scale, slack, and quotient-density entries remain
  open.
- 2026-06-03: Proved the positive/ideal trace-defect absorber theorem
  (`first-chart-trace-defect-absorber`) and added
  `verify_carter_trace_defect_absorber.py`. This is a new robustness layer for
  the first-chart source computation: a future Kerr density may equal the
  accepted trace plus nonnegative rational square defects and additional
  phase-free gauge/trapped ideal terms, with no loss of the `1/512` margin on
  the joint quotient. The included absorber is a candidate, not a `q_grav`
  proof, but it makes the final principal-symbol target significantly more
  flexible while remaining exactly machine-checkable.
- 2026-06-03: Proved the defect-transport propagation theorem
  (`first-chart-defect-transport`) and added
  `verify_carter_defect_transport_morphism.py`. This gives a finite
  local-to-local propagation gate: an accepted absorber on one chart can seed a
  target chart through a rational phase pullback, nonnegative/ideal transport
  defect identity, and phase-norm lower certificate. The included shadow
  transport has norm factor `1` and preserves the `1/512` margin; it is not a
  new Kerr atlas chart, but it makes the overlap-extension task exactly
  checkable.
- 2026-06-04: Proved the finite defect-transport graph theorem
  (`finite-defect-transport-graph`) and added
  `verify_carter_defect_transport_graph.py`. This upgrades single-edge
  transport into an ordered atlas propagation certificate: every edge must use
  an already certified source chart, verify the exact transport identity and
  norm lower certificate, and export the target margin as the source margin
  times the edge factor. The included shadow graph reaches two transported
  charts with product margin `1/512`; it is not a real Kerr atlas graph, but it
  turns future overlap propagation into a finite rational graph check.
- 2026-06-05: Proved the orbit-cohomology breakthrough theorem
  (`orbit-cohomology-breakthrough`) and added
  `verify_carter_orbit_cohomology_certificate.py`. This attacks the analytic
  trapped-superradiant blocker directly: nonresonant torus Fourier modes are
  removed by exact rational cohomological equations, resonant modes are reduced
  to finite rational average lower bounds, and approximation/partition/normal
  extension/quantization losses are checked against a positive margin. The
  included shadow torus certificate has minimum net margin `91/512`; it is not
  a Kerr Fourier derivation, but it turns the old orbit-average/globalization
  assumptions into a finite certificate target.
- 2026-06-05: Proved the tame orbit-cohomology theorem
  (`tame-orbit-cohomology`) and added
  `verify_carter_tame_orbit_cohomology.py`. This closes the main analytic
  loophole in Fourier division: nonresonant denominators must stay above a
  rational floor, the weighted escape-symbol derivative seminorm is checked
  explicitly, and the derivative loss is subtracted from the orbit-cohomology
  margin. The shadow tame certificate verifies denominator floor `1`, maximum
  escape seminorm `101/180`, and residual tame margin `3893/23040`.

## Failed Or Dangerous Moves
- Claiming full Kerr stability from the Carter atlas certificates fails: the
  Renormalized Carter cone coefficient construction, Kerr pseudodifferential
  margin certificate, nonlinear reconstruction, and gauge compatibility remain
  open.
- Treating the seed-level overlap policy as a complete angular-torus atlas
  fails: it covers the certified seed neighborhoods, not every angular chart
  needed for the full torus fibration.

## Next Action
Replace the remaining open entries in
`spin_half_clean_chart_exact_derivation_audit_candidate.json` by exact formulas
one family at a time. The next decisive target is the radial-Hessian packet:
derive the radial Hessian contribution from the gauge-reduced linearized Kerr
principal symbol in the normalized first-chart variables, prove its exact
symbolic packet identity, and then upgrade `radial_hessian_packet` from
`candidate_placeholder` to `exact_formula`. After that, derive the
superradiant-square and Carter-scale packets, followed by the rational slack and
quotient-density entries. In parallel, replace the shadow graph by genuine
neighboring chart transport morphisms using actual overlap pullbacks. The finite
graph checker will then propagate the seed margin as a product of rational norm
factors across the reached atlas charts. For the analytic trapped-superradiant
branch, derive the actual Kerr torus Fourier coefficients for `a_KY` and
`q_sr` on the certified Carter atlas and feed them into the orbit-cohomology
certificate; the nonresonant modes should be solved by Fourier division, while
the remaining resonant modes must satisfy strict rational average lower bounds.
Then attach a tame certificate proving denominator floors and derivative
seminorm bounds for the resulting escape symbol in the same Carter-Wick symbol
class.
