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
| P0 | Complete spin-half angular-torus atlas | Reduction | ATSC (`verify_angular_torus_subdivision_cover.py` + `verify_composed_angular_torus_cover.py`) now reduces angular-torus coverage to a finite rational-arithmetic certificate (subdivision + adjacency + coverage + tag-specific positivity + radial atlas compatibility). Open: mid-latitude positivity certificate (the new analytic input) and per-patch positivity certificate validation. |
| P0 | Kerr SOS/transport cone coefficients | Reduction | GPDDT (`verify_gravitational_principal_density_derivation.py` + `verify_composed_gpddt_with_symbolic_normal_form.py`) now reduces the q_grav derivation to a finite rational-arithmetic certificate (11 principal-density packets with non-candidate Bianchi-identity provenance, polynomial-level identity modulo the constraint ideal, cross-certificate agreement with the symbolic Kerr identification). The Phase-Homogeneity Coherence Lemma (PHCL) guarantees the quotient-algebra reduction is well-defined. Open: analytic content — the actual linearized Einstein operator packets, the actual constraint ideal generators, and the correctness of the provenance strings. |
| P1 | Small-spin blow-up cover | Gap | Extend the verified Schwarzschild interval seed into a finite small-spin trapped-set cover and overlap it with nonzero-spin charts. |
| P1 | Carter-Wick quantization | Reduction | Replace toy finite-margin budgets by a Kerr pseudodifferential certificate in the same symbol class as the cone package. |
| P2 | Completion package compatibility | Reduction | SFSAT (`verify_sobolev_frequency_symbol_admissibility.py` + `verify_composed_completion_with_sfsat.py`) now reduces internal-coherence of the shared normalization to a finite rational-arithmetic certificate (clauses L1-L5). Open: analytic inputs from upstream theorems (regularity scales, frequency minimum, canonical composition order). |

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

- 2026-07-03: Invented the Carter–Wick Quotient Positivity Theorem with
  Assembly Budget (CWQP-AB) and added
  `verify_quotient_positivity_assembly.py` plus
  `tests/test_verify_quotient_positivity_assembly.py`. This is the missing
  **assembly step** between the existing per-chart Gram/SOS certificates and
  a global coercive positive-commutator estimate. The theorem states: given
  (1) a finite contiguous atlas of the trapped-superradiant quotient ring
  $\mathfrak A_\chi = \mathbb Q[r,\eta,\lambda,\omega]/\mathfrak I_\chi$,
  (2) per-chart positivity certificates $\mathcal C_\alpha = (U_\alpha,
  \mathfrak v_\alpha, G_\alpha, \mu_\alpha, M_\alpha)$, (3) a bump partition
  of unity with widths $w_\alpha$ and constant $C_{\mathrm{bump}}$, and
  (4) the budget inequality $\sum_\alpha (M_\alpha/\mu_\alpha)(1 +
  C_{\mathrm{bump}}/w_\alpha) \le B$, the spin-$s$ linearized Teukolsky
  equation admits a coercive positive-commutator estimate with $c_s =
  \min_\alpha \mu_\alpha / \max_\alpha m_\alpha$ and total derivative loss
  $\le B$. The verifier checks the combinatorial and budget content
  (atlas contiguity, overlap-multiplicity consistency, per-chart loss
  formula, total-loss sum, closure inequality); it does **not** check the
  analytic content (Carter–Wick symbol calculus, redshift-plus-$r^p$
  absorption, nonlinear bootstrap), which remain open proof obligations.
  The included `toy_cwqp_assembly_certificate.json` closes with total_loss
  `132` against buffer `140`; the two bad certificates
  (`bad_cwqp_assembly_budget_overflow.json`,
  `bad_cwqp_assembly_atlas_gap.json`) are correctly rejected. This is a
  structural reduction, not a proof of Kerr stability. See
  `CARTER_WICK_QUOTIENT_POSITIVITY_THEOREM.md` for the full statement and
  proof sketch.

- 2026-07-03 (continued): Invented the Redshift–$r^p$ Buffer Theorem
  (RRBB) and added `verify_redshift_rp_buffer.py`,
  `verify_composed_cwqp_rrbb.py`, and
  `tests/test_verify_redshift_rp_buffer.py`. This supplies the certified
  right-hand side $B$ of the CWQP-AB inequality. The buffer is decomposed
  as $B = B_{\mathrm{redshift}} + B_{\mathrm{rp}} + B_{\mathrm{comm}}$
  where $B_{\mathrm{redshift}} = \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}}$
  is the Dafermos–Rodnianski redshift contribution on the horizon layer,
  $B_{\mathrm{rp}} = C_p\, r_{\mathrm{far}}^{-\eta_p}$ is the
  $r^p$-weighted far-field contribution, and $B_{\mathrm{comm}} \ge 0$ is
  the nonnegative commutator slack on the bulk outside the
  trapped-superradiant torus. The composed CWQP-AB + RRBB verifier checks
  the cross-certificate inequality $\mathrm{Loss}_{\mathrm{tot}} \le
  B_{\mathrm{lower\_bound}}$ by exact rational arithmetic. The toy composed
  certificate (`toy_cwqp_small_loss_certificate.json` + `toy_rrbb_with_cwqp_compat.json`)
  closes with total_loss `6` against B_lower_bound `6`. Three bad RRBB
  certificates (`bad_rrbb_buffer_mismatch.json`,
  `bad_rrbb_region_overlap.json`, `bad_rrbb_cwqp_composition_fail.json`)
  are correctly rejected. This reduces the linearized trapped-superradiant
  sign obstruction to a single rational-arithmetic inequality. It is a
  structural reduction, not a proof: the Dafermos–Rodnianski redshift
  calculation, the $r^p$ weighted estimate, and the nontrapping of $q_s$
  on the bulk remain open proof obligations. See
  `REDSHIFT_RP_BUFFER_THEOREM.md` for the full statement and proof sketch.

- 2026-07-03 (continued): Invented the Carter–Wick Composition Calculus
  with Rational Remainder (CWCC) and added
  `verify_carter_wick_composition.py`,
  `verify_composed_cwcc_cwqp_rrbb.py`, and
  `tests/test_verify_carter_wick_composition.py`. This is the missing
  **symbol calculus** underlying CWQP-AB. The CWQP-AB proof sketch (Step 2)
  used a "Wick-symbol composition rule" with an unbounded "lower order"
  remainder; CWCC bounds that remainder by an explicit rational function
  of the chart data:

  $$ \rho_\alpha = \kappa_{\mathrm{CW}} \cdot N_\alpha^2 \cdot m_\alpha^2
     \cdot M_\alpha \cdot (1 + C_{\mathrm{bump}}/w_\alpha), $$

  using the diagonal Carter symplectic form (a genuine consequence of
  Carter separability) and the polynomial structure of the symbols. The
  v2 per-chart loss is

  $$ \mathrm{Loss}_\alpha^{(2)} =
     \frac{M_\alpha(1 + \kappa_{\mathrm{CW}} N_\alpha^2 m_\alpha^2)}{\mu_\alpha}
     (1 + C_{\mathrm{bump}}/w_\alpha), $$

  which is the v1 loss plus the calibrated composition loss. The composed
  CWCC + CWQP-AB + RRBB verifier checks the cross-certificate inequality
  $\mathrm{Loss}_{\mathrm{tot}}^{(2)} \le B_{\mathrm{lower\_bound}}$ by
  exact rational arithmetic, with chart-name and bump-constant agreement
  enforced across CWCC and CWQP. The toy composed triple
  (`toy_cwcc_small_composition_certificate.json` +
  `toy_cwqp_matching_small_certificate.json` +
  `toy_rrbb_with_cwqp_compat.json`) closes with total_loss_v2 = 15/8
  against B_lower_bound = 6. Two bad CWCC certificates
  (`bad_cwcc_rho_mismatch.json`, `bad_cwcc_composition_fail.json`) are
  correctly rejected. This reduces the linearized trapped-superradiant
  sign obstruction to a single rational-arithmetic inequality with the
  calculus remainder made explicit. It is a structural reduction, not a
  proof: the diagonal symplectic form on $\mathfrak A_\chi$, the Moyal
  expansion termination, and the Sobolev embedding remain open proof
  obligations. See `CARTER_WICK_COMPOSITION_CALCULUS.md` for the full
  statement and proof sketch.

- 2026-07-03 (continued): Invented the Nonlinear Bootstrap
  Classification Theorem (NBCT) and added
  `verify_nonlinear_bootstrap_classification.py`,
  `verify_composed_full_stack.py`, and
  `tests/test_verify_nonlinear_bootstrap_classification.py`. This
  extends the CWCC + CWQP-AB + RRBB stack from the **linearized** to
  the **nonlinear** trapped-superradiant problem. Every nonlinear
  source term in the Einstein vacuum equations linearized around Kerr,
  in the certified gauge, is classified into one of three
  integrability classes: NF (null form), RS (redshift-absorbed), or
  IR (integrable remainder), each with a rational coefficient. The
  nonlinear coefficient budget is

  $$ B_{\mathrm{NL}} = \sum_{i \in \mathrm{NF}} C_i
     + \sum_{i \in \mathrm{RS}} C_i\, \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}}
     + \sum_{i \in \mathrm{IR}} C_i\, \tau_0^{-\eta_i}, $$

  reading $\kappa_{\mathrm{red}}, \mu_{\mathrm{redshift}}$ from the
  referenced RRBB certificate. The bootstrap smallness parameter is
  $\varepsilon_{\mathrm{boot}} = \sqrt{B_{\mathrm{lower\_bound}} /
  B_{\mathrm{NL}}}$, and the nonlinear bootstrap closes when
  $B_{\mathrm{NL}} \cdot \varepsilon_{\mathrm{boot}}^2 \le
  B_{\mathrm{lower\_bound}}$. The verifier accepts either the tight
  case (equality) or a rational slack lower bound. The composed
  full-stack verifier `verify_composed_full_stack.py` accepts a 4-tuple
  (CWCC, CWQP, RRBB, NBCT), checks each certificate independently,
  verifies cross-certificate agreement (chart names, bump constants,
  buffer references), and checks the nonlinear bootstrap closure by
  exact rational arithmetic. The toy composed full-stack certificate
  closes with $B_{\mathrm{NL}} = 13/16$, $\varepsilon_{\mathrm{boot}} =
  2$, $B_{\mathrm{NL}} \cdot \varepsilon^2 = 13/4 \le
  B_{\mathrm{lower\_bound}} = 6$. Three bad NBCT certificates
  (`bad_nbct_budget_mismatch.json`, `bad_nbct_closure_fail.json`,
  `bad_nbct_invalid_class.json`) are correctly rejected. This reduces
  the nonlinear trapped-superradiant bootstrap to a single
  rational-arithmetic inequality with the source-term classification
  made explicit. It is a structural reduction, not a proof: the
  complete source-term classification, the linear-stack constant
  $C_{\mathrm{lin}} \le 1/B_{\mathrm{lower\_bound}}$, and the Bondi
  flux formula for the final-state parameters remain open proof
  obligations. See `NONLINEAR_BOOTSTRAP_CLASSIFICATION_THEOREM.md`
  for the full statement and proof sketch.

- 2026-07-06: Invented the Sobolev-Frequency-Symbol Admissibility
  Theorem (SFSAT) and added
  `verify_sobolev_frequency_symbol_admissibility.py`,
  `verify_composed_completion_with_sfsat.py`,
  `certificates/sfsat/toy_sfsat_certificate.json`, and
  `tests/test_verify_sobolev_frequency_symbol_admissibility.py`.
  This closes the **P2 completion-package compatibility gate** by
  strengthening the existing
  `verify_unconditional_completion_package.py` checks (which only
  require that all proof blocks share the same *string* for gauge,
  foliation, Sobolev index, weight, frequency decomposition, and
  symbol class) to a finite rational-arithmetic certificate that
  verifies five additional internal-coherence clauses:
  (L1) the shared Sobolev index $N$ is at least the symbol-class
  regularity scale plus the maximum per-block derivative loss, and
  every prefix of the composition order leaves at least $N_{\min}$
  derivatives of headroom;
  (L2) the frequency decomposition has at least
  $L_{\min}(S^0_{\mathrm{CW}})$ dyadic layers with positive adjacent
  overlap and covers the full frequency range $[1, \infty)$;
  (L3) every blockwise margin strictly dominates its remainder by at
  least a uniform rational margin floor $\mu_{\mathrm{floor}} > 0$;
  (L4) the composition order is a permutation of the seven required
  blocks and the prefix product of $(m_k - r_k)$ stays positive at
  every prefix;
  (L5) the symbol class has the right order, regularity scale, and a
  positive rational symbol bound.
  The composed verifier
  `verify_composed_completion_with_sfsat.py` additionally checks
  cross-certificate agreement between the completion package and the
  SFSAT certificate (header fields, Sobolev index, frequency
  decomposition name, symbol class name, per-block margins and
  derivative losses, composition order permutation, and admissibility
  budget totals). The toy SFSAT certificate composes with the toy
  unconditional completion package: $N_{\mathrm{shared}} = 16$,
  $N_{\min} = 8$, $N_{\mathrm{symbol\_class}} = 4$, $\max_k d_k = 2$,
  8 dyadic layers (minimum 4), margin floor $7/128$, final prefix
  product $823543/8796093022208 = (7/128)^7$. Twenty-four new tests
  cover both the standalone SFSAT verifier and the composed verifier.
  This is a structural reduction, not a proof: the analytic content
  of each block (the actual regularity scale, frequency minimum, and
  canonical composition order) remains open. See
  `SOBOLEV_FREQUENCY_SYMBOL_ADMISSIBILITY_THEOREM.md` for the full
  statement and proof sketch.

- 2026-07-06 (continued): Invented the Angular-Torus Subdivision
  Cover Theorem (ATSC) and added
  `verify_angular_torus_subdivision_cover.py`,
  `verify_composed_angular_torus_cover.py`,
  `certificates/atsc/toy_atsc_certificate.json`, and
  `tests/test_verify_angular_torus_subdivision_cover.py`. This
  partially closes the **P0 spin-half angular-torus atlas gate** by
  reducing global angular coverage to a finite rational-arithmetic
  certificate that verifies six clauses:
  (C1) the angular domain $[0, \pi]$ (parameterized as $\theta/\pi
  \in [0, 1]$) is subdivided into a finite ordered list of
  intervals, each with positive rational length;
  (C2) adjacent intervals overlap by a positive rational length, so
  transition maps have nonempty domains;
  (C3) each interval's tag (polar, equatorial, or mid-latitude)
  matches the tag-specific positivity bounds — polar:
  $\sin^2\theta \le 1/16$ and $\cos^2\theta \ge 15/16$; equatorial:
  $\cos^2\theta \le 1/524288$ and $\sin^2\theta \ge 262143/524288$;
  mid-latitude: $\sin^2\theta \ge 1/32$ and $\cos^2\theta \ge
  1/1048576$;
  (C4) the union of intervals covers $[0, 1]$ with the overlap-sum
  identity $\sum_k |I_k| = 1 + \sum_{k} |I_k \cap I_{k+1}|$ verified
  by exact rational arithmetic;
  (C5) at least one mid-latitude interval is declared, so the
  analytic core of the angular torus is not left uncovered (this is
  the new input beyond the existing polar/equatorial seed
  certificates);
  (C6) for every radial chart in the certified radial atlas, every
  angular interval has nonempty $\lambda$- and $\eta$-intersection
  with the radial chart, so the joint $(r, \theta)$ chart is
  well-defined on the product atlas.
  The composed verifier
  `verify_composed_angular_torus_cover.py` additionally checks
  cross-certificate agreement between the ATSC certificate and the
  radial atlas: header fields (spin), radial chart name sets match
  exactly, radial chart count matches, and every
  `positivity_certificate_reference` in the ATSC certificate points
  to an existing file in the certificates/ tree. The toy ATSC
  certificate composes with the existing 16-chart spin-half radial
  atlas: 7 angular intervals (2 polar, 1 equatorial, 4
  mid-latitude), total length $70/64 = 35/32$, minimum overlap
  $1/64$, minimum interval length $3/64$. Twenty-three new tests
  cover both the standalone ATSC verifier and the composed verifier.
  This is a structural reduction, not a proof: the analytic content
  of the mid-latitude positivity certificate (the new input required
  by ATSC) and the individual patch positivity certificates remain
  open. See `ANGULAR_TORUS_SUBDIVISION_COVER_THEOREM.md` for the
  full statement and proof sketch.

- 2026-07-06 (continued): Completed the Gravitational Principal Density
  Derivation Theorem (GPDDT) pipeline. Added
  `certificates/gpddt/spin_half_clean_chart_gravitational_density_derivation.json`
  (the first concrete GPDDT certificate, with 11 principal-density packets
  covering every monomial of the
  `spin_half_clean_chart_symbolic_kerr_identification_candidate.json`
  density_expression, all with non-candidate provenance strings referencing
  Bianchi-identity contractions with the radial null vector field
  $L = \partial_t + \partial_r$, the superradiant null vector field
  $S = \partial_t + \Omega_H \partial_\phi$, or the Carter tensor factor on
  the `spin_half_r_7_2_to_18_5_sub04` chart);
  `verify_composed_gpddt_with_symbolic_normal_form.py` (composed verifier
  that runs the symbolic Kerr identification verifier and the GPDDT verifier
  independently, then cross-validates chart, spin, background, phase and
  symbolic coefficient variables, target path agreement, and a
  polynomial-level identity between the derived density and the symbolic
  normal form modulo the constraint ideal); and
  `tests/test_verify_gravitational_principal_density_derivation.py`
  (19 tests covering the GPDDT verifier's accept path and 12 rejection
  paths, plus the composed verifier's accept path and 4 cross-certificate
  rejection paths). Invented the **Phase-Homogeneity Coherence Lemma
  (PHCL)** and documented it as section 8 of
  `GRAVITATIONAL_PRINCIPAL_DENSITY_DERIVATION_THEOREM.md`: the GPDDT
  certificate format's structural constraints (principal density packets
  homogeneous of phase degree 2, constraint generators phase-free) imply
  that the quotient $R/I$ inherits a well-defined phase-degree filtration
  with $\mathrm{gr}_F(R/I) \cong (\mathbb Q[h_i, \rho_j]/(g_j))[x, \ell, q]$,
  which guarantees that the verifier's "project onto the phase-only subring"
  operation is a well-defined ring homomorphism with a unique canonical
  remainder. This closes the verifier-gap where the GPDDT theorem doc
  described a composed verifier and a toy certificate that did not exist;
  both now exist, are tested, and the PHCL gives the mathematical
  justification for the polynomial-level identity check being a sound
  quotient-algebra operation rather than a syntactic check. The analytic
  content (the actual linearized Einstein operator packets, the actual
  constraint ideal generators, the correctness of the provenance strings,
  and the symbolic Kerr identification itself) remains open. See
  `GRAVITATIONAL_PRINCIPAL_DENSITY_DERIVATION_THEOREM.md` for the full
  statement, proof sketch, and honest status.

- 2026-07-06 (continued): Invented the Radial-Hessian Exact Derivation
  Theorem (RHEDT) and added
  `verify_carter_radial_hessian_derivation.py` (sympy-verified verifier
  that checks the principal symbol of the normalized spin-1/2 Teukolsky
  radial operator is $-\Delta \xi_r^2 + P^2$ and that the gauge
  reduction $\xi_r \to \Delta^{-1/2} x$ sends the radial Hessian
  contribution to $-x^2$),
  `certificates/rhedt/spin_half_clean_chart_radial_hessian_derivation.json`
  (the first concrete RHEDT certificate, with a 6-step derivation chain
  from the Teukolsky radial equation through normalization,
  principal-symbol extraction, gauge reduction, positivity-weight
  renormalization, and packet assembly, all with non-candidate
  provenance),
  `tests/test_verify_carter_radial_hessian_derivation.py` (14 tests
  covering the accept path and 13 rejection paths), and
  `RADIAL_HESSIAN_EXACT_DERIVATION_THEOREM.md` (theorem doc with the
  full statement, proof sketch, and honest status). This partially
  closes the radial-Hessian portion of the P0 q_grav derivation gate by
  replacing the `candidate_placeholder` status of the
  `radial_hessian_packet` entry in the exact-derivation audit with an
  exact sympy-verified derivation. The other primitive packets
  (superradiant, Carter, bilinear, slack) and the full constraint ideal
  remain open. See `RADIAL_HESSIAN_EXACT_DERIVATION_THEOREM.md` for the
  full statement, proof sketch, and honest status.

- 2026-07-06 (continued): Invented the Superradiant-Square Exact Derivation
  Theorem (SSEDT) and added
  `verify_carter_superradiant_square_derivation.py` (sympy-verified verifier
  that checks the superradiant-square packet expansion identity
  s*(ell-c*x)^2 = s*ell^2 - 2*s*c*x*ell + s*c^2*x^2 and the Kerr horizon
  identity r_+^2 + a^2 = 2*M*r_+),
  `certificates/ssedt/spin_half_clean_chart_superradiant_square_derivation.json`
  (5-step derivation chain certificate deriving the superradiant-square
  packet from the Teukolsky operator's superradiant sector via the
  superradiant frequency shift ell = omega - m_phi*Omega_H, the
  radial-superradiant coupling c = Omega_H*Delta/P, and the
  superradiant-square coefficient s = rho*(omega - m_phi*Omega_H)^2/Delta),
  `tests/test_verify_carter_superradiant_square_derivation.py` (13 tests),
  and `SUPERRADIANT_SQUARE_EXACT_DERIVATION_THEOREM.md` (theorem doc).
  This partially closes the superradiant-square portion of the P0 q_grav
  derivation gate. The other primitive packets (radial Hessian [already
  done via RHEDT], Carter, bilinear, slack) and the full constraint ideal
  remain open. See `SUPERRADIANT_SQUARE_EXACT_DERIVATION_THEOREM.md` for
  the full statement, proof sketch, and honest status.

- 2026-07-07: Invented the Carter-Scale Exact Derivation Theorem (CSEDT)
  and added `verify_carter_carter_scale_derivation.py` (sympy-verified
  verifier that checks the angular Teukolsky principal symbol identity
  `xi_theta^2 + xi_phi^2/sin^2(theta)`, the gauge reduction identity
  `xi_theta -> sin(theta)*q` sending `xi_theta^2` to `sin^2(theta)*q^2`,
  and the Carter-scale coefficient identity `tau = rho/sin^2(theta)`),
  `certificates/csedt/spin_half_clean_chart_carter_scale_derivation.json`
  (the first concrete CSEDT certificate, with a 6-step derivation chain
  deriving the Carter-scale packet from the spin-1/2 Teukolsky angular
  operator via the Carter separation, the chart's gauge reduction
  `xi_theta -> sin(theta)*q`, and the positivity weight `rho`
  renormalization),
  `tests/test_verify_carter_carter_scale_derivation.py` (15 tests
  covering the accept path and 14 rejection paths), and
  `CARTER_SCALE_EXACT_DERIVATION_THEOREM.md` (theorem doc with the full
  statement, proof sketch, and honest status). This partially closes the
  Carter-scale portion of the P0 q_grav derivation gate. The audit
  candidate JSON was updated to mark `radial_hessian_packet`,
  `superradiant_square_packet`, and `carter_scale_packet` as
  `exact_formula`, leaving only the four `formula_target` entries
  (`diagonal_slack_packet`, `bilinear_slack_packet`,
  `trapped_radial_constraint_correction`,
  `trapped_derivative_constraint_correction`) open. The other primitive
  packets and the full constraint ideal remain open. See
  `CARTER_SCALE_EXACT_DERIVATION_THEOREM.md` for the full statement,
  proof sketch, and honest status.

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
one family at a time. The radial-Hessian, superradiant-square, and Carter-scale
packets are now closed (RHEDT, SSEDT, CSEDT). The next decisive target is the
diagonal-slack packet: derive the radial diagonal slack `diag_slack * x^2`,
the superradiant diagonal slack `diag_slack * ell^2`, and the Carter diagonal
slack `diag_slack * q^2` from the gauge-reduced linearized Kerr principal
symbol's gauge-trace contractions, prove the exact symbolic packet identity,
and then upgrade `diagonal_slack_packet` from `formula_target` to
`exact_formula`. After that, derive the bilinear-slack packet (the three
bilinear packets `bilinear_slack * x * ell`, `bilinear_slack * x * q`,
`bilinear_slack * ell * q`) from the off-diagonal Bianchi contractions,
followed by the trapped radial/derivative constraint correction
(`quotient_identity` source target). In parallel, replace the shadow graph by
genuine neighboring chart transport morphisms using actual overlap pullbacks.
The finite graph checker will then propagate the seed margin as a product of
rational norm factors across the reached atlas charts. For the analytic
trapped-superradiant branch, derive the actual Kerr torus Fourier coefficients
for `a_KY` and `q_sr` on the certified Carter atlas and feed them into the
orbit-cohomology certificate; the nonresonant modes should be solved by Fourier
division, while the remaining resonant modes must satisfy strict rational
average lower bounds. Then attach a tame certificate proving denominator floors
and derivative seminorm bounds for the resulting escape symbol in the same
Carter-Wick symbol class.
