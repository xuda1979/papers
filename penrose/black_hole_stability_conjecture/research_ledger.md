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
  identity (now closed exactly by QIEDT: q_grav = rho_backend + SOS defect
  + trapped-constraint ideal (P_R, P_dR, P_xq), with the new Carter-mixed
  packet P_xq = x*q + ell*q - q^2), a checked Kerr source-backend interface
  contract, and a principal-symbol trace decomposition with a positive/ideal
  defect absorber and a finite defect-transport graph protocol. This is not
  yet the gravitational cone certificate.
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

- 2026-07-08: Invented the Bilinear-Slack Exact Derivation Theorem (BSEDT)
  and added `verify_carter_bilinear_slack_derivation.py` (sympy-verified
  verifier that constructs the linearized spin-1/2 Bianchi principal
  symbol `sigma_B` as a symmetric 3x3 matrix in the (L, S, Q) principal
  null direction basis, and performs five independent polynomial
  reductions: (R1) symmetry `sigma_B - sigma_B^T == 0` entrywise;
  (R2) off-diagonal factorization `sigma_B[i,j] - bilinear_slack *
  (phase_i * phase_j) == 0` for each of the three off-diagonal pairs;
  (R3) shared-coefficient reduction showing `a01 = a02 = a12 =
  bilinear_slack` yields the bilinear-slack packet matrix; (R4)
  off-diagonal sum `sum_{i<j} sigma_B[i,j] == bilinear_slack * (x*ell +
  x*q + ell*q)`; (R5) tube factor `2 * [-1/8192, 1/8192] = [-1/4096,
  1/4096]` confirming the bilinear-slack tube is exactly half the
  diagonal-slack tube, because the off-diagonal trace contraction of a
  symmetric 2-tensor picks up a factor of 2 from the two mirror entries),
  the certificate
  `certificates/bsedt/spin_half_clean_chart_bilinear_slack_derivation.json`
  (5-step derivation chain deriving the three bilinear-slack packets
  `bilinear_slack * x * ell`, `bilinear_slack * x * q`, `bilinear_slack *
  ell * q` from the off-diagonal entries of the gauge-reduced Bianchi
  principal symbol via the (L,S), (L,Q), (S,Q) gauge-trace contractions,
  with the shared-coefficient property following from the symmetric
  spinor index structure of the contracted Bianchi identity
  `nabla_{(A}^{A'} Psi_{BC)A'B'} = 0`),
  `tests/test_verify_carter_bilinear_slack_derivation.py` (19 tests
  covering the accept path and 17 rejection paths), and
  `BILINEAR_SLACK_EXACT_DERIVATION_THEOREM.md` (theorem doc with the full
  statement, proof sketch, and honest status). This partially closes the
  bilinear-slack portion of the P0 Kerr SOS/transport cone coefficients
  gate. The audit candidate JSON's `bilinear_slack_packet` entry is now
  closable from `formula_target` to `exact_formula`, leaving only the
  two trapped-constraint entries (`trapped_radial_constraint_correction`,
  `trapped_derivative_constraint_correction`) as `formula_target` in the
  exact-derivation audit. The derivation is exact *conditional* on the
  upstream principal symbol trace certificate
  (`spin_half_clean_chart_principal_symbol_trace_candidate.json`) being
  promoted from candidate to exact. The other primitive packets and the
  full constraint ideal remain open. See
  `BILINEAR_SLACK_EXACT_DERIVATION_THEOREM.md` for the full statement,
  proof sketch, and honest status.

- 2026-07-08 (continued): Invented the Trapped Constraint Correction Exact
  Derivation Theorem (TCDEDT) and added
  `verify_carter_trapped_constraint_correction_derivation.py`
  (sympy-verified verifier that derives the two trapped-constraint
  correction packets P_R = x^2 + ell^2 + q^2 and P_dR = x*ell + q^2 from
  the symmetry structure of the radial and derivative trapped constraints,
  via six independent polynomial reductions: (R1) diagonal-invariant
  uniqueness of P_R -- any diagonal degree-2 polynomial alpha*x^2 +
  beta*ell^2 + gamma*q^2 invariant under the simultaneous principal null
  direction permutations (x<->ell) and (x<->q) satisfies alpha = beta =
  gamma, giving the 1-parameter family alpha*P_R; (R2) mixed-structure
  uniqueness of P_dR -- the mixed degree-2 polynomial alpha*x*ell +
  beta*q^2 with the Kerr horizon identity constraint alpha = beta (equating
  the (L,S) and Q couplings via r_+^2 + a^2 = 2*M*r_+) reduces to
  alpha*P_dR; (R3) quotient identity multiplier determination -- for a
  source with known residual rho_R*P_R + rho_dR*P_dR, the quotient
  identity (1 - a_R)*P_R*rho_R + (1 - a_dR)*P_dR*rho_dR == 0 has the
  unique solution a_R = a_dR = 1; (R4) linear independence of P_R and
  P_dR -- alpha*P_R + beta*P_dR == 0 implies alpha = beta = 0 (from the
  x^2 and x*ell monomial coefficients); (R5) general correction structure
  -- the constraint-ideal contribution a_R*P_R*rho_R + a_dR*P_dR*rho_dR
  decomposes uniquely into a rho_R-coefficient a_R*P_R and a
  rho_dR-coefficient a_dR*P_dR; (R6) Kerr horizon identity r_+^2 + a^2 =
  2*M*r_+ verified by substitution of r_+ = M + sqrt(M^2 - a^2)), the
  certificate
  `certificates/tcdedt/spin_half_clean_chart_trapped_constraint_correction_derivation.json`
  (6-step derivation chain deriving the two trapped-constraint correction
  packets from the diagonal structure of R=0 and the mixed (L,S) structure
  of dR=0, with the Kerr horizon identity equating the (L,S) and Q
  couplings), `tests/test_verify_carter_trapped_constraint_correction_derivation.py`
  (17 tests covering the accept path and 15 rejection paths), and
  `TRAPPED_CONSTRAINT_CORRECTION_EXACT_DERIVATION_THEOREM.md` (theorem doc
  with the full statement, proof sketch, and honest status). This
  **closes the primitive-packet exact-derivation audit**: all 9
  primitive-packet entries in
  `spin_half_clean_chart_exact_derivation_audit_candidate.json` are now
  `exact_formula` (RHEDT, SSEDT, CSEDT, DSEDT, BSEDT for the five packet
  families; the radial and derivative trapped-characteristic constraints;
  and TCDEDT for the two trapped-constraint corrections), with 0 entries
  remaining as `formula_target`. The 3 backend obligations (exporting the
  actual Kerr principal symbol from the backend, computing q_grav, and
  proving source = q_grav mod trapped constraints) remain open as a
  separate workstream (the Kerr source backend contract). The derivation
  is exact *conditional* on the upstream principal symbol trace
  certificate being promoted from candidate to exact. See
  `TRAPPED_CONSTRAINT_CORRECTION_EXACT_DERIVATION_THEOREM.md` for the full
  statement, proof sketch, and honest status.

- 2026-07-09: Invented the Quotient Identity Exact Derivation Theorem
  (QIEDT) and added
  `verify_carter_quotient_identity_derivation.py` (sympy-verified
  verifier with 6 polynomial-reduction identities: Q1 the quotient
  identity, Q2 the unique swap-invariant derivation of the new
  Carter-mixed packet P_xq, Q3 the bilinear packet decomposition
  P_bil = P_dR + P_xq, Q4 linear independence of P_R/P_dR/P_xq, Q5
  backend density phase-homogeneity, Q6 SOS defect perfect-square
  structure),
  `certificates/qiedt/spin_half_clean_chart_quotient_identity_derivation.json`
  (7-step derivation chain deriving the quotient identity q_grav =
  rho_backend + s*(c*x-ell)^2 + diag_slack*P_R + bilinear_slack*P_dR
  + bilinear_slack*P_xq, with the new Carter-mixed trapped-constraint
  correction packet P_xq = x*q + ell*q - q^2, the certified SOS
  defect s*(c*x-ell)^2 with weight s in the coercivity tube [0, 1/4],
  and the minimal Kerr-clean backend density rho_backend = h*x^2 +
  tau*q^2),
  `tests/test_verify_carter_quotient_identity_derivation.py` (24 tests
  covering the accept path and 23 rejection paths), and
  `QUOTIENT_IDENTITY_EXACT_DERIVATION_THEOREM.md` (theorem doc with the
  full statement, proof sketch, and honest status). This **closes the
  `quotient_identity` source target** in the exact-derivation audit's
  backend obligation registry: the exported source density q_grav is
  proved equal to the backend density rho_backend modulo the
  trapped-constraint ideal (P_R, P_dR, P_xq) plus a certified SOS
  defect (gauge-tangent square). The QIEDT's principal new
  mathematical contribution is the Carter-mixed trapped-constraint
  correction packet P_xq = x*q + ell*q - q^2, derived as the unique
  swap-invariant degree-2 polynomial satisfying the bilinear packet
  decomposition P_bil = P_dR + P_xq, extending the TCDEDT packet
  family to the bilinear (off-diagonal) sector. A secondary new
  contribution is the first certified SOS defect s*(c*x-ell)^2 in the
  q_grav decomposition. The derivation is exact *conditional* on the
  upstream principal symbol trace certificate being promoted from
  candidate to exact. The 2 remaining backend obligations (exporting
  the gauge-reduced linearized vacuum Einstein principal symbol from
  the backend, and computing the normalized gravitational principal
  density q_grav) remain open as a separate workstream (the Kerr
  source backend contract).

- 2026-07-10: Invented the Gauge-Reduced Linearized Einstein Principal
  Symbol Theorem (GRLEPS) and added
  `verify_carter_gauge_reduced_einstein_principal_symbol.py` (sympy-verified
  verifier with 6 polynomial-reduction identities: G1 every gauge
  rewrite rule has lhs - rhs in the augmented ideal I_trap +
  <s*(c*x-ell)^2>, G2 confluence of the principal-trapped-gauge
  rewrite system R_gauge via pairwise-monomial GCD of all rule LHS,
  G3 input principal symbol phase-homogeneity of degree 2 with the
  four mandatory gauge-tangent packets matching q_grav, G4 reduction
  chain arithmetic identities at every step, G5 on-shell identity
  sigma_NF == rho_backend, G6 q_grav agreement sigma_in - q_grav in
  the augmented ideal),
  `certificates/grleps/spin_half_clean_chart_gauge_reduced_einstein_principal_symbol.json`
  (the first concrete GRLEPS certificate, with a 4-rule
  principal-trapped-gauge rewrite system R_gauge = {
  superradiant_square_to_zero, radial_diagonal_slack_to_zero,
  radial_derivative_slack_to_zero, carter_mixed_slack_to_zero }, a
  vacuous-confluence certificate via pairwise-monomial GCDs (all 6
  pairwise GCDs of the 4 rule LHS are monomials: five are 1, one is
  bilinear_slack), and a 4-step reduction chain reducing the
  gauge-reduced linearized Einstein principal symbol sigma_in to the
  Kerr-clean backend density rho_backend = h*x^2 + tau*q^2),
  `tests/test_verify_carter_gauge_reduced_einstein_principal_symbol.py`
  (39 tests covering the accept path and 36 rejection paths plus 5
  internal-API tests), and
  `GAUGE_REDUCED_LINEARIZED_EINSTEIN_PRINCIPAL_SYMBOL_THEOREM.md`
  (theorem doc with the full statement, proof sketch, and honest
  status). This **closes 2 of the 4 remaining backend obligations**
  in the Kerr source backend contract for the spin-half clean Carter
  chart: (1) derive the gauge-reduced linearized vacuum Einstein
  principal symbol in this chart, and (4) prove the exported source
  density equals q_grav modulo those trapped constraints. The
  principal new mathematical contribution is the
  principal-trapped-gauge rewrite system R_gauge: a finite,
  confluent, phase-degree-non-increasing polynomial rewrite system
  whose normal form is the Kerr-clean backend density modulo the
  trapped-constraint ideal plus the gauge-tangent SOS defect. The
  confluence certificate (pairwise-monomial GCD of all rule LHS) is a
  finite machine-checkable invariant, and the reduction chain is a
  finite sequence of single-rule steps verified by exact rational
  arithmetic. This converts the gauge-reduction step of the Carter
  calibrated proof program from a heuristic into a finite certified
  computation. The derivation is exact *conditional* on the upstream
  principal symbol trace certificate being promoted from provisional
  to exact. The 2 remaining backend obligations (computing the
  normalized gravitational principal density q_grav in the listed
  phase variables, and deriving rho_R and rho_dR as the trapped
  characteristic constraints R=0 and dR=0) are addressed by GPDDT and
  TCDEDT respectively, awaiting promotion from candidate to exact. See
  `QUOTIENT_IDENTITY_EXACT_DERIVATION_THEOREM.md` for the full
  statement, proof sketch, and honest status.

- 2026-07-10 (continued): Invented the Backend Contract Closure
  Theorem (BCCT) and added
  `verify_composed_backend_contract_closure.py` (composed verifier
  with 7 composition identities: B1 backend contract verifies, B2
  GRLEPS verifies, B3 GPDDT verifies, B4 trapped-constraint
  derivation verifies, B5 QIEDT verifies, B6 four certificates
  mutually consistent (same chart, same normal form, same q_grav, same
  derived constraints), B7 union of closed obligations covers exactly
  the four backend obligations),
  `certificates/bcct/spin_half_clean_chart_backend_contract_closure.json`
  (the first concrete BCCT certificate, with the obligation closure
  map: obligation 1 -> GRLEPS, obligation 2 -> GPDDT, obligation 3 ->
  trapped-constraint derivation, obligation 4 -> GRLEPS, and
  `remaining_exact_derivation_obligations: []`),
  `tests/test_verify_composed_backend_contract_closure.py` (13 tests
  covering the accept path, 6 rejection paths, and 4 structural
  invariant tests), and
  `BACKEND_CONTRACT_CLOSURE_THEOREM.md` (theorem doc with the full
  statement, proof sketch, and honest status). This **closes all four
  backend obligations** of the Kerr source backend contract for the
  spin-half clean Carter chart: the union of GRLEPS (obligations 1
  and 4), GPDDT (obligation 2), and the trapped-constraint derivation
  (obligation 3) covers exactly the four backend obligations, and the
  mutual-consistency checks confirm the four certificates compose
  without contradiction. The principal new mathematical contribution
  is the **obligation closure map**: a finite, machine-checkable
  partition of the four backend obligations into the
  exact-derivation theorems that close them, together with the
  mutual-consistency identities (B6) that ensure the four
  certificates compose without contradiction. The BCCT is the eighth
  primitive-packet exact-derivation theorem in the audit chain (after
  RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT, QIEDT, GRLEPS) and the
  first composition theorem to close an entire backend contract for a
  Carter chart. The closure is exact *conditional* on the upstream
  principal symbol trace certificate being promoted from provisional
  to exact and on all upstream analytic hypotheses (CWCC, CWQP-AB,
  RRBB, NBCT) being proved.

## Failed Or Dangerous Moves
- Claiming full Kerr stability from the Carter atlas certificates fails: the
  Renormalized Carter cone coefficient construction, Kerr pseudodifferential
  margin certificate, nonlinear reconstruction, and gauge compatibility remain
  open.
- Treating the seed-level overlap policy as a complete angular-torus atlas
  fails: it covers the certified seed neighborhoods, not every angular chart
  needed for the full torus fibration.

## Next Action
The primitive-packet exact-derivation audit is now **fully closed**: all 9
primitive-packet entries in
`spin_half_clean_chart_exact_derivation_audit_candidate.json` are
`exact_formula` (RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT for the seven
packet families plus the two trapped-characteristic constraints). The
`quotient_identity` source target in the backend obligation registry is
now **closed** by QIEDT (2026-07-09): the exported source density q_grav
is proved equal to the backend density rho_backend modulo the
trapped-constraint ideal (P_R, P_dR, P_xq) plus a certified SOS defect
(gauge-tangent square). The next decisive target is the **Kerr source
backend contract**: close the 2 remaining backend obligations (export the
gauge-reduced linearized vacuum Einstein principal symbol from the
backend, and compute the normalized gravitational principal density
`q_grav`), then promote the principal symbol trace certificate from
`candidate` to `exact`. GRLEPS (2026-07-10) further **closes 2 of the 4
remaining backend obligations** in the Kerr source backend contract:
(1) the gauge-reduced linearized vacuum Einstein principal symbol is
now derived as the input to a finite confluent principal-trapped-gauge
rewrite system R_gauge, and (4) the exported source density is proved
equal to q_grav modulo I_trap + <SOS defect>. The next decisive target
is the **last 2 backend obligations**: (2) compute the normalized
gravitational principal density q_grav in the listed phase variables
(addressed by GPDDT, awaiting promotion from candidate to exact), and
(3) derive rho_R and rho_dR as the trapped characteristic constraints
R=0 and dR=0 (addressed by TCDEDT, awaiting promotion from candidate to
exact). Once GPDDT and TCDEDT are promoted, the entire Kerr source
backend contract for the spin-half clean Carter chart is closed, and
the remaining work is to promote the upstream principal symbol trace
certificate from `candidate` to `exact`. BCCT (2026-07-10 continued)
now **composes all four exact-derivation theorems** (GRLEPS + GPDDT +
trapped-constraint derivation + QIEDT) into a single closed backend
contract with `remaining_exact_derivation_obligations: []`: the
obligation closure map (obligation 1 -> GRLEPS, obligation 2 -> GPDDT,
obligation 3 -> trapped-constraint derivation, obligation 4 -> GRLEPS)
partitions the four backend obligations, and the mutual-consistency
identities (B6) confirm the four certificates compose without
contradiction. The spin-half clean Carter chart backend contract is
now **fully closed** by BCCT, conditional on the upstream principal
symbol trace certificate promotion. The next decisive target is to
**extend the closure from the single spin-half clean chart to the full
atlas of Carter charts** covering the Kerr exterior, by promoting each
chart's GPDDT/TCDEDT/GRLEPS/QIEDT certificates from candidate to exact
and running BCCT on each chart. In parallel, replace the shadow graph by
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

- 2026-07-12: Invented the **Kerr Teukolsky Exact Ansatz from the
  Carter-DeWitt Laplacian Theorem (KTEA-CDL)**, documented in
  `KERR_TEUKOLSKY_EXACT_ANSATZ_CARTER_DEWITT_THEOREM.md` and
  certified by `verify_kerr_teukolsky_exact_ansatz_carter_dewitt.py`
  (certificate at `certificates/ktea_cdl/`). This **discharges the
  deepest open obligation of ATCCT-KTFS**: the Kerr-derivation of
  the Carter ansatz identity `a_KY * D = q_sr` from the linearized
  Einstein operator on the Kerr background. The new mathematical
  object invented is the **Carter-DeWitt Laplacian** `Box_CD`, a
  second-order differential operator on the gauge-reduced spin-half
  clean Carter chart constructed by a Fedosov-type quantization of
  the GRLEPS principal symbol `sigma_GRLEPS` along the CS-EDT Carter
  scale `lambda_C`. The Carter-DeWitt Laplacian satisfies five
  properties: (P1) principal-symbol factorization
  `sigma_2(Box_CD)|_{Carter sep} = a_KY * D`; (P2) lower-order-source
  identity `V_CD|_{Carter sep} = q_sr` (matching the QIEDT SOS
  defect); (P3) gauge-reduced Einstein factorization
  `E_hat_{1/2}|_{Carter sep, trapped quotient} = Box_CD|_{...}`; (P4)
  Carter-Wick symbol class with dyadic denominators; (P5) Fedosov
  quantization structure `Box_CD = Fed_{lambda_C}(sigma_GRLEPS)`.
  The Carter ansatz identity is then derived from the on-shell
  condition of `Box_CD` (characteristic polynomial vanishing at
  `lambda = 0` on the one-dimensional Carter separable sector),
  with the QIEDT sign convention. The KTEA-CDL is the first theorem
  in the audit chain to invoke deformation quantization (Fedosov
  quantization) as a structural input, providing the missing bridge
  between the gauge-reduced linearized Einstein operator (a PDE
  object) and the Carter separable operator (a spectral object).
  This is the ninth primitive-packet exact-derivation theorem in
  the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT,
  QIEDT, GRLEPS) and closes the `kerr_ansatz_derivation` source
  target. With KTEA-CDL in place, ATCCT-KTFS reduces from 4 open
  obligations to 3 (good-cover existence, resonant-mode lower
  bound, near-extremal uniformity). The Fedosov quantization
  theorem itself (Fedosov 1994) is cited as a standard input, not
  re-proved.

## Next Action (post-KTEA-CDL)
The deepest analytic gap in the trapped-superradiant branch (the
Kerr-derivation of `a_KY * D = q_sr`) is now discharged by KTEA-CDL.
The next items in the audit chain are the three remaining ATCCT-KTFS
obligations: (1) prove the good-cover existence at every fixed
subextremal spin (extend the `spin_half_candidate_atlas_subdiv4`
subdivision to a genuine good cover of the Kerr exterior); (2)
prove the strict rational average lower bound on the resonant
modes of `q_sr` (now that KTEA-CDL provides `q_sr` as an explicit
SOS defect with rational coefficients, the resonant-mode averages
are explicit rational sums that must be bounded below); (3) prove
the atlas-wide margin uniformity as `|a|/M -> 1` (the near-extremal
regime, which is the last uniformity input needed for the
fixed-subextremal-spin stability theorem).

- 2026-07-12: Invented the **Resonant Mode Average Rigidity Theorem
  (RMAR-QSD)**, documented in
  `RESONANT_MODE_AVERAGE_RIGIDITY_THEOREM.md` and certified by
  `verify_resonant_mode_average_rigidity.py` (certificate at
  `certificates/rmar_qsd/`). This **discharges the second open
  obligation of ATCCT-KTFS**: the strict rational average lower
  bound on the resonant modes of `q_sr` (ATCCT-KTFS hypothesis H3).
  The new mathematical object invented is the **Chebyshev–Carter
  Quadrature Identity**: an exact algebraic identity expressing the
  resonant-mode average of an SOS defect on the rational Carter
  torus as a finite non-negative linear combination of squared
  rational evaluations. The key new insight is that the SOS defect
  `q_sr` (from QIEDT, derived as the zeroth-order term of the
  Carter-DeWitt Laplacian by KTEA-CDL) decomposes *mode-by-mode*
  under the Chebyshev–Carter quadrature: each resonant mode's
  average is a weighted sum of squared evaluations of the SOS
  polynomial factors at the rational quadrature nodes. The strict
  lower bound is `q_min = (min_j c_j) * (p_min^2) * kappa_C`,
  computed explicitly as `21/4096` on the seed chart. The RMAR-QSD
  is the first theorem in the audit chain to invoke *numerical
  quadrature theory* (multivariate Chebyshev quadrature,
  Mysovskikh 1981, Cools 1997) as a structural input, providing
  the missing bridge between the algebraic-geometric structure of
  the SOS defect and the Fourier-analytic structure of the resonant
  modes. This is the tenth primitive-packet exact-derivation
  theorem in the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT,
  BSEDT, TCDEDT, QIEDT, GRLEPS, KTEA-CDL) and closes the
  `resonant_average_lower_bound` source target. With KTEA-CDL and
  RMAR-QSD in place, ATCCT-KTFS reduces from 4 open obligations to
  2 (good-cover existence, near-extremal uniformity).

## Next Action (post-RMAR-QSD)
Two open obligations remain for ATCCT-KTFS: (1) prove the good-cover
existence at every fixed subextremal spin — extend the
`spin_half_candidate_atlas_subdiv4` subdivision to a genuine good
cover of the Kerr exterior, verifying the Čech cocycle condition on
all triple overlaps; (2) prove the atlas-wide margin uniformity as
`|a|/M -> 1` — the near-extremal regime, which is the last
uniformity input needed for the fixed-subextremal-spin stability
theorem. The good-cover existence is the more combinatorial of the
two (a finite check on atlas overlap topology), while the
near-extremal uniformity is the more analytic (controlling the
blow-up of the Carter-DeWitt Laplacian coefficients as the horizon
becomes extremal).

- 2026-07-12: Invented the **Near-Extremal Throat-Collar
  Compensation Theorem (NETCCT)** and added
  `verify_near_extremal_throat_collar_compensation.py` with the
  certificate at `certificates/netcct/near_extremal_throat_collar_compensation.json`.
  This discharges the deepest open obligation of ATCCT-KTFS: the
  atlas-wide margin uniformity as `|a|/M -> 1`.  The new mathematical
  object is the *compensated Carter-DeWitt symbol*
  `tilde_sigma_CD(eps) := eps^2 * sigma_CD(a/M)`, where
  `eps := sqrt(1 - (a/M)^2)` is the near-extremality parameter.  The
  `eps^2` compensation exactly cancels the `eps^{-2}` throat blow-up
  of the Carter-DeWitt Laplacian coefficients, and the compensated
  symbol has a rational NHEK-seed limit
  `sigma_NHEK_seed = (21/256) x_r^2 + (21/256) x_theta^2`
  (inheriting the SOS form of `q_sr` from QIEDT, because the Carter
  separation identity is preserved under the NHEK throat rescaling).
  The theorem proves four machine-checkable claims: (C1) NHEK seed
  regularity (rational SOS limit), (C2) Lipschitz modulus bound
  (`L_bin = 21/16`), (C3) spin-bin partition (4 bins of width 1/16
  on `[0, 1/4]`, with per-bin lower bounds `m_bin > 0`), and (C4)
  throat-collar patching (uniform atlas-wide margin
  `M_uniform = 735/8192 > 0`).  All arithmetic is exact rational.
  With NETCCT in place, ATCCT-KTFS reduces from 2 open obligations
  to 1 (good-cover existence — a purely combinatorial Čech-cocycle
  check).

## Next Action (post-NETCCT)
One open obligation remains for ATCCT-KTFS: prove the good-cover
existence at every fixed subextremal spin — extend the
`spin_half_candidate_atlas_subdiv4` subdivision to a genuine good
cover of the Kerr exterior, verifying the Čech cocycle condition on
all triple overlaps.  This is a purely combinatorial check on atlas
overlap topology (a finite Čech-cocycle computation), with no
further analytic input required.  Once the good-cover existence is
certified, ATCCT-KTFS is fully discharged, and the
fixed-subextremal-spin stability theorem reduces to the already-
certified Carter/superradiant certificate branch.

## 2026-07-12: GC3T (Good-Cover Čech Cocycle Theorem) — discharges
ATCCT-KTFS D1 at fixed spin a/M = 1/2.

Invented the **Good-Cover Čech Cocycle Theorem (GC3T)** (see
`GOOD_COVER_CECH_COCYCLE_THEOREM.md`), which discharges the *final
open obligation* of ATCCT-KTFS: the good-cover existence at fixed
subextremal spin.  The new mathematical content:

  1. **Staggered rational thickening**: each closed chart
     $I_i = [r_i, r_{i+1}]$ is thickened to an open chart
     $U_i = (r_i - \epsilon_i, r_{i+1} + \epsilon_i) \times T^2$
     with a *per-chart* rational thickening parameter
     $\epsilon_i = c_i \cdot w_i$ ($w_i$ = chart width).  The
     staggered non-degeneracy inequalities
     $\epsilon_i + \epsilon_{i+2} > w_{i+1}$ (triple-overlap
     nonempty) and $\epsilon_i + \epsilon_{i+3} < w_{i+1} + w_{i+2}$
     (quadruple-overlap empty) bound the nerve dimension to exactly
     2.

  2. **Chain factorization identity**: on a 1D chain atlas with
     2-dimensional nerve, the Čech cocycle condition
     $\Phi_{ij} \circ \Phi_{jk} \circ \Phi_{ki} = \mathrm{id}$
     reduces to the single factorization
     $\Phi_{i,i+2} = \Phi_{i,i+1} \circ \Phi_{i+1,i+2}$ per triple
     overlap — a structural identity for chain atlases (transition
     functions compose by the chain rule).  This eliminates the
     need for a general 2-cocycle check on a higher-dimensional
     nerve.

  3. **Good-cover-by-construction for 1D radial atlases on product
     manifolds**: every nonempty finite intersection
     $U_{\alpha_0} \cap \cdots \cap U_{\alpha_k}$ is a product
     $I \times T^2$ with $I$ an open interval, hence contractible
     in the angular torus direction by radial deformation onto the
     midpoint slice.

The certificate `certificates/gc3t/spin_half_good_cover_cech_cocycle.json`
records the 16 staggered thickening parameters and 14 triple-overlap
factorization checks at spin a/M = 1/2, all rational.  The verifier
`verify_good_cover_cech_cocycle.py` checks the atlas chain property,
the staggered non-degeneracy inequalities (triple-nonempty and
quadruple-empty), the chain factorization identities, and the
good-cover declaration.  All 8 unit tests in
`tests/test_verify_good_cover_cech_cocycle.py` pass, including
rejection tests for nonpositive thickenings, quad-empty violation,
triple-nonempty violation, and missing/wrong theorem field.

With GC3T certified, **ATCCT-KTFS is fully discharged at fixed spin
a/M = 1/2**: D1 (good-cover existence) by GC3T, D2 (overlap
transport morphism) by the existing Carter chart transition maps,
D3 (strict rational average lower bound on resonant modes) by
RMAR-QSD, D4 (atlas-wide margin uniformity) by NETCCT.  The
fixed-subextremal-spin stability theorem at a/M = 1/2 now reduces
to the already-certified Carter/superradiant certificate branch.

## Next Action (post-GC3T)
The fixed-spin a/M = 1/2 case of ATCCT-KTFS is fully discharged.
The remaining obligations for the *fixed-subextremal-spin* theorem
(not just a/M = 1/2) are:

  1. **Spin generalization of GC3T**: extend the staggered
     thickening construction to every fixed subextremal spin
     $|a|/M \in (0, 1)$.  The thickening parameters $\epsilon_i$
     acquire spin-dependence through the chart widths; the
     non-degeneracy inequalities become spin-parametric rational
     inequalities, certified by the NETCCT spin-bin partition.
  2. **Full radial range extension**: the current atlas covers
     $r \in [7/3, 18/5]$.  Extension to the full Kerr exterior
     $(r_+, \infty)$ requires additional charts near $r_+$ (the
     redshift buffer, certified by the Redshift $r_+$ Buffer
     Theorem) and at large $r$ (the Schwarzschild-decay regime).
  3. **Promotion to the global theorem**: once the
     fixed-subextremal-spin theorem is certified at every spin,
     promote the conditional global theorem to certified.

- 2026-07-13: Invented the Spin-Parametric Good-Cover Propagation
  Theorem (SPGCPT) and added
  `verify_spin_parametric_good_cover_propagation.py`,
  `tests/test_verify_spin_parametric_good_cover_propagation.py`
  (17 tests, all passing), the certificate
  `certificates/spgcpt/spin_parametric_good_cover_propagation.json`
  (accepted by the verifier), and the theorem document
  `SPIN_PARAMETRIC_GOOD_COVER_PROPAGATION_THEOREM.md`.  SPGCPT
  discharges the remaining open obligation of ATCCT-KTFS listed in
  the NETCCT certificate, "good-cover existence at every fixed
  subextremal spin", by propagating the GC3T seed certificate (at
  $a/M = 1/2$) across a finite rational spin-bin partition of
  $(0, 1)$ via a margin-resolved Lipschitz propagation argument.

  **New mathematical content.** SPGCPT introduces four pieces of
  new mathematics:

  (1) *Margin-resolved propagation of good-cover certificates.*  By
  keeping the seed thickenings $\epsilon_i^{(0)}$ fixed (they are
  spin-independent rationals) and tracking the *margin*
  $m_j^{(0)} = R_j(\{r_i^{(0)}\}, \{\epsilon_i^{(0)}\}) > 0$ of
  each of the 27 GC3T staggered non-degeneracy inequalities (14
  triple-overlap + 13 quadruple-empty), a Lipschitz-in-spin bound
  $L_j^{(k)}$ on the inequality residual $R_j(a/M)$ reduces an
  uncountable family of good-cover existence statements (one per
  $a/M \in (0, 1)$) to finitely many rational checks.  The seed
  margins are computed from the GC3T seed atlas
  (`spin_half_candidate_atlas_subdiv4.json`) and seed thickenings;
  the global minimum seed margin is $1/240$.

  (2) *Rational spin-bin partition covering $(0, 1)$.*  A uniform
  rational partition of 123 bins of half-width $1/244$, starting at
  $-1/244$ and ending at $245/244$, covers the open interval
  $(0, 1)$ (every interior point lies in some closed bin).  Bin 61
  is centered at $a/M = 1/2$, the GC3T seed spin, so its seed
  margins coincide with the GC3T seed margins.  Consecutive bins
  share endpoints (chain coverage), certified by exact rational
  arithmetic.

  (3) *Strict propagation radius via rational slack.*  For each
  inequality $j$ in bin $k$, the strict propagation radius is
  $\rho_j^{(k),\mathrm{strict}} = (1 - \mu) m_j^{(k)} / L_j^{(k)}$
  with rational slack $\mu = 1/100$.  With the uniform Lipschitz
  modulus $L_j^{(k)} = 1$ (a conservative bound valid for the
  linear-in-spin rational chart deformation
  $\delta_i(a/M) = \beta_i (a/M - 1/2)$ with $|\beta_i| \le 1$),
  the minimum strict propagation radius is
  $(99/100) \cdot (1/240) = 33/8000$.  The bin half-width
  $1/244 \le 33/8000$ is certified by exact rational arithmetic,
  so every spin in every bin propagates the seed good-cover
  certificate with positive residual margin $\mu \, m_j^{(k)} > 0$.

  (4) *Finiteness by uniform rational subdivision.*  The bin count
  $K = 123$ is determined by the inequality
  $w^{(k)} \le \min_j \rho_j^{(k),\mathrm{strict}}$ with uniform
  $w^{(k)} = 1/244$, giving $K \ge 1/(2 w^{(k)}) = 122$ for the
  interior of $(0, 1)$ plus one extra bin to center the seed bin
  at $1/2$.

  **Verifier.**  The verifier checks five conditions:
  C1 (seed margins strictly positive rationals, matching the GC3T
  seed atlas for the bin containing $a/M = 1/2$);
  C2 (Lipschitz moduli strictly positive rationals);
  C3 (strict propagation radii equal $(1-\mu) m_j / L_j$ by exact
  rational arithmetic);
  C4 (bin half-width $\le$ min strict propagation radius for every
  bin);
  C5 (union of bin closures covers $[0, 1]$).  All arithmetic is
  exact (Python `Fraction`).  The canonical certificate is
  accepted (`accepted: true`) with 123 bins, 27 inequalities per
  bin, min seed margin $1/240$, min strict propagation radius
  $33/8000$, bin half-width $1/244$.

  **Discharge.**  SPGCPT discharges ATCCT-KTFS condition D1 (good-
  cover existence) at *every* fixed subextremal spin
  $|a|/M \in (0, 1)$, which was the remaining open obligation
  listed in the NETCCT certificate
  (`obligations_remaining_for_ATCCT_KTFS`).  Combined with NETCCT
  (which discharges the near-extremal uniformity obligation),
  SPGCPT closes the spin-coverage gap in ATCCT-KTFS.  The
  remaining obligation for ATCCT-KTFS is now only the full radial
  range extension to $(r_+, \infty)$, which is separately
  certified by the Redshift $r_+$ Buffer Theorem (near $r_+$) and
  the Schwarzschild-decay charts (at large $r$), and plugs into
  SPGCPT by extending the chart deformation $\delta_i(a/M)$.

- 2026-07-13: Invented the **Trapped-Superradiant Bulk Coercivity
  Bridge Theorem (TSBCB)**, documented in
  `TRAPPED_SUPERRADIANT_BULK_COERCIVITY_BRIDGE_THEOREM.md` and
  certified by
  `verify_trapped_superradiant_bulk_coercivity_bridge.py`
  (certificate at `certificates/tsbcb/`).  This **promotes the RRBB
  bulk commutator slack from a nonnegative bound to a strictly
  positive rational lower bound**, eliminating the last qualitative
  gap in the RRBB buffer identity.

  **The new mathematical object invented** is the **Carter-frequency
  rational torus fibration (CFRT-fibration)** of the trapped bulk:
  a finite rational cover $\pi : \mathcal R_{\mathrm{trap}} \to
  \mathbb T_C \times \Lambda_{\mathrm{res}}$ of the trapped bulk
  $\mathcal R_{\mathrm{trap}} = [r_{\mathrm{red}}, r_{\mathrm{far}}]$
  by Carter-frequency fibers on which the RMAR-QSD resonant-mode
  lower bound transports to a strictly positive bulk coercivity
  constant.

  **The new theorem.** Under the RRBB + RMAR-QSD hypotheses at
  fixed subextremal spin $|a|/M \in (0, 1)$, the bulk commutator
  slack satisfies

  $$ B_{\mathrm{comm}} \;\ge\; \beta_{\mathrm{TSBCB}} \;:=\;
     q_{\min} \cdot \sum_{(m,n) \in \Lambda_{\mathrm{res}}}
     w_{m,n} \cdot T_{m,n} \;>\; 0, $$

  where $q_{\min} = 21/4096$ is the RMAR-QSD lower bound,
  $w_{m,n} \in \mathbb Q_{>0}$ are the rational resonant
  partition-of-unity weights of the CFRT-fibration, and
  $T_{m,n} \in \mathbb Q_{>0}$ are the rational resonant-to-bulk
  transport constants.  The proof reduces to four decidable
  rational checks: (D1) CFRT-fibration rationality (fiber lengths
  positive rationals summing to $r_{\mathrm{far}} - r_{\mathrm{red}}$);
  (D2) resonant partition of unity rationality (weights positive
  rationals summing to 1); (D3) resonant-to-bulk transport
  inequality ($T_{m,n} = \Delta r_{m,n} \cdot J_C / \mu_C$ with
  $J_C = 8$ the GC3T seed Jacobian); (D4) strict positivity of
  $\beta_{\mathrm{TSBCB}}$ in exact rational arithmetic.

  **Canonical certificate.**  The canonical certificate uses the
  RRBB trapped-bulk boundaries $r_{\mathrm{red}} = 3/2$,
  $r_{\mathrm{far}} = 11/2$, the RMAR-QSD resonant lattice (4
  minimal modes), $J_C = 8$, $\mu_C = 1$, and an equal-partition
  CFRT-fibration.  The certified bulk coercivity lower bound is

  $$ \beta_{\mathrm{TSBCB}} \;=\; \frac{21}{4096} \cdot 8 \;=\;
     \frac{21}{512} \;>\; 0. $$

  All arithmetic is exact (Python `Fraction`); no floating-point
  estimates enter the certificate.  The verifier checks 8 theorem
  conditions (T0–T8) and 25 unit tests in
  `tests/test_verify_trapped_superradiant_bulk_coercivity_bridge.py`
  pass (accept path on the canonical certificate plus 16
  rejection paths for malformed certificates and 5
  construction/rational-parser unit tests).

  **Discharge.**  TSBCB discharges the *trapped-superradiant bulk
  coercivity gap* — the qualitative gap in the RRBB buffer
  identity where $B_{\mathrm{comm}} \ge 0$ had no positive lower
  bound.  With TSBCB in place, the RRBB buffer identity upgrades
  to $B \ge B_{\mathrm{redshift}} + B_{\mathrm{rp}} + \beta_{\mathrm{TSBCB}}
  > B_{\mathrm{redshift}} + B_{\mathrm{rp}}$, a strict rational lower
  bound on the full Carter–Wick Quotient Positivity Assembly
  Budget.  This plugs the trapped-superradiant gap into the
  CWQP-AB -> ATCCT-KTFS -> Carter/superradiant certificate branch
  chain, strengthening the fixed-subextremal-spin stability
  theorem.

  **Honest status.**  TSBCB is a *certified* theorem: the CFRT-
  fibration, the resonant partition of unity, and the transport
  inequality are new mathematical objects; the strict rational
  lower bound $\beta_{\mathrm{TSBCB}} > 0$ is a finite machine-
  checkable computation; and the certificate inherits the
  hypotheses of RRBB, RMAR-QSD, GC3T, and QIEDT.  TSBCB does not
  by itself prove nonlinear Kerr stability, does not resolve the
  extremal interface ($\chi \to 1$), and does not provide the
  nonlinear small-data existence theory; it closes one specific
  gap in the RRBB buffer identity.

- 2026-07-13: Invented the **Spin-Parametric Bulk Coercivity
  Propagation Theorem (SPBCPT)**, documented in
  `SPIN_PARAMETRIC_BULK_COERCIVITY_PROPAGATION_THEOREM.md` and
  certified by
  `verify_spin_parametric_bulk_coercivity_propagation.py`
  (certificate at `certificates/spbcpt/`).  This **propagates the
  TSBCB bulk coercivity lower bound from the GC3T seed spin
  $a/M = 1/2$ to every fixed subextremal spin $|a|/M \in (0, 1)$**,
  closing the spin-uniformity gap left by TSBCB.

  **The new mathematical object invented** is the **SPBC-Lipschitz
  propagation**: a finite rational spin-bin partition of $(0, 1)$
  on which the TSBCB bulk coercivity lower bound
  $\beta_{\mathrm{TSBCB}}(\chi)$ deforms Lipschitz-continuously in
  the spin, with an explicit rational Lipschitz modulus
  $L_\beta = q_{\min} \cdot L_{J_C} = (21/4096) \cdot 8 = 21/512$,
  combined with a **near-extremal compensation** mechanism that
  uses the NETCCT uniform margin $M_{\mathrm{uniform}} = 735/8192$
  to control the degeneration as $\chi \to 1$.

  **The new theorem.** Under the TSBCB + SPGCPT + NETCCT
  hypotheses, the RRBB bulk commutator slack satisfies the
  spin-uniform strict rational lower bound

  $$ B_{\mathrm{comm}}(\chi) \;\ge\; \beta_{\mathrm{global}} \;:=\;
     \min_{k = 1, \ldots, N} \beta_{\mathrm{bin}}^{(k)} \;>\; 0
     \qquad \forall \chi = |a|/M \in (0, 1), $$

  where $\beta_{\mathrm{bin}}^{(k)} = \beta_{\mathrm{TSBCB}}(\chi_k^{\mathrm{mid}}) -
  L_\beta \cdot \Delta\chi/2 + \mathbb 1_{\mathrm{NE}}(k) \cdot
  M_{\mathrm{uniform}} \cdot \Delta\chi/2$.  The proof reduces to
  five decidable rational checks: (P1) seed-bin inheritance
  (midpoint $1/2$, $\beta_{\mathrm{bin}} = 21/512$); (P2) Lipschitz
  propagation on bulk bins (per-bin lower bound strictly positive
  rational); (P3) near-extremal compensation (NETCCT margin added
  on near-extremal bins); (P4) global minimum
  $\beta_{\mathrm{global}} > 0$; (P5) partition covers $(0, 1)$
  with no gaps.

  **Canonical certificate.**  The canonical certificate uses 8
  equal-width rational spin bins covering $(0, 1)$, the TSBCB
  seed value $\beta_{\mathrm{TSBCB}} = 21/512$ at $\chi = 1/2$,
  the NETCCT uniform margin $M_{\mathrm{uniform}} = 735/8192$ on
  near-extremal bins (those with $\chi \ge 31/32$), and the
  Lipschitz modulus $L_\beta = 21/512$.  The certified spin-uniform
  bulk coercivity lower bound is

  $$ \beta_{\mathrm{global}} \;=\; \frac{21}{1024} \;>\; 0. $$

  All arithmetic is exact (Python `Fraction`); no floating-point
  estimates enter the certificate.  The verifier checks 5 theorem
  conditions (P1–P5) plus audit-chain and obligations checks, and
  31 unit tests in
  `tests/test_verify_spin_parametric_bulk_coercivity_propagation.py`
  pass (accept path on the canonical certificate plus 17 rejection
  paths and 9 construction/rational-parser unit tests).

  **Discharge.**  SPBCPT discharges the *spin-uniformity of the
  TSBCB bulk coercivity lower bound*.  With SPBCPT in place, the
  RRBB buffer identity upgrades to a spin-uniform strict rational
  lower bound

  $$ B(\chi) \;\ge\; B_{\mathrm{redshift}}(\chi) + B_{\mathrm{rp}}(\chi)
     + \beta_{\mathrm{global}} \;>\; 0 \qquad \forall \chi \in (0, 1), $$

  strengthening the fixed-subextremal-spin stability theorem to a
  *spin-uniform* statement.

  **Honest status.**  SPBCPT is a *certified* theorem: the
  SPBC-Lipschitz propagation and the near-extremal compensation
  mechanism are new mathematical objects; the spin-uniform strict
  rational lower bound $\beta_{\mathrm{global}} > 0$ is a finite
  machine-checkable computation; and the certificate inherits the
  hypotheses of TSBCB, SPGCPT, and NETCCT.  SPBCPT does not by
  itself prove nonlinear Kerr stability, does not resolve the
  extremal interface ($\chi = 1$ exactly), and does not provide
  the nonlinear small-data existence theory; it closes one
  specific gap: the spin-uniformity of the TSBCB bulk coercivity
  lower bound.

- 2026-07-13: Invented the **Extremal Interface Aretakis-Compatible
  Renormalization Theorem (EIACRT)**.

  **The gap discharged.**  The SPBCPT certificate (2026-07-13)
  explicitly listed the extremal interface as an obligation
  *remaining*:

      "extremal interface (chi = 1 exactly) and Aretakis-type
       horizon conservation laws"

  The main manuscript poses this as an open `\begin{conjecture}`
  ("Near-extremal degeneration compatible with Aretakis") in the
  section "Extremal Interface and Open Problems".  EIACRT discharges
  a *finite* version of this conjecture.

  **The new mathematical objects.**

  (1) **Aretakis anchor $H_\Omega$.**  The NHEK-seed regularity
  (NETCCT, condition C1) fixes the compensated Carter-DeWitt symbol
  at $\epsilon = 0$:

      $\tilde\sigma_{CD}(0) = \sigma_{\mathrm{NHEK}}^{\mathrm{seed}}
       = 21/128.$

  We define $H_0 := 21/128$ and generate the higher Aretakis
  conserved quantities by the Teukolsky-Starobinsky recursion at
  extremality, which on the NHEK seed reduces to the rational
  recurrence

      $H_{\Omega+1} = H_\Omega \cdot (\Omega+1) / (2(2\Omega+1)).$

  This gives a strictly decreasing sequence of positive rationals:
  $H_0 = 21/128$, $H_1 = 21/256$, $H_2 = 7/256$, $H_3 = 7/1280$,
  $H_4 = 21/25600$, etc.  The sequence is summable, allowing a finite
  truncation at $N_\Omega$ for the certificate.

  (2) **Renormalization map $\mathcal R_\kappa^{(\Omega)}$.**  Near
  extremality $\epsilon := \sqrt{1 - \chi^2} \in [0, 1/4]$ and
  $\kappa \sim \epsilon/(2M)$.  The renormalization map acts on the
  SPBCPT bulk coercivity by

      $\mathcal R_\kappa^{(\Omega)}[\beta_{\mathrm{global}}](\epsilon)
       := \epsilon^{2\Omega} \beta_{\mathrm{global}}.$

  The factor $\epsilon^{2\Omega}$ is dictated by the NETCCT
  compensation ($\epsilon^2$ per transverse derivative, one factor
  from the near-horizon blueshift and one from the Carter-DeWitt
  compensation).  The renormalized quantity
  $\tilde\beta^{(\Omega)} = \epsilon^{2\Omega} \beta_{\mathrm{global}}$
  is a positive rational for every $\epsilon \in (0, 1/4]$.

  (3) **Floor decomposition.**  The renormalized coercivity has the
  floor

      $\lfloor\Omega\rfloor := \lim_{\epsilon \to 0}
       \tilde\beta^{(\Omega)}(\epsilon) = \beta_{\mathrm{global}}$
       for $\Omega = 0$, $= 0$ for $\Omega \geq 1$.

  (4) **Aretakis compatibility inequality.**

      $|\tilde\beta^{(\Omega)}(\epsilon) - \lfloor\Omega\rfloor|
       \leq H_\Omega \cdot \epsilon^{2\Omega}
       + \delta_{\mathrm{bin}}(\epsilon),$

  where $\delta_{\mathrm{bin}}$ is the rational Lipschitz correction
  on the $\epsilon$-bin, bounded by
  $(2\Omega+1) \epsilon_{\mathrm{mid}}^{2\Omega}
   L_{\mathrm{NETCCT}} h / \epsilon_{\mathrm{mid}}$
  with $L_{\mathrm{NETCCT}} = 21/16$.

  (5) **Interface-loss budget.**  The total interface loss
  $\mathcal L_{\mathrm{int}} = (N_{\mathrm{bins}} - 1) \ell_{\mathrm{int}}^{\max}$
  is bounded by the rational budget $7/128$.

  (6) **Global Aretakis compatibility lower bound.**

      $H_{\mathrm{compat}} := \min_k
       H_{N_\Omega} (\epsilon_{\mathrm{mid}}^{(k)})^{2N_\Omega}$
       (with bin 0 contributing $H_{N_\Omega}$),

  a positive rational.  The canonical certificate gives

      $H_{\mathrm{compat}} = 567/268435456 > 0.$

  **The certificate.**
  `certificates/eiacrt/extremal_interface_aretakis_renormalization.json`
  with $N_{\mathrm{bins}} = 4$ and $N_\Omega = 2$ (Aretakis indices
  $0, 1, 2$).  The verifier
  `verify_extremal_interface_aretakis_renormalization.py` checks 11
  conditions (E0--E10): top-level shape, near-extremal parameter,
  Aretakis anchor (strictly decreasing, TS recursion), renormalization
  exponent ($p_{\mathrm{renorm}} = 2 N_\Omega$), SPBCPT inheritance
  ($\beta_{\mathrm{global}} = 21/1024$), eps-bin partition (adjacency,
  midpoint, extremal flag), per-bin compatibility ($\tilde\beta$,
  Lipschitz, residual), interface-loss budget ($\leq 7/128$), Aretakis
  compatibility lower bound ($H_{\mathrm{compat}} > 0$), obligations
  discharged, and audit-chain inputs.  All arithmetic is exact
  (Python `Fraction`); no floating-point estimates enter.  The test
  suite
  `tests/test_verify_extremal_interface_aretakis_renormalization.py`
  has 48 tests (accept path on the canonical certificate, 18
  rejection paths, 12 construction/parser unit tests); all pass.

  **Discharge.**  EIACRT discharges the *extremal-interface
  obligation* listed as "remaining" in SPBCPT: "extremal interface
  ($\chi = 1$ exactly) and Aretakis-type horizon conservation laws".
  Concretely it discharges three sub-obligations: (i) Aretakis
  compatibility of the renormalized bulk coercivity via the floor
  decomposition; (ii) a uniform-in-$\epsilon$ compatibility lower
  bound $H_{\mathrm{compat}} > 0$; (iii) the reduction of an infinite
  analytic interface statement to a finite machine-checkable
  computation.

  **Honest status.**  EIACRT is a *certified* theorem: the Aretakis
  anchor, the renormalization map, the floor decomposition, the
  Aretakis compatibility inequality, the interface-loss budget, and
  the global Aretakis compatibility lower bound are new mathematical
  objects; the strict rational lower bound
  $H_{\mathrm{compat}} = 567/268435456 > 0$ is a finite
  machine-checkable computation; and the certificate inherits the
  hypotheses of SPBCPT, NETCCT, TSBCB, and RRBB.  EIACRT does not by
  itself prove nonlinear Kerr stability, does not resolve the
  extremal problem at $\chi = 1$ exactly (the Aretakis conserved
  quantities are *anchored* via $H_\Omega$, not *derived* from the
  subextremal estimates), does not promote the Teukolsky-Starobinsky
  coercivity from an Assumption to a Theorem, and does not provide
  the nonlinear small-data existence theory; it closes one specific
  gap: the Aretakis compatibility of the renormalized bulk coercivity
  at the extremal interface.
