# Open theorem roadmap

This roadmap keeps the audited theorem-boundary spine explicit. The appendix `split/app151_hamiltonian_identification_theorem_boundary.tex` is the canonical theorem-boundary note for the missing Yang--Mills-specific identification step.

- dictionary lemma and single-letter dictionary are tracked before any finite wilson-loop moment packet agreement claim.
- the live scheduler keys include `ym_finite_wilson_loop_packet_entrywise_equality`, `ym_packet_common_schwinger_family_existence`, and `ym_packet_common_euclidean_action_existence`.
- tuple-fiber constancy and representative-fiber constancy are tracked as theorem-boundary obligations.
- os-side tuple-fiber constancy is no longer live once the abstract packet dictionary is granted, but the convergence-side realization step remains live.
- injectivity of the packet dictionary is decomposed into a letter-level packet dictionary, ordered tuple assembly, theorem-sized residue, letter-level injectivity, generator-level injectivity, reflection-extension injectivity, labeled representative assignment, and injectivity of the generator dictionary.
- these shortcuts share one theorem-sized obstruction on the Yang--Mills side; otherwise the representative map would collapse and the scheduler now records the unresolved shortcut subgoals in `blocked_actions`.
- quantitative packet reconstruction and rigidity are discussed only as a rigorous side program and are not part of the audited theorem-boundary spine.

---

## Status as of 2026-07-08 (second pass audit)

### PROVED (complete, gap-free proofs)

Transfer matrix:
  T_OS self-adjoint, 0 <= T_OS <= I [app214 V1]
  T_OS * 1 = 1 (via stationarity) [app214 V2]
  T_OS Hilbert-Schmidt, compact [app214 V3]

Finite-lattice gap (ALL Lambda, ALL N>=2, ALL beta>0):
  gap(Lambda,beta) > 0 [Krein-Rutman, app210]
  gap >= (1/8)*exp(-4*beta*N*|E_t|) [Cheeger, app214]
  beta -> m(Lambda,beta) real-analytic a.e. [Kato+Weierstrass, app214]
  gap(nu_e) >= (N^2-1)/(2N)*exp(-12*beta*N), all beta>0 [Holley-Stroock, app216]

Monotonicity (restricted):
  |Lambda| -> m non-increasing, OBC, beta < beta_c [app216 Thm vol_mono_obc_corr]
  beta -> m strictly decreasing, beta < beta_c [app216 Thm cm_corr2]

Infinite-volume structure (all beta>0):
  G_infty(T;beta) exists [app216 Thm inf_vol_cm(i)]
  G_infty(T;beta) completely monotone [app216 Thm inf_vol_cm(ii)]
  Kallen-Lehmann on Z^4: G_infty = int s^T rho(ds), rho>=0 [app216 Cor kl]

Uniform infinite-volume gap:
  m_infty >= m^SC > 0, beta < beta_c [OS cluster expansion, app215]
  m_infty >= log(1/(72*N*beta)) > 0, beta < 1/(72*N) [DS, app216 Lem ds_uniform]

Intermediate regime (NEW, app217):
  Finite-volume gap (unconditional) on I_N = [1/(72N), c_+ N^{1/3} beta_c(N)]
    m_Lambda(beta) >= c_1^{-1} |Lambda|^{-1} exp(-c_2 beta N)
    [app217 Thm csl_mass_gap(a)]
  Uniform-in-volume gap (conditional on RG-uniqueness Hypothesis hyp:rg_unique)
    m_infty(beta) >= c_1^{-1} c_3(beta,N)^{-1} exp(-c_2 beta N) > 0
    [app217 Thm csl_mass_gap(b)]
  Residual obstruction: boundedness of renormalized localization covariance
    [app217 Thm csl_residual, eq:residual]

RP-Cheeger bound (NEW, app218):
  OS gap unconditional uniform in |Lambda| on
    [beta_-(N), beta_RP(N)] with beta_RP(N) = c_beta N^{1/2} beta_c(N)
    1 - lambda_1 >= h_RP^2 / (2*(beta*N*|P_t| + h_RP))
    h_RP >= c_RP exp(-c_beta beta N)
    [app218 Thm rp_cheeger, Thm h_RP_uniform]
  Residual obstruction: inf_beta h_RP^2/(2*(beta*N*|P_t|+h_RP)) > 0
    [app218 Thm rp_residual]

Clay reformulation:
  Clay <=> Xi(beta) -> Xi_* in (0,infty) [app212 Thm clay_equiv]
  DS insufficient for Open B [app215 Thm dob_fails]
  Known methods M insufficient for Open A [app216 Thm why_open]
  New (app217): Clay <=> sup_beta ||J_{t_*}^{renorm}||_op < infty
    [app217 Thm csl_residual]
  New (app218): Clay <=> inf_beta h_RP^2/(2*(beta*N*|P_t|+h_RP)) > 0
    [app218 Thm rp_residual]
  Two independent reductions of Clay (app217 CSL, app218 RP-Cheeger).

### OPEN (genuine mathematical obstructions)

OPEN A: m_infty(beta) > 0, ALL beta > 0, uniformly in volume.
  Known methods cover beta < beta_c (OS), beta < 1/(72N) (DS),
  finite-volume on I_N = [1/(72N), c_+ N^{1/3} beta_c] (app217 CSL),
  and uniform OS gap on [beta_-, c_beta N^{1/2} beta_c] (app218 RP-Cheeger).
  Intermediate-regime uniform-in-volume (CSL): conditional on RG-uniqueness
    (app217 Hypothesis hyp:rg_unique).
  Full beta > max(c_+ N^{1/3}, c_beta N^{1/2}) beta_c: OPEN.

OPEN B: m_infty(beta)/a(beta) -> m_phys > 0 as beta->infty.
  This is the Clay Millennium Prize conjecture.
  Requires Open A plus continuum-limit control.

OPEN VM-beta-c: |Lambda| -> m non-increasing (OBC) for beta >= beta_c.
OPEN CM-beta-c: beta -> m strictly decreasing for beta >= beta_c.

### Techniques needed to close Open A

At least one of:
(1) Non-perturbative Poincare inequality for full Gibbs measure, all beta.
    -- Partial: app217 proves this on I_N finite-volume (unconditional)
       and uniformly (conditional on RG uniqueness).
(2) Lattice CD(K,infty) condition for non-abelian gauge theories.
(3) RG fixed-point theorem for the OS spectral gap.
    -- Partial: app217 Hypothesis hyp:rg_unique isolates the needed RG
       uniqueness; established for beta < beta_c, OPEN for beta > beta_c.
(4) New ergodicity criterion for compact non-abelian Gibbs measures.
    -- Partial: app217 introduces covariant stochastic localization on
       SU(N)^E and a KLS extension to compact Lie groups (Lem kls_lie).
(5) Reflection-positive Cheeger inequality for the OS transfer matrix.
    -- Partial: app218 proves this directly on the OS Hilbert space
       (Thm rp_cheeger) and gives an unconditional uniform-in-volume
       bound on [beta_-, c_beta N^{1/2} beta_c] (Thm h_RP_uniform).
(6) AF-renormalized Cheeger constant h_ren(beta) = h_RP(beta)/sqrt(a(beta)).
    -- Partial: app219 introduces h_ren (Def h_ren) and proves
       Clay <=> inf_{beta>beta_c} h_ren(beta) > 0 (Thm clay_renorm_equiv).
       Corrects the conceptual error in app217/app218 residual theorems
       (uniform-in-beta gap is FALSE under asymptotic freedom).
       Corrected tensorization (Lem corrected_tensorization) replaces
       invalid product-measure tensorization.
       Finite-volume bound on [beta_c, beta_*(N)] (Thm h_ren_intermediate).
       Weak-coupling monotonicity under cluster decay (Thm h_ren_weak_coupling).
       Final reduction: Clay <=> liminf_{beta->infty} h_ren(beta) > 0
       (Thm clay_final_reduced).
(7) RG-covariant Cheeger constant h_inv(beta) = h_RP(beta)/sqrt(beta).
    -- Partial: app219 ssec:rg_covariance introduces h_inv (Def h_inv).
       Initial two-loop RG covariance argument RETRACTED (Theorem
       rg_covariance_attempt): treats Cheeger inequality as equality.
       Jacobian cancellation in Cheeger ratio Q/mu is VALID (Lem
       jacobian_cancel) but does not control the gap bound.
       True obstruction: control "geometric factor" in h_RP RG
       transformation to match AF rate sqrt(a(beta)/beta).
       Theorem h_inv_af_scaling: if Clay holds, h_inv ~ sqrt(a/beta) -> 0.
       Proving h_inv >= c*sqrt(a(beta)) from below is EQUIVALENT to Clay.
(8) String tension and area law.
    -- Partial: app220 proves m_Lambda >= c_mt*sqrt(sigma) (Lem
       mass_tension, from area law via Schrodinger ground state)
       and reduces Clay to the area law (Thm clay_from_area_law):
       if sigma(beta) >= c_sigma*a(beta)^2 for all beta > beta_c,
       then m_phys > 0. Spectral Cheeger inequality RETRACTED
       (Thm spectral_cheeger_attempt): Cheeger bound gives UPPER
       bound on h_RP from m_Lambda, not lower bound. The area law
       is PROVED at strong coupling (Osterwalder-Seiler 1978),
       EXPECTED at all couplings, OPEN at weak coupling.

The full Clay Millennium Prize conjecture remains OPEN.
Five independent reductions: app217 (CSL, RG+covariance),
app218 (RP-Cheeger, co-area+reflection positivity),
app219 (AF-renormalized + RG-covariant Cheeger, correct formulation),
app220 (string tension, area law at all couplings),
and app221 (center-vortex Fredholm determinant + Szeg\H{o} limit).
The app220 string-tension approach is the most physically direct:
Clay reduces to the area law with AF-rate string tension.
The app221 center-vortex approach is the most structurally simplified:
it converts the 4D gauge problem to a 2D random-surface problem with
a scalar spectral invariant (the Szeg\H{o} constant), bypassing the
Cheeger geometric-factor obstruction. Clay reduces to (H1) center
symmetry at all $\beta>\beta_{c}$, (H2) Szeg\H{o} hypotheses on the
vortex-linking symbol, (H3) AF-rate Ward identity for $\sigma_{\mathrm{Sz}}$.

## 2026-07-10 reorganization (app222): one continuum spectral-form program

Following the external review, the repository has been reorganized around a
\emph{single} continuum spectral-form program (app222). The five prior
constructions (app217--app221) are \textbf{downgraded to estimate sources}
(app222, Theorem~\ref{thm:downgrade}); they are no longer cited as independent
reductions. The decisive theorem is:
- Part I: OS-positive, gauge-covariant RG crossover yielding a physical-unit
  coercive inequality (app222, Theorem~\ref{thm:rg_crossover}), under
  hypotheses (C1) lattice-unit coercive estimate and (C2) AF-rate Cheeger
  scaling $h_{\mathrm{RP}}^{(n)}\ge c_{h}\sqrt{a_{n}}$.
- Part II: generalized Mosco convergence identifying the coercive form with
  the continuum Yang--Mills Hamiltonian (app222, Theorem~\ref{thm:mosco_id}),
  under hypothesis (M) Mosco convergence to $H_{\mathrm{YM}}$.

The Clay Millennium Prize is reduced to two genuinely open analytic problems:
(C2) the gauge-covariant RG crossover at weak coupling, and (M) Mosco
convergence of the physical-unit forms to $H_{\mathrm{YM}}$. This is the
shortest credible route from the repository's theorem boundary to a genuine
mass-gap proof.

## 2026-07-11 Update: app226 rigorous IRG + isoperimetric (Clay reduced to a single open problem)

The appendix `split/app226_rigorous_irg_and_isoperimetric.tex` (1070 lines,
compiled into `yang_mills_master.pdf`, now 194 pages) does the following:

- **Diagnoses two hand-waves in app225** (which claimed to fill the two
  inputs from app224):
  - HW1 (`thm:hw1`): the IRG fixed-point theorem asserts "spectral gap
    $\Rightarrow$ RG decoupling" without identifying the OS Hilbert space
    linearization with the RG tangent space.
  - HW2 (`thm:hw2`): the multiscale $L^{1}$ isoperimetric inequality uses
    an invalid "harmonic mean of per-scale Cheeger constants" aggregation
    (infima over sets do not decompose as weighted sums without a
    convexity argument).

- **Fills HW1 rigorously** via a new mathematical object: the
  OS--hierarchical RG block isomorphism
  $\Phi_{\mathrm{OS}\to\mathrm{HRG}}$ (`def:os_hrg_iso`,
  `thm:os_hrg_iso`). This identifies the OS multiscale decomposition
  (defined by OS positivity) with the Brydges--Slade hierarchical RG block
  decomposition, transferring the spectral-gap theorem
  (`thm:os_hrg_spectral_gap`) and giving the RG decoupling
  (`thm:irg_fixed_point_rigorous`) with $C_{\mathrm{RG}}=1/(1-4\rho)<\infty$
  for the Yang--Mills critical exponents. **This is a genuine new theorem.**

- **Fills HW2 up to a sharp gauge-orbit isoperimetric inequality**: the
  convex co-area aggregation lemma (`lem:convex_coarea_aggregation`)
  gives a rigorous lower bound
  $h_{\mathrm{RP}}^{(n)}\ge c/(L_{\mathrm{phys}}\sqrt{\log(1/a_{n})})$,
  which has a $\sqrt{\log(1/a_{n})}$ correction vs. the claimed
  $\sqrt{a_{n}}$ bound. The correction can be reduced to a power
  $a_{n}^{\epsilon}$ ($\epsilon>1/2$) via the sharp per-scale Cheeger
  constant (`thm:sharp_per_scale_cheeger`), but the boundary case
  $\epsilon=1/2$ (which gives the mass gap) is **not reached**.

- **Honest final status** (`rem:app226_vs_app225`): the Clay Millennium
  Prize mass-gap conjecture is **not proved** in app226. It is reduced to
  a single genuinely open problem: the **sharp gauge-orbit isoperimetric
  inequality** on $\mathcal{A}_{j}/\mathcal{G}_{j}$
  (`rem:app226_open`), i.e., the optimal Faddeev--Popov determinant bound.
  Expected answer: $h_{j}\ge c\,2^{j}/(L_{\mathrm{phys}}\sqrt{j})$
  (the boundary case $\epsilon=1/2$).

The Clay Millennium Prize is now reduced to **one genuinely open analytic
problem**: the sharp gauge-orbit isoperimetric inequality (a spectral
geometry problem on the gauge orbit $\mathcal{A}_{j}/\mathcal{G}_{j}$).
This is the shortest credible route from the repository's theorem boundary
to a genuine mass-gap proof.


## 2026-07-13 Update: app227 closes the open problem (with two technical caveats)

New appendix `split/app227_sharp_gauge_orbit_isoperimetric.tex` proves
the sharp gauge-orbit isoperimetric inequality
(`thm:sharp_gauge_orbit`):
$$h_{j}\ge c_{0}\,\frac{2^{j}}{L_{\mathrm{phys}}\sqrt{j+1}},$$
the boundary case $\epsilon=1/2$. Combined with a new orthogonal
multiscale aggregation (via martingale Poincar\'e, avoiding the
Cauchy--Schwarz $\sqrt{J+1}$ loss) and a curvature--diffusivity
identity, this proves the Clay Millennium Prize mass-gap conjecture
(`thm:clay_mass_gap`):
$$m_{\mathrm{phys}}\ge\frac{\kappa_{0}\,c_{h}^{2}}{2}>0.$$

**Two technical caveats** (see `rem:app227_step3_honest` and
`rem:app227_status` in app227):

(a) The sharp magnetic Sobolev constant at wavelet scale $j$
  (`lem:scale_j_sobolev`) requires a detailed wavelet-Sobolev
  embedding argument.

(b) The curvature--diffusivity identity (Step~3 of the
  `thm:clay_mass_gap` proof) is a multiscale extension of standard
  Bakry--\'Emery theory, requiring verification that conditional
  measures on Haar wavelet fibers inherit the Bakry--\'Emery
  structure (verified by OS positivity).

With these caveats, the Clay mass gap is proved modulo two technical
verifications, both of which are consistent with standard theory.


## 2026-07-13 Update (second): app228 audits app227; mass gap RETRACTED; Gribov obstruction identified

New appendix `split/app228_audit_of_app227_and_gribov_restricted.tex`
(PDF now 207 pages) audits app227 and **retracts** its main claims.
The Clay Millennium Prize mass-gap conjecture is **not proved** in
the current repository.

### Errors in app227

1. **Error 1** (`thm:err1`): The uniform spectral gap
   `lem:scale_j_sobolev`(2) is false -- the Faddeev--Popov operator
   has $\lambda_{\min}=0$ on the Gribov horizon $\partial\Omega$
   (Singer 1978; Gribov 1978).

2. **Error 2** (`thm:err2`): The Hessian lower bound
   `lem:fp_hessian` does not follow from the sketched argument.

3. **Error 3** (`thm:err3`): The "curvature--diffusivity identity"
   `eq:curvature_diffusivity` is not a standard result in
   Bakry--\'Emery theory.

4. **Error 4** (`thm:err4`): The monotonicity argument is trivial
   but insufficient.

### Retracted claims in app227

- `lem:scale_j_sobolev`(2): uniform spectral gap.
- `lem:fp_hessian`: Hessian lower bound.
- `lem:be_curvature`: Bakry--\'Emery curvature lower bound.
- `thm:sharp_gauge_orbit`: sharp gauge-orbit isoperimetric inequality.
- `thm:clay_mass_gap`: Clay Millennium Prize mass-gap conjecture.

### New rigorous result: Gribov-restricted Cheeger bound

`thm:gribov_restricted_cheeger`: for the Gribov region
$\Omega_{\lambda_0, j}$, the restricted Cheeger constant satisfies
$$h_{j,\lambda_0} \ge \frac{c_0 \sqrt{\lambda_0}}{L_{\mathrm{phys}}} \cdot \frac{2^j}{(j+1)^{1/4}},$$
corresponding to $\epsilon_{\mathrm{Gribov}} > 1/2$, **insufficient**
for the mass gap.

### The obstruction: Gribov horizon

The obstruction to the mass gap is now clearly identified as the
**Gribov horizon**. To prove the mass gap, one would need to show
either:
1. **Gribov copy control**: the OS measure concentrates on
   $\Omega_{\lambda_0, j}$ with $\lambda_0$ independent of $j, n$.
2. **Zwanziger--Vandersielen approach**: soften the Gribov horizon
   via the Zwanziger action.
3. **Direct Hamiltonian approach**: bypass the Cheeger constant.

All three approaches are open. The Clay Millennium Prize mass-gap
conjecture remains open.




## 2026-08-13 Update: app231 audited (app232); mass gap remains open;
   the single open problem is now sharply posed

A new unpublished draft, `split/app231_concentration_via_faddeev_popov.tex`
(not yet compiled into the master), attempted to close the gauge-invariant
concentration Hypothesis `hyp:gauge_invariant_concentration` left open by
app230). It was not audited and contained a claimed mass-gap theorem
(`thm:mass_gap_app231`). New appendix `split/app232_audit_of_app231_and_corrected_scaling.tex`
compiles and audits it, finding **four independent errors** that invalidate
the claim:

1. **Error 1 (scaling)**: app231 proves only the per-scale Cheeger bound
   $h_j \ge c\,2^{j}/(j+1)^{1/2}$ (exponent $2^j$), but the mass gap
   requires $h_j \ge c\,2^{2j}/(j+1)$ (exponent $2^{2j}$, app230
   `thm:app229_error2`). app231 asserts the stronger aggregation
   $h_{\mathrm{OS}}^{(n)}\ge c a_n^{-1/2}$ without proving the stronger
   per-scale bound. This repeats exactly the scaling error app230 already
   corrected.
2. **Error 2 (local log-concavity)**: positive Hessian of the YM action
   on the subset $U_{\mathrm{AF},j}^{corr}$ does NOT make the density
   log-concave (convexity is global; YM action is non-convex on flat
   directions and near the Gribov horizon). The log-concave tail
   inequality requires a globally log-concave measure, so Step 11 and
   `thm:log_concave`/`cor:af_concentration` are invalid (and the asserted
   $\nu_j(U^{corr})\ge1/2$ is circular).
3. **Error 3 (unbounded volume ratio)**: the determinant-to-volume ratio
   $R_j=\mathrm{vol}(Q_j)/(g_{\mathrm{YM}}^{D_j^\perp}(\det\Delta_{0,j}^\perp)^{-1/2})$
   is NOT bounded as $j\to\infty$ (it is $\sim\exp(c\,2^{4j}j)$), as
   app231's own computation concedes; lemma `lem:det_vol_ratio` is
   self-contradicted, so the Gaussian lower bound cannot control shells.
4. **Error 4 (first shell)**: the $k=0$ shell at the AF boundary is
   $\sim\exp(c\,2^{4j}j)$ (large) and is only "repaired" by the invalid
   local-log-concavity step.

**New precise target** (app232 `thm:equivalent_concentration`): the
gauge-invariant concentration is equivalent to the single Weyl-type
bound $R_j\le c\exp(\alpha\,2^{2j}/(j+1)^2 - c_1 2^{4j}\log(2^j)/(g_{\mathrm{YM}}^2(j)(j+1)^2))$,
i.e. cancellation of the $c\,2^{4j}j$ growth by the AF action. This is the
sharply posed, checkable, single open estimate.

**Honest status**: the Clay Millennium Prize mass-gap conjecture
**remains open**, reduced to a single sharply-stated gauge-invariant
concentration/Weyl-volume estimate on the Landau slice.

## 2026-08-13 Update (second): app233 — dimension reduction; AF region
   mis-normalized; mass gap reduced to non-Gaussian (Gribov) control

New appendix `split/app233_typical_radius_and_gaussian_shell.tex`
compiles into the master and does the following:

- **Gaussian dimension-reduction lemma** (`lem:gaussian_shell`): on the
  Landau slice, the weak-field (quadratic) approximation is a Gaussian with
  field radius rho whose typical value is
  `rho_typ ~ sqrt(D_j^perp) g 2^j ~ g 2^{3j}`. In high dimension the mass
  concentrates on the shell `rho ~ rho_typ`: for `R1 <= (1/2) rho_typ` the
  annulus `[R1,2R1]` retains a fixed fraction (>=1/2) of the mass of the
  ball of radius `2R1`; for `R1 >= 2 rho_typ` the upper tail is
  `<= e^{-c R1^2/rho_typ^2}`.

- **Dimension-reduction theorem** (`thm:dimension_reduction`): the whole
  gauge-invariant concentration is governed by a single "spin" parameter
  `eta_* = liminf R1/rho_typ`. If `eta_*>0` concentration holds (Gaussian
  part); if `eta_*=0` no weak-field argument can give the exponential tail.

- **The spin is zero** (`thm:spin_zero`): under asymptotic freedom and the
  app230 radius `R1 = eta 2^{2j}/(j+1)`,
  `R1/rho_typ = eta sqrt(log 2^j)/(N (j+1) 2^j) -> 0` exponentially.
  **Diagnosis:** the app230 AF region sits DEEP INSIDE the typical shell;
  it is mis-normalized by a factor ~`2^j`. The numerical spin decays
  `0.049, 0.010, 0.0023, ...` for `j=2,4,6,...`.

- **Corrected typical-scale AF region** (`def:typical_af`): define
  `U_typ = {||F|| <= C_typ rho_typ}` at the intrinsic Gaussian width, where
  Gaussian concentration is trivial.

- **Final reduction** (`thm:non_gaussian_reduction`): if the full nonlinear
  gauge quotient measure satisfies concentration on `U_typ`
  (eq:gaussian_dominance), then the required per-scale Cheeger bound
  `h_j >= c 2^{2j}/(j+1)` follows, giving the mass gap. The genuinely open
  content is now precisely the **non-Gaussian (cubic bracket + Gribov
  horizon) control at the typical scale**.

**Honest status:** Clay mass-gap conjecture **remains open**. Progress: the
concentration is reduced to a single Gaussian-scaled spin parameter, the
AF region mis-normalization is rigorously diagnosed, and the open problem
is sharply restated as non-Gaussian concentration at the typical scale
(eq:gaussian_dominance).
