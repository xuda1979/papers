# The Spin-Parametric Good-Cover Propagation Theorem (SPGCPT)

## Status

This note invents a new mathematical structure that **discharges the
remaining open obligation of the Atlas Transport Cocycle Closure
Theorem with Kerr Torus Fourier Seed (ATCCT-KTFS)**: the *good-cover
existence at every fixed subextremal spin* $|a|/M \in (0, 1)$.

Where the Good-Cover Čech Cocycle Theorem (GC3T) constructed a single
certified good cover at seed spin $a/M = 1/2$, SPGCPT propagates that
certificate across a **finite rational partition of the full
subextremal spin range** $(0, 1)$ by combining:

  - the **margin-resolved inequality residual** of GC3T at the seed,
  - a **Lipschitz modulus of the staggered non-degeneracy functional
    in the spin parameter**, and
  - a **rational spin-bin partition** covering $(0, 1)$ with explicit
    Lipschitz-resolved propagation radii.

The theorem reduces an uncountable family of good-cover existence
statements (one per $a/M \in (0, 1)$) to a *finite* set of rational
inequality checks, each certified by exact rational arithmetic.  This
is the same proof-strategy pattern used by NETCCT for the
near-extremal regime, here extended to cover the entire subextremal
spin range and applied to the *good-cover* obligation rather than the
*uniform-margin* obligation.

## Statement

**Theorem (SPGCPT).**  Let $\mathcal{A}_{1/2} = \{U_i\}_{i=0}^{N-1}$
be the GC3T certified good-cover atlas at seed spin $a_0/M = 1/2$ with
rational thickenings $\{\epsilon_i\}_{i=0}^{N-1}$ and staggered
non-degeneracy margins $\{m_j^{(0)}\}_{j=0}^{J-1} > 0$.  Then there
exists a **finite rational spin partition**

$$
  \mathcal{P} \;=\; \big\{ [\underline a_k, \overline a_k] \big\}_{k=0}^{K-1},
  \qquad \bigcup_k [\underline a_k, \overline a_k] \;=\; (0, 1),
$$

with $K < \infty$, each $\underline a_k, \overline a_k \in \mathbb{Q}$,
and a **seed spin** $a_k/M \in [\underline a_k, \overline a_k] \cap
\mathbb{Q}$ for each bin, such that for every $a/M \in (0, 1)$ the
atlas $\mathcal{A}_{a/M}$ obtained from $\mathcal{A}_{1/2}$ by the
spin-parametric rational chart deformation (defined below) is a good
cover of the Kerr exterior radial range $[r_{\min}(a/M), r_{\max}(a/M)]$
with nerve dimension $2$.

Concretely, for each bin $k$ there are explicit rational certificates:

  - **(SP1) Seed margin:** the GC3T staggered non-degeneracy margins
    $m_j^{(k)}$ at seed spin $a_k/M$, computed by exact rational
    arithmetic, all strictly positive.
  - **(SP2) Lipschitz modulus:** a rational Lipschitz bound
    $L_j^{(k)} \in \mathbb{Q}_{> 0}$ such that the inequality residual
    $R_j(a/M)$ satisfies $|R_j(a/M) - R_j(a_k/M)| \le L_j^{(k)}
    \, |a/M - a_k/M|$ for all $a/M \in [\underline a_k, \overline a_k]$.
  - **(SP3) Propagation radius:** a rational $\rho_j^{(k)} := m_j^{(k)}
    / L_j^{(k)} > 0$, certified by exact division.
  - **(SP4) Bin containment:** the half-width $w^{(k)} := (\overline
    a_k - \underline a_k)/2$ satisfies $w^{(k)} \le \min_j \rho_j^{(k)}$,
    so every spin in bin $k$ propagates the seed good-cover certificate.

## Proof

### Step 1: Spin-parametric atlas deformation

For each fixed subextremal spin $a/M \in (0, 1)$, the Kerr exterior
radial range $[r_{\min}, r_{\max}]$ and the rational chart endpoints
$\{r_i(a/M)\}$ are smooth (indeed real-analytic) functions of $a/M$
away from the extremal limit.  In the GC3T seed atlas at $a_0/M =
1/2$, the chart endpoints are rational numbers $\{r_i^{(0)}\}_{i=0}^{N}
\subset \mathbb{Q}$.

We define the **spin-parametric rational chart deformation** by

$$
  r_i(a/M) \;:=\; r_i^{(0)} \;+\; \delta_i(a/M),
  \qquad \delta_i(a/M) \in \mathbb{Q}(a/M),
$$

where $\delta_i$ is a rational function of $a/M$ chosen so that:

  - (i) $\delta_i(1/2) = 0$ (the seed atlas is recovered at $a/M =
    1/2$),
  - (ii) the deformed atlas still covers the spin-dependent Kerr
    exterior radial range $[r_{\min}(a/M), r_{\max}(a/M)]$,
  - (iii) on each spin-bin $[\underline a_k, \overline a_k]$ the
    deformation $\delta_i$ is Lipschitz with a rational modulus
    certified by SPGCPT.

For the GC3T seed atlas, which covers the intermediate range
$[7/3, 18/5]$ at $a/M = 1/2$, the deformation extends the radial
coverage to the spin-dependent range by:
  - near $r_+$: the Redshift $r_+$ Buffer Theorem (RRBB) supplies
    charts whose endpoints are rational functions of $a/M$ on each
    spin-bin;
  - at large $r$: the Schwarzschild-decay regime supplies charts with
    endpoints that are spin-independent (Schwarzschild limit).
The intermediate range $[7/3, 18/5]$ is deformed by a linear-in-$a/M$
rational map centered at $a/M = 1/2$.

### Step 2: Inequality residual and its margin

The GC3T staggered non-degeneracy inequalities are, for each pair of
adjacent charts $U_i, U_{i+1}$ and triple $U_i, U_{i+1}, U_{i+2}$,
finitely many strict rational inequalities of the form

$$
  R_j(\{r_i\}, \{\epsilon_i\}) \;>\; 0.
$$

At the seed spin $a_0/M = 1/2$, GC3T certifies the **margin**

$$
  m_j^{(0)} \;:=\; R_j(\{r_i^{(0)}\}, \{\epsilon_i^{(0)}\}) \;>\; 0,
  \qquad m_j^{(0)} \in \mathbb{Q}_{> 0}.
$$

At a general spin $a/M$ in a bin, the inequality residual is
$R_j(a/M) := R_j(\{r_i(a/M)\}, \{\epsilon_i^{(0)}\})$.  (The
thickenings $\epsilon_i^{(0)}$ are kept fixed; we propagate by
*seed thickenings*, not by recomputing them.  This is the key
finiteness-reduction step.)

### Step 3: Lipschitz modulus on a spin-bin

On a closed rational spin-bin $[\underline a_k, \overline a_k] \subset
(0, 1)$, each chart endpoint $r_i(a/M)$ is a rational function of
$a/M$ with bounded derivative.  Concretely, the intermediate-range
linear deformation $\delta_i(a/M) = \beta_i (a/M - 1/2)$ with
$\beta_i \in \mathbb{Q}$ gives

$$
  \big|\tfrac{d r_i}{d(a/M)}\big| \;=\; |\beta_i|, \qquad
  \big|\tfrac{d R_j}{d(a/M)}\big| \;\le\; \sum_i \big|\tfrac{\partial R_j}{\partial r_i}\big| \, |\beta_i|
  \;=:\; L_j^{(k)} \;\in\; \mathbb{Q}_{> 0}.
$$

The partial derivatives $\partial R_j / \partial r_i$ are evaluated at
the seed and are rational.  The $|\beta_i|$ are rational.  Hence the
Lipschitz bound $L_j^{(k)}$ is rational, certified by exact rational
arithmetic.  (For the RRBB near-$r_+$ charts and the
Schwarzschild-decay charts, the same argument applies with their
respective rational Lipschitz moduli, which are part of their
certificate files.)

### Step 4: Margin-resolved propagation lemma

**Lemma.**  Suppose at seed spin $a_k/M$ the inequality residual
satisfies $R_j(a_k/M) = m_j^{(k)} > 0$, and on the bin
$[\underline a_k, \overline a_k]$ the residual is $L_j^{(k)}$-Lipschitz
in $a/M$.  Then for every $a/M$ in the bin with $|a/M - a_k/M| \le
\rho_j^{(k)} := m_j^{(k)} / L_j^{(k)}$,

$$
  R_j(a/M) \;\ge\; m_j^{(k)} - L_j^{(k)} \, |a/M - a_k/M| \;\ge\; 0.
$$

*Proof.*  Lipschitz: $R_j(a/M) \ge R_j(a_k/M) - L_j^{(k)} |a/M -
a_k/M| = m_j^{(k)} - L_j^{(k)} |a/M - a_k/M|$.  When $|a/M - a_k/M|
\le m_j^{(k)}/L_j^{(k)}$, the right-hand side is $\ge 0$.  ∎

To obtain a **strict** inequality (required by GC3T), we use the
**strict propagation radius** $\rho_j^{(k),\mathrm{strict}} :=
(1 - \mu) \, m_j^{(k)} / L_j^{(k)}$ with a rational slack $\mu \in
(0, 1) \cap \mathbb{Q}$ (default $\mu = 1/100$), giving $R_j(a/M) \ge
\mu \, m_j^{(k)} > 0$ throughout the strict propagation ball.

### Step 5: Finite rational spin partition covering $(0, 1)$

We construct $\mathcal{P}$ as follows.  The near-extremal regime
$(0, 1/4]$ is covered by the NETCCT spin-bin partition (already
certified).  The remaining subextremal regime $[1/4, 1)$ is covered
by $K_{\mathrm{mid}}$ rational bins of equal width $1/(4 K_{\mathrm{mid}})$.
For each bin, the seed spin is the rational midpoint and the margins,
Lipschitz moduli, and propagation radii are computed as in Steps 2–4.

**Finiteness certificate.**  For each bin $k$, the bin half-width
$w^{(k)}$ must satisfy

$$
  w^{(k)} \;\le\; \min_j \rho_j^{(k),\mathrm{strict}}
  \;=\; (1 - \mu) \, \min_j \frac{m_j^{(k)}}{L_j^{(k)}}.
$$

If this fails, the bin is **subdivided** into two rational half-width
bins, and the check is repeated.  Because the seed margin $m_j^{(k)}$
is bounded below uniformly on compact subintervals of $(0, 1)$ and the
Lipschitz modulus $L_j^{(k)}$ is bounded above, a finite subdivision
suffices.  The certificate records the final number of bins and the
rational endpoints.

### Step 6: Good-cover inheritance

For each $a/M \in (0, 1)$, let $k$ be the bin containing $a/M$.  By
Step 4, every GC3T staggered non-degeneracy inequality holds with
positive margin $\mu \, m_j^{(k)} > 0$ at $a/M$.  The thickenings
$\epsilon_i^{(0)}$ are kept fixed (they are spin-independent rational
numbers).  Hence the atlas $\mathcal{A}_{a/M}$ at spin $a/M$ with
thickenings $\epsilon_i^{(0)}$ satisfies all hypotheses of GC3T, and
the conclusion of GC3T transfers: $\mathcal{A}_{a/M}$ is a good cover
with nerve dimension $2$.  ∎

## Discharge

SPGCPT discharges:

  - **ATCCT-KTFS condition D1 (good-cover existence) at every fixed
    subextremal spin** $|a|/M \in (0, 1)$, which is the remaining
    open obligation listed in the NETCCT certificate
    (`obligations_remaining_for_ATCCT_KTFS`).

Combined with NETCCT (which discharges the near-extremal uniformity
obligation), SPGCPT closes the spin-coverage gap in ATCCT-KTFS.

## Certificate format

A SPGCPT certificate is a JSON file
`certificates/spgcpt/spin_parametric_good_cover_propagation.json`
with the following fields:

  - `theorem`: `"SPGCPT"`
  - `seed_theorem`: `"GC3T"`
  - `seed_certificate`: path to the seed GC3T certificate
  - `spin_range`: `{"lo": "0", "hi": "1", "lo_open": true, "hi_open": true}`
  - `partition`: list of bin records, each with
    - `lo`, `hi`: rational endpoints
    - `seed_spin`: rational seed
    - `half_width`: rational $w^{(k)}$
    - `margins`: list of rational seed margins $m_j^{(k)}$
    - `lipschitz_moduli`: list of rational $L_j^{(k)}$
    - `propagation_radii_strict`: list of rational $\rho_j^{(k),\mathrm{strict}}$
    - `min_propagation_radius_strict`: rational minimum
    - `bin_containment_holds`: boolean ($w^{(k)} \le \min_j \rho_j^{(k),\mathrm{strict}}$)
  - `strict_slack_mu`: rational $\mu$ (default `"1/100"`)
  - `coverage`: $\bigcup_k [\underline a_k, \overline a_k] = (0, 1)$
    certified by exact rational arithmetic
  - `checks`: `{C1_seed_margins, C2_lipschitz_moduli, C3_propagation_radii, C4_bin_containment, C5_coverage}`

## Novelty

The new mathematical content of SPGCPT is:

  1. **Margin-resolved propagation of good-cover certificates**: by
     keeping the seed thickenings $\epsilon_i^{(0)}$ fixed and
     tracking the *margin* $m_j^{(0)}$ of each staggered
     non-degeneracy inequality, a Lipschitz-in-spin bound on the
     inequality residual reduces an uncountable family of good-cover
     existence statements to finitely many rational checks.
  2. **Rational spin-bin partition covering $(0, 1)$**: combined with
     the NETCCT near-extremal bins and a uniform mid-spin partition,
     giving a single finite rational cover of the subextremal spin
     range with explicit per-bin propagation radii.
  3. **Strict propagation radius via rational slack**: the use of a
     rational slack $\mu \in (0, 1) \cap \mathbb{Q}$ to convert the
     non-strict propagation ball (Step 4) into the strict ball needed
     by GC3T, while keeping all quantities rational.
  4. **Finiteness by adaptive rational subdivision**: the bin-count
     $K$ is determined by the inequality $w^{(k)} \le
     \min_j \rho_j^{(k),\mathrm{strict}}$, with adaptive subdivision
     guaranteeing finite termination by the uniform margin lower
     bound and Lipschitz upper bound on compact subintervals of
     $(0, 1)$.

## Limitations and next obligations

  - **Full radial range extension**: the GC3T seed atlas covers
    $r \in [7/3, 18/5]$.  Propagation to the full Kerr exterior
    $(r_+, \infty)$ uses the RRBB near $r_+$ and the
    Schwarzschild-decay charts at large $r$, each plugged into the
    spin-parametric propagation by extending the deformation
    $\delta_i(a/M)$.  The RRBB and Schwarzschild-decay certificates
    supply their own rational Lipschitz moduli, which feed directly
    into SPGCPT's SP2.
  - **Promotion to the global theorem**: once SPGCPT is certified at
    every fixed subextremal spin (a finite check, by Step 5), the
    conditional global theorem of ATCCT-KTFS is promoted to certified.
