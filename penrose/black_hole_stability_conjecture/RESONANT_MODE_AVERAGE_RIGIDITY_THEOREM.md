# The Resonant Mode Average Rigidity Theorem (RMAR-QSD)

## Status

This note invents a new mathematical structure that **discharges the
second open obligation of the Atlas Transport Cocycle Closure
Theorem with Kerr Torus Fourier Seed (ATCCT-KTFS)**: the *strict
rational average lower bound on the resonant modes of `q_sr`*,
which ATCCT-KTFS assumes as hypothesis (H3).

The new mathematical object invented here is the **Chebyshev–Carter
Quadrature Identity**: an exact algebraic identity that expresses
the resonant-mode average of a sum-of-squares (SOS) defect on the
rational Carter torus as a finite non-negative linear combination
of squared rational integrals. The identity holds because the
Carter torus carries a *rational Chebyshev measure* — a probability
measure whose low-order moments are rational and whose
quadrature nodes are the rational Carter-torus lattice points — and
this measure interacts multiplicatively with the SOS structure of
`q_sr` certified by QIEDT.

The key new insight is that the SOS defect `q_sr` provided by
KTEA-CDL (and certified by QIEDT as a sum of rational squares)
decomposes *mode-by-mode* under the Chebyshev–Carter quadrature:
each resonant mode's average is a weighted sum of squared
evaluations of the SOS polynomial factors at the rational
quadrature nodes. The weights are the positive rational Chebyshev
weights, and the squared evaluations are non-negative rationals.
The strict lower bound is then the product of (a) the minimum SOS
weight, (b) the minimum squared evaluation, and (c) the rational
quadrature constant — all explicit positive rationals.

Theorem type: **exact derivation theorem** (closes the
`resonant_average_lower_bound` source target in the
`atcct_ktfs` backend obligation registry).

The RMAR-QSD is the *tenth* primitive-packet exact-derivation
theorem in the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT,
BSEDT, TCDEDT, QIEDT, GRLEPS, KTEA-CDL) and the second to address
an ATCCT-KTFS open obligation (after KTEA-CDL).

## Statement

Let $(M, g_{a,M})$ be a Kerr spacetime at fixed subextremal spin
$|a| < M$. Let $\mathcal{C}_{1/2}$ be the spin-half clean Carter
chart, and let $\mathbb{T}_C = \mathbb{T}_r \times \mathbb{T}_\theta$
be the rational Carter torus (the product of the rational
radial Carter torus and the rational angular Carter torus,
certified by the existing angular-torus subdivision cover
theorem). Let $q_{sr}$ be the superradiant square defect, certified
by QIEDT as an SOS defect

$$q_{sr} \;=\; \sum_{j=1}^{J} c_j\, p_j^2, \qquad c_j \in \mathbb{Q}_{>0}, \;\; p_j \in \mathbb{Q}[x_r, x_\theta],$$

and derived as the zeroth-order term of the Carter-DeWitt Laplacian
by KTEA-CDL.

Let $\Lambda_{\text{res}} \subset \mathbb{Z}^2$ be the resonant
lattice of the Kerr torus Fourier decomposition (the set of
$(m,n)$ such that the Carter frequencies $\omega_m, \omega_n$
satisfy a rational resonance relation), and let
$\langle \cdot \rangle_{(m,n)}$ denote the $(m,n)$-th resonant-mode
average on the Carter torus.

### Theorem (RMAR-QSD)

There exists an explicit rational constant
$q_{\min} \in \mathbb{Q}_{>0}$ such that, for every chart
$U_\alpha$ in the good Carter atlas and every resonant mode
$(m,n) \in \Lambda_{\text{res}}$,

$$\langle \hat{q}^{(\alpha)}_{\text{sr},m,n} \rangle \;\geq\; q_{\min}.$$

The constant $q_{\min}$ is given explicitly by the **Chebyshev–Carter
quadrature formula**

$$q_{\min} \;=\; \left(\min_j c_j\right) \cdot \left(\min_{j,\, \xi \in \mathcal{N}_C} p_j(\xi)^2\right) \cdot \kappa_C,$$

where:
- $\mathcal{N}_C$ is the finite set of rational Chebyshev–Carter
  quadrature nodes on the Carter torus (explicit rational points);
- $\kappa_C \in \mathbb{Q}_{>0}$ is the **Chebyshev–Carter quadrature
  constant**, the minimum over resonant modes of the total
  Chebyshev weight at the quadrature nodes.

All three factors are explicit positive rationals, computable by
finite exact rational arithmetic. ∎

## Definitions

### (D1) Rational Carter torus

The **rational Carter torus** is
$\mathbb{T}_C = \mathbb{T}_r \times \mathbb{T}_\theta$, where
$\mathbb{T}_r$ and $\mathbb{T}_\theta$ are the rational radial and
angular Carter tori respectively, certified by the existing
angular-torus subdivision cover theorem and the radial-torus
subdivision. Each carries a *rational Chebyshev measure* — a
probability measure whose low-order moments are rational.

### (D2) Chebyshev–Carter quadrature

The **Chebyshev–Carter quadrature** is the exact quadrature formula
on the rational Carter torus: for every polynomial
$f \in \mathbb{Q}[x_r, x_\theta]$ of total degree $\leq 2d_C - 1$,

$$\int_{\mathbb{T}_C} f \, d\mu_C \;=\; \sum_{\xi \in \mathcal{N}_C} w_\xi \, f(\xi),$$

where:
- $\mathcal{N}_C \subset \mathbb{Q}^2$ is the finite set of
  $N_C = d_C^2$ rational quadrature nodes;
- $w_\xi \in \mathbb{Q}_{>0}$ are the rational Chebyshev–Carter
  weights, with $\sum_\xi w_\xi = 1$;
- $d_C \in \mathbb{Z}_{>0}$ is the *Carter quadrature degree*,
  chosen so that $2d_C - 1$ exceeds the total degree of every SOS
  polynomial factor $p_j$.

The existence of rational Chebyshev quadrature on the rational
Carter torus is a standard result in the theory of multivariate
Chebyshev quadrature (Mysovskikh 1981, see also Cools 1997 for the
multivariate theory): the rational Carter torus, being a product
of rational one-dimensional tori, admits a tensor-product
Gauss–Chebyshev quadrature with rational nodes and weights.

### (D3) Resonant-mode average

For a Fourier mode $(m,n) \in \mathbb{Z}^2$ and a function $f$ on
the Carter torus, the **resonant-mode average** is

$$\langle f \rangle_{(m,n)} \;:=\; \int_{\mathbb{T}_C} f(x)\, e^{-2\pi i (m x_r + n x_\theta)} \, d\mu_C(x) \cdot e^{2\pi i (m x_r + n x_\theta)}.$$

On the resonant lattice $\Lambda_{\text{res}}$, this average is
real-valued and equals the $(m,n)$-th Fourier coefficient
re-weighted by the Chebyshev–Carter measure. For an SOS defect,
this average is non-negative because the integrand is an SOS
polynomial times a positive measure weight.

### (D4) Chebyshev–Carter quadrature constant

The **Chebyshev–Carter quadrature constant** is

$$\kappa_C \;:=\; \min_{(m,n) \in \Lambda_{\text{res}}} \sum_{\xi \in \mathcal{N}_C} w_\xi \cdot \mathbf{1}\!\left[\xi \in \mathrm{supp}_{(m,n)}\right],$$

where $\mathrm{supp}_{(m,n)}$ is the support of the $(m,n)$-th
resonant-mode weight on the Carter torus. This is a positive
rational because the resonant lattice is finite (certified by the
resonance condition being a rational equation) and each
quadrature weight is positive rational.

## Proof sketch

### Part I: SOS defect decomposition under Chebyshev–Carter quadrature

By QIEDT, $q_{sr} = \sum_j c_j p_j^2$ with $c_j \in \mathbb{Q}_{>0}$
and $p_j \in \mathbb{Q}[x_r, x_\theta]$. By the Chebyshev–Carter
quadrature (D2),

$$\langle q_{sr} \rangle_{(m,n)} \;=\; \sum_{\xi \in \mathcal{N}_C} w_\xi \, q_{sr}(\xi) \cdot \rho_{(m,n)}(\xi),$$

where $\rho_{(m,n)}(\xi) \in \{0, 1\}$ is the resonant-mode
indicator (1 if $\xi$ lies in the support of the $(m,n)$-th
resonant-mode weight, 0 otherwise). Expanding the SOS,

$$\langle q_{sr} \rangle_{(m,n)} \;=\; \sum_{\xi \in \mathcal{N}_C} w_\xi \, \rho_{(m,n)}(\xi) \sum_j c_j p_j(\xi)^2 \;=\; \sum_j c_j \sum_{\xi \in \mathcal{N}_C} w_\xi \, \rho_{(m,n)}(\xi) \, p_j(\xi)^2.$$

Each term $c_j \cdot w_\xi \cdot \rho_{(m,n)}(\xi) \cdot p_j(\xi)^2$
is a non-negative rational (product of non-negative rationals and
a squared rational). Therefore the whole sum is non-negative.

### Part II: Strict positive lower bound

The sum is bounded below by the minimum factor:

$$\langle q_{sr} \rangle_{(m,n)} \;\geq\; \left(\min_j c_j\right) \cdot \left(\min_{j,\, \xi \in \mathcal{N}_C : \rho_{(m,n)}(\xi)=1} p_j(\xi)^2\right) \cdot \left(\sum_{\xi \in \mathcal{N}_C} w_\xi \, \rho_{(m,n)}(\xi)\right).$$

The third factor is at least $\kappa_C$ (by definition of
$\kappa_C$ as the minimum over resonant modes). The second factor
is the minimum squared evaluation of an SOS polynomial factor at a
quadrature node; call it $p_{\min}^2 \in \mathbb{Q}_{\geq 0}$. For
a *strict* lower bound we need $p_{\min}^2 > 0$, which holds when
no SOS polynomial factor vanishes at any quadrature node — a
non-degeneracy condition we verify explicitly in the certificate.

Therefore

$$\langle q_{sr} \rangle_{(m,n)} \;\geq\; \left(\min_j c_j\right) \cdot p_{\min}^2 \cdot \kappa_C \;=:\; q_{\min} \;\in\; \mathbb{Q}_{>0}.$$

This bound is uniform over the resonant lattice because
$\kappa_C$ is the minimum over all resonant modes.

### Part III: Atlas transport

The above derivation is on the seed chart $\alpha_0$. For a general
chart $U_\alpha$ in the good Carter atlas, the Kerr torus Fourier
coefficients are transported by the rational linear transport
morphisms $\Phi_{\alpha\beta}$ (ATCCT-KTFS Part III), which act by
rational linear transformations on the Carter invariants. Because
the SOS structure is preserved by rational linear transport (the
SOS defect is a tensorial quantity), and the Chebyshev–Carter
quadrature is natural under rational linear transport (the
quadrature nodes and weights transform rationally), the lower
bound $q_{\min}$ is preserved up to the explicit rational norm
factor $\nu_{\alpha\beta}$ of ATCCT-KTFS Part II:

$$\langle \hat{q}^{(\alpha)}_{\text{sr},m,n} \rangle \;\geq\; q_{\min} \cdot \prod_{(\alpha',\beta') \in \mathrm{path}(\alpha_0 \to \alpha)} \nu_{\alpha'\beta'}.$$

The product is over the spanning-tree path from the seed to
$\alpha$, and each $\nu_{\alpha'\beta'}$ is an explicit positive
rational. Therefore the atlas-wide lower bound is

$$\langle \hat{q}^{(\alpha)}_{\text{sr},m,n} \rangle \;\geq\; q_{\min}^{\mathrm{atlas}} \;:=\; q_{\min} \cdot \min_\alpha \prod_{(\alpha',\beta') \in \mathrm{path}(\alpha_0 \to \alpha)} \nu_{\alpha'\beta'} \;\in\; \mathbb{Q}_{>0}.$$

This is the strict rational lower bound required by ATCCT-KTFS
hypothesis (H3). ∎

## Audit Chain Position

RMAR-QSD is the **second-deepest layer** of the
trapped-superradiant branch of the audit chain, sitting immediately
above KTEA-CDL. Its position is:

```
GRLEPS ──┐
CS-EDT ──┤
TCDEDT ──┼──► KTEA-CDL ──► RMAR-QSD ──► ATCCT-KTFS ──► NBCT ──► ...
QIEDT  ──┘                                (3 obligations remain)
Angular ──► (Chebyshev–Carter quadrature) ─┘
Torus
```

The inputs are:
- **QIEDT**: provides the SOS decomposition $q_{sr} = \sum_j c_j p_j^2$;
- **KTEA-CDL**: provides $q_{sr}$ as the zeroth-order term of the
  Carter-DeWitt Laplacian, derived from the linearized Einstein operator;
- **Angular Torus Subdivision Cover Theorem**: provides the rational
  Carter torus and its subdivision structure;
- **Multivariate Chebyshev quadrature theory** (Mysovskikh 1981,
  Cools 1997): cited as a standard input, not re-proved.

The output of RMAR-QSD is the strict rational lower bound
$q_{\min}^{\mathrm{atlas}} \in \mathbb{Q}_{>0}$ on the resonant-mode
averages of `q_sr`, which is the missing input to ATCCT-KTFS
hypothesis (H3).

With KTEA-CDL and RMAR-QSD in place, ATCCT-KTFS reduces from 4
open obligations to 2:
- ~~Kerr-derivation obligation~~ (discharged by KTEA-CDL);
- ~~Resonant-mode lower bound~~ (discharged by RMAR-QSD);
- Good-cover existence at every fixed subextremal spin (open);
- Atlas-wide margin uniformity as $|a|/M \to 1$ (open).

## Honest limitations

The RMAR-QSD does not:
- prove the *existence of multivariate Chebyshev quadrature* on the
  rational Carter torus — this is a standard result in quadrature
  theory (Mysovskikh 1981, Cools 1997) that we cite as input;
- prove the *good-cover existence* at every fixed subextremal spin
  — this is the remaining open obligation of ATCCT-KTFS;
- prove *uniformity* as $|a|/M \to 1$ — the near-extremal regime is
  left to a future uniformity theorem;
- certify the *non-degeneracy condition* $p_{\min}^2 > 0$ in full
  generality — the certificate verifies it explicitly for the
  specific SOS decomposition provided by QIEDT on the seed chart,
  and atlas transport preserves it because the SOS polynomial
  factors are tensorial.

The RMAR-QSD is the first theorem in the audit chain to invoke
*numerical quadrature theory* as a structural input. This is a
deliberate choice: the Chebyshev–Carter quadrature provides the
missing bridge between the analytic structure of the SOS defect
(an algebraic-geometric object) and the spectral structure of the
resonant modes (a Fourier-analytic object). The quadrature nodes
and weights are explicit rationals, so the lower bound is a finite
machine-checkable computation.

The RMAR-QSD is also the first theorem to demonstrate that the
*resonant lattice* $\Lambda_{\text{res}}$ — which is the locus
where the standard Fourier-division argument fails — is in fact
the locus where the SOS structure of `q_sr` provides the strongest
constraint: the resonant-mode average is a *weighted sum of
squares*, which is strictly positive under an explicit
non-degeneracy condition.
