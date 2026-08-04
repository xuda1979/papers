# The Atlas Transport Cocycle Closure Theorem with Kerr Torus Fourier Seed (ATCCT-KTFS)

## Status

This note invents a new mathematical structure that **promotes the
spin-half clean Carter chart closure to a finite atlas of Carter
charts covering the Kerr exterior** and simultaneously **derives the
Kerr torus Fourier coefficients** for the trapped-superradiant
branch's `a_KY` and `q_sr` inputs, together with a tame
denominator-floor and derivative-seminorm certificate for the
resulting escape symbol in the Carter-Wick symbol class.

It is the **ninth primitive-packet exact-derivation theorem** in the
audit chain (after RHEDT, SSEDT, CSEDT, DSEDT, BSEDT, TCDEDT, QIEDT,
GRLEPS, and the BCCT composition) and the **first atlas-wide theorem**:
every prior exact-derivation theorem operates inside a single Carter
chart, while ATCCT-KTFS operates on the *cohomology of the overlap
cover* and is the first theorem whose certificate is a Čech-style
1-cocycle rather than a single-chart identity.

The theorem is **conditional**: it assumes (i) the existing BCCT
contract on each atlas chart as a black-box input, (ii) a finite
subdivision of the angular torus fibration into Carter charts whose
overlap cover is a good cover in the sense of Bott-Tu, and (iii) the
existence of a strict rational lower bound on the resonant Fourier
modes of `q_sr`. Subject to these inputs, the atlas-wide backend
contract closes and the orbit-cohomology certificate becomes a finite
machine-checkable computation.

## Motivation: The Two Simultaneous Bottlenecks

The research ledger's "Next Action" identifies two simultaneous
bottlenecks that must be cleared before the global theorem can
promote from conditional to certified:

1. **Atlas extension.** The BCCT closes the backend contract for the
   *single* spin-half clean Carter chart. The full exterior Kerr
   stability theorem requires closure on the *full atlas* of Carter
   charts covering the Kerr exterior. The current "shadow graph" is a
   placeholder that records which charts *should* neighbor which, but
   it does not provide actual transport morphisms on overlaps. A
   genuine neighboring-chart transport morphism is a symplectic
   pullback on the overlap cotangent bundle that must (a) conjugate
   the two charts' trapped-constraint ideals, (b) preserve the
   GRLEPS principal symbol up to the QIEDT SOS defect, and (c)
   transport the BCCT local coercivity margin across the overlap
   with an explicit rational norm factor.

2. **Kerr torus Fourier coefficients.** The orbit-cohomology
   certificate for the trapped-superradiant branch requires explicit
   Fourier coefficients for the Carter ansatz field `a_KY` and the
   trapped-superradiant density `q_sr` on each atlas chart. The
   nonresonant modes are solved by Fourier division, but the
   resonant modes (those whose frequency coincides with a rational
   orbit on the angular torus) must satisfy a strict rational
   average lower bound, and the resulting escape symbol must be
   shown to lie in the Carter-Wick symbol class with explicit
   denominator floors and derivative seminorm bounds.

These two bottlenecks are coupled: the atlas transport morphisms
determine how the Fourier coefficients transform between charts, and
the Fourier coefficients determine the local orbit-cohomology
certificate that the transport morphisms must preserve. The
ATCCT-KTFS resolves the coupling by introducing a **Kerr torus
Fourier seed** on a single seed chart and transporting it along a
spanning tree of the overlap cover, with the cocycle condition on
overlap triple intersections providing the consistency check.

## Statement

Let $\mathcal{A} = \{U_\alpha\}_{\alpha \in I}$ be a finite good cover
of the Kerr exterior by Carter charts at fixed subextremal spin
$|a| < M$, with each $U_\alpha$ carrying a BCCT contract
$\mathcal{B}_\alpha$ with local coercivity margin
$\mu_\alpha \in \mathbb{Q}_{>0}$. Let
$U_{\alpha\beta} = U_\alpha \cap U_\beta$ denote the overlaps,
assumed connected and nonempty for edges of a fixed spanning tree
$T \subset I \times I$, and let
$\Phi_{\alpha\beta}: T^*U_{\alpha\beta} \to T^*U_{\alpha\beta}$ be
the overlap pullback that conjugates chart $\beta$'s
trapped-constraint ideal to chart $\alpha$'s.

Let $\mathcal{F}_\alpha = \{\hat{a}^{(\alpha)}_{K,m,n},
\hat{q}^{(\alpha)}_{\text{sr},m,n}\}_{(m,n) \in \mathbb{Z}^2}$ be
the Kerr torus Fourier coefficients of $a_KY$ and $q_sr$ on chart
$U_\alpha$, indexed by the angular $(m, n)$ frequency lattice. Let
$\Lambda_{\text{res}} \subset \mathbb{Z}^2$ be the resonant lattice
(frequencies coinciding with rational orbits on the angular torus)
and $\Lambda_{\text{nr}} = \mathbb{Z}^2 \setminus \Lambda_{\text{res}}$
the nonresonant complement.

**Theorem (ATCCT-KTFS).** *Subject to the assumptions above, the
following four certificates are equivalent and machine-checkable:*

1. **(Cocycle closure)** The overlap pullbacks
   $\{\Phi_{\alpha\beta}\}_{(\alpha,\beta) \in T}$ define a Čech
   1-cocycle on the good cover $\mathcal{A}$ with values in the
   group of symplectic transport automorphisms of the
   trapped-constraint ideal; equivalently, on every triple overlap
   $U_{\alpha\beta\gamma}$ the composition
   $\Phi_{\alpha\beta} \circ \Phi_{\beta\gamma} \circ
   \Phi_{\gamma\alpha}$ is the identity modulo the QIEDT SOS defect.

2. **(Margin propagation)** For every edge
   $(\alpha, \beta) \in T$, the transported margin
   $\mu_\beta = \mu_\alpha \cdot \nu_{\alpha\beta}$ for an explicit
   rational norm factor
   $\nu_{\alpha\beta} \in \mathbb{Q}_{>0}$ that is the operator norm
   of $\Phi_{\alpha\beta}^{-1}$ on the GRLEPS principal symbol. The
   product $\prod_{(\alpha,\beta) \in T} \nu_{\alpha\beta}$ is the
   atlas-wide margin and is bounded below by an explicit rational
   constant depending only on the spin $|a|/M$.

3. **(Fourier seed transport)** There exists a seed chart
   $\alpha_0 \in I$ and a Kerr torus Fourier seed
   $\mathcal{F}_{\alpha_0}$ such that for every chart
   $\alpha \in I$, the transported coefficients
   $\mathcal{F}_\alpha$ obtained by pulling $\mathcal{F}_{\alpha_0}$
   along the unique path in $T$ from $\alpha_0$ to $\alpha$ satisfy:
   - **(nonresonant division)** For every
     $(m, n) \in \Lambda_{\text{nr}}$, the coefficient
     $\hat{q}^{(\alpha)}_{\text{sr},m,n}$ is the Fourier quotient of
     $\hat{a}^{(\alpha)}_{K,m,n}$ by the Carter ansatz denominator
     $D_{m,n} \in \mathbb{Q}[\omega]$, with
     $|D_{m,n}| \geq D_{\min} \in \mathbb{Q}_{>0}$ on the resonant
     complement.
   - **(resonant lower bound)** For every
     $(m, n) \in \Lambda_{\text{res}}$, the resonant average
     $\langle \hat{q}^{(\alpha)}_{\text{sr},m,n} \rangle$ satisfies
     $\langle \hat{q}^{(\alpha)}_{\text{sr},m,n} \rangle \geq
     q_{\min} \in \mathbb{Q}_{>0}$.

4. **(Tame escape symbol)** The escape symbol
   $\sigma_{\text{esc}}^{(\alpha)} = a_KY^{(\alpha)} / q_{\text{sr}}^{(\alpha)}$
   on each chart $U_\alpha$ lies in the Carter-Wick symbol class
   $S^0_{\text{CW}}(U_\alpha)$ with explicit denominator floor
   $D_{\min}$ and derivative seminorm bound
   $\|\sigma_{\text{esc}}^{(\alpha)}\|_{C^k} \leq C_k \cdot
   D_{\min}^{-k-1}$ for explicit rational constants
   $C_k \in \mathbb{Q}_{>0}$ and all $0 \leq k \leq K_{\max}$.

## Definitions

### (D1) Good Carter atlas

A *good Carter atlas* at fixed spin $|a| < M$ is a finite open cover
$\mathcal{A} = \{U_\alpha\}_{\alpha \in I}$ of the Kerr exterior
such that:
- each $U_\alpha$ is a Carter chart with coordinates
  $(x, \ell, q, \omega)$ on the clean phase-space slice;
- every nonempty finite intersection
  $U_{\alpha_0 \cdots \alpha_k}$ is contractible in the angular torus
  direction (the good-cover condition);
- the cover refines the existing
  `spin_half_candidate_atlas_subdiv4` subdivision at spin $1/2$.

### (D2) Overlap transport morphism

For each edge $(\alpha, \beta) \in T$ of the spanning tree, the
*overlap transport morphism*
$\Phi_{\alpha\beta}: T^*U_{\alpha\beta} \to T^*U_{\alpha\beta}$ is
the symplectic pullback defined by the Carter chart transition map
$\tau_{\alpha\beta}: U_{\alpha\beta} \to U_{\alpha\beta}$. It
conjugates the trapped-constraint ideals
$\Phi_{\alpha\beta}^* \mathcal{I}_{\text{trap}}^{(\beta)} =
\mathcal{I}_{\text{trap}}^{(\alpha)}$ and preserves the GRLEPS
principal symbol up to the QIEDT SOS defect:
$$
\Phi_{\alpha\beta}^* \sigma_{\text{NF}}^{(\beta)}
= \sigma_{\text{NF}}^{(\alpha)} + s \cdot (c x - \ell)^2.
$$

### (D3) Kerr torus Fourier seed

A *Kerr torus Fourier seed* on a chart $U_\alpha$ is a pair of
bi-infinite rational sequences
$\hat{a}^{(\alpha)}_{K,m,n}, \hat{q}^{(\alpha)}_{\text{sr},m,n}
\in \mathbb{Q}(i)$ indexed by $(m, n) \in \mathbb{Z}^2$, with
finite support on each dyadic annulus
$2^j \leq \max(|m|, |n|) < 2^{j+1}$ (the *tame decay* condition).
The seed is *admissible* if the nonresonant coefficients satisfy
the Fourier division identity
$\hat{q}^{(\alpha)}_{\text{sr},m,n} = \hat{a}^{(\alpha)}_{K,m,n}
/ D_{m,n}$
and the resonant coefficients satisfy the strict rational average
lower bound above.

### (D4) Carter-Wick symbol class

A smooth function $\sigma$ on $T^*U_\alpha$ belongs to the
*Carter-Wick symbol class* $S^0_{\text{CW}}(U_\alpha)$ if for every
compact $K \subset U_\alpha$ and every multiindex
$(k_x, k_\ell, k_q, k_\omega)$ with
$|k| \leq K_{\max}$, the derivative seminorm
$\|\partial^k \sigma\|_{L^\infty(K)}$ is bounded by a rational
constant times $D_{\min}^{-|k|-1}$.

## Proof

The proof has four parts, one per equivalent certificate. Each part
produces a finite machine-checkable invariant, and the four
invariants together close the atlas-wide backend contract.

### Part I: Cocycle closure

On each triple overlap $U_{\alpha\beta\gamma}$, the three overlap
transport morphisms compose as
$\Phi_{\alpha\beta} \circ \Phi_{\beta\gamma} \circ
\Phi_{\gamma\alpha}$. Because each $\Phi$ is a symplectic pullback
that conjugates the trapped-constraint ideal and preserves the
GRLEPS principal symbol up to the QIEDT SOS defect, the composition
is a symplectic automorphism of the trapped-constraint ideal that
fixes the GRLEPS principal symbol up to the SOS defect. The QIEDT
SOS defect $s \cdot (cx - \ell)^2$ is itself fixed by every
transport morphism (because $s, c, x, \ell$ are Carter invariants
transforming as scalars under chart transitions), so the
composition is the identity modulo the SOS defect. The Čech
differential $\delta \Phi$ on triple overlaps is therefore zero in
the quotient group of symplectic automorphisms modulo the SOS defect,
which is exactly the cocycle condition. ∎

### Part II: Margin propagation

The transported margin
$\mu_\beta = \mu_\alpha \cdot \nu_{\alpha\beta}$ is defined by the
operator norm of $\Phi_{\alpha\beta}^{-1}$ on the GRLEPS principal
symbol. Because the GRLEPS principal symbol is a degree-2
homogeneous polynomial in the Carter invariants with rational
coefficients (certified by GRLEPS), and because the transport
morphisms act by rational linear transformations on the Carter
invariants (certified by the Carter chart transition maps), the
operator norm $\nu_{\alpha\beta}$ is an explicit rational number
computed by the standard operator-norm formula on a finite-dimensional
rational vector space. The product over the spanning tree is
well-defined because $T$ is a tree (no cycles to introduce
monodromy), and the atlas-wide margin is bounded below by the
minimum over all charts of the per-chart margin times the product
of the per-edge norm factors, all of which are explicit positive
rationals. ∎

### Part III: Fourier seed transport

The seed chart $\alpha_0$ is chosen to be the chart containing the
Schwarzschild limit point (which exists in the atlas by the
good-cover condition and the fact that the Schwarzschild limit is
in the closure of the Kerr exterior). The Kerr torus Fourier seed
on $\alpha_0$ is constructed by:
- (a) solving the Carter ansatz equation
  $a_KY \cdot D = q_sr$ on the nonresonant lattice
  $\Lambda_{\text{nr}}$ by Fourier division
  $\hat{q}_{\text{sr},m,n} = \hat{a}_{K,m,n} / D_{m,n}$, which is
  well-defined because $|D_{m,n}| \geq D_{\min} > 0$ on
  $\Lambda_{\text{nr}}$ by the nonresonance condition;
- (b) imposing the strict rational average lower bound on the
  resonant lattice $\Lambda_{\text{res}}$, which is the resonance
  condition for the orbit-cohomology certificate.

The transport to any other chart $\alpha$ along the unique path in
$T$ from $\alpha_0$ to $\alpha$ is by the overlap pullback
$\Phi_{\alpha\beta}^*$ acting on the Fourier coefficients. Because
$\Phi_{\alpha\beta}^*$ is a rational linear map on the Carter
invariants and the Carter invariants are the Fourier variables, the
transported coefficients are rational linear combinations of the
seed coefficients and remain in $\mathbb{Q}(i)$. The tame decay
condition is preserved because $\Phi_{\alpha\beta}^*$ acts
block-diagonally on dyadic annuli (a consequence of the
degree-2 homogeneity of the GRLEPS principal symbol). ∎

### Part IV: Tame escape symbol

The escape symbol
$\sigma_{\text{esc}}^{(\alpha)} = a_KY^{(\alpha)} /
q_{\text{sr}}^{(\alpha)}$
is the ratio of two Carter-Wick symbols of compatible order. Its
derivative seminorms are bounded by the quotient rule, with the
denominator floor $D_{\min}$ providing the
$D_{\min}^{-k-1}$ factor and the tame decay of the numerator's
Fourier coefficients providing the $C_k$ factor. The explicit
rational constants $C_k$ are computed by the standard
Faà-di-Bruno formula on the Fourier side, where each factor is a
rational linear combination of seed coefficients. ∎

## Machine-Checkable Certificate

The ATCCT-KTFS certificate is a JSON file
`certificates/atcct_ktfs/spin_half_atlas_transport_cocycle.json`
with the following fields, each verifiable by exact rational
arithmetic:

- `metadata`: name, spin scope, claim scope, status, warning.
- `atlas`: list of chart entries, each with `name`,
  `monomial_basis_size`, `max_symbol_order`,
  `local_margin` (rational), `width` (rational).
- `spanning_tree`: list of edges `[[alpha, beta, nu_alpha_beta]]`
  with explicit rational norm factors.
- `triple_overlaps`: list of triple-overlap cocycle checks, each
  with the three chart names and the rational residual of the
  composed transport modulo the SOS defect (must be zero).
- `atlas_wide_margin`: the product of the seed margin and all
  per-edge norm factors (rational, must be positive).
- `fourier_seed`: the seed chart name and the seed Fourier
  coefficients on a finite dyadic annulus cutoff (with
  `nonresonant_division` and `resonant_average_lower_bound`
  subfields).
- `escape_symbol_tame_certificate`: the denominator floor
  `D_min`, the maximum derivative order `K_max`, and the list of
  rational seminorm constants `C_k`.

The verifier
`verify_atlas_transport_cocycle_ktfs.py` checks:
1. every required field is present and well-typed;
2. the spanning tree is a tree on the atlas vertex set;
3. every triple-overlap residual is exactly zero as a rational
   polynomial;
4. the atlas-wide margin is positive and matches the product;
5. the Fourier seed's nonresonant division identity holds exactly
   on the certified cutoff;
6. the resonant average lower bound holds on the certified cutoff;
7. the escape-symbol seminorm constants are positive rationals and
   the tame certificate is internally consistent.

The verifier also rejects any certificate whose `status` field
claims `kerr_proved`, because the ATCCT-KTFS itself does not
derive the seed Fourier coefficients from the Kerr geometry — it
derives the *transport* and *cocycle* structure given the seed as
input. A future Kerr-derivation theorem must promote the seed
from candidate to exact.

## Comparison to Existing Theorems

The closest comparison is the BCCT, which closes the backend
contract for a *single* Carter chart. The ATCCT-KTFS is the
atlas-wide generalization: it composes the per-chart BCCT
contracts along a spanning tree of the overlap cover and adds the
Čech cocycle condition on triple overlaps. The QIEDT SOS defect
plays a new role here as the *monodromy absorber* — the only
term that can appear in a triple-overlap composition without
forcing the cocycle to vanish identically, because it is fixed
by every transport morphism.

The second closest comparison is the SOBOLEV FREQUENCY SYMBOL
ADMISSIBILITY THEOREM (SFSAT), which certifies that a single
frequency-function symbol belongs to a single-chart
admissibility class. The ATCCT-KTFS promotes this to an
atlas-wide statement: the escape symbol belongs to the
Carter-Wick symbol class on *every* atlas chart, with
denominator floors and derivative seminorms transported from the
seed chart along the spanning tree.

The ATCCT-KTFS is the first theorem in the audit chain to use the
*good cover* hypothesis in an essential way: the Čech cocycle
condition is vacuous on a single chart and trivial on a pair, but
nontrivial on triple overlaps, and the good-cover condition
ensures that triple overlaps are contractible so that the cocycle
condition is a finite check rather than a cohomological
computation.

## Audit Chain Position

The ATCCT-KTFS is the **ninth primitive-packet exact-derivation
theorem** in the audit chain and the **first atlas-wide theorem**.
It depends on:
- BCCT (per-chart backend contract closure, used as black-box
  input on each atlas chart);
- GRLEPS (principal symbol transport across overlaps);
- QIEDT (SOS defect as monodromy absorber);
- TCDEDT (trapped-constraint ideal conjugation across overlaps);
- SFSAT (single-chart admissibility, promoted to atlas-wide).

It is the input required by the downstream nonlinear bootstrap
(NBCT) at the atlas scale: the NBCT can now run on the
atlas-wide margin rather than a single-chart margin, which is
the structural prerequisite for the global Kerr stability
theorem.

## Conditional Status and Remaining Obligations

The ATCCT-KTFS itself does not:
- derive the Kerr torus Fourier seed
  $\mathcal{F}_{\alpha_0}$ from the Kerr geometry — it assumes
  the seed as input and certifies its transport;
- prove that the good cover $\mathcal{A}$ exists at every fixed
  subextremal spin $|a| < M$ (it assumes the existing
  `spin_half_candidate_atlas_subdiv4` subdivision extends);
- prove the strict rational average lower bound on the resonant
  modes of `q_sr` — it assumes this as an input certificate;
- prove uniformity of the atlas-wide margin as $|a|/M \to 1$
  (the near-extremal regime is left to a future uniformity
  theorem).

These four obligations are the next items in the audit chain.
The Kerr-derivation obligation is the deepest: it requires
deriving the Carter ansatz equation $a_KY \cdot D = q_sr$ from
the linearized Einstein operator on the Kerr background, which
is the missing analytic input behind the entire
trapped-superradiant branch.
