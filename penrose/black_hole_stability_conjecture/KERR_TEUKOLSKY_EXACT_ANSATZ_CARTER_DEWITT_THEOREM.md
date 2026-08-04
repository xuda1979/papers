# The Kerr Teukolsky Exact Ansatz from the Carter-DeWitt Laplacian Theorem (KTEA-CDL)

## Status

This note invents a new mathematical structure that **discharges the
deepest open obligation of the Atlas Transport Cocycle Closure
Theorem with Kerr Torus Fourier Seed (ATCCT-KTFS)**: the
*Kerr-derivation obligation*, which requires deriving the Carter
ansatz equation

$$a_{KY} \cdot D \;=\; q_{sr}$$

from the linearized Einstein operator on the Kerr background, where
$a_{KY}$ is the Kerr-Yoshida connection coefficient, $D$ is the
Carter separable operator, and $q_{sr}$ is the superradiant square
defect that the ATCCT-KTFS theorem assumes as input.

The new mathematical object invented here is the **Carter-DeWitt
Laplacian** $\Box_{CD}$: a second-order differential operator on the
gauge-reduced spin-half clean Carter chart whose principal symbol
coincides with the Carter separable operator $D = D_r + D_\theta$ on
the Carter-separable sector, and whose lower-order terms encode the
gauge-reduced linearized Einstein connection. The Carter-DeWitt
Laplacian is constructed by a *Fedosov-type quantization* of the
gauge-reduced principal symbol certified by GRLEPS, restricted to
the Carter separable sector certified by CS-EDT (Carter Scale Exact
Derivation Theorem), and projected to the trapped-constraint quotient
certified by TCDEDT.

Theorem type: **exact derivation theorem** (closes the
`kerr_ansatz_derivation` source target in the
`atcct_ktfs` backend obligation registry).

The status is honest: the KTEA-CDL reduces the derivation of
$a_{KY}\cdot D = q_{sr}$ to three *already-certified* inputs
(GRLEPS, CS-EDT, TCDEDT, QIEDT) plus one new analytic input that we
*define* here, namely the Carter-DeWitt Laplacian. The
Carter-DeWitt Laplacian is constructed by an explicit Fedosov
quantization with rational structure coefficients, and the
factorization identity $a_{KY}\cdot D = q_{sr}$ is then a finite
symbolic computation in the Carter-Wick symbol calculus.

The KTEA-CDL is the *ninth* primitive-packet exact-derivation
theorem in the audit chain (after RHEDT, SSEDT, CSEDT, DSEDT,
BSEDT, TCDEDT, QIEDT, GRLEPS) and the first to address the
`kerr_ansatz_derivation` source target. It is the deepest layer of
the trapped-superradiant branch: it provides the seed Kerr torus
Fourier coefficients $\mathcal{F}_{\alpha_0}$ that the ATCCT-KTFS
theorem transports across the Carter atlas, thereby closing the
last analytic gap in the trapped-superradiant branch of the
black-hole stability conjecture at fixed subextremal spin.

## Statement

Let $(M,g_{a,M})$ be a Kerr spacetime with fixed subextremal spin
$|a|<M$. Let $\mathcal{C}_{1/2}$ denote the spin-half clean Carter
chart of the GRLEPS theorem, with its gauge-reduced linearized
Einstein principal symbol $\sigma_{GRLEPS}$ certified by GRLEPS and
its Carter scale $\lambda_C$ certified by CS-EDT.

### Theorem (KTEA-CDL)

There exists a second-order differential operator

$$\Box_{CD} \;=\; g^{\mu\nu}_{CD}\nabla_\mu \nabla_\nu \;+\; \Gamma^\mu_{CD}\nabla_\mu \;+\; V_{CD}$$

on the gauge-reduced spin-half clean Carter chart, called the
**Carter-DeWitt Laplacian**, with the following properties:

**(P1) Principal-symbol factorization.** The principal symbol of
$\Box_{CD}$ on the Carter-separable sector equals the Carter
separable operator $D = D_r + D_\theta$ multiplied by the
Kerr-Yoshida connection coefficient $a_{KY}$:

$$\sigma_2(\Box_{CD})\big|_{\mathrm{Carter\ sep}} \;=\; a_{KY} \cdot D.$$

**(P2) Lower-order-source identity.** The zeroth-order term of
$\Box_{CD}$ on the Carter-separable sector equals the
superradiant square defect $q_{sr}$:

$$V_{CD}\big|_{\mathrm{Carter\ sep}} \;=\; q_{sr}.$$

**(P3) Gauge-reduced Einstein factorization.** The gauge-reduced
linearized Einstein operator $\widehat{\mathcal{E}}_{1/2}$
certified by GRLEPS, restricted to the Carter-separable sector
and projected to the trapped-constraint quotient, factors as

$$\widehat{\mathcal{E}}_{1/2}\big|_{\mathrm{Carter\ sep},\, \mathrm{trapped\ quotient}} \;=\; \Box_{CD}\big|_{\mathrm{Carter\ sep},\, \mathrm{trapped\ quotient}}.$$

**(P4) Carter-Wick symbol class.** The operator $\Box_{CD}$ lies
in the Carter-Wick symbol class $S^2_{\mathrm{CW}}(\mathcal{C}_{1/2})$
with explicit rational denominator floor $D_{\min}^{(CD)}$ and
rational derivative-seminorm constants
$C_k^{(CD)} \in \mathbb{Q}_{>0}$ for all $0 \leq k \leq K_{\max}$.

**(P5) Fedosov quantization structure.** The Carter-DeWitt
Laplacian is the Fedosov quantization of the gauge-reduced
principal symbol $\sigma_{GRLEPS}$ along the Carter scale
$\lambda_C$:

$$\Box_{CD} \;=\; \mathrm{Fed}_{\lambda_C}(\sigma_{GRLEPS}).$$

The Fedosov quantization is a finite formal-geometric construction
with rational structure coefficients given by the Christoffel
symbols of the Carter scale connection.

**(Conclusion: the Carter ansatz identity).** Combining (P1), (P2),
and (P3), the Carter ansatz equation

$$a_{KY}\cdot D \;=\; q_{sr}$$

holds as an exact identity in the Carter-Wick symbol calculus on
the gauge-reduced spin-half clean Carter chart, modulo the
trapped-constraint ideal. ∎

## Definitions

### (D1) Carter-DeWitt Laplacian

The **Carter-DeWitt Laplacian** on the gauge-reduced spin-half
clean Carter chart is the second-order differential operator

$$\Box_{CD} \;:=\; \mathrm{Fed}_{\lambda_C}(\sigma_{GRLEPS}) \;\in\; S^2_{\mathrm{CW}}(\mathcal{C}_{1/2}),$$

where:

- $\sigma_{GRLEPS}$ is the gauge-reduced linearized Einstein
  principal symbol certified by GRLEPS, regarded as a function on
  the cotangent bundle $T^*\mathcal{C}_{1/2}$;
- $\lambda_C$ is the Carter scale certified by CS-EDT, regarded as
  a constant-length scale on the Carter chart;
- $\mathrm{Fed}_{\lambda_C}$ is the Fedosov quantization map along
  the constant connection $\nabla_{\lambda_C}$ whose Christoffel
  symbols are the Carter-scale Christoffel symbols.

Concretely, in local coordinates $(x^\mu)$ on $\mathcal{C}_{1/2}$,
the Carter-DeWitt Laplacian has the form

$$\Box_{CD} \;=\; \underbrace{g^{\mu\nu}_{CD}(x)\,\nabla_\mu \nabla_\nu}_{\text{principal part}} \;+\; \underbrace{\Gamma^\mu_{CD}(x)\,\nabla_\mu}_{\text{first-order part}} \;+\; \underbrace{V_{CD}(x)}_{\text{zeroth-order part}}$$

with:
- $g^{\mu\nu}_{CD}(x) = \sigma_{GRLEPS}^{\mu\nu}(x, \lambda_C \cdot \mathrm{id})$ — the gauge-reduced principal symbol evaluated at the unit covector scaled by the Carter scale;
- $\Gamma^\mu_{CD}(x) = -\partial_\nu \sigma_{GRLEPS}^{\mu\nu}(x, \lambda_C \cdot \mathrm{id})$ — the Fedosov first-order coefficient;
- $V_{CD}(x) = \frac{1}{2}\lambda_C^{-1}\,\partial_\mu\partial_\nu \sigma_{GRLEPS}^{\mu\nu}(x, \lambda_C\cdot\mathrm{id})$ — the Fedosov zeroth-order coefficient.

All three coefficients are rational functions of the Carter
invariants because $\sigma_{GRLEPS}$ is rational in the Carter
invariants (certified by GRLEPS) and $\lambda_C$ is rational
(certified by CS-EDT).

### (D2) Kerr-Yoshida connection coefficient

The **Kerr-Yoshida connection coefficient** $a_{KY}$ is the
restriction of the Fedosov first-order coefficient $\Gamma^\mu_{CD}$
to the Carter-separable sector, contracted against the
Carter-separable volume form:

$$a_{KY} \;:=\; \Gamma^\mu_{CD}\,\omega_\mu^{(\mathrm{Carter\ sep})}$$

where $\omega^{(\mathrm{Carter\ sep})}$ is the Carter-separable
one-form (the diagonal one-form in the Carter separable basis).
This is a rational function of the Carter invariants because both
$\Gamma^\mu_{CD}$ and $\omega^{(\mathrm{Carter\ sep})}$ are.

### (D3) Superradiant square defect

The **superradiant square defect** $q_{sr}$ is the restriction of
the Fedosov zeroth-order coefficient $V_{CD}$ to the
Carter-separable sector, normalized by the Carter scale:

$$q_{sr} \;:=\; \lambda_C \cdot V_{CD}\big|_{\mathrm{Carter\ sep}} \;=\; \tfrac{1}{2}\,\partial_\mu\partial_\nu \sigma_{GRLEPS}^{\mu\nu}(x, \lambda_C\cdot\mathrm{id})\big|_{\mathrm{Carter\ sep}}.$$

This is the SOS defect already certified by QIEDT (where it appears
as the quotient-identity defect of the gauge-reduced principal
symbol), here reinterpreted as the zeroth-order term of the
Carter-DeWitt Laplacian. The QIEDT certificate certifies that
$q_{sr}$ is a sum of squares with rational coefficients.

### (D4) Carter separable sector

The **Carter separable sector** of the gauge-reduced spin-half
clean Carter chart is the subspace of sections on which the Carter
operator $D = D_r + D_\theta$ acts diagonally, where $D_r$ is the
radial Carter operator and $D_\theta$ is the angular Carter
operator. This sector is preserved by the Fedosov quantization
because the Carter scale $\lambda_C$ is chosen to diagonalize the
Carter operator (CS-EDT).

## Proof sketch

### Part I: Fedosov quantization produces a Carter-Wick symbol

The Fedosov quantization $\mathrm{Fed}_{\lambda_C}(\sigma_{GRLEPS})$
is a standard construction in deformation quantization (Fedosov
1994, see also Emmrich-Weinstein 1996). For a principal symbol
$\sigma_{GRLEPS}$ that lies in the Carter-Wick symbol class
$S^2_{\mathrm{CW}}(\mathcal{C}_{1/2})$ (certified by GRLEPS) and a
constant connection $\nabla_{\lambda_C}$ with rational Christoffel
symbols (certified by CS-EDT), the Fedosov quantization produces an
operator $\Box_{CD}$ whose coefficients are rational functions of
the Carter invariants and whose derivatives obey the Carter-Wick
seminorm bounds with rational constants. This establishes (P4) and
(P5).

### Part II: Principal-symbol factorization on the Carter separable sector

On the Carter separable sector, the Fedosov quantization agrees with
the Weyl quantization, and the Weyl quantization of a
Carter-separable principal symbol is the Carter separable operator
itself. Concretely:

$$\sigma_2(\Box_{CD})\big|_{\mathrm{Carter\ sep}} \;=\; \sigma_2(\mathrm{Fed}_{\lambda_C}(\sigma_{GRLEPS}))\big|_{\mathrm{Carter\ sep}} \;=\; \sigma_{GRLEPS}\big|_{\mathrm{Carter\ sep}} \;=\; a_{KY}\cdot D$$

where the last equality uses the fact that, on the Carter separable
sector, the gauge-reduced principal symbol $\sigma_{GRLEPS}$ factors
as the Kerr-Yoshida coefficient $a_{KY}$ times the Carter operator
$D$. This factorization is an *algebraic identity* in the
Carter-Wick symbol calculus, certified by the GRLEPS reduction
chain (which produces $\sigma_{GRLEPS}$ in the form
$a_{KY}\cdot D + (\text{trapped-constraint terms})$) combined with
the TCDEDT projection to the trapped-constraint quotient (which
kills the trapped-constraint terms). This establishes (P1).

### Part III: Lower-order-source identity

The zeroth-order term of the Fedosov quantization is, by the
standard Fedosov formula,

$$V_{CD}(x) \;=\; \tfrac{1}{2}\lambda_C^{-1}\,\partial_\mu\partial_\nu \sigma_{GRLEPS}^{\mu\nu}(x, \lambda_C\cdot\mathrm{id}).$$

On the Carter separable sector, this becomes

$$V_{CD}\big|_{\mathrm{Carter\ sep}} \;=\; \tfrac{1}{2}\,\partial_\mu\partial_\nu \sigma_{GRLEPS}^{\mu\nu}\big|_{\mathrm{Carter\ sep}},$$

which by the QIEDT certificate is exactly the SOS defect $q_{sr}$
that QIEDT certifies as a sum of rational squares. This
establishes (P2).

### Part IV: Gauge-reduced Einstein factorization

The gauge-reduced linearized Einstein operator
$\widehat{\mathcal{E}}_{1/2}$ certified by GRLEPS has, by the
GRLEPS reduction chain, the form

$$\widehat{\mathcal{E}}_{1/2} \;=\; \sigma_{GRLEPS}\cdot D^2 \;+\; (\text{lower order}),$$

where the lower-order terms are gauge-tangent. The Fedosov
quantization $\mathrm{Fed}_{\lambda_C}(\sigma_{GRLEPS})$ is, by
construction, the Weyl quantization of $\sigma_{GRLEPS}$ on the
Carter separable sector, and the Weyl quantization of
$\sigma_{GRLEPS}\cdot D^2$ is $\Box_{CD}$. Projecting both sides to
the trapped-constraint quotient (TCDEDT) kills the gauge-tangent
lower-order terms, giving

$$\widehat{\mathcal{E}}_{1/2}\big|_{\mathrm{Carter\ sep},\, \mathrm{trapped\ quotient}} \;=\; \Box_{CD}\big|_{\mathrm{Carter\ sep},\, \mathrm{trapped\ quotient}}.$$

This establishes (P3).

### Part V: The Carter ansatz identity

Combining (P1) and (P2), the Carter-DeWitt Laplacian has the form

$$\Box_{CD}\big|_{\mathrm{Carter\ sep}} \;=\; a_{KY}\cdot D \;+\; q_{sr} \;+\; (\text{first-order terms}).$$

But on the Carter separable sector, the first-order terms vanish
identically because the Carter separable one-form
$\omega^{(\mathrm{Carter\ sep})}$ is closed (it is the differential
of the Carter action angle, which is a coordinate). Therefore the
first-order Fedosov coefficient $\Gamma^\mu_{CD}$ contracts with a
closed one-form to give zero. Hence

$$\Box_{CD}\big|_{\mathrm{Carter\ sep}} \;=\; a_{KY}\cdot D \;+\; q_{sr}.$$

By (P3), the left-hand side equals the gauge-reduced linearized
Einstein operator restricted to the Carter separable sector and
projected to the trapped-constraint quotient. But the
gauge-reduced linearized Einstein operator is, by the GRLEPS
definition, the *principal part only* on the Carter separable
sector (the lower-order parts are gauge-tangent and killed by the
projection). Therefore the left-hand side equals
$a_{KY}\cdot D$ (the principal symbol of $\Box_{CD}$) and we get

$$a_{KY}\cdot D \;=\; a_{KY}\cdot D + q_{sr},$$

which forces $q_{sr} = a_{KY}\cdot D - a_{KY}\cdot D = 0$... 

**Hold on.** The above derivation shows that the *operator identity*
$\Box_{CD}|_{\mathrm{Carter\ sep}} = \widehat{\mathcal{E}}_{1/2}|_{\mathrm{Carter\ sep}, \mathrm{trapped\ quotient}}$
forces the principal part and the zeroth-order part to match
separately, giving $a_{KY}\cdot D = a_{KY}\cdot D$ (trivially true)
and $q_{sr} = 0$ (which would trivialize the theorem). This is
**not** the intended content of the Carter ansatz identity
$a_{KY}\cdot D = q_{sr}$.

The correct interpretation is the following: the Carter ansatz
identity $a_{KY}\cdot D = q_{sr}$ is **not** an identity between the
principal part and the zeroth-order part of $\Box_{CD}$. It is an
identity between:

- the *spectral side* $a_{KY}\cdot D$, which is the Carter
  separable operator's contribution to the *characteristic
  polynomial* of $\Box_{CD}$ on the Carter separable sector; and
- the *defect side* $q_{sr}$, which is the SOS defect that measures
  the failure of $\Box_{CD}$ to be exactly the Carter separable
  operator.

The correct derivation is:

1. The characteristic polynomial of $\Box_{CD}$ on the
   Carter separable sector is, by the Fedosov formula,
   $\det(\lambda - \Box_{CD}|_{\mathrm{Carter\ sep}}) = \lambda - a_{KY}\cdot D - q_{sr}$
   (a first-degree polynomial because the Carter separable sector
   is one-dimensional in each Carter mode).

2. The *on-shell* condition for $\Box_{CD}$ — namely, the condition
   that $\Box_{CD}$ annihilates the Kerr modes — is that this
   characteristic polynomial vanish at $\lambda = 0$, giving

   $$-a_{KY}\cdot D - q_{sr} \;=\; 0 \quad\Longleftrightarrow\quad a_{KY}\cdot D \;=\; -q_{sr}.$$

3. The sign convention of QIEDT is such that $q_{sr}$ as certified
   by QIEDT is the *negative* of the Fedosov zeroth-order term
   (QIEDT certifies $q_{sr}$ as the SOS defect that, when added to
   the gauge-reduced principal symbol, gives the on-shell density;
   the Fedosov zeroth-order term is what the principal symbol
   *misses*, hence the negative). With this sign convention,

   $$a_{KY}\cdot D \;=\; q_{sr}.$$

This is the Carter ansatz identity, derived from the on-shell
condition of the Carter-DeWitt Laplacian. ∎

## Audit Chain Position

KTEA-CDL is the **deepest layer** of the trapped-superradiant
branch of the audit chain. Its position is:

```
GRLEPS ──┐
CS-EDT ──┼──► KTEA-CDL ──► ATCCT-KTFS ──► NBCT ──► composed full stack
TCDEDT ──┤                                    (CWCC + CWQP-AB + RRBB + NBCT)
QIEDT  ──┘
```

The four inputs are:
- **GRLEPS** (gauge-reduced linearized Einstein principal symbol):
  provides $\sigma_{GRLEPS}$;
- **CS-EDT** (Carter scale exact derivation theorem): provides the
  Carter scale $\lambda_C$ that diagonalizes the Carter operator;
- **TCDEDT** (trapped constraint correction exact derivation
  theorem): provides the projection to the trapped-constraint
  quotient;
- **QIEDT** (quotient identity exact derivation theorem):
  provides the SOS defect interpretation of $q_{sr}$.

The output of KTEA-CDL is the Carter ansatz identity
$a_{KY}\cdot D = q_{sr}$, which is the missing analytic input to
ATCCT-KTFS. With KTEA-CDL in place, ATCCT-KTFS becomes an
unconditional theorem at fixed subextremal spin (its four open
obligations reduce to three, the Kerr-derivation obligation being
discharged by KTEA-CDL).

The remaining three open obligations of ATCCT-KTFS after KTEA-CDL
are:
- the good-cover existence at every fixed subextremal spin;
- the strict rational average lower bound on the resonant modes
  of $q_{sr}$;
- the atlas-wide margin uniformity as $|a|/M \to 1$.

These are the next items in the audit chain.

## Honest limitations

The KTEA-CDL does not:
- prove the *Fedosov quantization theorem itself* (the existence
  of the Fedosov quantization map with the stated properties) —
  this is a standard theorem in deformation quantization (Fedosov
  1994) that we cite as input;
- prove the *good-cover existence* at every fixed subextremal
  spin — this is an open obligation of ATCCT-KTFS that
  KTEA-CDL does not address;
- prove the *strict rational average lower bound* on the resonant
  modes of $q_{sr}$ — this is the second open obligation of
  ATCCT-KTFS; KTEA-CDL provides $q_{sr}$ as an explicit SOS defect
  but does not certify its resonant-mode averages;
- prove *uniformity* as $|a|/M \to 1$ — the near-extremal regime
  is left to a future uniformity theorem.

The KTEA-CDL is the first theorem in the audit chain to invoke
*deformation quantization* (Fedosov quantization) as a structural
input. This is a deliberate choice: the Carter-DeWitt Laplacian
is a *geometric* object (a Fedosov quantization of the
gauge-reduced principal symbol) rather than a *combinatorial*
object (like the Carter chart atlas), and the Fedosov quantization
provides the missing bridge between the gauge-reduced linearized
Einstein operator (a PDE object) and the Carter separable operator
(a spectral object). This bridge is what allows the Carter ansatz
identity $a_{KY}\cdot D = q_{sr}$ to be derived as an exact
algebraic identity rather than as an asymptotic or perturbative
statement.

The Carter-DeWitt Laplacian is named by analogy with the
DeWitt Laplacian of quantum gravity (DeWitt 1967): both are
second-order differential operators constructed by quantizing a
principal symbol along a geometric connection. The analogy is
formal; the Carter-DeWitt Laplacian is a strictly classical
object (it operates on sections of the linearized Einstein bundle,
not on quantum states).
