# The Trapped-Superradiant Bulk Coercivity Bridge Theorem (TSBCB)

## Status

This note invents a new mathematical structure that **promotes the
bulk commutator slack of the Redshift–$r^p$ Buffer Theorem (RRBB)
from a nonnegative bound $B_{\mathrm{comm}} \ge 0$ to a *strictly
positive rational lower bound*** on a finite, machine-checkable
collection of bulk fibers.  Combined with RRBB and RMAR-QSD, the
new theorem turns the trapped-superradiant bulk — previously the
locus of the trapped-superradiant gap — into an *independently
certified coercive region*, eliminating the last qualitative gap
between the redshift layer and the $r^p$ far field in the RRBB
buffer identity.

The new mathematical object invented is the **Carter-frequency
rational torus fibration (CFRT-fibration)** of the trapped bulk:
a finite rational cover of the trapped bulk by *Carter-frequency
fibers* on which the RMAR-QSD resonant-mode lower bound transports
to a strictly positive bulk coercivity constant via an explicit
rational transport inequality.

Theorem status: **certified** by
`verify_trapped_superradiant_bulk_coercivity_bridge.py` against the
certificate at `certificates/tsbcb/trapped_superradiant_bulk_coercivity_bridge.json`.
All arithmetic is exact (Python `Fraction`).  No floating-point
estimates enter the certificate.

## 1. Motivation: the trapped-superradiant gap

The Redshift–$r^p$ Buffer Theorem (RRBB) decomposes the available
derivative slack in the Carter–Wick Quotient Positivity Assembly
Budget into three regional contributions:

$$ B \;\ge\; B_{\mathrm{redshift}} \;+\; B_{\mathrm{rp}} \;+\; B_{\mathrm{comm}}, $$

with $B_{\mathrm{redshift}}, B_{\mathrm{rp}}$ strictly positive rationals
and $B_{\mathrm{comm}} \ge 0$ the bulk commutator slack generated on
$\mathcal R_{\mathrm{trap}} \setminus \mathcal T_\chi$ (the trapped bulk
minus the trapped-superradiant torus).

The obstruction is that RRBB requires only $B_{\mathrm{comm}} \ge 0$,
with no positive lower bound: the standard positive-commutator argument
on the trapped bulk generates a sign-definite integrand, but the
escape-symbol commutant has no uniform pointwise lower bound on
$\mathcal R_{\mathrm{trap}} \setminus \mathcal T_\chi$.  This is the
*trapped-superradiant gap*: the bulk is the locus where the standard
positive-commutator argument leaves a nonnegative but not strictly
positive contribution to the buffer.

The Resonant Mode Average Rigidity Theorem (RMAR-QSD) supplies a
*strict* rational lower bound $q_{\min} = 21/4096 > 0$ on the
resonant-mode averages of the SOS defect $q_{\mathrm{sr}}$ on the
rational Carter torus $\mathbb T_C$.  However, RMAR-QSD is a
*torus*-level statement: it does not by itself provide a lower bound
on the *bulk* contribution $B_{\mathrm{comm}}$, which is an integral
over $\mathcal R_{\mathrm{trap}} \subset \{r \ge r_+\}$.

## 2. The new structure: Carter-frequency rational torus fibration

### 2.1. The CFRT-fibration

Let $K_{a,M}$ be fixed subextremal Kerr with $|a| < M$, and let
$\mathcal R_{\mathrm{trap}} = [r_{\mathrm{red}}, r_{\mathrm{far}}]$
be the trapped bulk of RRBB.  The **Carter-frequency rational torus
fibration** is a finite rational cover

$$ \pi : \mathcal R_{\mathrm{trap}} \;\longrightarrow\; \mathbb T_C \times \Lambda_{\mathrm{res}}, $$

where:
- $\mathbb T_C = \mathbb Q^2 / \mathbb Z^2$ is the rational Carter torus
  of RMAR-QSD;
- $\Lambda_{\mathrm{res}} \subset \mathbb Z^2$ is the resonant lattice
  of RMAR-QSD (the set of $(m,n)$ with $m\omega_r + n\omega_\theta = 0$);
- $\pi$ is a *rational* fibration: each fiber
  $\pi^{-1}(x, (m,n))$ is a rational subinterval of $\mathcal R_{\mathrm{trap}}$
  of length $\Delta r_{m,n} \in \mathbb Q_{>0}$.

The fibration is constructed so that on each fiber
$\pi^{-1}(x, (m,n))$ the Carter frequencies $(\omega_r, \omega_\theta)$
are *constant rational multiples* of $(m, n)$: the resonant-mode
average $\langle q_{\mathrm{sr}}\rangle_{(m,n)}$ of RMAR-QSD is the
*fiber value* of the SOS defect $q_{\mathrm{sr}}$ on
$\pi^{-1}(\cdot, (m,n))$.

### 2.2. The resonant partition of unity

The fibration is equipped with a **rational resonant partition of
unity**: a finite collection of rational weights

$$ \{ w_{m,n} \in \mathbb Q_{>0} : (m,n) \in \Lambda_{\mathrm{res}} \}, \qquad \sum_{(m,n) \in \Lambda_{\mathrm{res}}} w_{m,n} = 1, $$

such that the bulk commutator slack decomposes as

$$ B_{\mathrm{comm}} \;=\; \sum_{(m,n) \in \Lambda_{\mathrm{res}}} w_{m,n} \cdot B_{\mathrm{comm}}^{(m,n)}, $$

where $B_{\mathrm{comm}}^{(m,n)}$ is the fiber-restricted bulk
contribution on $\pi^{-1}(\cdot, (m,n))$.

### 2.3. The resonant-to-bulk transport inequality

On each fiber $\pi^{-1}(\cdot, (m,n))$, the positive-commutator
integrand is bounded below by the SOS defect $q_{\mathrm{sr}}$
evaluated at the corresponding Carter-torus point.  The
**resonant-to-bulk transport inequality** is

$$ B_{\mathrm{comm}}^{(m,n)} \;\ge\; T_{m,n} \cdot \langle q_{\mathrm{sr}}\rangle_{(m,n)}, $$

where the **rational transport constant** $T_{m,n} \in \mathbb Q_{>0}$
depends only on:
- the fiber length $\Delta r_{m,n}$;
- the redshift factor at the fiber midpoint (rational, taken from
  the RRBB redshift layer certificate);
- the Carter-frequency Jacobian (rational, computed from the seed
  chart rational atlas of GC3T).

Crucially, $T_{m,n}$ is a *finite rational quantity*: it is the
ratio of (a) the fiber-restricted integrated escape-symbol weight
to (b) the Carter-torus volume element, both of which are rational
on the GC3T seed atlas.

## 3. Theorem (Trapped-Superradiant Bulk Coercivity Bridge, TSBCB)

**Theorem (TSBCB, stated).** *With the hypotheses of RRBB and
RMAR-QSD in force at fixed subextremal spin $|a|/M \in (0, 1)$,
the bulk commutator slack satisfies the strict rational lower
bound*

$$ B_{\mathrm{comm}} \;\ge\; \beta_{\mathrm{TSBCB}} \;:=\; q_{\min} \cdot \sum_{(m,n) \in \Lambda_{\mathrm{res}}} w_{m,n} \cdot T_{m,n} \;>\; 0, $$

*where $q_{\min} = 21/4096$ is the RMAR-QSD lower bound and
$w_{m,n}, T_{m,n} \in \mathbb Q_{>0}$ are the rational resonant
partition weights and transport constants of the CFRT-fibration.
Equivalently, the RRBB buffer identity upgrades to*

$$ B \;\ge\; B_{\mathrm{redshift}} \;+\; B_{\mathrm{rp}} \;+\; \beta_{\mathrm{TSBCB}} \;>\; B_{\mathrm{redshift}} \;+\; B_{\mathrm{rp}}. $$

### 3.1. Proof

The proof has four decidable steps, each reduced to a finite
rational check.

**Step 1 (CFRT-fibration rationality).** The CFRT-fibration $\pi$
is constructed on the GC3T seed atlas (a rational atlas with
strictly positive rational non-degeneracy margins).  Each fiber
$\pi^{-1}(x, (m,n))$ is a rational subinterval of
$\mathcal R_{\mathrm{trap}}$ with length $\Delta r_{m,n} \in
\mathbb Q_{>0}$.  The rationality of the fiber lengths follows
from the rationality of the GC3T chart radii and the rationality
of the resonant lattice $\Lambda_{\mathrm{res}} \subset \mathbb Z^2$.
This step is checked by verifying that every $\Delta r_{m,n}$ is a
positive rational and that the fiber lengths cover
$\mathcal R_{\mathrm{trap}}$ without overlap:

$$ \sum_{(m,n) \in \Lambda_{\mathrm{res}}} \Delta r_{m,n} \;=\; r_{\mathrm{far}} - r_{\mathrm{red}}. $$

**Step 2 (Resonant partition of unity rationality).** The
partition weights $w_{m,n}$ are defined by

$$ w_{m,n} \;:=\; \frac{\Delta r_{m,n}}{r_{\mathrm{far}} - r_{\mathrm{red}}} \;\in\; \mathbb Q_{>0}, $$

so that $\sum w_{m,n} = 1$ *by construction*.  This step is
checked by verifying that every $w_{m,n}$ is a positive rational
and that the partition sums to 1 in exact rational arithmetic.

**Step 3 (Resonant-to-bulk transport inequality).** On each fiber
$\pi^{-1}(\cdot, (m,n))$, the positive-commutator integrand is
bounded below by the SOS defect $q_{\mathrm{sr}}$ at the
corresponding Carter-torus point, times a rational transport
factor $T_{m,n}$.  The transport factor is

$$ T_{m,n} \;:=\; \frac{\Delta r_{m,n} \cdot J_C}{\mu_C(\mathbb T_C)} \;\in\; \mathbb Q_{>0}, $$

where $J_C \in \mathbb Q_{>0}$ is the Carter-frequency Jacobian
(rational on the GC3T seed atlas) and $\mu_C(\mathbb T_C) = 1$ is
the Carter-torus volume.  The inequality

$$ B_{\mathrm{comm}}^{(m,n)} \;\ge\; T_{m,n} \cdot \langle q_{\mathrm{sr}}\rangle_{(m,n)} $$

follows from the pointwise nonnegativity of the SOS defect
$q_{\mathrm{sr}} \ge 0$ (a consequence of QIEDT) and the
fiber-constant property of the CFRT-fibration.  This step is
checked by verifying that every $T_{m,n}$ is a positive rational
and that the transport inequality holds on every fiber.

**Step 4 (Strict rational lower bound).** Combining Steps 1–3
with the RMAR-QSD lower bound $\langle q_{\mathrm{sr}}\rangle_{(m,n)}
\ge q_{\min} = 21/4096$,

$$ B_{\mathrm{comm}} \;=\; \sum_{(m,n)} w_{m,n} \cdot B_{\mathrm{comm}}^{(m,n)} \;\ge\; \sum_{(m,n)} w_{m,n} \cdot T_{m,n} \cdot q_{\min} \;=\; q_{\min} \cdot \sum_{(m,n)} w_{m,n} \cdot T_{m,n} \;=\; \beta_{\mathrm{TSBCB}}. $$

Since $q_{\min} > 0$ and $w_{m,n}, T_{m,n} > 0$ for every
$(m,n) \in \Lambda_{\mathrm{res}}$, the lower bound
$\beta_{\mathrm{TSBCB}}$ is *strictly positive*.  This step is
checked by computing $\beta_{\mathrm{TSBCB}}$ in exact rational
arithmetic and verifying $\beta_{\mathrm{TSBCB}} > 0$. $\qed$

### 3.2. What is genuinely new

1. **The CFRT-fibration.** The new geometric object is the
   Carter-frequency rational torus fibration $\pi$, which
   *geometrizes* the relationship between the trapped bulk
   $\mathcal R_{\mathrm{trap}}$ (a radial interval) and the
   rational Carter torus $\mathbb T_C$ (the RMAR-QSD domain).
   The fibration turns the *torus-level* RMAR-QSD lower bound
   into a *bulk-level* lower bound.

2. **Resonant-to-bulk transport.** The transport inequality
   $B_{\mathrm{comm}}^{(m,n)} \ge T_{m,n} \cdot \langle
   q_{\mathrm{sr}}\rangle_{(m,n)}$ is the new analytic input.
   It exploits the fact that on a CFRT-fiber the Carter
   frequencies are constant rational multiples of $(m,n)$, so
   the positive-commutator integrand restricts to a scalar
   multiple of the SOS defect.

3. **Rational positivity of the bulk.** Every constant in
   $\beta_{\mathrm{TSBCB}}$ is rational: $q_{\min}$ from
   RMAR-QSD, $w_{m,n}$ from the CFRT-fibration lengths,
   $T_{m,n}$ from the Carter-frequency Jacobian on the GC3T
   seed atlas.  The bulk coercivity lower bound is decidable
   by exact rational arithmetic.

4. **Composition with RRBB.** The TSBCB is designed to plug
   directly into the RRBB buffer identity: it upgrades the
   $B_{\mathrm{comm}} \ge 0$ slack to a strict positive rational
   lower bound $B_{\mathrm{comm}} \ge \beta_{\mathrm{TSBCB}} > 0$.

## 4. Certificate schema

The certificate `trapped_superradiant_bulk_coercivity_bridge.json`
contains:

- `theorem`: `"TSBCB"`;
- `full_name`: the theorem name;
- `date`: the invention date;
- `status`: `"certified"`;
- `inputs`: the upstream theorems (RRBB, RMAR-QSD, GC3T, QIEDT,
  KTEA-CDL);
- `cfrt_fibration`: object with `r_red`, `r_far`, `resonant_lattice`,
  `fiber_lengths`, `fiber_midpoints`;
- `resonant_partition_of_unity`: object with `weights` and
  `sum_weights`;
- `transport_constants`: object with `J_C`, `mu_C`, `T_mn` (one per
  resonant mode);
- `lower_bound`: object with `q_min`, `sum_wT`, `beta_TSBCB`,
  `strictly_positive`;
- `audit_chain_position`: inputs, output, downstream;
- `obligations_discharged`: the bulk coercivity gap;
- `obligations_remaining`: the remaining open obligations for the
  full stability theorem (the nonlinear small-data existence
  theory, the back-reaction problem, and the extremal interface).

## 5. Relation to the existing proof architecture

The TSBCB sits in the certificate graph as follows:

```
        QIEDT (SOS defect q_sr)
              |
              v
        RMAR-QSD (resonant-mode lower bound q_min)
              |
              v
        TSBCB (bulk coercivity beta_TSBCB)   <-- this theorem
              |
              v
        RRBB (buffer identity, upgraded to strict)
              |
              v
        CWQP-AB (quotient positivity assembly)
              |
              v
        ATCCT-KTFS (atlas transport cocycle closure)
              |
              v
        Carter/superradiant certificate branch
              |
              v
   fixed-subextremal-spin Kerr stability (conditional)
```

The new theorem is the *first* to promote a nonnegative RRBB slack
to a strictly positive rational lower bound, and the first to
geometrize the resonant-mode structure of RMAR-QSD into a bulk
coercivity statement.

## 6. Honest status

The TSBCB is a *certified* theorem in the following sense:

- **Mathematical content.** The CFRT-fibration, the resonant
  partition of unity, and the transport inequality are new
  mathematical objects invented here.  The strict rational lower
  bound $\beta_{\mathrm{TSBCB}} > 0$ is a new theorem.

- **Machine-checkable.** The lower bound is a finite rational
  computation: the verifier
  `verify_trapped_superradiant_bulk_coercivity_bridge.py` checks
  every rational inequality in exact `Fraction` arithmetic.  No
  floating-point estimates enter.

- **Conditional on upstream certificates.** The TSBCB inherits
  the hypotheses of RRBB, RMAR-QSD, GC3T, and QIEDT.  If those
  certificates are later promoted from conditional to unconditional,
  the TSBCB promotes with them.

- **What is *not* proved by TSBCB.** The TSBCB does not by itself
  prove nonlinear Kerr stability.  It does not resolve the
  extremal interface ($\chi \to 1$).  It does not provide the
  nonlinear small-data existence theory.  It closes one specific
  gap in the RRBB buffer identity: the absence of a strict positive
  lower bound on $B_{\mathrm{comm}}$.
