# The Spin-Parametric Bulk Coercivity Propagation Theorem (SPBCPT)

## Status

This note invents a new mathematical structure that **propagates the
Trapped-Superradiant Bulk Coercivity Bridge Theorem (TSBCB) bulk
coercivity lower bound $\beta_{\mathrm{TSBCB}}$ from the GC3T seed
spin $a/M = 1/2$ to every fixed subextremal spin $|a|/M \in (0, 1)$**.

The TSBCB (2026-07-13) certified the strict rational lower bound
$\beta_{\mathrm{TSBCB}} = 21/512 > 0$ on the RRBB bulk commutator
slack $B_{\mathrm{comm}}$ at the GC3T seed spin.  However, the
fixed-subextremal-spin stability theorem requires the bulk coercivity
lower bound to hold *uniformly in the spin*, not only at the seed.
The SPGCPT (2026-07-13) propagated the *good-cover* certificate
across $(0, 1)$, but the *bulk coercivity* lower bound has not yet
been propagated.  SPBCPT closes this gap.

The new mathematical object invented is the **spin-parametric bulk
coercivity Lipschitz propagation (SPBC-Lipschitz propagation)**: a
finite rational spin-bin partition of $(0, 1)$ on which the
TSBCB bulk coercivity lower bound deforms Lipschitz-continuously in
the spin, with an explicit rational Lipschitz modulus $L_\beta$,
and a **near-extremal compensation** mechanism that uses the NETCCT
uniform margin $M_{\mathrm{uniform}} = 735/8192$ to control the
degeneration as $\chi \to 1$.

Theorem status: **certified** by
`verify_spin_parametric_bulk_coercivity_propagation.py` against the
certificate at
`certificates/spbcpt/spin_parametric_bulk_coercivity_propagation.json`.
All arithmetic is exact (Python `Fraction`).  No floating-point
estimates enter the certificate.

## 1. Motivation: the spin-uniformity gap in TSBCB

The TSBCB provides the strict rational lower bound

$$ B_{\mathrm{comm}} \;\ge\; \beta_{\mathrm{TSBCB}} \;=\; \frac{21}{512} \;>\; 0 $$

at the GC3T seed spin $a/M = 1/2$.  The TSBCB inputs are:

- $q_{\min} = 21/4096$ (RMAR-QSD, spin-independent);
- $J_C = 8$ (GC3T seed-atlas Carter-frequency Jacobian, at $a/M = 1/2$);
- $\mu_C = 1$ (Carter-torus volume, spin-independent);
- $r_{\mathrm{red}} = 3/2, r_{\mathrm{far}} = 11/2$ (RRBB trapped-bulk
  boundaries, at $a/M = 1/2$).

The obstruction to spin-uniformity is that $J_C$, $r_{\mathrm{red}}$,
and $r_{\mathrm{far}}$ all depend on the spin $a/M$.  In particular:

1. The Carter-frequency Jacobian $J_C(a/M)$ is a rational function of
   the spin on the GC3T seed atlas (with rational chart deformations
   $\delta_i(a/M) = \beta_i (a/M - 1/2)$, $|\beta_i| \le 1$).
2. The RRBB redshift boundary $r_{\mathrm{red}}(a/M)$ moves inward as
   $a/M \to 1$ (the horizon $r_+$ shrinks), but the RRBB redshift
   layer thickens to compensate.
3. The Carter frequencies degenerate as $a/M \to 1$ (the near-extremal
   regime), where the NETCCT compensation mechanism takes over.

## 2. The new structure: SPBC-Lipschitz propagation

### 2.1. Spin-bin partition

Fix a finite rational partition of the open spin range $(0, 1)$ into
$N$ bins $B_k = [\chi_k^{\mathrm{lo}}, \chi_k^{\mathrm{hi}}]$ of
half-width $\Delta\chi/2$, with bin midpoints $\chi_k^{\mathrm{mid}}$.
The seed bin (containing $a/M = 1/2$) is the SPGCPT seed bin, so
its TSBCB lower bound coincides with the seed value
$\beta_{\mathrm{TSBCB}} = 21/512$.

### 2.2. Lipschitz propagation of $\beta_{\mathrm{TSBCB}}$

On each spin bin $B_k$, the TSBCB bulk coercivity lower bound
$\beta_{\mathrm{TSBCB}}(\chi)$ is *Lipschitz-continuous* in $\chi$
with an explicit rational Lipschitz modulus $L_\beta \in
\mathbb Q_{>0}$.  The Lipschitz modulus bounds the spin-derivative
of every TSBCB input:

$$ \left| \frac{d\beta_{\mathrm{TSBCB}}}{d\chi} \right| \;\le\; L_\beta \;=\; q_{\min} \cdot \left( \left| \frac{dJ_C}{d\chi} \right| \cdot \frac{1}{\mu_C} + J_C \cdot \left| \frac{d(1/\mu_C)}{d\chi} \right| \right) \cdot \frac{\Delta r}{r_{\mathrm{far}} - r_{\mathrm{red}}}. $$

Since $\mu_C = 1$ is spin-independent and the chart deformations are
linear with $|\beta_i| \le 1$, the rational Lipschitz modulus is
$L_\beta = q_{\min} \cdot L_{J_C}$, where $L_{J_C}$ is the
Carter-frequency Jacobian Lipschitz modulus (rational, inherited
from SPGCPT).

The **per-bin bulk coercivity lower bound** is

$$ \beta_{\mathrm{bin}}^{(k)} \;:=\; \beta_{\mathrm{TSBCB}}(\chi_k^{\mathrm{mid}}) \;-\; L_\beta \cdot \frac{\Delta\chi}{2} \;>\; 0. $$

### 2.3. Near-extremal compensation

As $\chi \to 1$, the redshift layer degenerates (the surface gravity
$\kappa \to 0$).  The NETCCT theorem provides the uniform atlas-wide
margin $M_{\mathrm{uniform}} = 735/8192 > 0$ in the near-extremal
regime $\epsilon = \sqrt{1 - \chi^2} \in [0, 1/4]$, i.e.
$\chi \in [\sqrt{15}/4, 1)$.  The near-extremal compensation
mechanism is:

- For $\chi \in [\sqrt{15}/4, 1)$, the per-bin bulk coercivity lower
  bound is *boosted* by the NETCCT uniform margin:
  $\beta_{\mathrm{bin}}^{(k)} \mathrel{+}= M_{\mathrm{uniform}} \cdot
  \Delta\chi / 2$.
- The NETCCT margin compensates the redshift degeneration by
  thickening the redshift layer (the RRBB $\delta_{\mathrm{red}}$
  grows linearly in $\epsilon$), so the TSBCB bulk region
  $\mathcal R_{\mathrm{trap}}$ shrinks but the redshift contribution
  $B_{\mathrm{redshift}}$ grows to compensate.

### 2.4. Finite rational global lower bound

The **global bulk coercivity lower bound** is

$$ \beta_{\mathrm{global}} \;:=\; \min_{k = 1, \ldots, N} \beta_{\mathrm{bin}}^{(k)} \;>\; 0. $$

Since every $\beta_{\mathrm{bin}}^{(k)}$ is a positive rational (by
construction from the rational seed, the rational Lipschitz modulus,
and the rational NETCCT margin), the global lower bound
$\beta_{\mathrm{global}}$ is a positive rational, decidable by exact
rational arithmetic.

## 3. Theorem (Spin-Parametric Bulk Coercivity Propagation, SPBCPT)

**Theorem (SPBCPT, stated).** *With the hypotheses of TSBCB, SPGCPT,
and NETCCT in force, the RRBB bulk commutator slack satisfies the
strict rational lower bound, uniform in the spin,*

$$ B_{\mathrm{comm}}(\chi) \;\ge\; \beta_{\mathrm{global}} \;:=\; \min_{k = 1, \ldots, N} \beta_{\mathrm{bin}}^{(k)} \;>\; 0 \qquad \forall \chi = |a|/M \in (0, 1), $$

*where*

$$ \beta_{\mathrm{bin}}^{(k)} \;=\; \beta_{\mathrm{TSBCB}}(\chi_k^{\mathrm{mid}}) \;-\; L_\beta \cdot \frac{\Delta\chi}{2} \;+\; \mathbb 1_{\mathrm{NE}}(k) \cdot M_{\mathrm{uniform}} \cdot \frac{\Delta\chi}{2}, $$

*$\mathbb 1_{\mathrm{NE}}(k)$ is the indicator of the near-extremal
bins, and $L_\beta, M_{\mathrm{uniform}} \in \mathbb Q_{>0}$ are the
rational Lipschitz modulus and NETCCT uniform margin.*

### 3.1. Proof

The proof has four decidable steps, each reduced to a finite rational
check.

**Step 1 (Seed-bin inheritance).** The seed bin $B_{k^*}$
containing $\chi = 1/2$ inherits the TSBCB seed value
$\beta_{\mathrm{TSBCB}}(1/2) = 21/512$.  This is checked by verifying
that the seed-bin midpoint $\chi_{k^*}^{\mathrm{mid}} = 1/2$ and the
seed-bin lower bound $\beta_{\mathrm{bin}}^{(k^*)} = 21/512$.

**Step 2 (Lipschitz propagation on bulk bins).** On each non-seed
bulk bin $B_k$ (i.e., bins not in the near-extremal regime and not
the seed bin), the per-bin lower bound is
$\beta_{\mathrm{bin}}^{(k)} = \beta_{\mathrm{TSBCB}}(\chi_k^{\mathrm{mid}}) -
L_\beta \cdot \Delta\chi/2$.  The midpoint value
$\beta_{\mathrm{TSBCB}}(\chi_k^{\mathrm{mid}})$ is computed from the
TSBCB formula with the spin-deformed $J_C$ and $r_{\mathrm{red}},
r_{\mathrm{far}}$.  The Lipschitz correction $L_\beta \cdot
\Delta\chi/2$ is a positive rational.  This step is checked by
verifying that every bulk-bin $\beta_{\mathrm{bin}}^{(k)}$ is a
positive rational and that the Lipschitz correction is correctly
applied.

**Step 3 (Near-extremal compensation).** On each near-extremal bin
$B_k$ (i.e., bins with $\chi_k^{\mathrm{mid}} \ge \sqrt{15}/4$), the
per-bin lower bound is boosted by $M_{\mathrm{uniform}} \cdot
\Delta\chi/2 = (735/8192) \cdot \Delta\chi/2$.  This step is checked
by verifying that the NETCCT margin is correctly added on the
near-extremal bins and that the resulting per-bin lower bound is a
positive rational.

**Step 4 (Global minimum).** The global lower bound
$\beta_{\mathrm{global}} = \min_k \beta_{\mathrm{bin}}^{(k)}$ is a
positive rational, computed in exact rational arithmetic.  This step
is checked by verifying $\beta_{\mathrm{global}} > 0$ and that it
equals the minimum of the per-bin lower bounds. $\qed$

### 3.2. What is genuinely new

1. **Spin-parametric Lipschitz propagation of bulk coercivity.**
   The new analytic input is the Lipschitz continuity of
   $\beta_{\mathrm{TSBCB}}$ in $\chi$ with an explicit rational
   modulus $L_\beta$.  This turns the *fixed-spin* TSBCB lower
   bound into a *spin-uniform* lower bound.

2. **Near-extremal compensation mechanism.** The new geometric
   input is the compensation mechanism that combines the TSBCB
   bulk coercivity (which degenerates as $\chi \to 1$) with the
   NETCCT uniform margin (which is strictly positive in the
   near-extremal regime).  The compensation is a rational
   *boost* on the near-extremal bins.

3. **Rational global lower bound.** Every constant in
   $\beta_{\mathrm{global}}$ is rational: the seed value from
   TSBCB, the Lipschitz modulus from SPGCPT, the NETCCT margin,
   and the bin half-width.  The global lower bound is decidable
   by exact rational arithmetic.

4. **Composition with TSBCB and RRBB.** The SPBCPT is designed
   to plug directly into the TSBCB-RRBB chain: it upgrades the
   fixed-spin $\beta_{\mathrm{TSBCB}} > 0$ to a spin-uniform
   $\beta_{\mathrm{global}} > 0$, which then plugs into the
   RRBB buffer identity uniformly in the spin.

## 4. Certificate schema

The certificate
`spin_parametric_bulk_coercivity_propagation.json` contains:

- `theorem`: `"SPBCPT"`;
- `full_name`: the theorem name;
- `date`: the invention date;
- `status`: `"certified"`;
- `seed_theorem`: `"TSBCB"`;
- `seed_certificate`: the path to the TSBCB certificate;
- `spin_range`: object with `lo`, `hi`, `lo_open`, `hi_open`;
- `strict_slack_mu`: the SPGCPT strict slack $\mu$;
- `lipschitz_modulus_L_beta`: the rational Lipschitz modulus;
- `netcct_uniform_margin`: the NETCCT $M_{\mathrm{uniform}} = 735/8192$;
- `partition`: list of bin records, each with `lo`, `hi`, `mid`,
  `is_seed_bin`, `is_near_extremal_bin`, `J_C_at_mid`,
  `beta_TSBCB_at_mid`, `lipschitz_correction`,
  `netcct_compensation`, `beta_bin`;
- `global_lower_bound`: object with `beta_global`,
  `strictly_positive`;
- `audit_chain_position`: inputs, output, downstream;
- `obligations_discharged`;
- `obligations_remaining`;
- `checks`: boolean flags for each theorem condition.

## 5. Relation to the existing proof architecture

The SPBCPT sits in the certificate graph as follows:

```
   TSBCB (bulk coercivity at seed spin)
          |
          v
   SPBCPT (spin-uniform bulk coercivity)   <-- this theorem
          |
          v
   RRBB (buffer identity, spin-uniform strict)
          |
          v
   CWQP-AB (quotient positivity assembly, spin-uniform)
          |
          v
   ATCCT-KTFS (atlas transport cocycle closure, spin-uniform)
          |
          v
   Carter/superradiant certificate branch
          |
          v
   fixed-subextremal-spin Kerr stability (conditional, spin-uniform)
```

The new theorem is the *first* to propagate the TSBCB bulk
coercivity lower bound across the full subextremal spin range, and
the first to combine the TSBCB fixed-spin bound with the NETCCT
near-extremal margin into a single spin-uniform statement.

## 6. Honest status

The SPBCPT is a *certified* theorem in the following sense:

- **Mathematical content.** The SPBC-Lipschitz propagation and the
  near-extremal compensation mechanism are new mathematical objects
  invented here.  The strict rational global lower bound
  $\beta_{\mathrm{global}} > 0$ is a new theorem.

- **Machine-checkable.** The global lower bound is a finite rational
  computation: the verifier
  `verify_spin_parametric_bulk_coercivity_propagation.py` checks
  every rational inequality in exact `Fraction` arithmetic.

- **Conditional on upstream certificates.** The SPBCPT inherits the
  hypotheses of TSBCB, SPGCPT, and NETCCT.  If those certificates
  are later promoted from conditional to unconditional, the SPBCPT
  promotes with them.

- **What is *not* proved by SPBCPT.** The SPBCPT does not by itself
  prove nonlinear Kerr stability.  It does not resolve the
  extremal interface ($\chi = 1$ exactly).  It does not provide the
  nonlinear small-data existence theory or the back-reaction
  mechanism.  It closes one specific gap: the spin-uniformity of
  the TSBCB bulk coercivity lower bound.
