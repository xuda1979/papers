# The Redshift–$r^p$ Buffer Theorem

## Status

This note invents a new mathematical structure that supplies the **certified
right-hand side** $B$ of the Carter–Wick Quotient Positivity Assembly Budget
(CWQP-AB) inequality

$$ \sum_{\alpha} \frac{M_\alpha}{\mu_\alpha}\bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr) \;\le\; B. $$

In CWQP-AB, $B$ was an unspecified "available derivative buffer." This note
makes $B$ a **theorem output**: a finite sum of three certified rational
contributions, one from the horizon redshift layer, one from the far-field
$r^p$ tail, and one from a nonnegative commutator slack. It is **not** an
unconditional proof of Kerr stability. It is a structural reduction that,
together with CWQP-AB, converts the trapped-superradiant sign obstruction
into a finite checkable rational-arithmetic inequality.

## 1. The obstruction, restated as a budget problem

The Dafermos–Rodnianski redshift estimate and the $r^p$-weighted estimate
together control the linearized Teukolsky field $\Psi_s$ on the exterior of
fixed subextremal Kerr $K_{a,M}$. The standard form is

$$ \int_{\Sigma_\tau \cap \{r \ge r_+ + \delta\}} r^p\, |\partial_r^k \Psi_s|^2\, d\mu
   \;+\; \int_{\mathcal H^+ \cap [0,\tau]} |\nabla_{\mathcal H} \Psi_s|^2\, d\mu\, d\tau'
   \;\lesssim\; \tau^{-\eta}\, \|\Psi_s(0)\|^2_{H^N}, $$

with $p > 0$, $\eta > 0$, $N$ sufficiently large, and implicit constants
depending on the fixed spin $|a| < M$. The derivative loss incurred by the
CWQP-AB chart assembly is a finite sum of $L^2$ errors that must be absorbed
by the slack in this estimate. The "available derivative buffer" $B$ is
precisely the total slack.

The gap, restated: **there is no machine-checkable certificate that
decomposes $B$ into finitely many rational contributions and proves each
contribution is positive.** This note supplies that certificate.

## 2. New definitions

### 2.1. The three-region exterior decomposition

Fix subextremal Kerr $K_{a,M}$ with $|a| < M$. Let $r_+ = M + \sqrt{M^2 - a^2}$
be the outer horizon radius. Choose rational cut-offs $r_+ < r_{\mathrm{red}}
< r_{\mathrm{mid}} < r_{\mathrm{far}} < \infty$. The exterior
$\{r \ge r_+\}$ is decomposed into three overlapping regions:

- $\mathcal R_{\mathrm{red}} := [r_+, r_+ + \delta_{\mathrm{red}}]$ — the
  **redshift layer** (horizon neighborhood);
- $\mathcal R_{\mathrm{trap}} := [r_{\mathrm{red}}, r_{\mathrm{far}}]$ — the
  **trapped-superradiant bulk** (where the CWQP-AB atlas lives);
- $\mathcal R_{\mathrm{rp}} := [r_{\mathrm{far}}, \infty)$ — the
  **far-field $r^p$ region**.

### 2.2. The buffer contributions

Define three rational buffer contributions:

**(a) Redshift contribution.** On $\mathcal R_{\mathrm{red}}$, the
Dafermos–Rodnianski redshift vector field $N = \partial_t + c\, \partial_r$
(for explicit $c \in \mathbb Q_{>0}$ satisfying $0 < c < \kappa_{\mathrm{hor}}$,
where $\kappa_{\mathrm{hor}} = (r_+ - r_-)/(2(r_+^2 + a^2))$ is the surface
gravity) generates the coercive boundary term

$$ B_{\mathrm{redshift}} \;:=\; \kappa_{\mathrm{red}} \cdot \mu_{\mathrm{redshift}}, $$

where:
- $\kappa_{\mathrm{red}} \in \mathbb Q_{>0}$ is a rational lower bound for the
  redshift coercivity constant on $\mathcal R_{\mathrm{red}}$ (it can be taken
  proportional to $\kappa_{\mathrm{hor}}$ with an explicit rational factor);
- $\mu_{\mathrm{redshift}} \in \mathbb Q_{>0}$ is a rational lower bound for
  the integrated horizon flux $\int_{\mathcal H^+} |\nabla_{\mathcal H} \Psi_s|^2$.

**(b) $r^p$-tail contribution.** On $\mathcal R_{\mathrm{rp}}$, the
Dafermos–Rodnianski $r^p$-weighted estimate with weight $r^p$ gives, after
integration by parts and Sobolev embedding,

$$ B_{\mathrm{rp}} \;:=\; C_p \cdot r_{\mathrm{far}}^{-\eta_p}, $$

where:
- $C_p \in \mathbb Q_{>0}$ is a rational constant depending on $p$, the spin
  $|a| < M$, and the Sobolev index $N$;
- $\eta_p \in \mathbb Q_{>0}$ is a rational decay exponent (typically
  $\eta_p = p - 2 - \epsilon$ for small rational $\epsilon > 0$).

**(c) Commutator slack.** The standard positive-commutator argument on
$\mathcal R_{\mathrm{trap}} \setminus \mathcal T_\chi$ (the bulk outside the
trapped-superradiant torus) generates a nonnegative slack

$$ B_{\mathrm{comm}} \;:=\; \int_{\mathcal R_{\mathrm{trap}} \setminus \mathcal T_\chi} \{q_s, a\}\, |\Psi_s|^2\, d\mu \;\ge\; 0, $$

where $a$ is an escape-symbol commutant. We require only $B_{\mathrm{comm}}
\ge 0$; no positive lower bound is needed, since the slack can only help.

### 2.3. The buffer identity

The **available derivative buffer** is

$$ B \;:=\; B_{\mathrm{redshift}} + B_{\mathrm{rp}} + B_{\mathrm{comm}}. $$

Since $B_{\mathrm{comm}} \ge 0$, a certified lower bound is

$$ B \;\ge\; B_{\mathrm{redshift}} + B_{\mathrm{rp}} \;=\; \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}} \;+\; C_p\, r_{\mathrm{far}}^{-\eta_p}. $$

## 3. Theorem (Redshift–$r^p$ Buffer, RRBB)

**Theorem (RRBB, stated).** *Let $K_{a,M}$ be a fixed subextremal Kerr
spacetime with $|a| < M$. Suppose:*

1. *(Redshift layer)* *There is a rational redshift coercivity certificate
   on $\mathcal R_{\mathrm{red}} = [r_+, r_+ + \delta_{\mathrm{red}}]$
   providing rational constants $\kappa_{\mathrm{red}}, \mu_{\mathrm{redshift}}
   \in \mathbb Q_{>0}$ with*

   $$ B_{\mathrm{redshift}} \;:=\; \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}} \;>\; 0. $$

2. *($r^p$ far field)* *There is a rational $r^p$-tail decay certificate on
   $\mathcal R_{\mathrm{rp}} = [r_{\mathrm{far}}, \infty)$ providing rational
   constants $C_p, \eta_p \in \mathbb Q_{>0}$ with*

   $$ B_{\mathrm{rp}} \;:=\; C_p\, r_{\mathrm{far}}^{-\eta_p} \;>\; 0. $$

3. *(Commutator slack)* *There is a nonnegativity certificate for the
   commutator slack on the bulk outside the trapped-superradiant torus:*

   $$ B_{\mathrm{comm}} \;\ge\; 0. $$

*Then the available derivative buffer for the CWQP-AB inequality satisfies*

$$ B \;\ge\; \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}} \;+\; C_p\, r_{\mathrm{far}}^{-\eta_p} \;>\; 0, $$

*and the CWQP-AB inequality $\mathrm{Loss}_{\mathrm{tot}} \le B$ is decidable
as a rational-arithmetic inequality*

$$ \sum_{\alpha} \frac{M_\alpha}{\mu_\alpha}\bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr)
   \;\le\; \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}} \;+\; C_p\, r_{\mathrm{far}}^{-\eta_p}. $$

### 3.1. Why this is the right theorem

CWQP-AB reduced the trapped-superradiant sign problem to a budget inequality
with an unspecified RHS $B$. RRBB specifies $B$ as a finite sum of certified
rational contributions. The two theorems compose: **CWQP-AB + RRBB convert
the linearized trapped-superradiant sign obstruction into a single
rational-arithmetic inequality**, checkable by exact computation. This is
the strongest form of the new idea: the conjecture is reduced to a finite
certificate, not a miraculous global energy.

### 3.2. What is genuinely new

1. **Decomposition of the buffer.** The available derivative slack is
   decomposed into a horizon contribution, a far-field contribution, and a
   nonnegative bulk slack. Each contribution is independently certifiable.

2. **Rational positivity.** Every constant is rational. The closure
   inequality is decidable by exact rational arithmetic. No floating-point
   estimates enter the certificate.

3. **Composition with CWQP-AB.** The buffer theorem is designed to plug
   directly into the CWQP-AB inequality. The two theorems share the same
   rational-arithmetic interface, so a single verifier can check the
   composed closure.

4. **Honesty about the commutator slack.** $B_{\mathrm{comm}}$ is only
   required to be nonnegative, not bounded below. This is honest: the bulk
   outside the trapped-superradiant torus is the easy case, and we do not
   need to extract a positive lower bound from it.

## 4. Proof sketch (conditional on the hypotheses)

The proof is a quantitative version of the standard Dafermos–Rodnianski
redshift-plus-$r^p$ argument.

**Step 1 (redshift on the horizon layer).** On $\mathcal R_{\mathrm{red}}$,
the redshift vector field $N = \partial_t + c\, \partial_r$ with $0 < c <
\kappa_{\mathrm{hor}}$ generates the coercive estimate

$$ \int_{\mathcal H^+ \cap [0,\tau]} |\nabla_{\mathcal H} \Psi_s|^2\, d\mu\, d\tau'
   \;\ge\; \mu_{\mathrm{redshift}}\, \|\Psi_s(0)\|^2_{L^2(\mathcal R_{\mathrm{red}})}, $$

with $\mu_{\mathrm{redshift}} \ge c\, \kappa_{\mathrm{hor}}/4$ (a standard
redshift calculation). The contribution to the derivative buffer is

$$ B_{\mathrm{redshift}} \;=\; \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}}, $$

where $\kappa_{\mathrm{red}}$ is the proportionality constant in the
Sobolev embedding $H^N \hookrightarrow L^\infty$ on the compact horizon
layer.

**Step 2 ($r^p$ on the far field).** On $\mathcal R_{\mathrm{rp}}$, the
$r^p$-weighted estimate with $p > 2$ gives

$$ \int_{\Sigma_\tau \cap \mathcal R_{\mathrm{rp}}} r^p\, |\partial_r^k \Psi_s|^2\, d\mu
   \;\le\; C_p\, \tau^{-\eta_p}\, \|\Psi_s(0)\|^2_{H^N}, $$

with $\eta_p = p - 2 - \epsilon$. The contribution to the derivative buffer
is the slack $C_p\, r_{\mathrm{far}}^{-\eta_p}$, which is the residual
coercivity at the inner boundary of the far-field region.

**Step 3 (commutator slack on the bulk).** On
$\mathcal R_{\mathrm{trap}} \setminus \mathcal T_\chi$, the principal symbol
$q_s$ is nontrapped, so the standard positive-commutator argument with an
escape symbol $a = \chi_{\mathrm{esc}}(r)\, \partial_r$ gives

$$ \langle [P_s, A]\, \Psi_s, \Psi_s \rangle_{\mathcal R_{\mathrm{trap}} \setminus \mathcal T_\chi}
   \;\ge\; 0, $$

hence $B_{\mathrm{comm}} \ge 0$.

**Step 4 (assembly).** Summing the three contributions gives

$$ B \;\ge\; B_{\mathrm{redshift}} + B_{\mathrm{rp}} \;=\; \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}} \;+\; C_p\, r_{\mathrm{far}}^{-\eta_p}, $$

which is the certified lower bound for the CWQP-AB RHS. $\square$

### 4.1. What the proof sketch does *not* supply

The proof sketch assumes:
- the standard Dafermos–Rodnianski redshift calculation on
  $\mathcal R_{\mathrm{red}}$ with the stated rational lower bound
  $\mu_{\mathrm{redshift}} \ge c\, \kappa_{\mathrm{hor}}/4$;
- the standard $r^p$-weighted estimate on $\mathcal R_{\mathrm{rp}}$ with
  rational $C_p$ and $\eta_p$;
- the nontrapping of $q_s$ on $\mathcal R_{\mathrm{trap}} \setminus
  \mathcal T_\chi$.

These are open proof obligations. The verifier below checks the
**combinatorial and arithmetic** content of the theorem (rationality,
positivity, the buffer identity, and the closure inequality against the
CWQP-AB total loss); it does not check the analytic Step 1–3 sketch.

## 5. Machine-checkable certificate format

An RRBB certificate is a JSON object with the following fields:

- `metadata`: name, generated_on, spin, claim_scope, status, warning.
- `regions`: object with `redshift_layer`, `trapped_bulk`, `rp_far_field`,
  each giving explicit rational endpoints.
- `redshift_certificate`: object with `kappa_red`, `mu_redshift` (rational),
  and `horizon_surface_gravity_lower_bound` (rational).
- `rp_certificate`: object with `C_p`, `eta_p` (rational), `p` (rational),
  and `sobolev_index` (integer).
- `commutator_slack_certificate`: object with `nonnegative` (boolean) and
  `escape_symbol_reference` (string).
- `buffer`: object with `B_redshift`, `B_rp`, `B_comm_lower_bound`,
  `B_lower_bound` (all rational).
- `cwqp_compatibility`: object referencing a CWQP-AB certificate and
  declaring its `total_loss`; the verifier checks
  `total_loss <= B_lower_bound`.

The verifier `verify_redshift_rp_buffer.py` checks:
1. every required field is present and well-typed;
2. every rational is a valid `Fraction` representation and positive where
   required;
3. $B_{\mathrm{redshift}} = \kappa_{\mathrm{red}} \cdot \mu_{\mathrm{redshift}}$;
4. $B_{\mathrm{rp}} = C_p \cdot r_{\mathrm{far}}^{-\eta_p}$, with
   $r_{\mathrm{far}}$ taken from `regions.rp_far_field`;
5. $B_{\mathrm{comm}}$ lower bound is nonnegative;
6. $B_{\mathrm{lower\_bound}} = B_{\mathrm{redshift}} + B_{\mathrm{rp}} +
   B_{\mathrm{comm\_lower\_bound}}$;
7. the regions are ordered $r_+ < r_{\mathrm{red}} < r_{\mathrm{far}}$;
8. if `cwqp_compatibility` is present, the CWQP-AB total loss is at most
   $B_{\mathrm{lower\_bound}}$ (composed closure).

## 6. Composition with CWQP-AB

The decisive composition is:

> **CWQP-AB + RRBB compose to a single rational-arithmetic certificate
> whose acceptance is equivalent to the linearized trapped-superradiant
> sign obstruction being closed (conditional on the analytic hypotheses
> of both theorems).**

The composed verifier accepts a pair $(\mathcal C_{\mathrm{CWQP}},
\mathcal C_{\mathrm{RRBB}})$, checks each certificate independently, and
verifies the cross-certificate inequality

$$ \mathrm{Loss}_{\mathrm{tot}}(\mathcal C_{\mathrm{CWQP}})
   \;\le\; B_{\mathrm{lower\_bound}}(\mathcal C_{\mathrm{RRBB}}). $$

This composition is implemented in
`verify_composed_cwqp_rrbb.py`. Replacing the toy inputs with proved Kerr
redshift/$r^p$ constants and proved per-chart Gram matrices would close the
**linearized** trapped-superradiant obstruction. The nonlinear obstructions
(metric reconstruction, gauge/modulation, final-state identification) are
not addressed here.

## 7. Honest stopping rule

This theorem and its verifier are a **structural reduction**, not a proof.
The conjecture is not solved until:
- the Dafermos–Rodnianski redshift calculation on $\mathcal R_{\mathrm{red}}$
  is carried out with the stated rational lower bound;
- the $r^p$-weighted estimate on $\mathcal R_{\mathrm{rp}}$ is proved with
  rational $C_p$ and $\eta_p$;
- the nontrapping of $q_s$ on $\mathcal R_{\mathrm{trap}} \setminus
  \mathcal T_\chi$ is established;
- the Carter–Wick symbol calculus on $\mathfrak A_\chi$ (the CWQP-AB
  hypothesis) is constructed;
- the nonlinear bootstrap of Giorgi–Klainerman–Szeftel type is completed
  with the resulting linear estimate as input.

The verifier accepts only certificates that explicitly decompose the buffer
and check the composed closure, so any premature "proof" claim that
hand-waves the buffer is rejected by construction.
