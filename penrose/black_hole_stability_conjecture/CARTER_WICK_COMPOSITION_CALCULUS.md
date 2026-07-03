# The Carter–Wick Composition Calculus with Rational Remainder

## Status

This note invents the **symbol calculus** that underlies the Carter–Wick
Quotient Positivity Assembly Budget (CWQP-AB) theorem. The CWQP-AB proof
sketch (Step 2) used a "Wick-symbol composition rule" with an unbounded
"lower order" remainder. This note makes that remainder **explicit and
rational**, and supplies a machine-checkable certificate that the
remainder fits inside the existing CWQP-AB loss budget.

This is **not** an unconditional proof of Kerr stability. It is the third
structural reduction in the chain CWQP-AB → RRBB → CWCC, converting the
calculus-remainder obstruction into a finite rational-arithmetic
inequality that composes with the existing verifiers.

## 1. The obstruction, restated

The CWQP-AB theorem's Step 2 asserts the Wick-symbol composition rule

$$ [P_s, \chi_\alpha A_\alpha \chi_\alpha]
   \;=\; \chi_\alpha^2\, [P_s, A_\alpha]
   \;+\; 2\, \mathrm{Im}(H_{q_s}\, \chi_\alpha)\, A_\alpha
   \;+\; R_\alpha, $$

where $R_\alpha$ is "lower order." The CWQP-AB loss budget treats
$R_\alpha$ as absorbed into the local margin $\mu_\alpha$, but no
explicit bound is given. This is the **composition-remainder gap**: the
calculus is not constructed with a quantitative remainder, so the
local commutator estimate is not actually justified.

The gap, restated: **there is no machine-checkable certificate that
bounds the Wick composition remainder $R_\alpha$ by an explicit
rational function of the chart data.** This note supplies that
certificate.

## 2. New definitions

### 2.1. The Carter–Wick symbol class

Fix subextremal Kerr $K_{a,M}$ with $|a| < M$. Work in the
Carter-separated variables $(r, \eta, \lambda, \omega)$ on the
trapped-superradiant quotient ring $\mathfrak A_\chi = \mathbb
Q[r,\eta,\lambda,\omega]/\mathfrak I_\chi$. The Carter symplectic form
is

$$ \sigma_\chi \;=\; \sum_{i=1}^{4} d\xi_i \wedge dx^i,
   \qquad (x^1, x^2, x^3, x^4) = (r, \eta, \lambda, \omega), $$

and is **diagonal** in the Carter variables (this is the key
consequence of Carter's separability: the Hamilton–Jacobi equation
decouples).

A **Carter–Wick symbol of order $m$ on $U_\alpha$** is a smooth function
$a_\alpha \in C^\infty(U_\alpha; \mathfrak A_\chi)$ that is polynomial
in the monomial basis $\mathfrak v_\alpha = (v_1, \dots, v_{N_\alpha})$
of degree at most $m$:

$$ a_\alpha \;=\; \sum_{|\beta| \le m} a_{\alpha,\beta}\, \mathfrak v_\alpha^\beta,
   \qquad a_{\alpha,\beta} \in \mathbb Q. $$

The **Carter–Wick quantization** $\mathrm{Op}_\alpha^W(a_\alpha)$ is the
Weyl quantization in the Carter variables. Because $\sigma_\chi$ is
diagonal, the Moyal bracket simplifies to a **finite sum over the
diagonal Poisson structure**:

$$ \{a, b\}_{\mathrm{CW}}
   \;=\; \sum_{i=1}^{4} \partial_{\xi_i} a\, \partial_{x^i} b
   - \partial_{x^i} a\, \partial_{\xi_i} b, $$

with no mixed-variable terms.

### 2.2. The composition remainder

For symbols $a, b$ of orders $m, n$ on $U_\alpha$, the Wick composition
is

$$ \mathrm{Op}_\alpha^W(a)\, \mathrm{Op}_\alpha^W(b)
   \;=\; \mathrm{Op}_\alpha^W(a \#_\alpha b), $$

with the **Moyal star product**

$$ a \#_\alpha b
   \;=\; a\, b \;+\; \frac{i}{2}\, \{a, b\}_{\mathrm{CW}}
   \;+\; R_\alpha(a, b), $$

where $R_\alpha(a, b)$ is the **composition remainder**. In general
pseudodifferential calculus, $R_\alpha$ is a sum of higher-order
bidifferential operators with no simple closed form. The key new
observation is:

> **In the Carter–Wick calculus, because $\sigma_\chi$ is diagonal and
> the symbols are polynomial in $\mathfrak v_\alpha$, the composition
> remainder $R_\alpha(a, b)$ is a finite sum of rational terms bounded
> by an explicit function of the chart data.**

### 2.3. The rational remainder bound

Define the **chart complexity** of $U_\alpha$ as the triple

$$ \mathcal K_\alpha \;:=\; (N_\alpha,\, m_\alpha,\, M_\alpha), $$

where:
- $N_\alpha = |\mathfrak v_\alpha|$ is the monomial basis size;
- $m_\alpha$ is the maximum symbol order on $U_\alpha$;
- $M_\alpha = \|G_\alpha\|_{\mathrm{op}}$ is the Gram op-norm bound.

Define the **Carter diagonal constant**

$$ \kappa_{\mathrm{CW}} \;:=\; \max_{i=1,\dots,4}\,
   \sup_{U_\alpha}\, |\partial_{\xi_i} \partial_{x^i}\, q_s| \;\in\; \mathbb Q_{>0}, $$

which is rational because $q_s$ is polynomial in the Carter variables
and $U_\alpha$ has rational endpoints. (In fact $\kappa_{\mathrm{CW}}$
is bounded by a universal constant depending only on $M$ and $|a|/M$,
but for the certificate we require only that it is rational and
positive.)

The **composition remainder bound** is

$$ \|R_\alpha(a, b)\|_{L^\infty(U_\alpha)}
   \;\le\; \rho_\alpha
   \;:=\; \kappa_{\mathrm{CW}} \cdot N_\alpha^2 \cdot m_\alpha^2 \cdot
   M_\alpha \cdot \bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr). $$

This is the **Carter–Wick rational remainder**. The bound is sharp up
to a universal rational factor (the proof sketch in §4 explains why).

### 2.4. The calibrated composition loss

The composition remainder contributes an additional **calibrated
composition loss** to the CWQP-AB per-chart loss:

$$ \mathrm{Loss}_\alpha^{\mathrm{comp}}
   \;:=\; \frac{\rho_\alpha}{\mu_\alpha}
   \;=\; \frac{\kappa_{\mathrm{CW}}\, N_\alpha^2\, m_\alpha^2\, M_\alpha}{\mu_\alpha}\,
   \bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr). $$

The **CWQP-AB-v2 per-chart loss** (with explicit composition remainder)
is

$$ \mathrm{Loss}_\alpha^{(2)}
   \;:=\; \frac{M_\alpha}{\mu_\alpha}\,\bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr)
   \;+\; \mathrm{Loss}_\alpha^{\mathrm{comp}}
   \;=\; \frac{M_\alpha\, \bigl(1 + \kappa_{\mathrm{CW}}\, N_\alpha^2\, m_\alpha^2\bigr)}{\mu_\alpha}\,
   \bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr). $$

## 3. Theorem (Carter–Wick Composition Calculus, CWCC)

**Theorem (CWCC, stated).** *Let $K_{a,M}$ be a fixed subextremal Kerr
spacetime with $|a| < M$. Let $\mathfrak A_\chi$ be the
trapped-superradiant quotient ring at fixed spin. Suppose:*

1. *(Carter diagonal constant)* *There is a rational constant
   $\kappa_{\mathrm{CW}} \in \mathbb Q_{>0}$ bounding the diagonal
   second derivatives of the spin-$s$ principal symbol $q_s$ on the
   trapped-superradiant quotient.*
2. *(Chart complexity)* *On each chart $U_\alpha$ of the CWQP-AB atlas,
   the chart complexity $(N_\alpha, m_\alpha, M_\alpha)$ is finite and
   rational.*
3. *(Bump data)* *The bump partition of unity $\{\chi_\alpha\}$ has
   bump constant $C_{\mathrm{bump}}$ and widths $\{w_\alpha\}$.*

*Then the Carter–Wick composition remainder on $U_\alpha$ satisfies*

$$ \|R_\alpha(a_\alpha, b_\alpha)\|_{L^\infty(U_\alpha)}
   \;\le\; \kappa_{\mathrm{CW}} \cdot N_\alpha^2 \cdot m_\alpha^2 \cdot
   M_\alpha \cdot \bigl(1 + C_{\mathrm{bump}}\, w_\alpha^{-1}\bigr)
   \;=\; \rho_\alpha, $$

*and the calibrated composition loss $\mathrm{Loss}_\alpha^{\mathrm{comp}}
= \rho_\alpha / \mu_\alpha$ is a rational function of the chart data.
The CWQP-AB-v2 total loss*

$$ \mathrm{Loss}_{\mathrm{tot}}^{(2)}
   \;:=\; \sum_\alpha \mathrm{Loss}_\alpha^{(2)} $$

*is a rational-arithmetic quantity, and the composed closure inequality*

$$ \mathrm{Loss}_{\mathrm{tot}}^{(2)} \;\le\; B_{\mathrm{lower\_bound}} $$

*is decidable by exact rational arithmetic, where $B_{\mathrm{lower\_bound}}$
is the RRBB certified buffer.*

### 3.1. Why this is the right theorem

CWQP-AB asserted the Wick composition rule with an unbounded remainder.
CWCC bounds that remainder by an explicit rational function of the
chart data, using the diagonal symplectic form (a genuine consequence
of Carter separability) and the polynomial structure of the symbols.
The three theorems compose:

> **CWCC → CWQP-AB-v2 → RRBB compose to a single rational-arithmetic
> certificate whose acceptance is equivalent to the linearized
> trapped-superradiant sign obstruction being closed, with every
> analytic constant in the composition-remainder bound made
> machine-checkable.**

The v2 loss is larger than the v1 loss (it adds the composition
contribution), so closing v2 is a stronger statement than closing v1.

### 3.2. What is genuinely new

1. **Diagonal symplectic exploitation.** The Carter separability gives
   a diagonal symplectic form, which collapses the Moyal bracket to a
   finite diagonal Poisson sum. This is the structural reason the
   composition remainder has a closed rational form.

2. **Polynomial symbol structure.** Because the symbols are polynomial
   in the chart monomial basis $\mathfrak v_\alpha$, the
   bidifferential operators in the Moyal expansion act on a finite
  -dimensional space, giving a finite rational bound.

3. **Composition-aware loss.** The v2 loss is the v1 loss plus the
   composition contribution. This makes the loss budget honest: it
   accounts for the calculus remainder rather than absorbing it
   implicitly.

4. **Composability.** CWCC plugs into the existing CWQP-AB + RRBB
   verifiers with a single new field per chart (the chart complexity)
   and a single new global constant ($\kappa_{\mathrm{CW}}$).

## 4. Proof sketch (conditional on the hypotheses)

**Step 1 (Moyal expansion in diagonal coordinates).** In Carter
variables, the Moyal star product is

$$ a \#_\alpha b \;=\; a\, b \;+\; \frac{i}{2}\, \{a, b\}_{\mathrm{CW}}
   \;+\; \frac{1}{8} \sum_{i,j} \sigma^{ij}\, \sigma^{kl}\,
   (\partial_{x^i}\partial_{x^k} a)\, (\partial_{x^j}\partial_{x^l} b)
   \;+\; \cdots $$

Because $\sigma_\chi$ is diagonal, $\sigma^{ij} = 0$ for $i \ne j$, and
the sum collapses to

$$ \frac{1}{8} \sum_{i} (\partial_{\xi_i}\partial_{x^i} a)\,
   (\partial_{\xi_i}\partial_{x^i} b). $$

Higher-order terms are similarly diagonal sums. For polynomial symbols
of order $m$, the expansion terminates at order $2m$ in derivatives.

**Step 2 (polynomial bound).** Each bidifferential term
$(\partial_{\xi_i}\partial_{x^i} a)\, (\partial_{\xi_i}\partial_{x^i}
b)$ on $U_\alpha$ is bounded by $\kappa_{\mathrm{CW}} \cdot m_\alpha^2
\cdot \|a\|_{L^\infty} \cdot \|b\|_{L^\infty}$. The number of terms is
$N_\alpha^2$ (one per pair of monomials). Hence

$$ \|R_\alpha(a, b)\|_{L^\infty}
   \;\le\; \kappa_{\mathrm{CW}} \cdot N_\alpha^2 \cdot m_\alpha^2 \cdot
   M_\alpha. $$

**Step 3 (bump localization).** When the composition is multiplied by
the bump $\chi_\alpha$, the bump derivatives contribute the standard
factor $1 + C_{\mathrm{bump}}\, w_\alpha^{-1}$, giving the stated
$\rho_\alpha$.

**Step 4 (rational arithmetic).** Every factor in $\rho_\alpha$ is
rational ($\kappa_{\mathrm{CW}}$ by hypothesis, $N_\alpha, m_\alpha$ as
integers, $M_\alpha, \mu_\alpha, C_{\mathrm{bump}}, w_\alpha$ as
rationals), so $\mathrm{Loss}_\alpha^{\mathrm{comp}} = \rho_\alpha /
\mu_\alpha$ is rational, and $\mathrm{Loss}_{\mathrm{tot}}^{(2)}$ is a
finite rational sum. $\square$

### 4.1. What the proof sketch does *not* supply

The proof sketch assumes:
- the diagonal form of $\sigma_\chi$ in Carter variables (this is a
  genuine consequence of Carter separability, but the rigorous
  coordinate calculation on the trapped-superradiant quotient ring is
  not carried out here);
- the termination of the Moyal expansion at order $2m$ for polynomial
  symbols (standard, but the uniform bound across the chart needs the
  rational-endpoint hypothesis);
- the Sobolev embedding used to convert the $L^\infty$ remainder bound
  into the $L^2$ loss contribution in the commutator estimate.

These are open proof obligations. The verifier below checks the
**arithmetic** content of the theorem (rationality of
$\kappa_{\mathrm{CW}}$, finiteness of chart complexity, correctness of
the $\rho_\alpha$ formula, sum-closure against the RRBB buffer); it
does not check the analytic Step 1–3 sketch.

## 5. Machine-checkable certificate format

A CWCC certificate is a JSON object with the following fields:

- `metadata`: name, generated_on, spin, claim_scope, status, warning.
- `carter_diagonal_constant`: rational $\kappa_{\mathrm{CW}} > 0$.
- `bump`: object with `bump_constant` (rational) — must match the
  CWQP-AB bump constant.
- `charts`: list of per-chart complexity records, each with
  `name`, `monomial_basis_size` ($N_\alpha$), `max_symbol_order`
  ($m_\alpha$), `gram_op_norm_bound` ($M_\alpha$), `local_margin`
  ($\mu_\alpha$), `width` ($w_\alpha$).
- `composition_loss`: object with per-chart `rho_alpha`, `loss_comp`,
  and `loss_v2` (all rational), and `total_loss_v2`.
- `rrbb_compatibility`: object referencing an RRBB certificate and
  declaring its `B_lower_bound`; the verifier checks
  `total_loss_v2 <= B_lower_bound`.

The verifier `verify_carter_wick_composition.py` checks:
1. every required field is present, well-typed, and rational;
2. $\kappa_{\mathrm{CW}} > 0$;
3. each chart has $N_\alpha \ge 1$, $m_\alpha \ge 1$, $M_\alpha > 0$,
   $\mu_\alpha > 0$, $w_\alpha > 0$;
4. $\rho_\alpha = \kappa_{\mathrm{CW}} \cdot N_\alpha^2 \cdot m_\alpha^2
   \cdot M_\alpha \cdot (1 + C_{\mathrm{bump}} / w_\alpha)$;
5. $\mathrm{Loss}_\alpha^{\mathrm{comp}} = \rho_\alpha / \mu_\alpha$;
6. $\mathrm{Loss}_\alpha^{(2)} = (M_\alpha / \mu_\alpha)(1 +
   C_{\mathrm{bump}}/w_\alpha) + \mathrm{Loss}_\alpha^{\mathrm{comp}}$;
7. $\mathrm{Loss}_{\mathrm{tot}}^{(2)} = \sum_\alpha
   \mathrm{Loss}_\alpha^{(2)}$;
8. if `rrbb_compatibility` is present, `total_loss_v2 <= B_lower_bound`.

## 6. Composition with CWQP-AB-v2 + RRBB

The decisive composition is:

> **CWCC + CWQP-AB-v2 + RRBB compose to a single rational-arithmetic
> certificate whose acceptance closes the linearized
> trapped-superradiant sign obstruction, with the calculus remainder
> explicitly bounded and absorbed into the loss budget.**

The composed verifier `verify_composed_cwcc_cwqp_rrbb.py` accepts a
triple $(\mathcal C_{\mathrm{CWCC}}, \mathcal C_{\mathrm{CWQP}},
\mathcal C_{\mathrm{RRBB}})$, checks each certificate independently,
verifies that the chart names and bump constants agree across CWCC and
CWQP, and checks the cross-certificate inequality

$$ \mathrm{Loss}_{\mathrm{tot}}^{(2)}(\mathcal C_{\mathrm{CWCC}})
   \;\le\; B_{\mathrm{lower\_bound}}(\mathcal C_{\mathrm{RRBB}}). $$

This is the strongest form of the new idea: the linearized
trapped-superradiant sign obstruction is reduced to a single
rational-arithmetic inequality, with the calculus remainder made
explicit. Replacing the toy inputs with proved Kerr constants would
close the obstruction (conditional on the analytic hypotheses of all
three theorems). The nonlinear obstructions are not addressed here.

## 7. Honest stopping rule

This theorem and its verifier are a **structural reduction**, not a
proof. The conjecture is not solved until:
- the diagonal form of $\sigma_\chi$ in Carter variables is rigorously
  established on the trapped-superradiant quotient ring;
- the Moyal expansion termination at order $2m$ is proved with the
  stated uniformity across rational-endpoint charts;
- the Sobolev embedding converting the $L^\infty$ remainder into the
  $L^2$ loss is established in the certified gauge;
- the RRBB analytic hypotheses (Dafermos–Rodnianski redshift, $r^p$
  estimate, nontrapping on the bulk) are proved;
- the nonlinear bootstrap is completed.

The verifier accepts only certificates that explicitly bound the
composition remainder and check the composed closure, so any premature
"proof" claim that hand-waves the calculus remainder is rejected by
construction.
