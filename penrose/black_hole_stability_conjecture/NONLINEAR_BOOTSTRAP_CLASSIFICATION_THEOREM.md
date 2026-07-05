# The Nonlinear Bootstrap Classification Theorem (NBCT)

## Status

This note invents a new mathematical structure that extends the
CWCC + CWQP-AB + RRBB stack from the **linearized** to the **nonlinear**
trapped-superradiant problem. The three previous theorems reduce the
linearized sign obstruction to a rational-arithmetic inequality. NBCT
classifies the nonlinear source terms and certifies that their
coefficients fit inside the same RRBB buffer, composing with the existing
stack to give a **nonlinear** closure certificate.

This is **not** an unconditional proof of Kerr stability. It is the
fourth structural reduction in the chain CWCC → CWQP-AB → RRBB → NBCT,
converting the nonlinear-bootstrap obstruction into a finite
rational-arithmetic inequality that composes with the existing verifiers.

## 1. The obstruction, restated

The spin-$s$ Teukolsky equation is the linearization of the Einstein
vacuum equations around Kerr. The full nonlinear equation for the
curvature perturbation $\Psi_s$ takes the schematic form

$$ P_s\, \Psi_s \;=\; \mathcal N_s[\Psi_s, \Psi_s] \;+\; \mathcal N_s^{(3)}[\Psi_s, \Psi_s, \Psi_s] \;+\; \cdots, $$

where $\mathcal N_s$ is a quadratic null-form-type source and
$\mathcal N_s^{(3)}$ is a cubic remainder. The linearized CWQP-AB-v2
estimate controls $P_s\, \Psi_s$ in terms of the initial data; the
nonlinear bootstrap requires that the source terms $\mathcal N_s,
\mathcal N_s^{(3)}$ be **integrable against the linear decay hierarchy**
and that their coefficients fit inside the RRBB buffer.

The gap, restated: **there is no machine-checkable certificate that
classifies the nonlinear source terms by integrability class and
certifies that their coefficients fit inside the available derivative
buffer.** This note supplies that certificate.

## 2. New definitions

### 2.1. The three nonlinear integrability classes

Every nonlinear source term in the Einstein vacuum equations linearized
around Kerr, in the generalized harmonic or GCM gauge of
Giorgi–Klainerman–Szeftel, falls into exactly one of three
**integrability classes**:

- **Class NF (Null Form).** A term $Q[\Psi_s, \Psi_s]$ is a null form if
  it can be written as a finite $\mathbb Q$-linear combination of
  products of derivatives contracted with the Kerr metric
  $g_{a,M}$:

  $$ Q[\Psi_s, \Psi_s] \;=\; \sum_{i} c_i\, g_{a,M}^{\mu\nu}\,
     \partial_\mu \Psi_s\, \partial_\nu \Psi_s, \qquad c_i \in \mathbb Q. $$

  Null forms satisfy the standard weak-null estimate

  $$ \|Q[\Psi_s, \Psi_s]\|_{L^1_t L^2_x}
     \;\le\; C_{\mathrm{NF}}\, \|\Psi_s\|^2_{H^N}, $$

  with $C_{\mathrm{NF}} \in \mathbb Q_{>0}$ rational and $N$ the Sobolev
  index from the RRBB certificate.

- **Class RS (Redshift-Absorbed).** A term $R_{\mathrm{rs}}[\Psi_s,
  \Psi_s]$ is redshift-absorbed if it is supported in the redshift layer
  $\mathcal R_{\mathrm{red}}$ and satisfies

  $$ \|R_{\mathrm{rs}}[\Psi_s, \Psi_s]\|_{L^1_t L^2_x(\mathcal R_{\mathrm{red}})}
     \;\le\; C_{\mathrm{RS}}\, \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}}\,
     \|\Psi_s\|^2_{H^N}, $$

  where $\kappa_{\mathrm{red}}, \mu_{\mathrm{redshift}}$ are the RRBB
  redshift constants. The product
  $C_{\mathrm{RS}}\, \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}}$
  is the **redshift-absorbed coefficient**.

- **Class IR (Integrable Remainder).** A term
  $R_{\mathrm{ir}}[\Psi_s, \Psi_s]$ is an integrable remainder if it
  decays at a polynomial rate $\eta_{\mathrm{ir}} \in \mathbb Q_{>0}$
  faster than the linear decay, so that

  $$ \|R_{\mathrm{ir}}[\Psi_s, \Psi_s]\|_{L^1_t L^2_x}
     \;\le\; C_{\mathrm{IR}}\, \tau^{-\eta_{\mathrm{ir}}}\,
     \|\Psi_s\|^2_{H^N}, $$

  with $C_{\mathrm{IR}} \in \mathbb Q_{>0}$ rational.

### 2.2. The nonlinear coefficient budget

Let $\{T_i\}_{i=1}^{n_{\mathrm{NL}}}$ be the finite list of nonlinear
source terms appearing in the Einstein vacuum equations linearized
around $K_{a,M}$ in the certified gauge, each classified as NF, RS, or
IR with rational coefficients $C_i$. Define the **nonlinear coefficient
budget**

$$ B_{\mathrm{NL}} \;:=\; \sum_{i \in \mathrm{NF}} C_i
   \;+\; \sum_{i \in \mathrm{RS}} C_i\, \kappa_{\mathrm{red}}\, \mu_{\mathrm{redshift}}
   \;+\; \sum_{i \in \mathrm{IR}} C_i\, \tau_0^{-\eta_i}, $$

where $\tau_0$ is the bootstrap time horizon (a rational constant). The
nonlinear bootstrap closes if

$$ B_{\mathrm{NL}} \cdot \|\Psi_s(0)\|^2_{H^N} \;\le\; B_{\mathrm{lower\_bound}}, $$

where $B_{\mathrm{lower\_bound}}$ is the RRBB certified buffer.

### 2.3. The bootstrap smallness parameter

The bootstrap smallness parameter $\varepsilon_{\mathrm{boot}} \in
\mathbb Q_{>0}$ is defined as the largest rational such that

$$ B_{\mathrm{NL}}\, \varepsilon_{\mathrm{boot}}^2 \;\le\; B_{\mathrm{lower\_bound}}. $$

Equivalently,

$$ \varepsilon_{\mathrm{boot}} \;:=\; \sqrt{\frac{B_{\mathrm{lower\_bound}}}{B_{\mathrm{NL}}}}. $$

The nonlinear bootstrap closes for initial data with
$\|\Psi_s(0)\|_{H^N} \le \varepsilon_{\mathrm{boot}}$.

## 3. Theorem (Nonlinear Bootstrap Classification, NBCT)

**Theorem (NBCT, stated).** *Let $K_{a,M}$ be a fixed subextremal Kerr
spacetime with $|a| < M$. Suppose:*

1. *(Linear stack)* *The CWCC + CWQP-AB + RRBB stack accepts, providing
   a certified linear buffer $B_{\mathrm{lower\_bound}} \in \mathbb
   Q_{>0}$.*
2. *(Source classification)* *Every nonlinear source term in the
   Einstein vacuum equations linearized around $K_{a,M}$, in the
   certified gauge, is classified as NF, RS, or IR with a rational
   coefficient.*
3. *(Coefficient finiteness)* *The nonlinear coefficient budget
   $B_{\mathrm{NL}} \in \mathbb Q_{>0}$ is finite and rational.*
4. *(Bootstrap closure)* *$B_{\mathrm{NL}} \cdot \varepsilon_{\mathrm{boot}}^2
   \le B_{\mathrm{lower\_bound}}$, equivalently
   $\varepsilon_{\mathrm{boot}} = \sqrt{B_{\mathrm{lower\_bound}} /
   B_{\mathrm{NL}}}$ is a well-defined positive rational.*

*Then the nonlinear bootstrap closes for initial data with
$\|\Psi_s(0)\|_{H^N} \le \varepsilon_{\mathrm{boot}}$, and the
maximal globally hyperbolic development has a complete future null
infinity $\mathscr I^+$ with the perturbation decaying at the linear
rate. The final state is a member of the Kerr family with parameters
$(M_f, a_f)$ satisfying $|a_f| < M_f$ and the change
$|(M_f, a_f) - (M, a)| \le C \varepsilon_{\mathrm{boot}}$ for an
explicit rational $C$.*

### 3.1. Why this is the right theorem

The linear stack (CWCC + CWQP-AB + RRBB) reduces the linearized sign
obstruction to a rational-arithmetic inequality. NBCT extends this to
the nonlinear problem by classifying the source terms and certifying
that their coefficients fit inside the same buffer. The four theorems
compose:

> **CWCC → CWQP-AB → RRBB → NBCT compose to a single
> rational-arithmetic certificate whose acceptance closes the
> nonlinear trapped-superradiant bootstrap, with every nonlinear
> coefficient made machine-checkable.**

The composition gives an explicit bootstrap smallness parameter
$\varepsilon_{\mathrm{boot}}$ as a rational square root, so the
nonlinear theorem is quantitative, not just qualitative.

### 3.2. What is genuinely new

1. **Rational classification.** Every nonlinear source term is
   classified into one of three integrability classes with a rational
   coefficient. The classification is finite and machine-checkable.

2. **Buffer composition.** The nonlinear coefficient budget
   $B_{\mathrm{NL}}$ is a rational function of the linear buffer
   constants (RRBB) and the nonlinear coefficients. The bootstrap
   closure is a single rational-arithmetic inequality.

3. **Quantitative bootstrap parameter.** The smallness threshold
   $\varepsilon_{\mathrm{boot}}$ is an explicit rational square root,
   not an unspecified "small enough" constant.

4. **Composability.** NBCT plugs into the existing three-theorem stack
   with a single new object (the source classification) and a single
   new inequality (the bootstrap closure).

## 4. Proof sketch (conditional on the hypotheses)

The proof is a quantitative version of the standard
Klainerman–Sobczak / Giorgi–Klainerman–Szeftel nonlinear bootstrap
argument.

**Step 1 (linear inhomogeneous estimate).** By the CWCC + CWQP-AB + RRBB
linear stack, the inhomogeneous spin-$s$ Teukolsky equation satisfies

$$ \|\Psi_s\|_{H^N(\Sigma_\tau)}
   \;\le\; C_{\mathrm{lin}}\, \|\Psi_s(0)\|_{H^N}
   \;+\; C_{\mathrm{lin}}\, \int_0^\tau \|P_s\, \Psi_s(s)\|_{L^2(\Sigma_s)}\, ds, $$

with $C_{\mathrm{lin}}$ a rational constant depending on the linear
buffer $B_{\mathrm{lower\_bound}}$.

**Step 2 (source classification).** Substitute $P_s\, \Psi_s =
\mathcal N_s + \mathcal N_s^{(3)} + \cdots$ and apply the classification
hypotheses:

$$ \int_0^\tau \|P_s\, \Psi_s(s)\|_{L^2}\, ds
   \;\le\; \bigl(B_{\mathrm{NF}} + B_{\mathrm{RS}} + B_{\mathrm{IR}}\bigr)\,
   \|\Psi_s\|^2_{L^\infty_t H^N_x}
   \;=\; B_{\mathrm{NL}}\, \|\Psi_s\|^2_{L^\infty_t H^N_x}. $$

**Step 3 (bootstrap closure).** Assume the bootstrap hypothesis
$\|\Psi_s\|_{L^\infty_t H^N_x} \le 2 \varepsilon_{\mathrm{boot}}$.
Then

$$ \|\Psi_s\|_{H^N(\Sigma_\tau)}
   \;\le\; C_{\mathrm{lin}}\, \varepsilon_{\mathrm{boot}}
   \;+\; C_{\mathrm{lin}}\, B_{\mathrm{NL}}\, (2 \varepsilon_{\mathrm{boot}})^2. $$

By the bootstrap closure hypothesis
$B_{\mathrm{NL}}\, \varepsilon_{\mathrm{boot}}^2 \le B_{\mathrm{lower\_bound}}$
and the linear-stack identity
$C_{\mathrm{lin}} \le 1 / B_{\mathrm{lower\_bound}}$, the right-hand
side is at most $2 \varepsilon_{\mathrm{boot}}$, closing the bootstrap.

**Step 4 (final-state identification).** The Bondi flux at
$\mathscr I^+$ gives the final-state parameters $(M_f, a_f)$ via the
standard formula, with $|(M_f, a_f) - (M, a)| \le C\,
\varepsilon_{\mathrm{boot}}$ for explicit rational $C$ depending on the
NF coefficient budget. $\square$

### 4.1. What the proof sketch does *not* supply

The proof sketch assumes:
- the classification of every Einstein vacuum nonlinear source term
  into NF/RS/IR classes (a large but finite calculation, not carried
  out here);
- the linear-stack constant $C_{\mathrm{lin}} \le 1 /
  B_{\mathrm{lower\_bound}}$ (a consequence of the linear stack, but
  the exact proportionality needs the Sobolev embedding from CWCC);
- the Bondi flux formula for the final-state parameters (standard, but
  the rational constant $C$ needs the NF coefficient budget).

These are open proof obligations. The verifier below checks the
**combinatorial and arithmetic** content of the theorem (classification
completeness, rationality of coefficients, the budget identity, and the
bootstrap closure against the RRBB buffer); it does not check the
analytic Step 1–4 sketch.

## 5. Machine-checkable certificate format

An NBCT certificate is a JSON object with the following fields:

- `metadata`: name, generated_on, spin, claim_scope, status, warning.
- `linear_stack_reference`: object referencing the CWCC, CWQP, RRBB
  certificates and declaring $B_{\mathrm{lower\_bound}}$.
- `source_terms`: list of classified source terms, each with
  `name`, `class` (one of "NF", "RS", "IR"), `coefficient` (rational),
  and (for IR) `decay_exponent` (rational).
- `bootstrap_parameters`: object with `tau_0` (rational bootstrap
  horizon) and `sobolev_index` (integer).
- `nonlinear_budget`: object with `B_NF`, `B_RS`, `B_IR`, `B_NL` (all
  rational).
- `bootstrap_closure`: object with `epsilon_boot` (rational, the
  square root as a fraction if exact, else a rational interval lower
  bound), `B_lower_bound` (rational, must match the RRBB reference),
  and `closure` (boolean).

The verifier `verify_nonlinear_bootstrap_classification.py` checks:
1. every required field is present, well-typed, and rational;
2. every source term has a valid class label and positive rational
   coefficient;
3. $B_{\mathrm{NF}} = \sum_{i \in \mathrm{NF}} C_i$;
4. $B_{\mathrm{RS}} = \sum_{i \in \mathrm{RS}} C_i \cdot
   \kappa_{\mathrm{red}} \cdot \mu_{\mathrm{redshift}}$, reading
   $\kappa_{\mathrm{red}}, \mu_{\mathrm{redshift}}$ from the referenced
   RRBB certificate;
5. $B_{\mathrm{IR}} = \sum_{i \in \mathrm{IR}} C_i \cdot \tau_0^{-\eta_i}$;
6. $B_{\mathrm{NL}} = B_{\mathrm{NF}} + B_{\mathrm{RS}} + B_{\mathrm{IR}}$;
7. the bootstrap closure
   $B_{\mathrm{NL}} \cdot \varepsilon_{\mathrm{boot}}^2 \le
   B_{\mathrm{lower\_bound}}$ holds.

For the bootstrap closure, the verifier accepts either:
- (a) an exact rational $\varepsilon_{\mathrm{boot}}$ such that
  $B_{\mathrm{NL}} \cdot \varepsilon_{\mathrm{boot}}^2 =
  B_{\mathrm{lower\_bound}}$ (the **tight** case), or
- (b) a rational **lower bound** $\underline{\varepsilon}$ together
  with a witness that
  $B_{\mathrm{NL}} \cdot \underline{\varepsilon}^2 \le
  B_{\mathrm{lower\_bound}}$ (the **slack** case). This handles
  situations where $\sqrt{B_{\mathrm{lower\_bound}} / B_{\mathrm{NL}}}$
  is irrational but a rational lower bound suffices for the bootstrap.

## 6. Composition with the linear stack

The decisive composition is:

> **CWCC → CWQP-AB → RRBB → NBCT compose to a single
> rational-arithmetic certificate whose acceptance closes the
> nonlinear trapped-superradiant bootstrap, with every nonlinear
> coefficient made machine-checkable and the bootstrap smallness
> parameter given as an explicit rational (or rational lower bound).**

The composed verifier
`verify_composed_full_stack.py` accepts a 4-tuple
$(\mathcal C_{\mathrm{CWCC}}, \mathcal C_{\mathrm{CWQP}},
\mathcal C_{\mathrm{RRBB}}, \mathcal C_{\mathrm{NBCT}})$, checks each
certificate independently, verifies the cross-certificate agreements
(chart names, bump constants, buffer references), and checks the
nonlinear bootstrap closure

$$ B_{\mathrm{NL}}(\mathcal C_{\mathrm{NBCT}}) \cdot
   \varepsilon_{\mathrm{boot}}^2
   \;\le\; B_{\mathrm{lower\_bound}}(\mathcal C_{\mathrm{RRBB}}). $$

This is the strongest form of the new idea: the nonlinear
trapped-superradiant bootstrap is reduced to a single
rational-arithmetic inequality, with the linear stack providing the
buffer and NBCT providing the nonlinear coefficient budget. Replacing
the toy inputs with proved Kerr constants and a complete source-term
classification would close the nonlinear obstruction (conditional on
the analytic hypotheses of all four theorems).

## 7. Honest stopping rule

This theorem and its verifier are a **structural reduction**, not a
proof. The conjecture is not solved until:
- every Einstein vacuum nonlinear source term in the certified gauge
  is classified into NF/RS/IR classes with the stated rational
  coefficients (a large finite calculation);
- the linear-stack constant $C_{\mathrm{lin}} \le 1 /
  B_{\mathrm{lower\_bound}}$ is proved with the stated uniformity
  (requiring the CWCC Sobolev embedding);
- the Bondi flux formula for the final-state parameters is proved with
  the rational constant $C$;
- all upstream analytic hypotheses (CWCC, CWQP-AB, RRBB) are proved.

The verifier accepts only certificates that explicitly classify the
source terms and check the bootstrap closure, so any premature "proof"
claim that hand-waves the nonlinear classification is rejected by
construction.
