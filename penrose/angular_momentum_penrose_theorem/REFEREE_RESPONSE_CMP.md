# Response to Referee Report for Communications in Mathematical Physics

**Manuscript:** The Angular Momentum Penrose Inequality: A Complete Proof via the Twist Contribution Method  
**Author:** Da Xu  
**Date:** December 17, 2025

---

## CRITICAL UPDATE: Response to Dimensional Analysis Concerns

We thank the referee for their exceptionally careful dimensional analysis of the Critical Inequality derivation in Section 8 (The Coupled Monotonicity Method). **The referee has identified fundamental dimensional inconsistencies that invalidate the current proof.** We acknowledge these errors and provide a detailed response below.

---

## Summary of Dimensional Errors Identified

The referee has correctly identified **four critical dimensional inconsistencies**:

| Error | Location | Issue |
|-------|----------|-------|
| **Error 1** | Lemma 8.3, Eq. (55) | Twist integral bound: LHS dimensionless, RHS has units $[L]^2$ |
| **Error 2** | Lemma 8.4, Eq. (100) | AMO derivative: LHS has units $[L]^2$, RHS has units $[L]$ |
| **Error 3** | Lemma 8.6, Eq. (103) | Gradient-area relation: LHS has units $[L]$, RHS has units $[L]^2$ |
| **Error 4** | Lemma 8.3, Step (ii) | Condition $\rho \leq 1$ is not scale-invariant |

---

## Detailed Response to Each Error

### Response to Error 1: Dimensional Mismatch in Twist Integral Bound (Lemma 8.3)

**Referee's Concern:**
> The inequality $\int_\Sigma \frac{|d\psi|^2}{\rho^4} d\sigma \geq \frac{64\pi^2 J^2}{A}$ is dimensionally impossible.
> - LHS: Integrand $\frac{|d\psi|^2}{\rho^4} \sim \frac{L^2}{L^4} = L^{-2}$. Integrating over area ($L^2$) gives dimensionless.
> - RHS: $J^2/A \sim L^4/L^2 = L^2$.

**Acknowledgment:** The referee is **correct**. This is a fundamental dimensional error.

**Analysis:**
In geometric units ($G = c = 1$):
- Angular momentum $J$ has dimensions $[L]^2$ (mass × length)
- The twist potential $\psi$ satisfies $|d\psi|^2 = \rho^2 |J|^2$ where $J^i$ is the angular momentum current
- Since $J^i \sim [L]^0$ (a vector density), we have $|d\psi| \sim [L]$
- The integrand $|d\psi|^2/\rho^4 \sim L^2/L^4 = L^{-2}$
- Integrating over area gives a **dimensionless** quantity on the LHS
- But $J^2/A \sim L^4/L^2 = L^2$ on the RHS

**Conclusion:** The equation as written is dimensionally inconsistent. **The proof of Lemma 8.3 is invalid.**

**Required Correction:** The correct dimensionally-consistent version should involve the moment of inertia $I_\Sigma = \int_\Sigma \rho^2 \, d\sigma$:
$$\int_\Sigma \frac{|d\psi|^2}{\rho^4} \, d\sigma \geq \frac{64\pi^2 J^2}{I_\Sigma}$$
where both sides are dimensionless. The claim $I_\Sigma \leq A$ used to derive the area-normalized bound is **scale-dependent** (see Error 4).

---

### Response to Error 2: Dimensional Mismatch in AMO Derivative (Lemma 8.4)

**Referee's Concern:**
> The equation $\frac{dm_H^2}{dt} \geq \frac{(1-W)}{8\pi} \int_{\Sigma_t} \frac{R_{\tilde{g}}}{|\nabla u|} d\sigma$ has:
> - LHS: $[L]^2$ (derivative of squared mass w.r.t. dimensionless parameter)
> - RHS: $L^{-2}/L^{-1} \times L^2 = [L]$ (scalar curvature/gradient × area)

**Acknowledgment:** The referee is **correct**. There is a missing dimensional factor.

**Analysis:**
- The Hawking mass squared $m_H^2$ has dimensions $[L]^2$
- The parameter $t \in [0,1]$ is dimensionless
- Thus $dm_H^2/dt$ has dimensions $[L]^2$
- Scalar curvature $R_{\tilde{g}} \sim [L]^{-2}$
- Gradient $|\nabla u| \sim [L]^{-1}$ (since $u$ is dimensionless)
- Area element $d\sigma \sim [L]^2$
- RHS: $[L]^{-2}/[L]^{-1} \times [L]^2 = [L]$

**Conclusion:** There is a missing factor of $\sqrt{A}$ or similar length scale. **The AMO derivative formula as applied to $m_H^2$ requires correction.**

**Required Correction:** The standard AMO formula \cite{amo2022} applies to $m_H$ (not $m_H^2$) and involves careful tracking of the mean curvature. The derivative formula for $m_H^2$ must be re-derived with proper dimensional analysis.

---

### Response to Error 3: Inconsistency in Gradient-Area Relation (Lemma 8.6)

**Referee's Concern:**
> The equation $\int_{\Sigma_t} |\nabla u| \, d\sigma \leq C_W \sqrt{A(t) \cdot A'(t)}$ has:
> - LHS: $[L]^{-1} \times [L]^2 = [L]$
> - RHS: $\sqrt{L^2 \times L^2} = [L]^2$

**Acknowledgment:** The referee is **correct**. This bound is dimensionally inconsistent.

**Analysis:**
- $|\nabla u| \sim [L]^{-1}$
- $d\sigma \sim [L]^2$
- LHS: $\int |\nabla u| \, d\sigma \sim [L]$
- $A(t) \sim [L]^2$, $A'(t) = dA/dt \sim [L]^2$ (since $t$ is dimensionless)
- RHS: $\sqrt{A \cdot A'} \sim [L]^2$

**Conclusion:** The inequality bounds a length by an area. **This is scale-dependent and fails under metric rescaling $g \to \lambda^2 g$.**

**Required Correction:** The Cauchy-Schwarz argument must be re-examined. The correct bound should be:
$$\int_{\Sigma_t} |\nabla u| \, d\sigma \leq \sqrt{A(t)} \cdot \sqrt{\int_{\Sigma_t} |\nabla u|^2 \, d\sigma}$$
with the second factor requiring independent dimensional analysis.

---

### Response to Error 4: Non-Scale-Invariant Condition (Lemma 8.3, Step ii)

**Referee's Concern:**
> The condition "$\rho \leq 1$" is physically meaningless in general relativity because it depends on the choice of units. For large black holes ($\rho \gg 1$), the inequality $I_\Sigma \leq A$ reverses.

**Acknowledgment:** The referee is **absolutely correct**. This is a fundamental error.

**Analysis:**
- The cylindrical radius $\rho$ has dimensions $[L]$
- The condition $\rho \leq 1$ is unit-dependent (1 meter? 1 kilometer? 1 Planck length?)
- Under rescaling $g \to \lambda^2 g$: $\rho \to \lambda \rho$, $A \to \lambda^2 A$, $I_\Sigma \to \lambda^4 I_\Sigma$
- The ratio $I_\Sigma/A \to \lambda^2 I_\Sigma/A$ is **not scale-invariant**
- For $\lambda \gg 1$ (large black holes), $I_\Sigma \gg A$, destroying the required bound

**Conclusion:** The argument relies on a coordinate artifact, not a geometric invariant. **This invalidates the derivation of the Critical Inequality.**

**Required Correction:** A valid geometric inequality must use scale-invariant ratios. The correct approach should compare $I_\Sigma/A^2$ (dimensionless) rather than $I_\Sigma$ vs $A$.

---

## Summary of Impact on the Proof

The referee's dimensional analysis reveals that **the derivation of the Critical Inequality in its current form is mathematically invalid**. Specifically:

1. **The twist integral bound** (Lemma 8.3, Eq. 55) equates quantities with different dimensions
2. **The AMO derivative formula** (Lemma 8.4, Eq. 100) has a missing length factor
3. **The gradient-area bound** (Lemma 8.6, Eq. 103) violates scale invariance
4. **The moment of inertia comparison** relies on a non-geometric (coordinate-dependent) condition

**Consequence:** The chain of lemmas leading to Proposition 8.X (Critical Inequality) does not establish the claimed result. The monotonicity of $m_{H,J}^2(t)$ is **not proven** by the current argument.

---

## Path Forward: Required Corrections

To salvage the proof, the following corrections are necessary:

### Correction 1: Dimensionally Consistent Twist Bound

Replace Equation (55) with the dimensionally correct version:
$$\int_\Sigma \frac{|d\psi|^2}{\rho^4} \, d\sigma \geq \frac{64\pi^2 J^2}{I_\Sigma}$$

The passage from $I_\Sigma$ to $A$ requires a **geometric** (not coordinate) bound on the ratio $I_\Sigma/A$. This may require additional hypotheses on the horizon geometry (e.g., near-round conditions) or the use of the Dain-Reiris inequality in a more sophisticated way.

### Correction 2: Re-derive AMO Formula for $m_H^2$

The standard AMO monotonicity applies to $m_H$, not $m_H^2$. The correct derivative formula should be:
$$\frac{dm_H}{dt} = \frac{1}{16\pi m_H} \int_{\Sigma_t} \left( R_{\tilde{g}} + |H|^2 - |\mathring{A}|^2 \right) \frac{d\sigma}{|\nabla u|}$$

For $m_H^2$, use the chain rule:
$$\frac{dm_H^2}{dt} = 2 m_H \frac{dm_H}{dt}$$

This introduces the factor $m_H \sim [L]$ that restores dimensional consistency.

### Correction 3: Scale-Invariant Gradient Bound

Replace Lemma 8.6 with a scale-invariant version. The Cauchy-Schwarz inequality gives:
$$\left( \int_{\Sigma_t} |\nabla u| \, d\sigma \right)^2 \leq A(t) \cdot \int_{\Sigma_t} |\nabla u|^2 \, d\sigma$$

The second factor must be related to geometric invariants. Using the co-area formula and AMO energy estimates, one can show:
$$\int_{\Sigma_t} |\nabla u|^2 \, d\sigma = A'(t) + \text{(curvature terms)}$$

### Correction 4: Geometric Moment of Inertia Bound

The key challenge is to establish a scale-invariant bound relating $I_\Sigma$ to $A$. Define the **normalized moment of inertia**:
$$\bar{I}_\Sigma := \frac{I_\Sigma}{A^2}$$

This is dimensionless and scale-invariant. For the Kerr horizon:
$$\bar{I}_{\text{Kerr}} = \frac{1}{A^2} \int_{\Sigma} \rho^2 \, d\sigma$$

A geometric bound of the form $\bar{I}_\Sigma \leq C$ for some universal constant $C$ would restore the proof, but such a bound requires careful derivation.

---

## Status of the Main Theorem

Given the identified errors, the current manuscript **does not provide a valid proof** of the Angular Momentum Penrose Inequality. The status is:

| Component | Status |
|-----------|--------|
| Stage 1 (Jang equation) | ✓ Valid |
| Stage 2 (AM-Lichnerowicz) | ✓ Valid |
| Stage 3 (AMO framework) | ✓ Valid |
| Critical Inequality derivation | ✗ **Invalid** (dimensional errors) |
| Main Theorem | ✗ **Not proven** |

---

## Commitment to Correction

We thank the referee for this detailed and constructive criticism. The dimensional errors identified are fundamental and must be addressed before the proof can be considered complete. We commit to:

1. **Revising the manuscript** to correct all dimensional inconsistencies
2. **Re-deriving the Critical Inequality** using scale-invariant quantities
3. **Clearly stating any additional hypotheses** required for the corrected proof
4. **Verifying the corrected proof** against the Kerr solution (which must still saturate the bound)

The corrected version will be submitted as a revised manuscript.

---

## Original Referee Assessment (For Reference)

The following sections document the original referee assessment before the dimensional errors were identified.

---

## Original Summary of Referee's Assessment

We thank the referee for their thorough and positive review, which recommends **Accept/Publish**. The referee's assessment validates the key technical contributions:

1. **The Twist Perturbation (Stage 1):** Correctly handled, with proper scaling arguments and axis regularity
2. **The Conformal Mass Bound (Stage 2):** Resolution of the supersolution issue via energy identity is "robust and correct"
3. **Angular Momentum Conservation (Stage 3):** Vacuum necessity correctly identified with appropriate physical justification

The referee requests verification of two specific technical points. We address these below.

---

## Response to Verification Point 1: The Constant Factor $(64\pi^2)$

### Referee's Concern

> The inequality relies on the bound $\int_\Sigma \frac{|d\psi|^2}{\rho^4} \, d\sigma \geq \frac{64\pi^2 J^2}{A}$.
> 
> *Check:* This constant $(64\pi^2)$ must align exactly with the Komar angular momentum normalization $(J = \frac{1}{8\pi} \oint \dots)$. A mismatch of a factor of 2 here would invalidate the monotonicity.

### Our Response

The referee is correct that the constant calibration is crucial. We confirm that the constants are correctly aligned:

**1. Komar Angular Momentum Definition (Theorem 1.1, equation for $J$):**
$$J := \frac{1}{8\pi} \int_\Sigma K(\eta, \nu) \, d\sigma$$

**2. Stream Function Relation (Proposition 4.11):**
The stream function $\psi$ satisfies $|d\psi|^2 = \rho^2 |J|^2$, and the Komar integral gives:
$$J = \frac{1}{4}[\psi]_{\text{poles}}$$

**3. Derivation of the Bound (Lemma 5.12):**
By Cauchy-Schwarz on the orbit space integral:
$$\left(\int_{\Sigma} \frac{|d\psi|}{\rho^2} \, d\sigma\right)^2 \leq A \cdot \int_{\Sigma} \frac{|d\psi|^2}{\rho^4} \, d\sigma$$

The left-hand side equals $(8\pi J)^2 = 64\pi^2 J^2$ by the Komar normalization, giving:
$$\int_\Sigma \frac{|d\psi|^2}{\rho^4}\, d\sigma \geq \frac{64\pi^2 J^2}{A}$$

**4. Verification via Kerr Saturation (Section 2):**
For Kerr spacetime with parameters $(M, a)$:
- Area: $A = 8\pi M(M + \sqrt{M^2 - a^2})$  
- Angular momentum: $J = aM$
- Sub-extremality bound: $A = 8\pi|J|$ at extremality ($a = M$)

Squaring: $A^2 = 64\pi^2 J^2$ at extremality, confirming $(8\pi)^2 = 64\pi^2$.

The explicit computation in Theorem 2.1 shows Kerr saturates the bound $M^2 = A/(16\pi) + 4\pi J^2/A$ for all spin values, which requires the $64\pi^2$ constant to be correct.

**Conclusion:** The constant $64\pi^2$ arises from $(8\pi)^2$ when squaring the Komar normalization factor. This is verified independently by Kerr saturation.

---

## Response to Verification Point 2: Double Limit Interchange ($p \to 1^+$)

### Referee's Concern

> The author asserts (Lemma 6.2/Remark 6.3) that [the double limit] commutes with the geometric limits. The uniform bounds required for this (Mosco convergence) are standard but must be rigorously checked in Section 6.

### Our Response

The double limit interchange is addressed in multiple places in the manuscript. We summarize the justification:

**1. Proposition 1.20 (Statement):**
The passage $p \to 1^+$ in the AMO $p$-harmonic approximations may be interchanged with the monotonicity limit under uniform energy bounds. This is justified via:
- Mosco convergence of energy functionals (Theorem 5.26)
- Moore-Osgood interchange argument (Remark 5.23)

**2. Theorem 5.19 (Double Limit Interchange for $p$-Harmonic Monotonicity):**
For the regularized AM-Hawking mass $\mathcal{M}_{p,J,\epsilon}(t)$, we prove:
$$\lim_{\epsilon \to 0} \lim_{p \to 1^+} \mathcal{M}_{p,J,\epsilon}(t) = \lim_{p \to 1^+} \lim_{\epsilon \to 0} \mathcal{M}_{p,J,\epsilon}(t) = \mathcal{M}_{J}(t)$$

**3. Uniform Bounds (Remark 5.23, conditions (C1)-(C6)):**
The interchange requires:
- **(C1) Uniform Hölder bounds:** $\|u_p\|_{C^{1,\alpha}} \leq C$ independent of $p$
- **(C2) Uniform Harnack inequality:** $\sup u_p \leq C \inf u_p$ on compact sets
- **(C3) Uniform regularization error:** $|A_{p,\epsilon}(t) - A_p(t)| \leq C\epsilon^{\beta_0}$
- **(C4) Uniform capacity bounds:** $(p-1)\text{Cap}_p(\Sigma) \leq C$
- **(C5) Quantitative convergence rate:** $|A_p(t) - A_1(t)| \leq C(p-1)^\gamma$
- **(C6) Moore-Osgood verification:** Uniform convergence in $\epsilon$ combined with pointwise convergence in $p$

**4. Mosco Convergence (Remark 5.24):**
The $p$-energies $\mathcal{E}_p[u] = \frac{1}{p}\int |\nabla u|^p dV$ Mosco-converge to the total variation $\mathcal{E}_1[u] = |Du|(\tilde{M})$. This implies:
- Convergence of minimizers (Theorem 5.26)
- Convergence of level set areas: $A_p(t) \to A_1(t)$ for a.e. $t$

**5. Preservation of Monotonicity:**
Since monotonicity is a closed condition (non-negative derivative in weak sense is preserved under uniform limits), taking the double limit preserves:
$$\frac{d}{dt}\mathcal{M}_{J}^2(t) \geq 0$$

**References to Literature:**
The Mosco convergence framework follows:
- Dal Maso (1993), *An Introduction to Γ-convergence*
- Tolksdorf (1984), stability estimates for $p$-harmonic functions
- Agostiniani-Mazzieri-Oronzio (2022), Section 4 for the non-rotating case

**Conclusion:** The double limit interchange is rigorously justified via standard variational convergence theory. The key is that all estimates are uniform in $p$ near 1, which is verified by explicit computation in Remark 5.23.

---

## Additional Remarks

### On the Referee's Positive Assessment

We appreciate the referee's recognition of:
- The "defensive rigor" in preemptively addressing potential critiques
- The physical consistency demonstrated by the dust shell counterexample (Remark 1.15)
- The completeness of the theory including Kerr saturation and rigidity

### Minor Clarifications

1. **Lemma 6.2/Remark 6.3 Reference:** The referee refers to "Lemma 6.2" which corresponds to our Section 6 content. In the current manuscript numbering, the relevant results are:
   - Theorem 5.19 (Double Limit Interchange)
   - Remark 5.23 (Quantitative Uniform Bounds)
   - Remark 5.24 (Mosco Convergence)

2. **Section Numbering:** The "Section 6" mentioned by the referee corresponds to our Section 5 (Rigorous Justification of Limiting Arguments) in the current manuscript structure.

---

## Summary

Both verification points raised by the referee are addressed in the manuscript:

| Verification Point | Location in Manuscript | Status |
|---|---|---|
| Constant factor $64\pi^2$ | Lemma 5.12, Remark~5.13 (new), Section 2 (Kerr verification) | ✓ Verified |
| Double limit interchange | Theorem 5.19, Remarks 5.23-5.24, Remark~5.25 (new summary) | ✓ Verified |

**Manuscript changes in response to this review:**
1. Added **Remark 5.13** (Constant Factor Calibration) explicitly explaining the origin of $64\pi^2$ from the Komar normalization and its independent verification via Kerr saturation.
2. Added **Remark 5.25** (Summary: Rigorous Justification of the Double Limit) consolidating the double limit verification for ease of reference.

We thank the referee for the careful review and recommendation to publish.

---

*Da Xu*  
*December 2025*

---

# APPENDIX: Technical Notes on Dimensional Consistency

This appendix provides detailed notes on dimensional analysis in geometric relativity, for reference in the manuscript revision.

## Conventions

In geometric units ($G = c = 1$):
- Mass $M$ has dimensions $[L]$
- Angular momentum $J$ has dimensions $[L]^2$
- Area $A$ has dimensions $[L]^2$
- Scalar curvature $R$ has dimensions $[L]^{-2}$
- The twist potential $\psi$ derived from $d\psi = \star J$ has dimensions $[L]^2$
- The gradient $|d\psi|$ has dimensions $[L]$

## Scale Transformation Properties

Under the rescaling $g \to \lambda^2 g$ for constant $\lambda > 0$:

| Quantity | Transformation | Dimensions |
|----------|---------------|------------|
| Length $\ell$ | $\ell \to \lambda \ell$ | $[L]$ |
| Area $A$ | $A \to \lambda^2 A$ | $[L]^2$ |
| Volume $V$ | $V \to \lambda^3 V$ | $[L]^3$ |
| Scalar curvature $R$ | $R \to \lambda^{-2} R$ | $[L]^{-2}$ |
| Mass $M$ | $M \to \lambda M$ | $[L]$ |
| Angular momentum $J$ | $J \to \lambda^2 J$ | $[L]^2$ |
| Cylindrical radius $\rho$ | $\rho \to \lambda \rho$ | $[L]$ |
| Moment of inertia $I_\Sigma$ | $I_\Sigma \to \lambda^4 I_\Sigma$ | $[L]^4$ |
| Twist gradient $|d\psi|$ | $|d\psi| \to \lambda |d\psi|$ | $[L]$ |

## Scale-Invariant Ratios

The following combinations are dimensionless and scale-invariant:

1. **Dimensionless mass**: $M^2/A$ (appears in Penrose inequality)
2. **Dimensionless angular momentum**: $J^2/A^2$
3. **Normalized moment of inertia**: $I_\Sigma/A^2$
4. **Twist integral**: $\int_\Sigma |d\psi|^2/\rho^4 \, d\sigma$ (integrand $\sim L^{-2}$, area element $\sim L^2$)

## The Kerr Consistency Check

For Kerr spacetime with parameters $(M, a = J/M)$:
- Horizon area: $A = 8\pi M(M + \sqrt{M^2 - a^2})$
- Angular momentum: $J = aM$
- At extremality ($a = M$): $A = 8\pi M^2$ and $J = M^2$

The Penrose bound $M^2 = A/(16\pi) + 4\pi J^2/A$ requires:
$$M^2 = \frac{8\pi M^2}{16\pi} + \frac{4\pi M^4}{8\pi M^2} = \frac{M^2}{2} + \frac{M^2}{2} = M^2 \quad \checkmark$$

This dimensional consistency check must be satisfied by any corrected proof.

---

*End of Response Document*
