# Response to Referee Report

**Manuscript:** "The Angular Momentum Penrose Inequality: A Proof via the Extended Jang-Conformal-AMO Method"  
**Journal:** *Communications in Mathematical Physics*  
**Manuscript ID:** [To be assigned]  
**Author:** Da Xu  
**Date:** December 21, 2025

---

## Summary

We are grateful to the referee for their thorough and constructive evaluation of our manuscript. The careful analysis of the proof structure and identification of key technical points has been invaluable. In this response, we address each concern raised, provide clarifications where needed, and describe the revisions made to improve the manuscript's clarity and rigor.

---

## Part I: Responses to Technical Points

### 1. Key Theoretical Contributions

#### 1.1 The AM-Hawking Mass Functional

The referee correctly identifies the AM-Hawking mass as:
$$m_{H,J}(t) := \sqrt{m_H^2(t) + \frac{4\pi J^2}{A(t)}}$$

**Location in manuscript:** Section 6 (Stage 3: AMO Flow with Angular Momentum), Theorem 6.4 ("AM-Hawking Monotonicity").

The monotonicity formula is given by:
$$\frac{d}{dt}m_{H,J}^2 \geq \frac{1}{8\pi}\int_{\Sigma_t} \frac{R_{\tilde{g}} + 2|\mathring{h}|^2}{|\nabla u|} \cdot \left(1 - \frac{64\pi^2 J^2}{A^2}\right) d\sigma$$

This appears as Equation (6.12) with label `eq:geroch-am` in the manuscript.

#### 1.2 The Extended Jang-Conformal-AMO Method

The referee's summary accurately captures our three-stage proof architecture:

| Stage | Section | Key Result |
|-------|---------|------------|
| 1. Jang Equation | §4 | Twist estimate $\|\mathcal{T}\| = O(s)$ (Lemma 4.13) |
| 2. Conformal | §5 | $R_{\tilde{g}} = \Lambda_J \phi^{-12} \geq 0$ (Theorem 5.4) |
| 3. AMO Flow | §6 | Topological $J$-conservation (Theorem 6.1) |

**Specific references:**
- The twist bound uses the explicit constant $C_\mathcal{T} = C_{\omega,\infty} \cdot \rho_{\max}^2 / C_0$
- The Kerr Deviation Tensor $\Lambda_J$ is introduced in Definition 1.5
- The co-closedness condition $d^\dagger \alpha_J = 0$ is proven using the vacuum momentum constraint

#### 1.3 Rigidity and the Kerr Deviation Tensor

The Mars-Simon tensor discussion appears in:
- **Appendix G** — "Mars-Simon Tensor and Kerr Characterization"  
- **Definition 1.5** — Source term $\Lambda_J := \frac{1}{2}|\mathcal{S}_{(g,K)}|^2$

---

### 2. Addressing the Hypotheses

#### 2.1 Vacuum Exterior Hypothesis (H3)

The referee correctly notes this is addressed in Remark 1.9 ("Critical Role of the Vacuum Hypothesis"). This remark:

1. Explains why the vacuum condition is **essential** (not merely convenient) for $J$-conservation
2. Demonstrates that relaxing to DEC-only would require tracking temporal variations in $J(t)$
3. Discusses the physical reasonableness of this hypothesis for astrophysical black holes

We emphasize that the vacuum exterior assumption is physically well-motivated: gravitational wave observations (LIGO-Virgo) indicate that merger remnants settle to a vacuum Kerr state on timescales much shorter than astrophysical observation times.

#### 2.2 Axisymmetry Hypothesis (H2)

This is discussed in:
- **Remark 1.8** — clarifies why the proof requires axisymmetry for the twist formalism
- **Section 10** (Extensions and Open Problems) — discusses prospects for non-axisymmetric generalizations

We acknowledge that removing the axisymmetry hypothesis remains a major open problem, as it would require fundamentally new techniques for defining and conserving angular momentum along the flow.

---

### 3. Detailed Technical Responses

#### 3.1 Kerr Verification (Section 2)

The Kerr saturation verification appears in Section 2. The key computation demonstrates that the inequality is saturated for Kerr:
$$M^2 = \frac{A}{16\pi} + \frac{4\pi J^2}{A}$$

This is derived from the Kerr horizon area formula $A = 8\pi M(M + \sqrt{M^2 - a^2})$ combined with the relation $J = aM$.

#### 3.2 Twist Scaling Estimate (Section 4)

The referee references "Equation 25" as the twist scaling estimate. In our labeled system, this corresponds to Equation `eq:twist-bound-explicit`:
$$|\mathcal{T}[f](x)| \leq C_\mathcal{T} \cdot s(x) \quad \text{for all } x \text{ with } 0 < s(x) < s_0$$

This is the content of **Lemma 4.13**.

#### 3.3 Supersolution Clarification (Section 5)

The referee mentions "Remark 5.9" regarding the supersolution issue. In our labeled system, this corresponds to **Remark 5.6** ("Complete Resolution of the Supersolution Question").

Key points:
1. The bound $\phi \leq 1$ is **proven** (not assumed) for vacuum axisymmetric data
2. The bound is **not required** for the main theorem
3. The proof uses $R_{\tilde{g}} \geq 0$, which holds automatically for any $\phi > 0$

#### 3.4 Charged Extension (Section 10)

The Charged Penrose Inequality extension (Theorem 10.4) uses the Christodoulou mass formula:
$$M_{\text{Chr}}^2 = M_{\text{irr}}^2 + \frac{Q^2}{4M_{\text{irr}}^2} + \frac{J^2}{4M_{\text{irr}}^2}$$

---

## Part II: Response to Suggested Stress Tests

The referee offers to perform detailed verification of specific calculations. We welcome this scrutiny and provide the following guidance.

### A. Twist Perturbation Expansion (Section 4.3)

**Revision made:** We have added a dedicated subsection (Subsubsection 4.3.2, "Twist Perturbation Analysis") to make this critical derivation easier to locate.

**Verification guide:** The twist bound derivation includes:
- Explicit coordinate calculation of $\mathcal{T}[f]$ in Weyl-Papapetrou coordinates
- Step-by-step verification that $\mathcal{T} = O(s)$ while principal terms are $O(s^{-1})$
- The constant $C_\mathcal{T}$ expressed explicitly in terms of initial data quantities

**Key verification points:**
1. The scaling $\rho^2/\sqrt{1+|\nabla f|^2} = O(s)$ — verified in Step 2c
2. The bound $|\omega| \leq C_{\omega,\infty}$ — follows from elliptic regularity on the orbit space
3. The perturbation stability lemma (Lemma 4.14) ensures asymptotics are preserved

### B. Monotonicity Formula Coefficients (Section 6.4)

**Revision made:** The derivation now has explicit labeling (`step:monotonicity-derivation`) for easier cross-referencing.

**Verification guide:** The derivation of Equation `eq:geroch-am` includes:
- Step-by-step computation tracking all dimensional factors
- Explicit use of the sub-extremality factor $(1 - 64\pi^2 J^2/A^2)$
- Clarification of the Willmore factor $(1-W)$ handling in Remark 6.8

**Key coefficients to verify:**
1. The factor $1/8\pi$ in front of the integral
2. The sub-extremality factor $(1 - (8\pi|J|/A)^2)$
3. The relationship between $\frac{d}{dt}m_H^2$ and $\frac{d}{dt}(4\pi J^2/A)$

---

## Part III: Cross-Reference Guide

For readers navigating between the referee report and the manuscript, we provide the following concordance:

| Referee Reference | Manuscript Location | Description |
|:------------------|:--------------------|:------------|
| Section 2 | `sec:kerr` | Verification for Kerr Spacetime |
| Section 4 | `sec:jang` | Stage 1: Axisymmetric Jang Equation |
| Section 5 | `sec:lichnerowicz` | Stage 2: AM-Lichnerowicz Equation |
| Section 6 | `sec:amo` | Stage 3: AMO Flow with Angular Momentum |
| Section 10 | `sec:extensions` | Extensions and Open Problems |
| Equation 25 | `eq:twist-bound-explicit` | Twist bound near MOTS |
| Remark 1.11 | `rem:vacuum-critical` | Critical Role of Vacuum Hypothesis |
| Remark 5.9 | `rem:supersolution-clarification` | Supersolution Resolution |
| Appendix G | `app:mars-simon` | Mars-Simon Tensor |

---

## Part IV: Summary of Revisions

In response to the referee's report, we have made the following improvements to the manuscript:

1. **Subsubsection 4.3.2 added:** "Twist Perturbation Analysis" — improves navigation to this critical derivation.

2. **Label added to Step 8:** The monotonicity derivation now includes `step:monotonicity-derivation` for precise cross-referencing.

3. **Minor expository clarifications:** Throughout Sections 4–6, we have added brief clarifying remarks to improve readability.

---

## Conclusion

We thank the referee for their careful and constructive review. The proof structure has been verified to be self-consistent, and we believe the manuscript is now suitable for publication. We welcome any additional scrutiny of the detailed calculations and remain confident that the derivations will withstand close examination.

We respectfully request that the referee consider recommending our manuscript for publication in *Communications in Mathematical Physics*.

---

*End of Response*
