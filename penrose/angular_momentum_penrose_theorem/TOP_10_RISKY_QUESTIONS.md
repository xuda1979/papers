# 10 MOST RISKY QUESTIONS FOR REFEREE REVIEW
## Angular Momentum Penrose Inequality Paper - Critical Analysis

**Date:** December 11, 2025
**Purpose:** Pre-submission preparation - identify and address the 10 highest-risk questions referees will ask

---

## QUESTION 1: Is the mass bound $M(\tilde{g}) \leq M(g)$ rigorously proven without assuming $\phi \leq 1$?

### Risk Level: **CRITICAL**

### The Concern:
The AM-Lichnerowicz equation has a $\Lambda_J \phi^{-7}$ term that distinguishes it from the standard equation. The naive supersolution argument ($\phi^+ = 1$) requires $R_{\bar{g}} \geq \Lambda_J$, which is not automatic.

### Current Paper Status:
**ADDRESSED** in Remark 4.3, Lemma 4.5, and Remark 4.12.

### The Resolution:
The paper establishes that **$\phi \leq 1$ is NOT required** for the main theorem:

1. **What's actually needed:** 
   - Existence of $\phi > 0$ solving AM-Lichnerowicz ✓
   - $R_{\tilde{g}} = \Lambda_J \phi^{-12} \geq 0$ (automatic) ✓
   - Mass bound $M(\tilde{g}) \leq M(g)$ ✓

2. **Energy identity proof (Lemma 4.5):**
   $$\mathcal{I}[\phi] := \int_{\bar{M}}\left(8|\nabla\phi|^2 + R_{\bar{g}}\phi(\phi-1) - \Lambda_J\phi^{-7}(\phi-1)\right)dV = 0$$
   
   This identity holds for ANY positive solution by multiplying the equation by $(\phi-1)$ and integrating by parts. The boundary terms vanish:
   - At infinity: decay $\phi - 1 = O(r^{-\tau})$
   - At cylindrical end: exponential decay $|\phi - 1| = O(e^{-\kappa t})$

3. **Mass formula consequence:** From $\mathcal{I}[\phi] = 0$, the conformal mass formula gives $M(\tilde{g}) \leq M(\bar{g}) \leq M(g)$.

### Referee-Ready Response:
> "The mass bound does not require $\phi \leq 1$. Lemma 4.5 establishes $M(\tilde{g}) \leq M(g)$ via the energy identity $\mathcal{I}[\phi] = 0$, which holds for any bounded positive solution. The verification steps (V1)-(V3) explicitly check convergence of boundary terms. See Remark 4.3 for the complete clarification."

---

## QUESTION 2: How can angular momentum be conserved when the Komar integral uses metric $g$ but level sets live on $(\tilde{M}, \tilde{g})$?

### Risk Level: **CRITICAL**

### The Concern:
The Komar angular momentum $J = \frac{1}{8\pi}\int_\Sigma K(\eta, \nu) d\sigma$ is defined using $(g, K)$, but the AMO flow and level sets $\Sigma_t$ are defined on the conformally transformed manifold $(\tilde{M}, \tilde{g})$. Is $J(\Sigma_t)$ even well-defined?

### Current Paper Status:
**ADDRESSED** in Remark 5.5 and Theorem 5.4.

### The Resolution:

1. **Manifold identity:** $M = \bar{M} = \tilde{M}$ as smooth manifolds. The Jang and conformal constructions are diffeomorphisms, not changes of the underlying space.

2. **Metric-independence of flux integrals:** 
   The Komar 1-form $\alpha_J = \frac{1}{8\pi}K(\eta, \cdot)^\flat_g$ is defined using the **physical** metric $g$. Its Hodge dual $\star_g \alpha_J$ is a **closed 2-form** (when vacuum + axisymmetry holds):
   $$d(\star_g \alpha_J) = 0$$
   
3. **Conservation via topology:** 
   - All level sets $\Sigma_t$ are homologous (Lemma 5.3)
   - By Stokes' theorem: $\int_{\Sigma_{t_2}} \star_g \alpha_J - \int_{\Sigma_{t_1}} \star_g \alpha_J = \int_W d(\star_g \alpha_J) = 0$

4. **The integral is metric-independent:** Part A of Theorem 5.4 proves that $\int_\Sigma \alpha_J(\nu_\gamma) d\sigma_\gamma$ gives the same value for ANY choice of metric $\gamma$ used to define the normal and area element.

### Referee-Ready Response:
> "The conservation is topological, not metric-dependent. The integral $J(\Sigma_t) = \int_{\Sigma_t} \star_g \alpha_J$ uses a closed 2-form defined from $(g,K)$. By Stokes' theorem and homology of level sets, the flux is independent of $t$. See Theorem 5.4 Part A for the metric-independence proof."

---

## QUESTION 3: What happens at the poles where $\Sigma$ intersects the rotation axis ($\rho = 0$)?

### Risk Level: **HIGH**

### The Concern:
The MOTS $\Sigma \cong S^2$ must intersect the axis $\Gamma = \{\eta = 0\}$ at two poles. At these points, the orbit radius $\rho \to 0$ and coordinate expressions become singular. Does this break the twist perturbation estimates?

### Current Paper Status:
**ADDRESSED** in Lemma 3.3, Lemma 3.4, and Remark 3.5.

### The Resolution:

1. **Topological necessity (Lemma 3.3):** An axisymmetric sphere MUST intersect the axis at exactly two poles. If $\Sigma \cap \Gamma = \emptyset$, the $U(1)$-action would be free, but $S^2$ cannot fiber over $S^1$.

2. **Twist scaling (Lemma 3.4):** The twist term scales as $\rho^2$, NOT $\rho$:
   $$|\mathcal{T}[\bar{f}](x)| \leq C \cdot \rho(x)^2 \cdot |\bar{\nabla}\bar{f}|(x) \leq C' \cdot d(x,p)^2$$
   
   **At the poles:** $\mathcal{T}(p_N) = \mathcal{T}(p_S) = 0$ since $\rho = 0$ there.

3. **Physical interpretation (Remark 3.6):** The $\rho^2$ factor reflects that angular momentum density scales as the square of the lever arm. At the axis, there are no orbits to "drag," so twist contributions vanish.

4. **Axis regularity conditions (AR1)-(AR3):** The paper specifies explicit regularity conditions ensuring the twist potential extends smoothly to the axis.

### Referee-Ready Response:
> "The twist term scales as $\rho^2$ (not $\rho$), so it vanishes quadratically at the poles. Lemma 3.4 proves $|\mathcal{T}| \leq C\rho^2$, and at poles $\rho = 0$ implies $\mathcal{T} = 0$. The perturbation analysis is unaffected by axis intersection."

---

## QUESTION 4: Why is the vacuum hypothesis essential, not just convenient?

### Risk Level: **HIGH**

### The Concern:
Prior Penrose inequality proofs (Huisken-Ilmanen, Bray) required only DEC. Is the vacuum hypothesis genuinely necessary for the rotating case, or is it an artifact of the proof technique?

### Current Paper Status:
**ADDRESSED** in Remark 1.10 (extensive discussion).

### The Resolution:

1. **Essential for $J$-conservation:** The momentum constraint is:
   $$D^j K_{ij} = D_i(\text{tr}K) + 8\pi j_i$$
   
   Co-closedness $d^\dagger \alpha_J = 0$ requires $j_i = 0$ (vacuum). With matter, $j_\phi \neq 0$ would cause $J(t)$ to vary.

2. **Why $J = 0$ case differs:** For non-rotating data, there's no angular momentum to conserve. Huisken-Ilmanen and Bray don't need vacuum because matter doesn't affect the $J = 0$ Hawking mass.

3. **Explicit obstruction:** Even with DEC + axisymmetric matter, angular momentum transport can occur (e.g., magnetized fluids). The inequality could fail without vacuum.

4. **Physical reasonableness:** Vacuum exterior is natural for isolated black holes—matter cannot remain in equilibrium just outside the horizon.

### Referee-Ready Response:
> "Vacuum is essential, not convenient. Angular momentum conservation (Theorem 5.4) requires $d^\dagger \alpha_J = 0$, which follows from the momentum constraint only when $j_i = 0$. For $J \neq 0$, matter contributions could cause $J(t)$ to vary, breaking the argument. See Remark 1.10 for detailed discussion."

---

## QUESTION 5: Is the spectral gap argument for Fredholm theory rigorous?

### Risk Level: **MEDIUM-HIGH**

### The Concern:
Lemma 4.2 claims the operator $L = -8\Delta_{\bar{g}} + R_{\bar{g}}$ is Fredholm of index zero. This requires the principal eigenvalue $\lambda_0$ of $-8\Delta_\Sigma + R_\Sigma$ to be strictly positive. Is this proven?

### Current Paper Status:
**ADDRESSED** in Lemma 4.2 (Steps 3-6).

### The Resolution:

1. **Eigenvalue positivity proof (by contradiction):**
   - Suppose $\lambda_0 = 0$ with eigenfunction $\varphi_0 > 0$
   - Then $\int_\Sigma R_\Sigma \varphi_0 = 8\int_\Sigma \Delta_\Sigma \varphi_0 = 0$
   - But stable MOTS have $R_\Sigma = 2K_\Sigma \geq 0$ (Galloway-Schoen)
   - And $\int_\Sigma R_\Sigma = 8\pi > 0$ (Gauss-Bonnet)
   - Therefore $\int R_\Sigma \varphi_0 > 0$, contradiction!

2. **Explicit lower bound:** Using Hersch's inequality:
   $$\lambda_0 \geq \frac{8\pi}{A(\Sigma)}$$
   giving $\gamma_0 = \sqrt{\lambda_0/8} \geq \sqrt{\pi/A}$.

3. **Valid weight interval:** For $\beta \in (-\gamma_0, 0)$, no indicial roots exist, so Lockhart-McOwen gives Fredholm of index zero.

### Referee-Ready Response:
> "The spectral gap is proven rigorously in Lemma 4.2, Step 3. For stable MOTS, Galloway-Schoen implies $R_\Sigma \geq 0$, and Gauss-Bonnet gives $\int R_\Sigma = 8\pi > 0$. A variational argument then shows $\lambda_0 > 0$. Explicit bounds: $\lambda_0 \geq 8\pi/A$."

---

## QUESTION 6: How is the Kerr deviation tensor $\mathcal{S}_{(g,K)}$ well-defined for non-stationary data?

### Risk Level: **MEDIUM-HIGH**

### The Concern:
Definition 1.8 introduces the Kerr deviation tensor by comparing the data's Weyl tensor to a "reference Kerr Weyl tensor." How is this comparison coordinate-independent for non-stationary data?

### Current Paper Status:
**ADDRESSED** in Definition 1.8 and Appendix A.

### The Resolution:

1. **Intrinsic Weyl tensors:** $E_{ij}$ and $B_{ij}$ are defined purely from $(g, K)$:
   - $E_{ij} = R_{ij} - \frac{1}{3}Rg_{ij} + (\text{tr}K)K_{ij} - K_{ik}K^k{}_j$
   - $B_{ij} = \epsilon_i{}^{kl}\nabla_k K_{lj}$
   
   These are intrinsic to the initial data, not requiring any spacetime development.

2. **Reference Kerr tensor:** Determined by asymptotic matching using ADM mass $M$ and angular momentum $J$—both coordinate-independent quantities.

3. **Propagation:** The Bianchi constraints propagate the Weyl structure throughout $M$, making the comparison well-defined everywhere.

4. **Key property:** $\mathcal{S}_{(g,K)} = 0$ iff the data is a Kerr slice (Mars uniqueness theorem).

### Referee-Ready Response:
> "The Kerr deviation tensor is intrinsically defined using the electric and magnetic Weyl tensors $(E, B)$ constructed from $(g, K)$. The reference Kerr tensor is determined by asymptotic matching to $(M, J)$. See Definition 1.8 and Appendix A for the Mars-Simon characterization."

---

## QUESTION 7: What is the precise monotonicity formula and how does sub-extremality enter?

### Risk Level: **MEDIUM**

### The Concern:
The paper claims $m_{H,J}(t)$ is monotone, but the formula must handle the angular momentum term. What is the exact formula and why is it non-negative?

### Current Paper Status:
**ADDRESSED** in Theorem 5.3 and Table 2.

### The Resolution:

**The AM-Geroch formula:**
$$\frac{d}{dt}m_{H,J}^2 \geq \frac{1}{8\pi}\int_{\Sigma_t}\frac{R_{\tilde{g}} + 2|\mathring{h}|^2}{|\nabla u|}\left(1 - \frac{64\pi^2 J^2}{A(t)^2}\right)d\sigma$$

1. **Each factor is non-negative:**
   - $R_{\tilde{g}} = \Lambda_J \phi^{-12} \geq 0$ ✓
   - $|\mathring{h}|^2 \geq 0$ ✓
   - $|\nabla u| > 0$ (no critical points) ✓

2. **Sub-extremality factor:** 
   $$\left(1 - \frac{64\pi^2 J^2}{A^2}\right) = \left(1 - \left(\frac{8\pi|J|}{A}\right)^2\right) \geq 0$$
   
   This holds when $A \geq 8\pi|J|$, which is the **Dain-Reiris inequality** (Theorem 6.1).

3. **Preservation:** Since $A'(t) \geq 0$ (area increases), if $A(0) \geq 8\pi|J|$ at the MOTS, then $A(t) \geq 8\pi|J|$ for all $t$.

### Referee-Ready Response:
> "The monotonicity formula (Eq. 5.3) includes a sub-extremality factor $(1 - 64\pi^2J^2/A^2)$, which is non-negative by the Dain-Reiris inequality $A \geq 8\pi|J|$ for stable MOTS. Since area increases along the flow, sub-extremality is preserved."

---

## QUESTION 8: Is there circular logic in using Dain-Reiris?

### Risk Level: **MEDIUM**

### The Concern:
The Dain-Reiris inequality $A \geq 8\pi|J|$ is used to establish sub-extremality. Does its proof rely on assumptions that create circularity?

### Current Paper Status:
**ADDRESSED** in Remark 6.2 (no circularity).

### The Resolution:

1. **Dain-Reiris is independent:** The inequality is proven using:
   - Stability of the MOTS
   - The dominant energy condition
   - A variational argument on the MOTS
   
   It does NOT use the Penrose inequality or any mass bound.

2. **Logical flow:**
   - Step 1: Stable MOTS + DEC → Dain-Reiris gives $A(0) \geq 8\pi|J|$ (INDEPENDENT)
   - Step 2: Monotonicity + Dain-Reiris → $m_{H,J}'(t) \geq 0$
   - Step 3: Boundary values → $M_{\text{ADM}} \geq m_{H,J}(0)$

3. **Different theorems:** Dain-Reiris bounds area-angular momentum ratios on horizons. Penrose inequality bounds mass in terms of horizon quantities. Neither implies the other.

### Referee-Ready Response:
> "No circularity. Dain-Reiris (Theorem 6.1) is proven independently using MOTS stability and DEC—it doesn't use any mass bound. The logical chain is: Dain-Reiris → sub-extremality → monotonicity → Penrose inequality. See Remark 6.2."

---

## QUESTION 9: Does the perturbation stability lemma (Lemma 3.7) actually apply?

### Risk Level: **MEDIUM**

### The Concern:
Lemma 3.7 is a general perturbation result. Are its hypotheses (P1)-(P3) actually verified for the axisymmetric Jang equation with twist?

### Current Paper Status:
**ADDRESSED** in Step 2 of Theorem 3.1 and Lemma 3.9.

### The Resolution:

**Verification of (P1):** Blow-up asymptotics
- Han-Khuri [Prop. 4.5] gives $f_0 = C_0 \ln s^{-1} + O(s^\alpha)$ with $C_0 = |\theta^-|/2$ ✓

**Verification of (P2):** Coercivity
- Lemma 3.9 shows indicial roots unchanged by twist (since $D\mathcal{T}|_f = O(e^{-t})$)
- Lockhart-McOwen + MOTS stability → Fredholm inverse exists ✓

**Verification of (P3):** Perturbation bound
- Lemma 3.6: $|\mathcal{T}[f]| \leq C_\mathcal{T} \cdot s$ where $C_\mathcal{T} = C_{\omega,\infty} \cdot \rho_{\max}^2/C_0$ ✓
- This is $O(s^{1+0})$ with $\gamma = 0$, satisfying (P3)

**Explicit constants:** Remark 3.8 provides quantitative formulas.

### Referee-Ready Response:
> "Hypotheses (P1)-(P3) are verified in Theorem 3.1, Step 2. The twist bound (Lemma 3.6) gives $|\mathcal{T}| = O(s)$, the indicial roots are unchanged (Lemma 3.9), and the unperturbed blow-up is from Han-Khuri. See Remark 3.8 for explicit constants."

---

## QUESTION 10: How is rigidity (equality iff Kerr) established?

### Risk Level: **LOW-MEDIUM**

### The Concern:
The paper claims equality holds if and only if the data is a Kerr slice. How is this proven?

### Current Paper Status:
**ADDRESSED** in Theorem 8.1 and Appendix A.

### The Resolution:

1. **Equality requires $\Lambda_J = 0$:**
   - If $M = \sqrt{A/(16\pi) + 4\pi J^2/A}$, monotonicity is saturated
   - This requires $R_{\tilde{g}} = \Lambda_J \phi^{-12} = 0$
   - Since $\phi > 0$, we need $\Lambda_J = 0$

2. **$\Lambda_J = 0$ characterizes Kerr:**
   - $\Lambda_J = \frac{1}{8}|\mathcal{S}_{(g,K)}|^2$
   - Mars uniqueness theorem: $\mathcal{S}_{(g,K)} = 0$ iff data is Kerr slice

3. **Kerr saturates the bound:**
   - Section 2 (Theorem 2.1) explicitly verifies equality for all $|a| \leq M$
   - Worked example (Example 2.2) computes numerically for $a/M = 0.9$

### Referee-Ready Response:
> "Equality requires $\Lambda_J = \frac{1}{8}|\mathcal{S}_{(g,K)}|^2 = 0$. By the Mars-Simon characterization (Appendix A), this holds iff the data is a Kerr slice. Section 2 independently verifies Kerr saturates the bound."

---

## SUMMARY: All 10 Questions Are Addressed

| # | Question | Risk | Status | Key Reference |
|---|----------|------|--------|---------------|
| 1 | Mass bound without $\phi \leq 1$ | CRITICAL | ✅ ADDRESSED | Lemma 4.5, Remark 4.3 |
| 2 | $J$-conservation across manifolds | CRITICAL | ✅ ADDRESSED | Theorem 5.4, Remark 5.5 |
| 3 | Axis pole singularities | HIGH | ✅ ADDRESSED | Lemma 3.4, Remark 3.6 |
| 4 | Vacuum necessity | HIGH | ✅ ADDRESSED | Remark 1.10 |
| 5 | Spectral gap / Fredholm | MED-HIGH | ✅ ADDRESSED | Lemma 4.2 |
| 6 | Kerr deviation tensor | MED-HIGH | ✅ ADDRESSED | Def. 1.8, Appendix A |
| 7 | Monotonicity formula | MED | ✅ ADDRESSED | Theorem 5.3, Table 2 |
| 8 | Dain-Reiris circularity | MED | ✅ ADDRESSED | Remark 6.2 |
| 9 | Perturbation lemma | MED | ✅ ADDRESSED | Thm 3.1 Step 2, Lemma 3.9 |
| 10 | Rigidity | LOW-MED | ✅ ADDRESSED | Theorem 8.1, Appendix A |

**CONCLUSION:** The paper has proactively addressed all 10 critical/risky questions with detailed proofs, explicit bounds, and clear explanations. No additional modifications are required.
