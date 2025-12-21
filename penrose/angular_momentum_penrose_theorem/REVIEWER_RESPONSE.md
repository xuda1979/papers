# Response to Referee Review (December 21, 2025)

We thank the referee for their thorough and insightful review of our manuscript "The Angular Momentum Penrose Inequality: A Proof via the Extended Jang–Conformal–AMO Method." Below we address each concern raised in the review.

---

## Response to Concern A: The Kerr Deviation Tensor Construction

**Reviewer's Concern:** The well-definedness lemma for the Kerr deviation tensor $\mathcal{S}_{(g,K)}$ should be more explicit about function spaces, boundary data, and gauge independence.

**Our Response:** We have significantly expanded the proof of Lemma 5.3 (`lem:Lambda-J-welldef`) in Section 5 to provide complete details. The enhanced proof now includes:

1. **Function spaces:** We explicitly state that $(E, B) \in C^{k-2,\alpha}_{-\tau-2}(M; S^2 T^*M)$ when $(g, K) \in C^{k,\alpha}_{-\tau} \times C^{k-1,\alpha}_{-\tau-1}$ for $k \geq 3$, $\tau > 1/2$.

2. **Bianchi constraint propagation:** We provide a rigorous PDE analysis showing that the Bianchi system is a first-order elliptic system in harmonic gauge. The solution exists and is unique in weighted Sobolev spaces $H^s_\delta(M; S^2_0 T^*M)$ with $s \geq 2$ and $\delta \in (-2, -1/2)$.

3. **Coordinate-independent asymptotic matching:** The parameters $(M, J)$ are defined by coordinate-independent integrals (ADM mass and Komar angular momentum). The reference Kerr Weyl tensor is constructed via asymptotic matching in the ADM frame, then extended uniquely via the Bianchi constraints.

4. **Gauge independence:** We explicitly enumerate the gauge freedom (asymptotic translations, rotations, boosts) and explain how each is fixed by standard conventions (ADM center of mass, Killing field alignment, vanishing linear momentum).

5. **Characterization of Kerr:** The rigidity statement $\mathcal{S}_{(g,K)} = 0 \Leftrightarrow$ Kerr follows from the Ionescu–Klainerman uniqueness theorem.

The complete technical details remain in Appendix G (`app:mars-simon`), but the main body now contains a self-contained argument that directly addresses the referee's concerns about well-definedness.

---

## Response to Concern B: Extension to Non-Vacuum Matter

**Reviewer's Suggestion:** Add a remark or proposition clarifying whether the argument extends to non-vacuum exteriors under a symmetry-compatible "no angular momentum flux" condition.

**Our Response:** This is already addressed in Section 10.4.1, specifically Proposition 10.x (`prop:non-vacuum-extension`), which states:

**Proposition (Conditional Extension):** The AM-Penrose inequality holds under the weaker hypothesis (H3'): $j_\phi := g(j, \eta) = 0$ in $M_{\mathrm{ext}}$ (vanishing azimuthal momentum flux), rather than full vacuum.

The proof shows that under (H3'), the Komar 1-form remains co-closed: $d^\dagger \alpha_J = 0$, which is the essential property for $J$-conservation. The physical interpretation is that matter does not carry angular momentum flux through axisymmetric surfaces.

**Physical examples satisfying (H3'):** 
- Co-rotating perfect fluids with 4-velocity parallel to the timelike Killing field
- Electrovacuum with axisymmetric fields where the Poynting vector has no azimuthal component
- Axisymmetric scalar field matter

We also explain in Remark 10.y (`rem:full-nonvacuum`) why the fully general non-vacuum case remains difficult: when $j_\phi \neq 0$, the joint evolution of $A(t)$ and $J(t)$ is uncontrolled.

---

## Response to Concern C: The $p \to 1$ Limit and "Almost All $t$" Issues

**Reviewer's Concern:** Clarify exactly which parts of the monotonicity hold "for a.e. $t$" versus "for all $t$", and why that's sufficient.

**Our Response:** This is comprehensively addressed in Remark 6.x (`rem:ae-vs-all`) in Section 6, highlighted in a yellow box for emphasis. The key points are:

1. **What holds for a.e. $t$:**
   - Level sets $\Sigma_t$ are smooth embedded surfaces for a.e. $t$ (Sard's theorem)
   - The derivative formula $\frac{d}{dt}m_{H,J}^2(t) \geq 0$ holds at regular values

2. **What holds for ALL $t$:**
   - The functions $A(t)$, $m_H(t)$, $m_{H,J}(t)$ are continuous and absolutely continuous
   - Boundary values $m_{H,J}(0)$ and $m_{H,J}(1)$ are well-defined
   - Global monotonicity $m_{H,J}(t_1) \leq m_{H,J}(t_2)$ holds for ALL $t_1 < t_2$

3. **Why a.e. suffices:** By the fundamental theorem of calculus for absolutely continuous functions:
   $$m_{H,J}^2(1) - m_{H,J}^2(0) = \int_0^1 \frac{d}{dt}m_{H,J}^2(t)\, dt \geq 0$$
   The singular set has measure zero, so its contribution vanishes.

**Additional technical details:**
- Lemma 6.y (`lem:gradient-lower-bound`) proves that critical points of $u_p$ are isolated (dimension 0)
- Remark 6.z (`rem:distributional-rigor`) provides complete justification for the $p \to 1^+$ limit via Moore–Osgood theorem
- Lemma 6.w (`lem:uniform-p-estimates`) establishes uniform-in-$p$ $C^{1,\alpha}$ bounds

---

## Response to Concern D: MOTS Boundary Value

**Reviewer's Suggestion:** Summarize the boundary argument in a self-contained "mini proof" box in Section 8.

**Our Response:** This is already present! Section 8 contains a highlighted yellow box labeled "**Mini Proof: MOTS Boundary Value**" (`box:mots-mini-proof`) that provides a self-contained 5-step derivation:

1. **Minimality in $\bar{g}$:** Cylindrical slices are asymptotically totally geodesic, so $H_{\bar{g}}|_\Sigma = 0$
2. **Neumann boundary for $\phi$:** Exponential decay $\phi = 1 + O(e^{-\kappa t})$ implies $\partial_\nu\phi|_\Sigma = 0$
3. **Hawking mass of minimal surface:** For $H_{\tilde{g}}|_\Sigma = 0$: $m_H(\Sigma) = \sqrt{A/(16\pi)}$
4. **Area preservation:** Since $df|_\Sigma$ is purely normal and $\phi|_\Sigma = 1$: $A_{\tilde{g}}(\Sigma) = A$
5. **Conclusion:** $m_{H,J}(0) = \sqrt{A/(16\pi) + 4\pi J^2/A}$

The mini-proof explicitly notes: "This is an equality, not merely a lower bound."

Additional context is provided in Remark 8.x (`rem:t0-clarification`) explaining the limiting procedure from the cylindrical end.

---

## Response to Concern E: Scope Limitations

**Reviewer's Suggestion:** Emphasize early that the result does not address the fully general AM-Penrose conjecture.

**Our Response:** This is already prominently displayed. Immediately after Theorem 1.1 (the main result), we include a **red-bordered highlighted box** titled "Scope of This Result" that explicitly states:

> Theorem 1.1 establishes the Angular Momentum Penrose Inequality **only** for initial data satisfying:
> - **Axisymmetry:** A Killing field $\eta = \partial_\phi$ must exist;
> - **Vacuum exterior:** The region outside the MOTS must satisfy $\mu = |j| = 0$.
>
> This does **not** resolve the fully general AM-Penrose conjecture (non-axisymmetric data, matter present).

This scope statement appears on page 2-3 of the manuscript, ensuring readers understand the limitations from the outset. The box also explains *why* each restriction is needed (defining Komar $J$, orbit-space reduction, $J$-conservation).

---

## Summary of Changes Made

1. **Section 5 (Lichnerowicz equation):** Expanded Lemma 5.3 proof with complete details on function spaces, Bianchi system well-posedness, and gauge independence.

2. **Existing content verified:** The manuscript already contains:
   - Non-vacuum extension (Section 10.4.1, Proposition 10.x)
   - MOTS boundary mini-proof box (Section 8)
   - Scope limitation box (after Theorem 1.1)
   - Extensive $p \to 1$ limit analysis (Section 6, Remarks on a.e. vs all t)
   - Critical point control and uniform-in-$p$ estimates

We believe these revisions address all the referee's concerns while maintaining the paper's logical structure and readability.

---
---
---

# Previous Response to Internal Review

## Executive Summary

This document responds to the earlier detailed internal review. After careful examination, the key concerns raised are already thoroughly addressed in the current version.

---

## Response to Potential Areas for Scrutiny

### 1. The Twist Estimate (Lemma 4.12 → Now Lemma `lem:twist-bound`)

**Reviewer Concern:** The claim that the twist term is a "lower-order perturbation" is the linchpin of the existence theory. A referee would likely check the derivation of O(s³) [Note: actually O(s)] line-by-line to ensure no circular reasoning regarding the dependence on f.

**Response:** The paper addresses this concern comprehensively in `sec04-jang.tex`:

1. **Explicit derivation in Step 2c (Lines 382-420):** The paper provides a detailed line-by-line derivation showing:
   - The twist 1-form ω is uniformly bounded: |ω| ≤ C_{ω,∞} (from elliptic regularity)
   - The orbit radius satisfies ρ(s,y) = ρ_Σ(y) + O(s) near the MOTS
   - The key scaling: T[f] = O(ρ²)/√(1 + |∇f|²) · O(1) = O(s)

2. **No circular reasoning:** The constant C_T depends **only on the initial data** (g, K) and the blow-up coefficient C₀ = |θ⁻|/2, which is determined by the MOTS geometry **before** solving the Jang equation. Crucially:
   - C_T does **not** depend on higher derivatives ∇^k f for k ≥ 2
   - The twist term T[f] involves **no second derivatives** of f
   - The bound holds uniformly for any function with logarithmic blow-up

3. **Lemma `lem:twist-bound` (Lines 426-478):** Formalizes the scaling analysis with explicit constants.

4. **Lemma `lem:perturbation-stability` (Lines 506-571):** Provides the general perturbation principle using contraction mapping in weighted Sobolev spaces.

**Status:** ✅ Already thoroughly addressed

---

### 2. Regularity at the Poles

**Reviewer Concern:** The paper notes that the coordinate system degenerates at the poles of the sphere. The analysis of the weighted Hölder spaces at these points (Section 4.1) is critical.

**Response:** The paper explicitly addresses pole regularity:

1. **Lemma `lem:mots-axis` (Lines 119-152):** Proves that:
   - MOTS must intersect the axis at exactly two poles (topological necessity)
   - The orbit radius vanishes quadratically at poles: ρ(x) = O(dist(x, p_±))
   - Mean curvature H remains finite at poles (detailed l'Hôpital calculation)

2. **Remark `rem:axis-correction` (Line 154):** Explicitly acknowledges this was corrected from earlier versions.

3. **Lemma `lem:twist-bound-poles` (Lines 157-248):** Dedicated lemma for twist perturbation at poles showing:
   - |T[f̄](x)| ≤ C · ρ(x)² · |∇̄f̄|(x) ≤ C' · d(x,p)² as x → p
   - T(p_N) = T(p_S) = 0 since ρ(p_±) = 0
   - Integrability: ∫_Σ |T| dA < ∞

4. **Axis regularity conditions (AR1)-(AR3) (Lines 199-231):** Explicit conditions for twist potential regularity at the axis including:
   - Twist potential Ω extends smoothly to axis
   - Component regularity: ω_r = O(r), ω_z = O(1) as r → 0
   - Hölder regularity in weighted spaces

**Status:** ✅ Already thoroughly addressed

---

### 3. The Vacuum Constraint

**Reviewer Concern:** The limitation to vacuum exteriors is significant. It restricts the generality compared to the Riemannian Penrose Inequality (which only requires DEC).

**Response:** The paper extensively discusses why vacuum is necessary:

1. **Theorem `thm:J-conserve` (sec06-amo.tex, Lines 210-230):** States that vacuum is required for the conservation argument.

2. **Key identity (Lines 237-242):** For vacuum axisymmetric data, the Komar 1-form satisfies:
   ```
   d†α_J = 0 ⟺ d(⋆_g α_J) = 0
   ```
   This co-closedness follows from the momentum constraint with j_i = 0 (vacuum).

3. **Remark `rem:vacuum-critical` and cross-reference (Line 500):** Explicitly notes why vacuum is essential.

4. **Remark `rem:non-vacuum` (Lines 503-560):** Discusses:
   - For non-vacuum data, J is NOT conserved along the flow
   - States a conjecture for non-vacuum AM-Penrose
   - Discusses the electrovacuum (Kerr-Newman) special case
   - Explains that j^(EM) ≠ 0 breaks co-closedness

5. **Physical interpretation (Lines 225-230):** Explains that vacuum + axisymmetry implies absence of angular momentum flux.

**Status:** ✅ Already thoroughly addressed with explicit discussion of why this limitation exists

---

### 4. Rigidity Characterization (Mars-Simon Tensor vs TT-Tensor)

**Reviewer Concern:** The paper addresses and corrects common misconceptions, such as the idea that σ^{TT} = 0 characterizes Kerr.

**Response:** The paper correctly uses the Mars-Simon/Kerr deviation tensor throughout:

1. **Definition in sec01-introduction.tex (Lines 158-179):**
   - Defines Λ_J = (1/8)|S_{(g,K)}|² using the Kerr deviation tensor
   - Explicitly constructs S_{(g,K)} via electric/magnetic Weyl tensors (E,B)

2. **Critical clarification in sec09-rigidity.tex (Lines 31-38):**
   > "The characterization σ^{TT} = 0 appearing in earlier versions was **incorrect**. Kerr slices generically have σ^{TT} ≠ 0. The correct characterization uses the Mars-Simon/Kerr deviation tensor S_{(g,K)}, which vanishes for **any** slice of Kerr regardless of the slicing choice."

3. **Appendix `app05-mars-simon.tex`:** Complete coordinate-independent construction including:
   - Killing Initial Data (KID) equations
   - Ernst-like potentials on initial data
   - Electric and magnetic Weyl tensor definitions
   - Simon tensor definition (Eq. simon-tensor)
   - Simon-Kerr theorem: S_{ij} = 0 ⟺ data is Kerr slice

4. **Remark `rem:tt-vs-deviation` in sec05-lichnerowicz.tex (Lines 343-351):** Explicitly reconciles TT-tensor and Kerr deviation tensor.

**Status:** ✅ Already thoroughly and correctly addressed

---

## Summary Table

| Scrutiny Area | Status | Location in Paper |
|--------------|--------|-------------------|
| Twist Estimate (non-circular) | ✅ Complete | sec04-jang.tex, Lines 350-571 |
| Pole Regularity | ✅ Complete | sec04-jang.tex, Lines 119-248 |
| Vacuum Constraint Justification | ✅ Complete | sec06-amo.tex, Lines 159-560 |
| Mars-Simon (not TT) for Rigidity | ✅ Complete | sec09-rigidity.tex, app05-mars-simon.tex |

---

## Conclusion

The reviewer's assessment that this is "a high-quality, ambitious mathematical physics paper" is well-founded. All three areas identified as requiring scrutiny have been thoroughly addressed in the current manuscript:

1. **Twist estimates** are derived without circular reasoning, with explicit constants depending only on initial data
2. **Pole regularity** includes dedicated lemmas for the degenerate ρ → 0 case
3. **Vacuum constraint** is explicitly justified with discussion of why non-vacuum fails
4. **Kerr characterization** correctly uses Mars-Simon tensor with explicit disclaimers about σ^{TT}

The paper is ready for submission with these technical details properly documented.
