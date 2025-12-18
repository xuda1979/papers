# SHORT VERSION OUTLINE
## Angular Momentum Penrose Inequality Paper
### Target: 50-60 pages for Communications in Mathematical Physics

**Created:** December 15, 2025  
**Purpose:** Condensed submission version; full version on arXiv

---

# STRATEGY

## Current Paper Stats
- **Full version:** 8398 lines (~170 pages double-spaced)
- **Target:** 50-60 pages (2500-3000 lines)
- **Reduction:** ~65% content moved to arXiv supplementary

## Guiding Principles
1. **Keep:** Core theorem, essential proofs, rigidity analysis
2. **Condense:** Duplicate explanations, verbose remarks
3. **Move to arXiv:** Extensions, numerical verification, detailed examples
4. **Reference:** "See [arXiv:XXXX.XXXXX] for details" for technical lemmas

---

# PROPOSED STRUCTURE

## PART I: FRONT MATTER (5-6 pages)

### §1. Introduction (5-6 pages)

#### 1.1 Main Result (1 page)
- **Theorem 1.1** (Main): Full statement with hypotheses (H1)-(H4)
- State the inequality: $M_{ADM} \geq \sqrt{A/(16\pi) + 4\pi J^2/A}$
- State rigidity: equality iff Kerr

#### 1.2 Historical Context (1 page)
- Brief history: Penrose 1973 → Huisken-Ilmanen/Bray → this work
- Prior partial results: Dain 2008, Dain-Reiris 2011
- One paragraph on why vacuum hypothesis is necessary

#### 1.3 Proof Overview (1.5 pages)
- Four-stage method diagram:
  ```
  Original Data (M,g,K) 
      → Stage 1: Jang (M̄,ḡ) 
      → Stage 2: AM-Lichnerowicz (M̃,g̃) 
      → Stage 3: AMO + J-conservation 
      → Stage 4: Dain-Reiris bound
  ```
- State the Critical Inequality (key innovation)
- Explain the AM-Hawking mass $m_{H,J}(t)$

#### 1.4 Notation and Conventions (0.5 pages)
- Define: $M_{ADM}$, $A$, $J$, $\eta$, $\Sigma$
- MOTS stability operator $L_\Sigma$
- Twist 1-form $\omega$ and potential $\psi$

**OMIT from full version:**
- Detailed hypothesis explanations (Remarks 1.4-1.7)
- Multi-component discussion (Remark 1.6) → arXiv
- Red-Blue analysis section → arXiv
- Executive summary boxes → integrate into text

---

## PART II: TECHNICAL FOUNDATIONS (10-12 pages)

### §2. Preliminaries (3-4 pages)

#### 2.1 Asymptotically Flat Initial Data (1 page)
- Definition of AF data with decay $\tau > 1/2$
- ADM mass formula
- Dominant Energy Condition (one sentence)

#### 2.2 MOTS and Stability (1 page)
- Definition of MOTS: $\theta^+ = 0$
- Stability operator $L_\Sigma$ and $\lambda_1 > 0$
- Galloway-Schoen: $\Sigma \cong S^2$

#### 2.3 Axisymmetric Structure (1-2 pages)
- Killing field $\eta$, orbit space quotient
- Komar angular momentum: $J = \frac{1}{8\pi}\int_\Sigma K(\eta,\nu) d\sigma$
- Twist 1-form: $\omega = d\psi$ (by vacuum + cohomology)

**OMIT:**
- Weighted Hölder spaces details → arXiv
- Axis regularity conditions (AR1)-(AR3) → arXiv appendix
- Bounded geometry verification → arXiv

---

### §3. Stage 1: Axisymmetric Jang Equation (3-4 pages)

#### 3.1 The Jang Equation with Twist (1 page)
- State equation: $H_\Gamma - \tr_\Gamma K = S_\omega[f]$
- Explain twist as lower-order perturbation

#### 3.2 Existence Theorem (1.5 pages)
- **Theorem 3.1** (Jang Existence): Statement only
- Key properties: cylindrical end at MOTS, $f \sim t + O(e^{-\beta t})$
- Mass bound: $M_{ADM}(\bar{g}) \leq M_{ADM}(g)$

#### 3.3 Scalar Curvature Identity (1 page)
- $R_{\bar{g}} \geq 0$ from DEC
- Twist contribution to Jang-conformal curvature

**OMIT:**
- Full proof of existence → "See [Han-Khuri 2013] and arXiv"
- Indicial root analysis → arXiv
- Weighted Sobolev setup → arXiv

---

### §4. Stage 2: AM-Lichnerowicz Equation (3-4 pages)

#### 4.1 The Modified Equation (1 page)
- State: $-8\Delta_{\bar{g}}\phi + R_{\bar{g}}\phi = \Lambda_J \phi^{-7}$
- Boundary conditions: $\phi|_\Sigma = 1$, $\phi \to 1$ at $\infty$

#### 4.2 Existence and Mass Bound (1.5 pages)
- **Theorem 4.1** (AM-Lich Existence): Statement
- **Lemma 4.2** (Mass Chain): $M_{ADM}(\tilde{g}) \leq M_{ADM}(\bar{g})$
- Sketch energy identity argument

#### 4.3 Properties of Conformal Metric (1 page)
- $R_{\tilde{g}} = \Lambda_J \phi^{-12} \geq 0$
- MOTS becomes minimal surface: $H_{\tilde{g}} = 0$ at inner boundary
- Area preserved at $t = 0$

**OMIT:**
- Detailed Fredholm theory → arXiv
- Exponential decay verification → arXiv
- Energy identity full proof → arXiv appendix

---

## PART III: CORE PROOF (15-18 pages)

### §5. Stage 3: Monotonicity and the Critical Inequality (8-10 pages)

**THIS IS THE HEART OF THE PAPER**

#### 5.1 The AMO Flow and AM-Hawking Mass (2 pages)
- $p$-harmonic potential $u_p$ with level sets $\Sigma_t$
- Hawking mass: $m_H(t) = \sqrt{A(t)/(16\pi)}(1 - W(t))$
- AM-Hawking mass: $m_{H,J}(t) := \sqrt{m_H^2(t) + 4\pi J^2/A(t)}$

#### 5.2 Angular Momentum Conservation (2 pages)
- **Theorem 5.1** (J-Conservation): $J(t) = J$ constant
- Proof via Stokes' theorem + vacuum condition
- Why vacuum is necessary (flux formula)

#### 5.3 The Critical Inequality (3-4 pages) ⭐ KEY SECTION ⭐
- **Proposition 5.2** (Critical Inequality):
  $$\frac{dm_H^2}{dt} \geq \frac{4\pi J^2}{A^2} \frac{dA}{dt}$$

- **Full proof outline:**
  - Step 1: AMO monotonicity formula
  - Step 2: Twist contribution $R_{\tilde{g}} \geq |d\psi|^2/(2\rho^4)$
  - Step 3: Cauchy-Schwarz for twist integral:
    $$\int_{\Sigma_t} \frac{|d\psi|^2}{\rho^4} d\sigma \geq \frac{64\pi^2 J^2}{A(t)}$$
  - Step 4: Combine with area evolution

- **Lemma 5.3** (Gradient-Area Bound): ← NEW, per Attack #28
  $$\int_{\Sigma_t} |\nabla u| d\sigma \leq C(W_{max}) \cdot A(t)$$

#### 5.4 Integration to Main Theorem (1.5 pages)
- Integrate Critical Inequality from $t = 0$ to $t = 1$
- Boundary values: $m_{H,J}(1) = M_{ADM}$, $m_{H,J}(0) = \sqrt{A/(16\pi) + 4\pi J^2/A}$
- **Conclusion:** Main inequality proved

**OMIT:**
- Mosco convergence details → arXiv
- $p \to 1$ limit technicalities → arXiv
- Weighted measure analysis → arXiv

---

### §6. Stage 4: Sub-Extremality (2-3 pages)

#### 6.1 The Dain-Reiris Bound (1.5 pages)
- **Theorem 6.1** (Sub-Extremality): $A \geq 8\pi|J|$
- State result from [Dain-Reiris 2011]
- Why needed: ensures $m_{H,J}$ is real

#### 6.2 Preservation Along Flow (1 page)
- Area non-decreasing: $A'(t) \geq 0$
- $J$ constant
- Sub-extremality improves: $A(t)/|J| \geq A(0)/|J| \geq 8\pi$

**OMIT:**
- Independent proof of Dain-Reiris → reference
- Detailed preservation analysis → arXiv

---

### §7. Rigidity: Equality Implies Kerr (4-5 pages)

#### 7.1 Equality Forces Stationarity (2 pages)
- Equality $\Rightarrow$ $R_{\tilde{g}} = 0$ everywhere
- $\Lambda_J = \frac{1}{8}|\sigma^{TT}|^2 = 0$
- Moncrief correspondence: $\sigma^{TT} = 0 \Rightarrow$ stationary

#### 7.2 Kerr Uniqueness (2 pages)
- Carter-Robinson theorem (cite Chruściel-Costa for smooth version)
- MOTS $\subset$ Event Horizon (Andersson-Mars-Simon)
- **Theorem 7.1**: Equality iff Kerr

#### 7.3 Kerr Saturation Verification (1 page)
- Direct calculation: Kerr satisfies equality
- Table of values for $a/M = 0, 0.5, 0.9, 0.99$

**OMIT:**
- Detailed Kerr algebra → arXiv appendix
- Numerical verification scripts → arXiv
- Higher-dimensional analogues → arXiv

---

## PART IV: CONCLUSION (3-4 pages)

### §8. Discussion (2-3 pages)

#### 8.1 Summary of Method
- Four-stage Jang-conformal-AMO framework
- Critical Inequality as key innovation
- Role of twist potential

#### 8.2 Open Problems (1 page)
- Non-axisymmetric case
- Multiple horizons
- Kerr-Newman (charged + rotating)
- Dynamical horizons

**OMIT:**
- Extensive corollaries → arXiv
- Thermodynamic interpretations → arXiv
- Charged Penrose proof → arXiv (or separate paper)

### §9. Acknowledgments (0.25 pages)

### References (2-3 pages)
- Target: 40-50 references (currently ~90)
- Keep: Foundational (Penrose, Bray, HI, AMO, Dain-Reiris, Chruściel-Costa)
- Reduce: Historical citations, tangential results

---

## APPENDICES (5-8 pages)

### Appendix A: Key Estimates Summary (2 pages)
- Table of critical bounds and their locations
- Cross-reference to arXiv for full proofs

### Appendix B: Twist Integral Computation (2-3 pages)
- Detailed Cauchy-Schwarz application
- Axis regularity verification
- Gradient-Area Bound proof

### Appendix C: Technical Clarifications (2-3 pages)
- Condensed version of Red-Blue analysis
- Common objections and resolutions
- $(1-W)$ factor tracking

---

# SECTION-BY-SECTION COMPARISON

| Full Version Section | Pages | Short Version | Pages | Status |
|---------------------|-------|---------------|-------|--------|
| §1 Introduction | 25 | §1 Introduction | 6 | Condense |
| §2 Proof Overview | 15 | Merged into §1.3 | 1.5 | Merge |
| §3 Jang Equation | 30 | §3 Jang | 4 | Heavy condense |
| §4 AM-Lichnerowicz | 20 | §4 AM-Lich | 4 | Heavy condense |
| §5 AMO + J-conserve | 25 | §5 Monotonicity | 10 | Keep core |
| §6 Sub-extremality | 10 | §6 Sub-ext | 3 | Condense |
| §7 Rigidity | 15 | §7 Rigidity | 5 | Condense |
| §8 Extensions | 20 | §8.2 Open problems | 1 | Heavy condense |
| §9 Charged Penrose | 10 | OMIT | 0 | → arXiv |
| §10-13 Appendices | 30 | App A-C | 8 | Select key items |
| **Total** | **~170** | **~50-55** | **~50-55** | |

---

# WHAT TO MOVE TO arXiv SUPPLEMENTARY

## Complete Sections to Move
1. **§8 Extensions** (except open problems summary)
2. **§9 Charged Penrose Inequality** (separate paper or arXiv)
3. **Numerical Verification Section** (all Python results)
4. **Multi-component discussion** (Remark 1.6)
5. **Red-Blue analysis full version**

## Technical Details to Move
1. Weighted Hölder/Sobolev space definitions
2. Axis regularity conditions (AR1)-(AR3)
3. Bounded geometry verification lemma
4. Indicial root analysis
5. Fredholm theory for cylindrical ends
6. Mosco convergence details
7. $p \to 1$ limit analysis
8. Detailed Kerr algebra
9. Corollaries (entropy bounds, extractable energy, etc.)

## What to Keep with "[See arXiv]" Reference
- Jang existence proof sketch → full proof arXiv
- AM-Lich existence proof sketch → full proof arXiv
- Energy identity verification → full details arXiv
- Numerical tables → full data arXiv

---

# SHORT VERSION TABLE OF CONTENTS

```
ABSTRACT (200 words)

1. INTRODUCTION
   1.1 Main Result
   1.2 Historical Context
   1.3 Proof Overview
   1.4 Notation

2. PRELIMINARIES
   2.1 Asymptotically Flat Initial Data
   2.2 MOTS and Stability
   2.3 Axisymmetric Structure

3. STAGE 1: JANG EQUATION
   3.1 The Equation with Twist
   3.2 Existence Theorem
   3.3 Scalar Curvature Identity

4. STAGE 2: AM-LICHNEROWICZ
   4.1 The Modified Equation
   4.2 Existence and Mass Bound
   4.3 Properties of Conformal Metric

5. STAGE 3: MONOTONICITY (⭐ Main Section)
   5.1 AMO Flow and AM-Hawking Mass
   5.2 Angular Momentum Conservation
   5.3 The Critical Inequality
   5.4 Integration to Main Theorem

6. STAGE 4: SUB-EXTREMALITY
   6.1 The Dain-Reiris Bound
   6.2 Preservation Along Flow

7. RIGIDITY
   7.1 Equality Forces Stationarity
   7.2 Kerr Uniqueness
   7.3 Kerr Saturation

8. DISCUSSION
   8.1 Summary
   8.2 Open Problems

ACKNOWLEDGMENTS

APPENDIX A: Key Estimates
APPENDIX B: Twist Computation
APPENDIX C: Technical Clarifications

REFERENCES (~45 items)
```

---

# CRITICAL WRITING TASKS

## Priority 1: Must Write for Short Version
1. **Condensed §1** (rewrite from scratch, 6 pages)
2. **Lemma 5.3** (Gradient-Area Bound) - NEW
3. **Condensed proof of Critical Inequality** (keep all steps, reduce verbiage)
4. **Appendix C** (condensed Red-Blue)

## Priority 2: Heavy Editing
1. §3 Jang: Keep theorem statements, remove proofs
2. §4 AM-Lich: Keep theorem statements, sketch energy identity
3. §7 Rigidity: Streamline logical chain

## Priority 3: Structural
1. Remove all "Executive Summary" boxes
2. Consolidate duplicate explanations
3. Forward references → inline brief mentions
4. Reduce bibliography by ~50%

---

# TIMELINE ESTIMATE

| Task | Time | 
|------|------|
| Create structure/skeleton | 2 hours |
| Write condensed §1 | 4 hours |
| Condense §3-4 | 4 hours |
| Polish §5 (keep mostly intact) | 2 hours |
| Condense §6-7 | 3 hours |
| Write §8 discussion | 1 hour |
| Write Appendices A-C | 4 hours |
| Final editing/polish | 4 hours |
| **Total** | **~24 hours** |

---

# FINAL CHECKLIST

## Before Starting Short Version
- [ ] Verify all theorems/lemmas have clear statements
- [ ] Identify all forward references to resolve
- [ ] List all technical details safe to omit
- [ ] Confirm arXiv version will be complete

## Short Version Requirements
- [ ] Self-contained (no essential proofs only on arXiv)
- [ ] All main theorems fully stated
- [ ] Critical Inequality proof complete
- [ ] Rigidity argument complete
- [ ] References sufficient for verification

## CMP Specific
- [ ] Abstract ≤ 200 words
- [ ] Keywords included
- [ ] MSC codes included
- [ ] Double-spaced
- [ ] 50-60 pages total

---

*Short Version Outline Complete*
*December 15, 2025*
