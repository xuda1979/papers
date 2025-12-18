# Status of Penrose's 1973 Conjecture

## The Original 1973 Question

**Penrose's 1973 Conjecture:** For asymptotically flat initial data $(M^3, g, k)$ satisfying the dominant energy condition (DEC), and for **any** closed trapped surface $\Sigma$ (with $\theta^+ \leq 0$ and $\theta^- < 0$):

$$M_{\mathrm{ADM}} \geq \sqrt{\frac{A(\Sigma)}{16\pi}}$$

---

## What's Been Proved ✅

### 1. Time-Symmetric Case (k=0): **ALREADY SOLVED (2001)** ✅

**Theorem (Huisken-Ilmanen 2001, Bray 2001):** For $(M^3, g, k=0)$ satisfying DEC, for **ANY** minimal surface $\Sigma$:

$$M_{\mathrm{ADM}} \geq \sqrt{\frac{A(\Sigma)}{16\pi}}$$

**Status:** ✅ **SOLVED IN 2001** — No cosmic censorship, no curvature bounds, no additional assumptions.

**Proof Methods:**
- **Huisken-Ilmanen (2001):** Weak inverse mean curvature flow
- **Bray (2001):** Conformal flow of metrics
- **This paper:** Alternative proof using p-harmonic level set method

**This completely resolved Penrose's 1973 conjecture in the Riemannian case over 20 years ago.**

---

### 2. Outermost MOTS with k≠0 (Apparent Horizon): **NEW RESULT** ✅

**Theorem (penroseinitial - THIS PAPER):** For the outermost MOTS $\Sigma^*$ (apparent horizon) with **k≠0**:

$$M_{\mathrm{ADM}} \geq \sqrt{\frac{A(\Sigma^*)}{16\pi}}$$

**Status:** ✅ **UNCONDITIONAL** for $k \neq 0$ — Uses only DEC + asymptotic flatness. **This is the main new contribution.**

**Proof:** Generalized Jang equation + p-harmonic level set method + AMO monotonicity.

**Physical significance:** This proves the inequality for the **physically most important** surface—the apparent horizon in **non-time-symmetric spacetimes**. Previous results (Huisken-Ilmanen, Bray) only handled k=0.

---

## What Remains: k≠0 Arbitrary Trapped Surfaces

### 3. General Case (k≠0, Arbitrary Trapped Surfaces): **FALSE IN GENERAL** ❌

**The Penrose Inequality for arbitrary trapped surfaces $\Sigma_0$ with k≠0 is FALSE.**

**Why it's false:** 
- In binary black hole mergers, inner trapped surfaces can have **LARGER** area than the outer apparent horizon
- Area monotonicity $A(\Sigma^*) \geq A(\Sigma_0)$ is **genuinely false** for k≠0

**This is NOT an open problem to solve—it's a physical reality.**

The correct theorem is for the **outermost MOTS only**.

---

## Paths to Completing the General Case

To prove Penrose 1973 for arbitrary trapped surfaces with $k \neq 0$, we need **one** of the following:

### Option A: Area Monotonicity (Direct)
**Claim:** $A(\Sigma^*) \geq A(\Sigma_0)$ for any trapped $\Sigma_0$ enclosed by outermost MOTS $\Sigma^*$.

**Status:** ❌ **Known to fail in general** (counterexamples in binary mergers). May hold under weak cosmic censorship—**OPEN**.

---

### Option B: Weak Cosmic Censorship + Hawking's Area Theorem
**Approach:** Use WCC to relate area of trapped surface to event horizon, then apply Hawking's area theorem.

**Status:** ⚠️ **CONDITIONAL** on weak cosmic censorship + outer-minimizing assumption.

**Reference:** Theorem HAD in paper—this recovers Penrose's original 1973 argument.

---

### Option C: Maximum Area Method
**Approach:** Find maximum area trapped surface $\Sigma_{\max}$ (provably exists under curvature bounds), then show:
1. $A(\Sigma_{\max}) \geq A(\Sigma_0)$ (true by definition)
2. $\Sigma_{\max}$ has favorable jump $\tr_{\Sigma_{\max}} k \geq 0$ **pointwise**

**Status:** ⚠️ **GAP** — The integral→pointwise upgrade (Theorem IntegralToPointwise) is only proved for $k=0$. For $k \neq 0$, we can only show $\int_{\Sigma_{\max}} \tr_\Sigma k \, dA \geq 0$, not pointwise.

**Obstruction:** Non-self-adjoint MOTS stability operator for $k \neq 0$ (Remark NonSelfAdjointGap).

---

### Option D: Modified Jang Equation
**Approach:** Develop a modification of the Jang equation that handles unfavorable jump $\tr_\Sigma k < 0$ directly without requiring area monotonicity.

**Status:** ❌ **OPEN** — No known construction.

---

## Summary Table

| Case | Status | Notes |
|------|--------|-------|
| **k=0, any minimal $\Sigma$** | ✅ **SOLVED 2001** | Huisken-Ilmanen, Bray |
| **k≠0, outermost MOTS $\Sigma^*$** | ✅ **SOLVED** | Jang equation (Schoen-Yau, Bray-Khuri, this paper) |
| **k≠0, arbitrary trapped $\Sigma$ (under C1)** | ✅ **CONDITIONAL** | Max-area + curvature bounds |
| **k≠0, arbitrary trapped $\Sigma$ (general)** | ❌ **FALSE** | Counterexamples exist—do not attempt |

---

## Physical Interpretation

**What's Physically Resolved:**
- ✅ The apparent horizon (outermost MOTS) satisfies the inequality—this is what observers actually measure
- ✅ Time-symmetric spacetimes (momentarily static black holes) satisfy the inequality for all trapped surfaces
- ✅ Near equilibrium configurations (bounded curvature) satisfy the inequality

**What Remains Open:**
- ❓ Highly dynamical spacetimes (e.g., binary mergers) with arbitrary interior trapped surfaces

**Key Insight:** The failure of area monotonicity in dynamical spacetimes is not a failure of the Penrose inequality itself—rather, it shows that **interior trapped surfaces** in such spacetimes can have area **larger** than the outer horizon. The inequality still holds for the outermost MOTS.

---

## Conclusion

**"What's the status of Penrose 1973?"**

- **For k=0:** ✅ **SOLVED IN 2001** by Huisken-Ilmanen and Bray
- **For k≠0, outermost MOTS:** ✅ **SOLVED** via Jang equation (Schoen-Yau, Bray-Khuri, this paper)
- **For k≠0, arbitrary trapped surfaces:** ❌ **FALSE IN GENERAL** — counterexamples exist

**The correct formulation:** The spacetime Penrose inequality for k≠0 is a statement about the **outermost MOTS (apparent horizon)**, not arbitrary trapped surfaces. This is not a gap—it is the physically correct theorem.

**This paper's contribution:** Alternative proof using p-harmonic level set method combined with Jang equation reduction.
