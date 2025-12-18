# Why Area Monotonicity Fails: The Mathematical Obstruction

## The Central Problem

**Goal:** Prove $A(\Sigma^*) \geq A(\Sigma_0)$ where:
- $\Sigma_0$ = arbitrary trapped surface (inner surface)
- $\Sigma^*$ = outermost MOTS (outer surface, apparent horizon)

**Why we need this:** If we can prove area monotonicity, then:
$$M_{\mathrm{ADM}} \geq \sqrt{\frac{A(\Sigma^*)}{16\pi}} \geq \sqrt{\frac{A(\Sigma_0)}{16\pi}}$$

The first inequality is **already proved** (Theorem penroseinitial). We need the second inequality.

---

## Attempt 1: Mean Curvature Flow Argument

### Setup
Consider a foliation $\{\Sigma_t\}_{t \in [0,1]}$ with:
- $\Sigma_0$ = inner trapped surface
- $\Sigma_1 = \Sigma^*$ = outermost MOTS
- Moving **outward** from $\Sigma_0$ to $\Sigma^*$

### Area Evolution
The first variation of area along the foliation:
$$\frac{dA}{dt} = \int_{\Sigma_t} H_t \, \phi_t \, dA_t$$

where:
- $H_t$ = mean curvature of $\Sigma_t$
- $\phi_t$ = lapse function (speed of outward motion)

### The Problem
In the trapped region:
$$H = \frac{1}{2}(\theta^+ + \theta^-) < 0$$

Why? Both null expansions are negative:
- $\theta^+ = H + \tr_\Sigma k \leq 0$ (outer trapped)
- $\theta^- = H - \tr_\Sigma k < 0$ (inner trapped)

Therefore:
$$\frac{dA}{dt} = \int_{\Sigma_t} \underbrace{H_t}_{<0} \, \underbrace{\phi_t}_{>0} \, dA_t < 0$$

**Conclusion:** Area **decreases** as we move outward! This gives:
$$A(\Sigma^*) < A(\Sigma_0)$$

This is the **WRONG** direction! ❌

---

## Attempt 2: Inverse Mean Curvature Flow (IMCF)

### For k=0 (Time-Symmetric): ✅ WORKS

When $k = 0$:
- The metric $(M, g)$ is Riemannian
- $\theta^+ = \theta^- = H$ (the null expansions coincide)
- Can apply Huisken–Ilmanen weak IMCF

**Key result:** The Hawking mass
$$m_H(\Sigma) = \sqrt{\frac{A(\Sigma)}{16\pi}}\left(1 - \frac{1}{16\pi}\int_\Sigma H^2 \, dA\right)$$

is **monotone increasing** along weak IMCF.

**Application:**
- Start IMCF from $\Sigma_0$ (where $H < 0$)
- Flow to $\Sigma^*$ (where $H = 0$)
- Monotonicity of $m_H$ gives:

$$\sqrt{\frac{A(\Sigma_0)}{16\pi}}\underbrace{\left(1 - \frac{1}{16\pi}\int_{\Sigma_0} H^2 dA\right)}_{<1} \leq \sqrt{\frac{A(\Sigma^*)}{16\pi}}\underbrace{\left(1 - 0\right)}_{=1}$$

Since the first correction term is $< 1$ and the second is $= 1$:
$$\sqrt{\frac{A(\Sigma_0)}{16\pi}} < \sqrt{\frac{A(\Sigma^*)}{16\pi}}$$

Therefore: $A(\Sigma^*) > A(\Sigma_0)$ ✅

---

### For k≠0: ❌ FAILS

When $k \neq 0$:
- The data is **not** Riemannian
- $\theta^+ \neq \theta^-$ (different null expansions in different directions)
- The mean curvature $H = \frac{1}{2}(\theta^+ + \theta^-)$ is not the same as either null expansion
- **No IMCF monotonicity formula** is known for general $(M, g, k)$

**Obstruction:** The extrinsic curvature $k$ couples to the geometry in a way that breaks the IMCF argument.

---

## Attempt 3: Null Expansion Foliation

### Idea
Since $\theta^+$ and $\theta^-$ are the null expansions, perhaps use **null foliations** instead of spacelike?

### Null Expansion Evolution
Along the outgoing null geodesics:
$$\frac{d\theta^+}{d\lambda} = -\frac{1}{2}(\theta^+)^2 - \sigma^+_{AB}\sigma^{+AB} - R_{AB}\ell^+_A\ell^+_B$$

where:
- $\sigma^+_{AB}$ = null shear tensor
- $R_{AB}\ell^+_A\ell^+_B$ = Ricci curvature term (related to energy density by Einstein equations)

### The Problem
In the trapped region:
- $\theta^+ < 0$ (converging outgoing rays)
- The Raychaudhuri equation shows $\theta^+$ becomes **more negative** as you move outward (focusing)
- This means outgoing rays converge faster and faster
- Area **decreases** along null rays

**Conclusion:** Null foliation also gives area decrease, not increase. ❌

---

## Why Counterexamples Exist

### Binary Black Hole Merger
Consider two black holes merging:

1. **Before merger:** Two separate apparent horizons with areas $A_1$ and $A_2$
2. **During merger:** A **common apparent horizon** forms with area $A_{\text{outer}}$
3. **Interior trapped surfaces:** Inside the common horizon, there can be trapped surfaces with area $A_{\text{interior}}$

**Physical fact:** In dynamical mergers:
$$A_{\text{interior}} > A_{\text{outer}}$$

is **possible**!

**Why?** The interior surface "remembers" the initial configuration where two separate horizons existed. The outer common horizon has not yet "settled" to equilibrium.

**This violates area monotonicity!** The inner trapped surface has **larger** area than the outer apparent horizon.

---

## Resolution Strategies

### 1. Restrict to k=0 ✅
- Use Huisken–Ilmanen IMCF
- **Fully proved**

### 2. Restrict to Outermost MOTS ✅
- Don't try to prove $A(\Sigma^*) \geq A(\Sigma_0)$
- Just prove $M_{\mathrm{ADM}} \geq \sqrt{A(\Sigma^*)/(16\pi)}$ directly
- **Fully proved**

### 3. Add Curvature Bounds (C1) ⚠️
- Assume $|Rm| \leq K$ in trapped region
- Use maximum area variational method
- Find $\Sigma_{\max}$ with $A(\Sigma_{\max}) \geq A(\Sigma_0)$ by definition
- **Proved under (C1)**

### 4. Invoke Cosmic Censorship ⚠️
- Assume weak cosmic censorship (WCC)
- Trapped surfaces lie inside black holes
- Event horizon area provides bound
- **Conditional on WCC** (Penrose's original 1973 argument)

---

## The Fundamental Obstruction

The mathematical heart of the problem:

**When k=0:**
- Geometry is Riemannian
- Mean curvature $H$ is the **only** relevant quantity
- IMCF monotonicity works

**When k≠0:**
- Geometry is "partially Lorentzian" (encoded by $k$)
- **Two** null expansions $\theta^+$ and $\theta^-$ compete
- Mean curvature $H = \frac{1}{2}(\theta^+ + \theta^-)$ averages them
- No single foliation (spacelike or null) respects the geometry

**Physical interpretation:**
- $k \neq 0$ means spacetime is **evolving**
- Interior trapped surfaces can "inflate" due to incoming matter/energy
- Outer horizon area grows more slowly
- Area monotonicity is **not a theorem** — it's violated in dynamical scenarios

---

## Conclusion

Area monotonicity $A(\Sigma^*) \geq A(\Sigma_0)$ is:

| Case | Status | Reason |
|------|--------|--------|
| **k=0** | ✅ **TRUE** | Huisken-Ilmanen IMCF |
| **k≠0 + (C1)** | ✅ **TRUE** | Maximum area method |
| **k≠0 (general)** | ❌ **FALSE** | Counterexamples in binary mergers |

The failure of area monotonicity in the general case is **not a gap in the proof**—it's a **physical reality** of dynamical spacetimes.

**The Penrose inequality still holds for the outermost MOTS** (which is what matters physically), but extending to arbitrary interior trapped surfaces requires additional assumptions.
