# Bootstrap Path: Detailed Task Specifications

## Overview

This document provides actionable specifications for each task in the Bootstrap sub-framework. Each task has:
- Precise mathematical statement
- Input/output specification
- Estimated difficulty and time
- Dependencies

---

## Module 2: Finite-Volume Spectral Gap

### Task 2.1.1: Discretize Configuration Space

**Mathematical Setup:**
```
Configuration space: SU(N)^{|E|} where |E| = 4 * L_0^4 edges
For L_0 = 4: |E| = 4 * 256 = 1024 edges
Full space dimension: (N²-1) * 1024 ≈ 8192 for SU(3)
```

**Discretization Options:**

1. **Character Expansion (Recommended):**
   - Expand observables in characters: O(U) = Σ_R c_R χ_R(U)
   - Truncate at representation j_max
   - For SU(2): j_max = 10 gives error < 10^{-6} at β = 1
   
2. **Finite Subgroup:**
   - Use largest finite subgroup of SU(N)
   - For SU(2): Binary icosahedral group (order 120)
   - For SU(3): Σ(360×3) (order 1080)

**Deliverable:** Code implementing truncated character expansion

**Time:** 1-2 weeks

---

### Task 2.2.1: Monte Carlo Autocorrelation

**Algorithm Specification:**

```python
# Pseudocode for gap estimation

def estimate_gap(beta, L0, N_samples=10000, N_thermalize=1000):
    # Initialize lattice
    U = initialize_cold_start(L0, gauge_group="SU(N)")
    
    # Thermalize
    for i in range(N_thermalize):
        U = hmc_step(U, beta)
    
    # Collect samples
    plaquette_values = []
    for i in range(N_samples):
        U = hmc_step(U, beta)
        plaquette_values.append(average_plaquette(U))
    
    # Compute autocorrelation
    autocorr = compute_autocorrelation(plaquette_values)
    
    # Fit exponential decay
    tau_int = fit_exponential_decay(autocorr)
    
    # Spectral gap estimate
    gap = 1 / (2 * tau_int)
    
    return gap, error_estimate
```

**Parameters:**
- L_0 = 4
- β range: [0.15, 2.5] in steps of 0.1
- N_samples: 10,000 - 100,000 per β value
- Observable: Average plaquette or Wilson loop

**Deliverable:** Table of Δ_{L_0}(β) ± error for β ∈ [0.15, 2.5]

**Time:** 2-4 weeks (mostly computation)

---

### Task 2.2.2: Cheeger Constant Bound

**Mathematical Statement:**

For gauge-invariant measure μ_β on SU(N)^{|E|}/G (gauge orbits):

```
h = inf_{A: 0 < μ(A) ≤ 1/2} [μ(∂A) / μ(A)]

Δ ≥ h²/2  (Cheeger inequality)
```

**Approach for Yang-Mills:**

1. **Gauge fix** to Coulomb gauge: reduces to SU(N)^{|E|-|V|+1}

2. **Product structure:** After gauge fixing, measure is NOT a product but has correlations only within plaquettes

3. **Recursive bound:** Use that Cheeger constant for weakly coupled systems satisfies:
   ```
   h_total ≥ h_single / (1 + interaction_strength)
   ```

4. **Single-link Cheeger:** For SU(N) with heat kernel measure:
   ```
   h_{SU(N)} ≥ c_N > 0  (known constant)
   ```

**Key Lemma to Prove:**
```
For L_0 = 4 and β ∈ [β_c, β_G]:
h(μ_β) ≥ h_0 / (1 + C β L_0^3)

where h_0 is the Cheeger constant for independent Haar measure
```

**Deliverable:** Rigorous lower bound h ≥ h_min > 0

**Time:** 3-4 weeks

---

### Task 2.3.1: Continuity Argument

**Theorem to Prove:**

```
Δ_{L_0}(β) is continuous (in fact, analytic) in β for β > 0
```

**Proof Sketch:**
1. Transfer matrix T_β depends analytically on β
2. Eigenvalues of analytic family are analytic (Kato)
3. Gap Δ = log(λ_0/λ_1) is analytic

**Application:**
```
Given: Δ_{L_0}(β_i) ≥ 2δ at grid points {β_1, ..., β_n}
Given: |dΔ/dβ| ≤ M
If: max|β_i - β_{i+1}| < δ/M
Then: Δ_{L_0}(β) ≥ δ for all β ∈ [β_1, β_n]
```

**Deliverable:** Proof + explicit M for L_0 = 4

**Time:** 1-2 weeks

---

## Module 3: Correlation Decay

### Task 3.1.1: Reflection Positivity Statement

**Theorem (Osterwalder-Seiler):**

Let Λ ⊂ Z^4 be symmetric under reflection θ through hyperplane {x_4 = 0}.

For any gauge-invariant observable O supported on {x_4 > 0}:
```
⟨O* · θ(O)⟩_β ≥ 0
```

where θ(O)(U) = O(θU) with (θU)_{x,μ} = U_{θx,μ}^{†} for μ ≠ 4.

**Corollary (Spectral Representation):**
```
⟨O(0) O(x)⟩ = ∫_0^∞ e^{-m|x_4|} dρ_O(m)

where dρ_O is a positive measure (spectral measure)
```

**Deliverable:** Clean statement adapted to our setting

**Time:** 1 week

---

### Task 3.2.3: Explicit Decay Rate

**Target Theorem:**

For Wilson loops W(C_1), W(C_2) at distance d:
```
|⟨W(C_1) W(C_2)^*⟩_c| ≤ C exp(-σ(β) · Area(S_min))
```

where S_min is minimal surface spanning C_1 ∪ C_2.

**For point correlators at distance r:**
```
|⟨O(0) O(x)⟩_c| ≤ C exp(-m_0 |x|)

where m_0 ≥ c √σ(β) for glueball operators
```

**Key Input:** String tension lower bound
```
For β ∈ [β_c, β_G]:
σ(β) ≥ σ_min > 0

Strong coupling: σ(β) ≈ -log(β/2N²) for β < β_c
Numerical:      σ(β) ≈ 0.05 - 0.2 for β ∈ [0.15, 2.5]
```

**Deliverable:** Explicit m_0 > 0 with proof

**Time:** 2-3 weeks

---

## Module 4: Bootstrap Synthesis

### Task 4.2.1: Martinelli-Olivieri Theorem

**Full Statement:**

Let (Ω, μ) be a probability space with:
- Ω = ∏_{i ∈ Z^d} Ω_i (product of local spaces)
- μ has finite-range interactions (range R)

**Conditions:**
1. **Block gap:** For any L_0-block B and boundary condition ξ:
   ```
   Gap(μ_{B|ξ}) ≥ δ
   ```

2. **Weak mixing:** For sets A, B with dist(A,B) > L_0:
   ```
   |Cov_μ(f_A, g_B)| ≤ ||f_A||_∞ ||g_B||_∞ · C exp(-m_0 · dist(A,B))
   ```

**Conclusion:**
```
Gap(μ) ≥ δ · (1 - e^{-m_0 L_0}) / (1 + C' δ L_0^d)
```

**Reference:** Martinelli-Olivieri, Comm. Math. Phys. 161 (1994), 447-486

**Time:** 1 week to state and verify conditions

---

### Task 4.2.3: Explicit Bound Computation

**Inputs:**
```
δ = 0.1      (from Module 2, finite-volume gap)
m_0 = 0.1   (from Module 3, correlation decay rate)
L_0 = 4     (block size)
d = 4       (dimension)
```

**Martinelli-Olivieri Formula:**
```
Δ_∞ ≥ δ (1 - e^{-m_0 L_0}) / (1 + C δ L_0^d)
     = 0.1 × (1 - e^{-0.4}) / (1 + C × 0.1 × 256)
     = 0.1 × 0.33 / (1 + 25.6 C)
```

For C = 1: Δ_∞ ≥ 0.033 / 26.6 ≈ 0.0012

**Optimization:** Choose L_0 to maximize bound
```
L_0 = 2: Δ_∞ ≥ 0.1 × 0.18 / (1 + 1.6) ≈ 0.007
L_0 = 3: Δ_∞ ≥ 0.1 × 0.26 / (1 + 8.1) ≈ 0.003
L_0 = 4: Δ_∞ ≥ 0.1 × 0.33 / (1 + 25.6) ≈ 0.001
```

**Conclusion:** Smaller L_0 gives better bound, but Module 2 computation is harder.

**Deliverable:** Optimized Δ_∞ lower bound

**Time:** 1-2 weeks

---

## Module 5: Continuum Limit

### Task 5.1.1: Asymptotic Scaling

**Two-loop β-function for SU(N):**
```
β(g) = -b_0 g³ - b_1 g⁵ + O(g⁷)

b_0 = 11N / (48π²)
b_1 = 34N² / (3(16π²)²)
```

**Lattice spacing vs coupling:**
```
a(β) Λ = exp(-1/(2b_0 g²)) × (b_0 g²)^{-b_1/(2b_0²)} × (1 + O(g²))

For lattice: g² = 2N/β, so:

a(β) Λ = exp(-β/(4Nb_0)) × (β/(2Nb_0))^{-b_1/(2b_0²)} × (1 + O(1/β))
```

**For SU(3) (N=3):**
```
a(β) Λ ≈ (6β/11)^{-51/121} exp(-3β/11) × (1 + O(1/β))
```

**Deliverable:** Explicit a(β) formula with error bounds

**Time:** 1 week

---

### Task 5.2.3: Physical Mass Gap

**Theorem Statement:**

```
m_phys = lim_{β→∞} Δ(β) / a(β)

Given: Δ(β) ≥ Δ_min > 0 for all β (from Module 4)
Given: a(β) ~ exp(-cβ) → 0 as β → ∞
```

**Key Issue:** Does Δ(β)/a(β) have a finite limit?

**Resolution:**
1. Correlation length ξ(β) = 1/Δ(β) in lattice units
2. Physical correlation length: ξ_phys = ξ(β) × a(β)
3. Mass gap: m_phys = 1/ξ_phys = Δ(β)/a(β)

**By asymptotic freedom:**
```
ξ(β) × a(β) → ξ_phys = const/Λ as β → ∞
```

**Therefore:**
```
m_phys = Λ / ξ_phys = c × Λ > 0
```

**Deliverable:** Rigorous proof that m_phys > 0

**Time:** 2-3 weeks

---

## Dependency Graph

```
Module 1 (DONE) ─────────────────────────┐
                                         │
Module 2 ──────────────────┐             │
  Task 2.1.1 → 2.1.2       │             │
  Task 2.2.1, 2.2.2, 2.2.3 │             │
  Task 2.3.1 → 2.3.2       │             │
           ↓               │             │
        δ > 0 ─────────────┼─────────────┤
                           │             │
Module 3 ──────────────────┤             │
  Task 3.1.1 → 3.1.2       │             │
  Task 3.2.1 → 3.2.2 → 3.2.3             │
  Task 3.3.1 → 3.3.2       │             │
           ↓               │             │
        m_0 > 0 ───────────┘             │
                           ↓             ↓
                     Module 4 ←──────────┘
                       Task 4.1.1 → 4.1.2
                       Task 4.2.1 → 4.2.2 → 4.2.3
                       Task 4.3.1, 4.3.2, 4.3.3
                              ↓
                         Δ_∞ > 0
                              ↓
                     Module 5
                       Task 5.1.1 → 5.1.2
                       Task 5.2.1 → 5.2.2 → 5.2.3
                              ↓
                         m_phys > 0
```

---

## Critical Path Timeline

| Week | Module 2 | Module 3 | Module 4 | Module 5 |
|------|----------|----------|----------|----------|
| 1-2  | 2.1.1    | 3.1.1    | -        | -        |
| 3-4  | 2.1.2    | 3.1.2, 3.2.1 | -    | -        |
| 5-6  | 2.2.1    | 3.2.2    | -        | -        |
| 7-8  | 2.2.2    | 3.2.3    | -        | -        |
| 9-10 | 2.2.3, 2.3.1 | 3.3.1 | -       | -        |
| 11-12| 2.3.2    | 3.3.2    | 4.1.1, 4.1.2 | -    |
| 13-14| -        | -        | 4.2.1-4.2.3 | -     |
| 15-16| -        | -        | 4.3.1-4.3.3 | 5.1.1 |
| 17-18| -        | -        | -        | 5.1.2, 5.2.1 |
| 19-20| -        | -        | -        | 5.2.2, 5.2.3 |

**Total: ~20 weeks with parallel execution of Modules 2 and 3**

---

## Success Metrics

### Module 2 Success
- [ ] Code produces transfer matrix
- [ ] Monte Carlo gives Δ_{L_0}(β) > 0.05 for all β ∈ [0.15, 2.5]
- [ ] Cheeger bound confirms δ > 0.01
- [ ] Uniformity proven with explicit M

### Module 3 Success
- [ ] Reflection positivity verified
- [ ] Spectral representation derived
- [ ] m_0 > 0.05 proven for intermediate coupling

### Module 4 Success
- [ ] MO conditions verified
- [ ] Δ_∞ > 0.001 computed
- [ ] All-β coverage complete

### Module 5 Success
- [ ] Asymptotic scaling with error bounds
- [ ] OS axioms verified
- [ ] m_phys = c Λ > 0 proven

---

## Final Checklist for Clay Prize

- [ ] All modules complete with rigorous proofs
- [ ] No unverified numerical claims
- [ ] Explicit constants throughout
- [ ] Self-contained document
- [ ] Independent verification by 3+ experts
- [ ] Addresses all Clay problem requirements
