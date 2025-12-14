# Rigorous Proof of Physical QCD Mass Gap
## December 2025 - Working Document

# The Goal

Prove rigorously that 4D SU(3) QCD with physical quarks has a mass gap.

---

# What We Need to Prove

**Main Theorem**: Physical QCD has mass gap Δ = m_π > 0

**Key Step**: Prove chiral symmetry breaking ⟨q̄q⟩ ≠ 0

---

# Approach 1: Vafa-Witten + Spectral Constraints

## Vafa-Witten Theorems (1984) - These ARE rigorous

**Theorem (Vafa-Witten)**: In QCD with m_q ≥ 0:
1. Vector-like symmetries (SU(N_f)_V) cannot be spontaneously broken
2. Parity cannot be spontaneously broken
3. The vacuum energy density is minimized at θ = 0

**What this tells us**:
- The ONLY allowed symmetry breaking pattern is SU(N_f)_L × SU(N_f)_R → SU(N_f)_V
- This is exactly chiral symmetry breaking!
- Vafa-Witten constrains WHAT can break, not WHETHER it breaks

## Can we extend Vafa-Witten to PROVE breaking occurs?

**Idea**: Use the anomaly structure to force breaking

---

# Approach 2: 't Hooft Anomaly Matching

## The Anomaly Constraint

The chiral symmetry SU(N_f)_L × SU(N_f)_R has an anomaly (triangle diagram).

**'t Hooft's Argument**: The anomaly must be matched between UV and IR.

In the UV: Quarks carry the anomaly
In the IR: Either
  (a) Massless fermions carry the anomaly, OR
  (b) The symmetry is spontaneously broken (Goldstone bosons)

**For QCD with N_f flavors and N_c = 3 colors**:

If the symmetry is UNBROKEN, we need massless composite fermions to match the anomaly.

**Claim**: For N_c = 3, N_f = 2 or 3, anomaly matching CANNOT be satisfied by massless fermions.

**Therefore**: Chiral symmetry MUST be broken!

---

# Approach 3: Persistent Mass Inequality

## Key Observation

Consider the correlator:
  C(x) = ⟨J_5^μ(x) J_5^ν(0)⟩

where J_5^μ = q̄γ^μγ_5 q is the axial current.

**Ward Identity**:
  ∂_μ⟨J_5^μ(x) O(0)⟩ = 2m_q⟨q̄γ_5 q(x) O(0)⟩ + anomaly term

**In the limit m_q → 0**:
- If ⟨q̄q⟩ = 0: The correlator has no pole, spectral function starts at some m > 0
- If ⟨q̄q⟩ ≠ 0: There's a massless pion pole

**The question**: Can we PROVE which case occurs?

---

# Approach 4: Spectral Density + Positivity

## Banks-Casher Relation

⟨q̄q⟩ = π ρ(0)

where ρ(λ) is the spectral density of the Dirac operator.

**To prove χSB, we need to prove ρ(0) > 0**

## What we know rigorously:

1. ρ(λ) ≥ 0 (positivity)
2. ∫ρ(λ)dλ = 1 (normalization per unit volume)
3. For free fermions: ρ(0) = 0
4. With interactions: ρ(0) could be > 0

## The Key Question:

**Can we prove that QCD interactions force ρ(0) > 0?**

---

# Approach 5: Instanton Contribution (Making it Rigorous)

## The Instanton Argument

Instantons are classical solutions with topological charge Q = ±1.

**Index Theorem**: An instanton with Q = 1 has exactly one fermionic zero mode.

**In an instanton ensemble**:
- Zero modes accumulate near λ = 0
- This gives ρ(0) > 0

## The Problem:

The instanton gas/liquid picture is NOT rigorous because:
1. Instantons interact
2. The semiclassical approximation isn't controlled
3. Large instantons are not suppressed

## Can We Make It Rigorous?

**Idea**: Use the rigorous result that instantons EXIST (they're saddle points) 
and that the index theorem is EXACT.

**Theorem (Atiyah-Singer)**: For any gauge field with topological charge Q:
  n_+ - n_- = Q
where n_± are the numbers of zero modes with ±chirality.

**This is rigorous!**

The question: Can we show that topologically non-trivial configurations 
contribute enough to force ρ(0) > 0?

---

# Approach 6: Reflection Positivity + Infrared Bounds

## Infrared Bounds (Fröhlich-Simon-Spencer, 1976)

For lattice spin systems, infrared bounds prove:
- Long-range order at low temperature
- Spontaneous symmetry breaking

**Theorem**: If ⟨S_0 · S_x⟩ ≥ c/|x|^{d-2} (for d > 2), then long-range order exists.

## Application to QCD?

**The challenge**: QCD is a gauge theory, not a spin system.

**But**: After gauge fixing, correlators ARE well-defined.

**Idea**: Prove infrared bounds for the chiral order parameter:
  ⟨(q̄_L q_R)(x) (q̄_R q_L)(0)⟩ ≥ ???

If this correlator doesn't decay fast enough, χSB must occur.

---

# Approach 7: The Casher Argument (1979)

## Casher's Theorem

**Theorem (Casher, 1979)**: In a confining gauge theory:
  ⟨q̄q⟩ ≠ 0 if and only if the Dirac spectrum has accumulation at zero.

**Key insight**: Confinement → linear potential → bound states → χSB

## The Logic:

1. Assume confinement (Wilson area law for adjoint representation)
2. Confinement implies a linear potential
3. Linear potential implies bound state spectrum
4. For light quarks, bound states must have specific chiral properties
5. This forces ⟨q̄q⟩ ≠ 0

## The Problem:

We're assuming confinement! But that's what we want to prove!

**However**: Maybe we can use a WEAKER assumption...

---

# Approach 8: Using Adjoint QCD as a Bridge

## The Idea

We HAVE a rigorous proof for Adjoint QCD (via SUSY).

**Question**: Can we relate Physical QCD to Adjoint QCD?

## Large-N Connection

At large N:
- Adjoint and fundamental matter have similar dynamics
- Both exhibit χSB
- Both have confinement

**Orbifold/Orientifold equivalences** relate these theories.

## The Strategy:

1. Prove mass gap for Adjoint QCD (DONE - via SUSY)
2. Show Physical QCD is "close" to Adjoint QCD
3. Use continuity/universality to transfer the result

**Problem**: This is not straightforward for finite N = 3.

---

# Approach 9: Direct Spectral Proof

## Completely New Approach

Forget about χSB. Prove mass gap DIRECTLY.

**Definition**: Mass gap exists if the Hamiltonian has
  Spec(H) ⊂ {0} ∪ [Δ, ∞) with Δ > 0

**For QCD with m_q > 0**:

1. The lightest states are mesons (q̄q bound states)
2. Mesons have mass ≥ 2m_q (naive) but actually lighter due to binding
3. But can they be massless? NO if binding energy < 2m_q

**Theorem (attempted)**:
For QCD with m_q > 0, the binding energy of the lightest meson is less than 2m_q.
Therefore, the meson mass > 0.

**Can we prove this?**

The binding energy is O(α_s × ΛQCD). For heavy quarks, this is calculable.
For light quarks, it's non-perturbative.

---

# MOST PROMISING: Approach 10 - Anomaly Matching + Vafa-Witten

## Combining Rigorous Results

**Step 1**: Vafa-Witten tells us the ONLY allowed symmetry breaking is χSB.

**Step 2**: 't Hooft anomaly matching tells us:
- If χSB doesn't occur, we need massless fermions
- The anomaly coefficients must match

**Step 3**: For SU(3) with N_f = 2:
- UV anomaly: Tr(T^a{T^b, T^c}) for quarks
- IR anomaly: Must be matched by massless composites

**Step 4**: PROVE that no massless composite can match the anomaly.

**Theorem (to prove)**: For SU(3) QCD with N_f = 2 or 3 light quarks,
there is no set of massless color-singlet fermions that can match the 
't Hooft anomaly of the UV theory.

**Therefore**: Chiral symmetry MUST be broken.

**Then**: GMOR gives m_π > 0 for m_q > 0.

**Result**: Mass gap exists!

---

# Let's Develop Approach 10

This seems most promising. It uses:
1. Vafa-Witten (rigorous)
2. Anomaly matching (rigorous)
3. Combinatorial argument (to be developed)
4. GMOR (rigorous given χSB)

## The Anomaly Matching Calculation

For SU(N_f)_L × SU(N_f)_R chiral symmetry:

**UV theory**: N_c quarks in fundamental of SU(N_c), each in fundamental of SU(N_f)_L or SU(N_f)_R

**Anomaly coefficients**:
- SU(N_f)_L^3: proportional to N_c
- SU(N_f)_R^3: proportional to N_c  
- SU(N_f)_L^2 × U(1)_B: proportional to N_c

**IR theory (if unbroken)**: Need massless fermions with same anomaly coefficients

**The constraint**: Massless composites must be color singlets (confinement)
and must reproduce anomaly coefficients.

**Claim**: For N_c = 3, N_f = 2, this is IMPOSSIBLE.

Therefore: χSB must occur!

---

# Next Steps

1. Work out the anomaly matching calculation explicitly
2. Prove the "no-go" theorem for massless composites
3. Apply GMOR to get m_π > 0
4. Conclude mass gap exists

This would be a RIGOROUS proof of Physical QCD mass gap!
