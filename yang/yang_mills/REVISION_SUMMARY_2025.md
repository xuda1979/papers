# Revision Summary: Response to Referee Report (December 2025)

This document summarizes all changes made to `yang_mills.tex` in response to the referee's detailed review.

## 1. LaTeX Compilation Errors (FIXED)

| Error | Location | Fix |
|-------|----------|-----|
| `\end{corollary>` typo | Line 31824 | Changed to `\end{corollary}` |
| `-latex'` curly quote | Line 32052 | Changed to `-latex` |
| Unicode ✓ (U+2713) | Lines 32938-32946 | Changed to `$\checkmark$` |

**Result:** Document now compiles successfully (457 pages).

## 2. Internal Consistency Issues (FIXED)

### Problem
The manuscript contained contradictory claims: some sections claimed "rigorous proof" while others admitted "framework requiring verification."

### Resolution
- **Theorem `thm:ym-lsi`** (Yang-Mills Spectral Gap): Renamed to "What Is Rigorous" with explicit CAVEAT that infinite-volume statement is conditional
- **Step 3** of the gap extension: Changed from claimed "proof" to labeled "CONDITIONAL---THE CORE OPEN PROBLEM"
- **Corollary `cor:uniform-gap-lsi`**: Changed to **Conjecture `conj:uniform-gap`**
- Added **Remark** on what would complete the proof (4 possible approaches)
- Updated **Introduction** status assessment to correctly separate rigorous from conditional

## 3. String Tension Derivation (FIXED)

### Problem
The paper claimed σ(β) > 0 for all β using the Bessel bound σ ≥ -log(I₁/I₀), but this:
1. Is only valid at strong coupling where cluster expansion converges
2. Gives σ ~ 1/(2β) at large β, which goes to zero (inconsistent with claimed bounds)

### Resolution
- Added prominent **WARNING BOX** at the start of Section 7 (String Tension via GKS)
- Made clear distinction:
  - **RIGOROUS:** σ_L(β) > 0 for finite L, all β
  - **RIGOROUS:** σ(β) > 0 for β < β₀ (strong coupling) in infinite volume
  - **CONDITIONAL:** σ(β) > 0 for all β requires same uniform bounds as main problem
- Fixed **Corollary `cor:quantitative-sigma`** to clarify this is finite-volume only
- Added explanation that σ_lat → 0 at large β is consistent with confinement (σ_phys = σ_lat/a² can remain positive)

## 4. Scaling/Continuum-Limit Errors (VERIFIED CORRECT)

### Original Concern
Referee cited apparent error: "ξ_lattice → 0 while ξ_physical remains finite"

### Verification
The document actually states (correctly):
- ξ_lattice → ∞ as β → ∞ (because ξ_lat = ξ_phys/a and a → 0)
- ξ_physical remains finite
- Scaling clarification remark already present

**No change needed** - the document was correct.

## 5. Δ(β) > 0 Argument (FIXED)

### Problem
The "extension to all β via continuity" argument claimed:
> "Δ(β*) = 0 implies second-order phase transition... excluded by center symmetry preservation"

This is a major logical leap.

### Resolution
- Replaced claimed proof with explicit **Conjecture `conj:uniform-gap`**
- Listed 3 specific reasons why the argument is incomplete:
  1. "Δ = 0 ⇒ phase transition" is not a theorem in this generality
  2. Center symmetry doesn't exclude correlation length divergence for all observables
  3. Analyticity proven only at strong coupling
- Added **Remark** listing 4 approaches that would complete the proof

## 6. RG Bridge Section (FIXED)

### Problem
Section claimed to be "rigorous" but made unproven assumptions:
- That σ_phys can be evaluated at any β
- That the continuum limit is unique
- That there are no phase transitions along RG flow

### Resolution
- Added large **WARNING BOX** at start of Section 5.10 stating it's a framework, not a proof
- Changed title from "Rigorous RG Bridge" to "RG Bridge Framework (CONDITIONAL)"
- Changed the green "Why This Proof is Rigorous" box to orange "CRITICAL GAP: Why This Proof Is NOT Complete"
- Changed **Corollary `cor:final-mass-gap`** to state "IF the RG bridge assumptions are valid, THEN..."

## 7. Toy-Model Formulas (VERIFIED CORRECT)

### Original Concern
Formula for Δ using Bessel ratios looks like 1D reduction, not 4D.

### Verification
**Theorem `thm:constructive-final`** already contains:
- Explicit label "Strong Coupling"
- Critical caveat: "This formula is NOT the exact 4D mass gap"
- Note that weak-coupling asymptotics are inconsistent with dimensional transmutation
- Clear statement that extension requires RG analysis

**No change needed** - caveats already present.

## 8. Conclusion Section (FIXED)

### Problem
Second conclusion section (around line 12434) claimed continuum results as theorems.

### Resolution
- Complete rewrite of Summary of Results
- Each result now labeled **RIGOROUS** or **CONDITIONAL**
- Explicit statement: "This requires verifying the uniform bounds---the core open problem"
- Changed "Key Mathematical Innovations" to "Key Mathematical Contributions"

---

## Summary of What Is Now Claimed

### Rigorously Proven
1. **Finite-volume spectral gap:** Δ_L(β) > 0 for all β > 0, finite L (Perron-Frobenius)
2. **Strong-coupling infinite-volume gap:** Δ(β) > 0 for β < β₀ (cluster expansion)
3. **Finite-volume string tension:** σ_L(β) > 0 for all β > 0, finite L
4. **Strong-coupling analyticity:** Free energy analytic for β < β₀
5. **Center symmetry:** ⟨P⟩ = 0 (Polyakov loop vanishes)
6. **Finite-volume Giles-Teper:** Δ_L ≥ c_N √σ_L

### Conditional/Framework
1. **All-β infinite-volume gap:** Δ(β) > 0 for all β requires proving no critical point
2. **Continuum mass gap:** Δ_phys > 0 requires uniform-in-L bounds at weak coupling
3. **RG bridge:** Evaluating σ_phys at strong coupling requires uniqueness of continuum limit

---

## Remaining Work for Millennium Prize

The paper is now honest that a complete proof requires:

1. **Uniform Log-Sobolev inequality** at all β, OR
2. **RG flow control** preserving spectral gap from weak to strong coupling, OR  
3. **Bootstrap argument** showing Δ(β) > 0 ⇒ Δ(β + ε) > 0, OR
4. **Direct spectral estimate** λ₁(L, β) ≤ 1 - c uniformly in L

None of these are completed in the manuscript.
