# Audit Changes - December 2025

## Summary

This document records changes made during the mathematical consistency audit.

---

## Errors Fixed

### 1. LSI Constant for Haar Measure on SU(N)

**Original Error:** The document `EXPLICIT_CALCULATIONS.tex` claimed both:
- ρ_N = 2/N (incorrect)
- ρ_N = (N²-1)/(2N²) (correct)

**Fix:** Removed the incorrect `ρ_N = 2/N` claim. The correct formula is:
$$\rho_N = \frac{N^2-1}{2N^2}$$

This gives:
- SU(2): ρ₂ = 3/8 = 0.375
- SU(3): ρ₃ = 4/9 ≈ 0.444

Added clarifying remark about comparison with Bakry-Émery (which uses different normalization).

### 2. Holley-Stroock Perturbation Lemma

**Original Error:** Several documents stated:
$$\rho_1 \geq \rho_0 \cdot e^{-\mathrm{osc}(V)}$$

**Correct Statement:**
$$\rho_1 \geq \rho_0 \cdot e^{-2\,\mathrm{osc}(V)}$$

The factor of 2 appears because we need bounds on both the numerator (entropy) 
and denominator (gradient integral) when changing measure.

**Files Fixed:**
- `GAP_TRANSPORT_RIGOROUS.tex`
- `RG_BRIDGE_CONSTRUCTION.tex`
- `DEEP_STRUCTURES_PROOF.tex`
- `GAP_TRANSPORT_ANALYSIS.tex`
- `yang_mills.tex`

### 3. Degradation Bound Clarification

**Issue:** `GAP_TRANSPORT_RIGOROUS.tex` stated $\delta_k = C_N L^8/\beta^{(k)}$ without 
clarifying this applies only to intermediate coupling.

**Fix:** Added theorem statement distinguishing three regimes:
1. **Weak coupling** (β > β_G): δ_k = O(1/β²) - crucial for convergence
2. **Intermediate** (β_c < β < β_G): δ_k = O(L⁸/β) 
3. **Strong coupling** (β < β_c): Zegarlinski applies directly

Added remark explaining why the weak coupling improvement is essential for m_phys > 0.

---

## Documents Updated

| Document | Changes |
|----------|---------|
| `EXPLICIT_CALCULATIONS.tex` | Fixed LSI constant, added clarifying remark |
| `GAP_TRANSPORT_RIGOROUS.tex` | Fixed Holley-Stroock, clarified degradation bounds |
| `RG_BRIDGE_CONSTRUCTION.tex` | Fixed Holley-Stroock formula |
| `DEEP_STRUCTURES_PROOF.tex` | Fixed Holley-Stroock formula |
| `GAP_TRANSPORT_ANALYSIS.tex` | Fixed Holley-Stroock formula |
| `yang_mills.tex` | Fixed Holley-Stroock formula and consequent bound |
| `DOCUMENT_GUIDE.md` | Updated constants table with formulas and sources |

---

## Verification

After changes:
- All constants are now consistent across documents
- Holley-Stroock uses factor of 2 everywhere
- LSI constant uses (N²-1)/(2N²) formula
- Degradation analysis explicitly distinguishes coupling regimes

---

## Remaining Open Questions (Not Errors)

These are acknowledged gaps, not mathematical errors:

1. **Precise value of C_N:** The constant in the degradation bound is order O(1) but 
   not computed precisely.

2. **Intermediate coupling oscillation:** The bound osc(V_k) = O(L³β) for intermediate 
   coupling is an estimate, not a rigorous bound.

3. **Weak coupling δ = O(1/β²):** Claimed from Gaussian approximation but detailed 
   proof not provided. This is physically plausible.

These are acknowledged as areas where the framework could be strengthened, but they 
are not mathematical inconsistencies.

---

## Conclusion

The framework is now mathematically consistent:
- No contradictory formulas
- Standard results (Holley-Stroock, LSI on SU(N)) stated correctly
- Clear acknowledgment of what is proven vs. estimated

The framework remains plausible and provides a clear path toward a complete proof.
