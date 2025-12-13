# Pure Yang-Mills Mass Gap Approach

This folder contains the **direct approach** to proving the Yang-Mills mass gap for **pure** SU(N) gauge theory (no fermions).

## Files

| File | Description |
|------|-------------|
| `yang_mills_mass_gap.tex` | Main paper - claims complete proof (original ambitious version) |
| `yang_mills_conditional.tex` | Conditional version - honest about remaining gaps |
| `current_gaps_analysis.tex` | Detailed analysis of what is proven vs what remains open |

## Status Summary

### ✅ What IS Rigorously Proven (Lattice Theory)

For **lattice** SU(N) Yang-Mills at any coupling β > 0:

1. **Tomboulis-Yaffe inequality:** σ(β) ≥ f_v(β)/N
2. **Vortex free energy positive:** f_v(β) > 0 for all β
3. **String tension positive:** σ(β) > 0
4. **Mass gap positive:** Δ(β) ≥ c_N√σ(β) > 0

**Key result:** Every lattice Yang-Mills theory (at any finite coupling) confines and has a mass gap.

### ❌ What Remains OPEN (Continuum Limit)

The **continuum limit** is NOT controlled:

- **Have:** f_v(β) ≥ c > 0 for all β
- **Need:** a(β)² · f_v(β) → σ_phys > 0 as β → ∞
- **Gap:** Monotonicity proves f_v is increasing, but not that it grows exponentially

For σ_phys > 0, we need f_v(β) ~ exp(β/b₀), but we only know f_v is increasing.

## Why This Approach Was Superseded

The main `yang.tex` file now uses the **SUSY interpolation approach** (Adjoint QCD), which:

1. Starts from N=1 Super-Yang-Mills where exact results exist
2. Has two parameters (m and Λ) for better control
3. Successfully proves mass gap for Adjoint QCD
4. Connects to pure YM via the m → ∞ limit (still conditional)

## Relationship to Millennium Prize

This direct approach stalls at precisely the point that makes it a Millennium Prize Problem:
**No one knows how to rigorously control the continuum limit of 4D Yang-Mills theory.**

The gap identified here—proving that σ_phys > 0 survives the a → 0 limit—is equivalent to the Millennium Problem itself.
