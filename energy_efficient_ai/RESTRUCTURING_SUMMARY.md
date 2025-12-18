# Paper Restructuring Progress Summary

## Completed Changes ✅

### 1. Title and Abstract
- ✅ Changed title to focus on SSA: "Spectral Sparse Attention: Subquadratic Long-Context Modeling via Cluster-Based Sparsification with Provable Guarantees"
- ✅ Rewrote abstract to emphasize SSA algorithm, O(N^{3/2}) complexity, and concrete results
- ✅ Removed broad "framework" language; removed thermodynamics/complexity from abstract

### 2. Introduction (Section 1)
- ✅ Restructured to lead with long-context challenge and SSA as solution
- ✅ Added explicit "What SSA does" description with three-step algorithm outline
- ✅ Highlighted key idea (spectral graph + clusterability assumption)
- ✅ Stated main result clearly (eigenspace preservation, O(N^{3/2}) regime)
- ✅ Reorganized related work into focused subsections (sparse attention, SSMs, spectral sparsification)
- ✅ Updated "Overview of Main Results" to separate main contributions from appendix material

### 3. SSA Algorithm Section (Section 2) - NEW
- ✅ Created comprehensive Section 2 with:
  * Notation subsection
  * Algorithm 1 pseudocode 
  * Complexity analysis (Proposition on edge budget)
  * Two-stage sampling discussion
  * **Theory-Implementation Mapping subsection** (addresses Review Concern #2)
    - Explicitly maps algorithm objects (intra-cluster softmax, sampling, renormalization) to theoretical objects (W, P, Laplacian)
    - Clarifies that SSA approximates W then normalizes to get P
    - Explains role of symmetrization and reversibility assumption
    - Lists key assumptions needed for mapping

### 4. Spectral Theory Section (Section 3) - RESTRUCTURED
- ✅ Merged Part II (Spectral Geometry) and Part IV (Sparsification Theory) into unified Section 3
- ✅ Changed from "Part" to "Section" hierarchy
- ✅ Created subsection structure:
  * 3.1: Attention Graph and Laplacian (from old Part II)
  * 3.2: **Regularity Assumptions** (NEW - addresses Review Concern #3)
  * 3.3: Information Propagation and Mixing Time  
  * 3.4: Main Approximation Theorem

### 5. Regularity Assumptions (Section 3.2) - NEW
- ✅ Added explicit Assumption with four conditions:
  1. Cluster separation (ε_cluster bound)
  2. Bounded degree ratios (κ_D condition number)
  3. Sampling distortion (κ factor for two-stage sampler)
  4. Spectral gap (δ_r > 0)
- ✅ Added Remark explaining why each assumption matters
- ✅ Connected κ in Assumption to κ in Theorem statement
- This directly addresses Review Concern #3: "Theorem looks plausible but under-specified"

### 6. Experiments Section (Section 4)
- ✅ Changed from "Part VII" to "Section 4"
- ✅ Simplified header to "Experimental Validation"
- ✅ Added upfront statement about NumPy prototype nature

### 7. Discussion Section (Section 5) - NEW
- ✅ Created Section 5 "Discussion" combining:
  * Energy scaling intuition (brief, from old Part V material)
  * Limitations subsection (moved from old "Conclusion" part)
- ✅ De-emphasized energy claims (now "scaling intuition" not "main result")
- ✅ Mentioned quantization synergy with forward reference to appendix

## Remaining Critical Changes 🔄

### 8. Move Material to Appendices
Still need to convert these Parts to Appendices:
- [ ] Part II sections not yet moved (Riemannian structure, Cheeger inequality, etc.) → Appendix A or B
- [ ] **Part III (Thermodynamic Theory)** → Appendix A: "Variational and Thermodynamic Foundations"
- [ ] **Part V (Circuit Complexity)** → Appendix B: "Computational Complexity Theory"
- [ ] **Part VI (Quantization/BitNet)** → Appendix C: "Ternary Quantization and BitNet Theory"
- [ ] Other detailed proofs → Appendix D or E

### 9. Fix Experimental Issues (HIGH PRIORITY)
From Review Concern #4 and #6:
- [ ] **Fix needle-in-haystack task**: Ensure local attention cannot access needle by construction
  - Review says local baseline scores 0.985-0.987 at all N, which shouldn't be possible if needle is outside local window
  - Need to verify task setup: query token should NOT carry the value
- [ ] **Add clusterability validation**: Empirically measure spectral gap or cluster quality on real attention heads
- [ ] Consider adding at least one real benchmark (even small-scale LRA task)

### 10. Symmetrization Discussion
From Review Concern #6:
- [ ] Add empirical check: measure |W - W^T| / |W| in trained models
- [ ] OR: test SSA with tied (W_Q = W_K) vs untied projections
- [ ] Show sensitivity of spectral diagnostics to asymmetry
- Could go in Section 3.2 (Regularity) or Section 4 (Experiments)

### 11. Clean Up Cross-References
- [ ] Update all theorem/section references after restructuring
- [ ] Ensure forward references to appendices are correct
- [ ] Check that moved content has appropriate back-references to main text

### 12. BitNet/Quantization Claims
From Review Concern #5:
- [ ] Move BitNet section to Appendix C
- [ ] Either: tone down to "informal intuition" 
- [ ] Or: add precise distributional assumptions for ε_QAT = O(ε_PTQ²) claim
- [ ] Remove "field" language (ternary arithmetic is NOT a field)

## Document Structure (Target)

```
Main Body (~20-25 pages):
  Section 1: Introduction ✅
  Section 2: SSA Algorithm ✅
  Section 3: Spectral Theory ✅ (mostly done)
  Section 4: Experimental Validation ✅ (needs fixes)
  Section 5: Discussion ✅

Appendices (~25-30 pages):
  Appendix A: Axiomatic Foundations ⏳ (exists but needs header update)
  Appendix B: Variational/Thermodynamic Theory 🔄 (needs move from Part III)
  Appendix C: Circuit Complexity 🔄 (needs move from Part V)
  Appendix D: Quantization Theory (BitNet) 🔄 (needs move from Part VI)
  Appendix E: Additional Proofs ⏳ (exists)
  Appendix F: Notation Glossary ⏳ (exists)
```

## Next Immediate Steps

1. **Move Part III to Appendix** (thermodynamic theory)
2. **Move Part V to Appendix** (circuit complexity)
3. **Move Part VI to Appendix** (BitNet quantization)
4. **Fix needle-in-haystack task** in experiments
5. **Add clusterability validation** (spectral gap measurement on real data)
6. **Compile and check** for broken references
7. **Test symmetrization assumption** empirically

## Key Messages for Reviewers

When submitting revision, emphasize:
1. ✅ "Substantially restructured to focus on SSA (main body now ~25 pages vs 50)"
2. ✅ "Added explicit theory-implementation mapping (Section 2.3)"
3. ✅ "Made regularity assumptions explicit (Section 3.2)"
4. 🔄 "Moved foundational theory to appendices (thermodynamics, complexity, quantization)"
5. 🔄 "Fixed experimental task to prevent local attention shortcut"
6. 🔄 "Added empirical validation of clusterability assumption"
7. 🔄 "Tested sensitivity to symmetrization assumption"

Legend: ✅ Done | 🔄 In Progress | ⏳ Pending | ❌ Not Started
