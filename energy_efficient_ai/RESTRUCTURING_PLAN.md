# Paper Restructuring Plan: Option 1 (Focused SSA Paper)

## Review's Main Critique
The paper is "three papers stitched together" - SSA algorithm + thermodynamics + circuit complexity. This dilutes the core contribution and makes it hard for reviewers to assess novelty.

## Changes Implemented So Far

### ✅ Title and Abstract
- Changed title to focus on SSA: "Spectral Sparse Attention: Subquadratic Long-Context Modeling via Cluster-Based Sparsification with Provable Guarantees"
- Rewrote abstract to lead with SSA algorithm, complexity, and results
- Removed mention of thermodynamics and circuit complexity from abstract

### ✅ Introduction (Section 1)
- Restructured to lead with the long-context challenge
- Highlighted SSA as the main contribution early
- Added clear "Key idea" and "Main theoretical result" paragraphs
- Reorganized related work to compare SSA directly with sparse attention, SSMs, and spectral sparsification
- Updated "Overview of Main Results" to separate main content from appendices

### ✅ SSA Algorithm Section (Section 2)
- Created new Section 2 with Algorithm 1, complexity analysis, and theory-implementation mapping
- Added explicit "Theory-Implementation Mapping" subsection addressing review concern #2
- Clarified what objects SSA approximates (W vs P) and how renormalization works

## Remaining Changes Needed

### High Priority (Critical for Review Response)

1. **Move foundational material to appendices**
   - Current Parts I (Axiomatic Foundations), III (Thermodynamics), V (Circuit Complexity), VI (Quantization)
   - These should become Appendices A, B, C, D
   - Keep only forward references in main text

2. **Restructure main document to**:
   ```
   Section 1: Introduction (DONE)
   Section 2: SSA Algorithm (DONE)  
   Section 3: Spectral Theory (reorganize current Part II + Part IV)
      - 3.1: Attention Graph and Laplacian
      - 3.2: Regularity Assumptions (NEW - address review concern #3)
      - 3.3: Main Approximation Theorem
      - 3.4: Mixing Time and Information Propagation
   Section 4: Experiments (current Part VII)
      - Fix needle-in-haystack task (review concern #4)
      - Add discussion of clusterability validation
   Section 5: Discussion
      - Energy scaling (brief, based on current Part V)
      - Limitations and assumptions
      - Future work
   ```

3. **Add Regularity Assumption subsection (Section 3.2)**
   - Explicitly state:
     * Bounded cluster separation
     * Lower bounds on sampling probabilities  
     * Bounded degree ratios
     * Connection to κ parameter
   - This addresses review concern #3 about the theorem being "under-specified"

4. **Fix experimental issues**
   - Needle-in-haystack: ensure local attention cannot succeed by construction
   - Add validation of cluster ability/spectral gap assumption on real attention heads
   - This addresses review concerns #4 and #6

### Medium Priority

5. **Add symmetrization discussion subsection**
   - Empirical check: measure |W - W^T| / |W| in trained models
   - Or test SSA with tied vs untied projections
   - Addresses review concern #6

6. **Tighten BitNet/quantization claims**
   - Either move entirely to appendix with informal framing
   - Or add precise distributional assumptions
   - Addresses review concern #5

### Lower Priority

7. **Proofread for consistency**
   - Update all cross-references after restructuring
   - Ensure theorem numbering is consistent
   - Check that moved content has appropriate forward/backward references

## File Structure After Restructuring

```
Main Body (~25 pages):
- Introduction  
- SSA Algorithm
- Spectral Theory (consolidated)
- Experiments
- Discussion

Appendices (~20-25 pages):
- Appendix A: Axiomatic Foundations
- Appendix B: Thermodynamic/Variational Theory
- Appendix C: Circuit Complexity and Turing Completeness
- Appendix D: Quantization Theory (BitNet)
- Appendix E: Additional Proofs
```

## Next Steps

1. Complete Section 3 restructuring (combine spectral geometry + approximation theory)
2. Add explicit Regularity Assumptions subsection
3. Move foundational parts to appendices
4. Fix experiments
5. Write Discussion section
6. Update all cross-references
7. Compile and verify

## Key Messages for Response to Reviewers

1. "We have substantially restructured the paper to focus on SSA as the primary contribution."
2. "Foundational theory (axioms, thermodynamics, complexity) moved to appendices."
3. "Added explicit theory-implementation mapping (Section 2.3) clarifying how SSA relates to theoretical objects."
4. "Added Regularity Assumption (Section 3.2) making theorem conditions explicit."
5. "Fixed needle-in-haystack task to prevent local attention shortcut."
6. "Added empirical validation of clusterability assumption."
