# C1: Restricted Local Hamiltonian Variants: Sharp Complexity Frontiers

## Paper Idea

**Math Core:** Classify when ground-state energy estimation stays hard/easy under constraints (geometry, stoquasticity, commuting terms).

## Paper-Shaped Deliverable

- A new reduction proving hardness in a more "physical" regime
- Or an algorithmic result for a special family (e.g., planar + commuting + bounded degree)

## Mathematical Tools

- Reductions
- Perturbation gadgets
- Promise problems
- Spectral theory

## Key References

- [Complexity of geometrically local stoquastic Hamiltonians](https://arxiv.org/abs/2407.15499)

## Proposed Main Theorem Structure

**Theorem (Hardness):** The Local Hamiltonian problem remains QMA-complete for:
- 2D square lattice
- 2-local interactions
- Interaction strength ratio $\leq \text{poly}(n)$
- [Additional physical constraint]

**Theorem (Algorithm):** For Hamiltonians satisfying:
- Planar interaction graph
- Commuting terms
- Bounded local dimension $d$

the ground state energy can be approximated to precision $\epsilon$ in time $n^{O(d^2)} \cdot \text{poly}(1/\epsilon)$.

**Corollary:** Sharp boundary: removing any one constraint from the algorithmic case yields QMA-hardness.

## Research Plan

1. Survey existing hardness results and identify gaps
2. Design perturbation gadgets for new constraints
3. Prove completeness of reduction
4. For tractable cases, develop specialized algorithms
5. Map the complete complexity landscape
