# C3: Commuting Hamiltonians and Algebraic Structure

## Paper Idea

**Math Core:** Commuting projector models relate to operator algebras/topological order.

## Paper-Shaped Deliverable

- Classification results in a restricted setting (2D locality + symmetries)
- New invariants for phases that also control computational properties

## Mathematical Tools

- C*-algebras
- von Neumann algebras
- Tensor categories (if you go topological)

## Key References

- Topological order and quantum error correction literature
- Classification of gapped phases

## Proposed Main Theorem Structure

**Theorem (Classification):** For 2D commuting projector Hamiltonians with:
- Translation invariance
- Finite-range interactions
- Unique ground state on torus

the ground state is in the same phase as a stabilizer state tensored with a topological state from list $\mathcal{T}$.

**Theorem (Invariants):** The following quantities are invariants of the gapped phase:
1. Ground state degeneracy on genus-$g$ surface: $\text{GSD}(g) = |Z|^{2g}$ for abelian theory
2. Topological entanglement entropy: $S_{top} = \log |Z|$
3. Anyon content: determined by modular tensor category $\mathcal{C}$

**Corollary (Computational):** Ground states of commuting Hamiltonians in phase $\mathcal{P}$ can be prepared by:
- Stabilizer operations if $\mathcal{P}$ is trivial
- Stabilizer + constant depth circuit if $\mathcal{P}$ is topological

## Research Plan

1. Define equivalence relations on commuting Hamiltonians
2. Develop algebraic invariants
3. Prove classification theorem
4. Connect to computational properties
5. Extend to symmetry-protected phases
