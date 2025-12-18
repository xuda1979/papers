# D3: Average-Case Hardness and Random Circuit Sampling (Math Version)

## Paper Idea

If you like probability + complexity: show when a distribution over circuits becomes hard/easy under a structural parameter.

## Paper-Shaped Deliverable

- Hardness conditioned on a complexity conjecture, but with a new clean parameter regime
- Or a new classical algorithm for a restricted ensemble (which is also publishable)

## Mathematical Tools

- Random graphs
- Concentration
- Complexity-theoretic reductions
- Tensor network bounds

## Key References

- Random circuit sampling and quantum supremacy literature
- Complexity theory of sampling problems

## Proposed Main Theorem Structure

**Theorem (Hardness):** Assuming the Polynomial Hierarchy does not collapse, sampling from the output distribution of random circuits with:
- Depth $D = \omega(\sqrt{n})$
- 2D geometry
- Haar-random 2-qubit gates

is classically intractable (no polynomial-time algorithm within total variation distance $< 1/4$).

**Theorem (Tractability Boundary):** For random circuits with depth $D = o(\log n)$:
There exists a classical algorithm sampling within TV distance $\epsilon$ in time $n^{O(D)} \cdot \text{poly}(1/\epsilon)$.

**Theorem (Concentration):** The hardness holds with probability $\geq 1 - n^{-\omega(1)}$ over the choice of random circuit.

## Research Plan

1. Define circuit ensemble and output distribution
2. Establish anti-concentration properties
3. Prove worst-to-average-case reduction
4. Identify tractability boundary via tensor network analysis
5. Characterize the phase transition
