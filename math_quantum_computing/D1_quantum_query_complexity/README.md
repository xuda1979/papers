# D1: Quantum Query Complexity Lower Bounds for Graph Properties

## Paper Idea

**Math Core:** Prove lower bounds using adversary / polynomial methods.

## Paper-Shaped Deliverable

- New tight bounds for a specific property (connectivity variants, subgraph containment under promises, etc.)
- Separations between classical randomized and quantum query complexity for a natural problem variant

## Mathematical Tools

- Semidefinite programs (adversary method)
- Approximation theory
- Fourier analysis on Boolean cube

## Key References

- Quantum query complexity literature
- Adversary method papers

## Proposed Main Theorem Structure

**Theorem (Lower Bound):** For the graph property $P$ (e.g., detecting a $k$-clique in a graph with promise of ≤1 clique):

$$Q(P) = \Omega(n^{\alpha_k})$$

where $\alpha_k$ is an explicit function of $k$.

**Theorem (Separation):** There exists a graph property $P$ such that:
$$R(P) = \Theta(n^2) \quad \text{but} \quad Q(P) = O(n^{3/2})$$

providing a polynomial separation.

**Theorem (Tight Bound):** For $k$-vertex connectivity:
$$Q(\text{$k$-connected}) = \Theta(n^{1 + (k-1)/2k})$$

## Research Plan

1. Identify candidate graph properties
2. Formulate adversary SDP
3. Find optimal adversary matrix
4. Prove lower bound
5. Match with algorithm (if possible) or identify gap
