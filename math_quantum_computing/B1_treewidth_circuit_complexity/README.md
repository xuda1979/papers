# B1: Complexity Bounds via Treewidth / Pathwidth of Circuit Graphs

## Paper Idea

**Math Core:** Map a circuit family to a graph invariant; prove simulation cost bounds (upper/lower).

## Paper-Shaped Deliverable

- Identify a nontrivial circuit class where contraction is provably efficient (or provably hard)
- Provide new inequalities between circuit depth/connectivity and graph width measures

## Mathematical Tools

- Treewidth
- Branchwidth
- Graph separators
- Minors
- Parameterized complexity

## Key References

- [Tensor networks for quantum computing - Nature](https://www.nature.com/articles/s42254-025-00853-1)

## Proposed Main Theorem Structure

**Theorem (Upper Bound):** For any quantum circuit $C$ on $n$ qubits with circuit graph $G_C$, the output amplitude can be computed in time:

$$T(C) = 2^{O(\text{tw}(G_C))} \cdot \text{poly}(n)$$

**Theorem (Lower Bound):** Under the Exponential Time Hypothesis (ETH), for circuit family $\mathcal{C}_n$ with $\text{tw}(G_{C_n}) = \Theta(n^\alpha)$:

$$T(C_n) = 2^{\Omega(n^\alpha / \log n)}$$

**Corollary:** Circuit depth $D$ and connectivity $\kappa$ satisfy:
$$\text{tw}(G_C) \leq D \cdot \kappa + O(\log n)$$

## Research Plan

1. Formalize circuit-to-graph mapping
2. Relate circuit properties to graph width measures
3. Prove upper bounds via dynamic programming on tree decomposition
4. Prove lower bounds via reductions from known hard problems
5. Identify circuit families at the boundary
