# E3: Nonlocal Games and Operator-Algebraic Relaxations

## Paper Idea

**Math Core:** Bound quantum correlations via SDPs or operator algebra tools.

## Paper-Shaped Deliverable

- New Tsirelson-type bounds for a structured game family
- New hierarchy convergence guarantees in a special case

## Mathematical Tools

- SDP hierarchies
- Operator spaces
- Representation theory

## Key References

- Tsirelson bounds and nonlocal games
- NPA hierarchy and operator algebraic methods

## Proposed Main Theorem Structure

**Theorem (Tsirelson Bound):** For the game family $\mathcal{G}_k$ (e.g., generalized CHSH with $k$ inputs):
$$\omega^*(\mathcal{G}_k) = \cos^2(\pi/(4k))$$

with explicit optimal strategy construction.

**Theorem (Hierarchy Convergence):** For games with:
- $n$ inputs per player
- $m$ outputs per player
- Symmetry group $G$

the NPA hierarchy converges at level $\ell = O(nm/|G|)$.

**Theorem (Separation):** There exists a game $G$ with:
$$\omega^{ns}(G) > \omega^*(G) > \omega^{qa}(G) > \omega(G)$$

where $\omega, \omega^{qa}, \omega^*, \omega^{ns}$ are classical, quantum approximate, quantum, and non-signaling values respectively.

## Research Plan

1. Identify structured game families
2. Set up SDP relaxations
3. Analyze convergence using symmetry
4. Prove tight bounds
5. Construct explicit strategies achieving bounds
