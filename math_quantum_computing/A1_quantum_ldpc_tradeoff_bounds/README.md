# A1: New Tradeoff Bounds for Quantum LDPC Parameters

## Paper Idea

**Math Core:** Prove inequalities linking code parameters (rate, distance) to structural constraints (check weight, geometry, expansion, locality).

## Paper-Shaped Deliverable

- A new impossibility bound under a *specific* constraint (e.g., bounded check weight + geometric locality + bounded degree interaction graph)
- Or a sharper bound for a known regime (even improving constants can be publishable if clean)

## Mathematical Tools

- Expander graphs
- Tanner graphs
- Spectral methods
- Isoperimetry
- Probabilistic method

## Key References

- [Quantum Low-Density Parity-Check Codes | PRX Quantum](https://link.aps.org/doi/10.1103/PRXQuantum.2.040101)

## Proposed Main Theorem Structure

**Theorem:** For any quantum LDPC code with check weight ≤ w, interaction graph of degree ≤ Δ, and geometric locality constraint L, the distance d and rate R satisfy:

$$d \cdot R^{\alpha} \leq f(w, \Delta, L) \cdot n^{\beta}$$

for explicit constants α, β depending on the constraint regime.

## Research Plan

1. Survey existing bounds (BPT bound, Bravyi-Terhal, etc.)
2. Identify gap in constraints not yet addressed
3. Develop proof technique combining expansion and locality
4. Prove main theorem
5. Provide examples showing bound is tight or near-tight
