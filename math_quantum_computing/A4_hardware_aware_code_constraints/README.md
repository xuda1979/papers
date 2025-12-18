# A4: "Hardware-Aware" Code Constraints as a Math Problem

## Paper Idea

Some newer work studies LDPC families compatible with limited long-range connectivity (e.g., neutral-atom layouts).

## Paper-Shaped Deliverable

- Formalize a graph/layout constraint (interaction radius, forbidden minors, bounded treewidth patches)
- Prove limits on achievable distance/rate under that constraint
- Or prove a constructive embedding result ("any code in family F can be laid out on architecture G with overhead h(n)")

## Mathematical Tools

- Graph minors
- Embeddings
- Geometric group theory / lattice geometry
- Combinatorial topology

## Key References

- [High-rate quantum LDPC codes for long-range-connected neutral atom registers](https://www.nature.com/articles/s41467-025-56255-5)

## Proposed Main Theorem Structure

**Theorem (Impossibility):** For any quantum LDPC code embeddable in a $d$-dimensional lattice with interaction radius $r$, the parameters satisfy:

$$\frac{d \cdot k}{n} \leq C(d) \cdot r^{\alpha(d)}$$

**Theorem (Constructive):** Any code from family $\mathcal{F}$ with parameters $[[n, k, d]]$ can be embedded in architecture $\mathcal{A}$ with:
- Physical qubit overhead: $O(n \cdot \text{polylog}(n))$
- Gate depth overhead: $O(\text{polylog}(n))$

## Research Plan

1. Define formal hardware constraint models
2. Derive impossibility bounds via graph-theoretic arguments
3. Design embedding algorithms for specific code families
4. Analyze overhead complexity
5. Optimize for practical architectures (neutral atoms, superconducting, ion traps)
