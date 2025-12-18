# A2: Homological-Product / Iterated-Product Constructions with Explicit Asymptotics

## Paper Idea

There's a big family of quantum LDPC constructions using **homological products** and related topological/combinatorial gadgets.

## Paper-Shaped Deliverable

- Give a *simpler* construction with explicit parameter bounds
- Analyze distance/rate under additional constraints (e.g., bounded geometry, bounded-degree, implementability constraints)
- Prove new lemmas about when iterating products preserves expansion or improves distance scaling

## Mathematical Tools

- Chain complexes
- Homology theory
- CSS codes
- Expander-like complexes

## Key References

- [Quantum LDPC Codes of Almost Linear Distance via Iterated Homological Products](https://people.eecs.berkeley.edu/~venkatg/pubs/papers/CCC2025-iterated-homological-products.pdf)

## Proposed Main Theorem Structure

**Theorem:** Let $\mathcal{C}_1, \mathcal{C}_2$ be chain complexes with expansion parameters $\lambda_1, \lambda_2$. The iterated homological product $\mathcal{C}_1 \otimes^{(k)} \mathcal{C}_2$ yields a quantum LDPC code with:

- Rate $R \geq R_0 \cdot f(k)$
- Distance $d \geq d_0 \cdot g(k, \lambda_1, \lambda_2)$
- Check weight $w \leq w_0 \cdot h(k)$

where $f, g, h$ are explicit functions we derive.

## Research Plan

1. Study existing homological product constructions
2. Identify simplification opportunities
3. Derive explicit bounds via spectral analysis
4. Prove distance scaling under iteration
5. Compare with existing constructions
