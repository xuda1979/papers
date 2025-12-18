# D2: Fine-Grained Quantum Complexity for Linear Algebra Primitives

## Paper Idea

Instead of "HHL exists," ask: what if the matrix is sparse + well-conditioned + structured?

## Paper-Shaped Deliverable

- A lower bound or no-go theorem in a realistic input model
- A sharper dependence on condition number / precision for a restricted family

## Mathematical Tools

- Information-based complexity
- Reductions
- Condition number analysis

## Key References

- HHL algorithm and variants
- Quantum linear algebra literature

## Proposed Main Theorem Structure

**Theorem (Lower Bound):** For solving $Ax = b$ where $A$ is:
- $s$-sparse
- Condition number $\kappa$
- Given via sparse access oracle

any quantum algorithm requires:
$$\Omega(\kappa \cdot \text{polylog}(n/\epsilon))$$

queries to achieve precision $\epsilon$.

**Theorem (Structured Upper Bound):** For matrices $A$ in class $\mathcal{M}$ (e.g., circulant, Toeplitz + positive definite):
$$T(Ax = b) = O(\sqrt{\kappa} \cdot \text{polylog}(n/\epsilon))$$

**Corollary (Gap):** There exist matrix classes where quantum provides exponential speedup, and classes where speedup is at most polynomial.

## Research Plan

1. Define precise input model and complexity measure
2. Prove information-theoretic lower bounds
3. Identify structure enabling improved algorithms
4. Characterize speedup landscape
5. Connect to practical applications
