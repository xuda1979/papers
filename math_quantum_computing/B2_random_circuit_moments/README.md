# B2: Exact or Symbolic Moment Calculations for Random/Local Circuits

## Paper Idea

There is active work computing **exact moments** of local random circuits with tensor methods.

## Paper-Shaped Deliverable

- Closed-form expressions for low-order moments for a new architecture (1D/2D/brickwork with boundaries)
- Prove anti-concentration or design-like behavior under conditions you can verify

## Mathematical Tools

- Representation theory
- Weingarten calculus
- Symmetric groups
- Random matrix methods

## Key References

- [Computing exact moments of local random quantum circuits via tensor network diagrams](https://link.springer.com/article/10.1007/s42484-024-00187-8)

## Proposed Main Theorem Structure

**Theorem:** For a 2D brickwork circuit of depth $D$ on an $L \times L$ lattice with open boundary conditions, the $k$-th moment of the output distribution satisfies:

$$\mathbb{E}[|\langle x | U | 0^n \rangle|^{2k}] = \frac{(k!)^2}{(2^n)^k} \cdot \left(1 + \sum_{j=1}^{k-1} c_j(L, D) \cdot 2^{-\alpha_j D}\right)$$

where $c_j(L, D)$ are explicit combinatorial coefficients.

**Corollary (Anti-concentration):** For $D \geq D_c(L) = O(L \log L)$, the circuit forms an $\epsilon$-approximate 2-design with $\epsilon = 2^{-\Omega(D/L)}$.

## Research Plan

1. Set up tensor network representation of moments
2. Apply Weingarten calculus for Haar integration
3. Compute boundary effects explicitly
4. Derive closed-form expressions
5. Prove design convergence rate
