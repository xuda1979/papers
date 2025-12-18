# A3: Decoding Guarantees for a Code Family (Provable, Not Just Empirical)

## Paper Idea

Instead of "we ran BP and it worked," aim for: "Under noise model X and graph property Y, decoding succeeds with probability ≥ 1−ε."

## Paper-Shaped Deliverable

- Prove a threshold / finite-length guarantee for a restricted noise model (erasure channel is a common "first rigorous" target)
- Analyze degeneracy effects (distinctive to quantum LDPC)

## Mathematical Tools

- Probabilistic combinatorics
- Percolation-style arguments
- Belief propagation analysis
- Union-find analysis
- Peeling processes

## Key References

- [Degenerate Quantum LDPC Codes With Good Finite Length Performance](https://quantum-journal.org/papers/q-2021-11-22-585/)

## Proposed Main Theorem Structure

**Theorem:** For the code family $\mathcal{F}$ with Tanner graph satisfying expansion property $\lambda < \lambda_c$, under the quantum erasure channel with erasure rate $p < p_c(\lambda)$:

1. The peeling decoder succeeds with probability $\geq 1 - n^{-\Omega(1)}$
2. The threshold $p_c(\lambda)$ satisfies $p_c(\lambda) = \frac{1}{2} - O(\sqrt{\lambda})$

**Corollary:** Degeneracy improves the effective threshold by factor $(1 + \delta)$ where $\delta$ depends on the code's degeneracy structure.

## Research Plan

1. Define precise noise model and decoder
2. Establish graph-theoretic conditions for success
3. Apply probabilistic analysis to bound failure events
4. Characterize degeneracy contribution
5. Compute explicit threshold estimates
