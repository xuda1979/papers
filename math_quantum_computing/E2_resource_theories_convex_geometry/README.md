# E2: Resource Theories as Convex Geometry

## Paper Idea

**Math Core:** Define/compute monotones, dual norms, or SDP relaxations for resources (entanglement, magic, coherence).

## Paper-Shaped Deliverable

- A new monotone with additivity/subadditivity proof
- New bounds between different monotones (sandwiching)

## Mathematical Tools

- Convex analysis
- Duality
- Semidefinite programming
- Majorization

## Key References

- Resource theory literature
- Entanglement measures and magic state theory

## Proposed Main Theorem Structure

**Theorem (New Monotone):** Define the resource measure:
$$M(\rho) = \min_{\sigma \in \mathcal{F}} D_{\alpha}(\rho \| \sigma)$$

for free states $\mathcal{F}$ and Rényi divergence $D_{\alpha}$. This satisfies:
1. Monotonicity under free operations
2. Additivity: $M(\rho \otimes \sigma) = M(\rho) + M(\sigma)$
3. Computability via SDP for stabilizer/Gaussian free states

**Theorem (Sandwiching):** For magic states:
$$\frac{1}{c_1} \cdot \log \text{RoM}(\rho) \leq M(\rho) \leq c_2 \cdot \log \text{RoM}(\rho)$$

where RoM is robustness of magic and $c_1, c_2$ are universal constants.

**Theorem (Operational):** $M(\rho)$ equals the optimal rate of resource distillation in the asymptotic limit.

## Research Plan

1. Define candidate resource measure
2. Prove monotonicity and convexity properties
3. Establish additivity/subadditivity
4. Derive bounds relating to existing measures
5. Connect to operational tasks
