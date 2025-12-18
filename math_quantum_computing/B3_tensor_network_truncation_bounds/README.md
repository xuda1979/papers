# B3: Rigorous Error Bounds for Tensor-Network Truncation

## Paper Idea

**Math Core:** When you approximate a contraction by truncating bond dimensions, what is the provable error?

## Paper-Shaped Deliverable

- New a priori error bounds in terms of entanglement measures or singular-value decay assumptions
- Worst-case lower bounds showing when truncation must fail

## Mathematical Tools

- Matrix analysis
- Operator norms
- Schatten norms
- Concentration inequalities
- Entanglement entropy bounds

## Key References

- [Tensor networks for quantum computing - Nature](https://www.nature.com/articles/s42254-025-00853-1)

## Proposed Main Theorem Structure

**Theorem (Upper Bound):** Let $|\psi\rangle$ be a state with entanglement spectrum $\{\lambda_i\}$ across cut $A|B$. Truncating to bond dimension $\chi$ yields approximation $|\tilde{\psi}\rangle$ with:

$$\| |\psi\rangle - |\tilde{\psi}\rangle \|_2 \leq \sqrt{\sum_{i > \chi} \lambda_i^2}$$

**Theorem (Structured Decay):** If the entanglement spectrum satisfies $\lambda_i \leq C \cdot i^{-\alpha}$ for $\alpha > 1/2$, then:

$$\| |\psi\rangle - |\tilde{\psi}\rangle \|_2 \leq \frac{C}{\sqrt{2\alpha - 1}} \cdot \chi^{-\alpha + 1/2}$$

**Theorem (Lower Bound):** There exist states with entanglement entropy $S$ such that any $\chi$-truncation has error:

$$\| |\psi\rangle - |\tilde{\psi}\rangle \|_2 \geq e^{-O(S)} \cdot \chi^{-1/2}$$

## Research Plan

1. Formalize truncation operations precisely
2. Derive error bounds from singular value properties
3. Connect to entanglement measures
4. Prove lower bounds via explicit constructions
5. Apply to specific circuit families
