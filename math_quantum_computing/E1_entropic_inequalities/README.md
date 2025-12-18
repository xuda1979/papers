# E1: New Entropic Inequalities or Equality Characterizations

## Paper Idea

**Math Core:** Characterize when inequalities are tight (strong subadditivity, recoverability bounds, etc.).

## Paper-Shaped Deliverable

- A new inequality for a restricted family (e.g., Gaussian states, stabilizer states) that becomes sharp and computable
- A new equivalence between equality cases and algebraic structure

## Mathematical Tools

- Operator convexity
- Matrix inequalities
- Functional calculus

## Key References

- Quantum entropy and information theory literature
- Strong subadditivity and extensions

## Proposed Main Theorem Structure

**Theorem (New Inequality):** For tripartite stabilizer states $\rho_{ABC}$:
$$I(A:C|B) \geq f(\text{rank structure})$$

where $f$ is an explicit function of the stabilizer group structure.

**Theorem (Equality Characterization):** For the approximate recoverability bound:
$$S(A|B) \approx -I(A:R)_{\Phi}$$

equality holds if and only if $\rho_{AB}$ has the Markov structure $A - B - R$.

**Theorem (Computability):** For Gaussian states, all relevant entropic quantities can be computed in polynomial time in the number of modes, and the following bounds are tight:
$$S(A) + S(B) - S(AB) \leq g(\text{covariance matrix})$$

## Research Plan

1. Study known entropic inequalities and equality cases
2. Identify restricted state classes with more structure
3. Derive tighter inequalities using algebraic constraints
4. Characterize equality cases algebraically
5. Develop efficient computation methods
