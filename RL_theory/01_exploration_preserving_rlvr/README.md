# Exploration-Preserving RLVR

## Research Focus
Attacking the phenomenon that **RLVR improves Pass@1 but can shrink the reasoning boundary**.

## Problem Statement

The *Limit of RLVR* paper highlights a critical issue:
- RLVR improves small-budget performance (e.g., Pass@1)
- But **base models can surpass RL-trained models at large Pass@k**
- RLVR often **biases toward rewarded paths but reduces exploration/coverage**

This means RLVR may be "mode-collapsing" to a narrow set of successful strategies.

## Proposed Solutions

### 1. Support-Preserving Constraints

Enforce a lower bound on policy probability for difficult problems:

$$\pi_{\text{new}}(y|x) \geq \alpha \cdot \pi_{\text{base}}(y|x)$$

for some subset of tokens/trajectories, preventing "hard problems" from losing all probability mass.

**Key Questions:**
- How to identify the subset requiring protection?
- Optimal choice of α?
- Computational overhead of constraint enforcement?

### 2. Reverse-KL or Mixture Regularization

- **Forward KL** to reference: encourages mode-covering
- **Reverse KL** to reference: encourages mode-seeking
- **Controlled combination**: maintain coverage while sharpening success

$$L = L_{\text{RLVR}} + \beta_F \cdot D_{KL}(\pi_{\text{ref}} || \pi) + \beta_R \cdot D_{KL}(\pi || \pi_{\text{ref}})$$

### 3. Diversity-Regularized GRPO

Add a repulsion term inside the group to avoid collapse:

$$L_{\text{diversity}} = -\lambda \sum_{i \neq j} D_{KL}(\pi(\cdot|y_i) || \pi(\cdot|y_j))$$

Or alternatively:
- Entropy floor constraints
- N-gram novelty bonuses
- Pairwise cosine dissimilarity in embedding space

## Evaluation Strategy (Compute-Feasible)

### Reproduce "Pass@k Curve Crossing"
- Use 7B model on math/coding benchmarks
- Sample k up to 128 or 256
- Compare base model vs RLVR-trained model

### Metrics to Report
| Metric | Description |
|--------|-------------|
| Pass@1 | Single-sample success rate |
| Pass@k | Best-of-k success rate (k=8,32,64,128) |
| Pass@k Boundary | The k where base model starts winning |
| Unique Solution Rate | Fraction of distinct correct solutions |
| Entropy | Policy entropy over completions |
| N-gram Novelty | New n-grams vs training set |

### Datasets
- GSM8K (math)
- MATH (harder math)
- HumanEval / MBPP (coding)

## Theoretical Analysis

### Questions to Address
1. Under what conditions does RLVR provably reduce coverage?
2. Can we prove support-preserving constraints maintain Pass@k?
3. What is the optimal regularization strength β as a function of k?

### Proposed Theorems
- **Theorem 1**: Under binary rewards, GRPO with group size G reduces support entropy by O(1/G)
- **Theorem 2**: Support-preserving constraints with α guarantee Pass@k ≥ α^k · Pass@k_base
- **Theorem 3**: Mixture KL regularization achieves Pareto-optimal Pass@1 vs Pass@k tradeoff

## Implementation Plan

### Phase 1: Reproduction (Week 1-2)
- Implement Pass@k evaluation pipeline
- Reproduce base vs RLVR comparison
- Confirm curve crossing phenomenon

### Phase 2: Constraint Methods (Week 3-4)
- Implement support-preserving constraint
- Test different α values
- Measure computational overhead

### Phase 3: Regularization Methods (Week 5-6)
- Implement mixture KL regularization
- Implement diversity-regularized GRPO
- Ablation studies

### Phase 4: Theory + Paper (Week 7-8)
- Formalize theoretical guarantees
- Complete experiments
- Write paper

## Key References

1. [Limit of RLVR](https://limit-of-rlvr.github.io/)
2. [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
3. [RLVR Success Amplification](https://arxiv.org/abs/2503.06639)

## Why This is High-Impact

- The limitation is **well-documented but not solved**
- Directly addresses a known failure mode of RLVR
- **Does not require massive models** (7B sufficient)
- High relevance to reasoning/coding community
- Clear evaluation metrics (Pass@k curves)

## Expected Contributions

1. **Empirical**: Reproduce and characterize Pass@k boundary shrinkage
2. **Methodological**: New regularization techniques preserving exploration
3. **Theoretical**: Formal analysis of exploration-exploitation in RLVR
4. **Practical**: Guidelines for when to use base model vs RLVR model
