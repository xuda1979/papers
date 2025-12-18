# Per-Token Reward Shaping from Policy Entropy

## Research Focus
Using **policy entropy as a proxy for cognitive effort** to enable fine-grained per-token credit assignment "for free".

## Key Idea

**Dynamic Entropy Weighting** treats high policy entropy as a heuristic for:
- "Cognitive effort" - harder decisions
- "Pivotal tokens" - tokens that matter for correctness
- "Uncertainty" - model is less confident

This enables **per-token credit assignment** without training a separate reward model.

## Background: GTPO and GRPO-S

Recent work proposes:
- **GTPO (Group Token Policy Optimization)**: Token-level advantage from group comparison
- **GRPO-S**: Sequence-level reward shaping based on entropy

### Entropy-Based Weighting

For token t at position i:
$$w_i = f(H_i) = f\left(-\sum_v \pi(v|x, y_{<i}) \log \pi(v|x, y_{<i})\right)$$

Higher entropy → higher weight (more "important" token)

### Motivation
- High entropy = model uncertain = decision is non-trivial
- Low entropy = obvious next token = less important for learning

## Research Questions

### 1. Is Entropy a Good Proxy for Importance?

**Formal definition of "pivotal tokens"**:
Tokens with high **causal effect** on final correctness.

**Counterfactual measure**:
$$\text{Pivotality}(t_i) = |P(\text{correct} | y) - P(\text{correct} | y_{\neg t_i})|$$

where $y_{\neg t_i}$ replaces token $t_i$ with alternatives.

**Key Question**: Does entropy correlate with pivotality?

### 2. Testing the Correlation

**Experimental Protocol**:
1. For each token in correct/incorrect completions
2. Compute entropy at that position
3. Compute counterfactual effect (replace token, re-verify)
4. Measure correlation

**Hypothesis**: High entropy tokens have higher counterfactual effect

### 3. Alternative Importance Proxies

If entropy is imperfect, what else can we use?

| Proxy | Formula | Intuition |
|-------|---------|-----------|
| Entropy | H(π(·\|context)) | Model uncertainty |
| Fisher Information | E[∇log π · ∇log π^T] | Sensitivity of likelihood |
| Gradient Norm | \|\|∇_θ log π(t\|context)\|\| | How much token affects parameters |
| Disagreement | Var_δ[π_{θ+δ}(t\|context)] | Sensitivity to perturbation |
| Attention | Σ_head attention_weight | What model "looks at" |

## Proposed Method: Principled Token Weighting (PTW)

### Framework

For each token, compute importance weight $w_i$ and use in loss:
$$L = -\sum_i w_i \cdot A \cdot \log \pi_\theta(t_i | x, y_{<i})$$

where A is the trajectory-level advantage (from GRPO).

### Importance Estimators

**Option 1: Entropy-Based (Baseline)**
$$w_i^{\text{ent}} = \frac{H_i - H_{\min}}{H_{\max} - H_{\min}}$$

**Option 2: Fisher Information**
$$w_i^{\text{fisher}} = \text{tr}(F_i) = \mathbb{E}\left[\|\nabla_\theta \log \pi_\theta(t_i|...)\|^2\right]$$

**Option 3: Gradient Norm**
$$w_i^{\text{grad}} = \|\nabla_\theta \log \pi_\theta(t_i | x, y_{<i})\|$$

**Option 4: Perturbation Disagreement**
$$w_i^{\text{perturb}} = \text{Var}_{\delta \sim \mathcal{N}(0, \sigma^2 I)}[\pi_{\theta+\delta}(t_i | ...)]$$

### Theoretical Justification

**Theorem (Optimal Weighting)**
Under linear function approximation, the optimal per-token weight for minimizing gradient variance is:
$$w_i^* \propto \sqrt{\text{Var}[\nabla_\theta \log \pi_\theta(t_i | ...)]}$$

which is related to Fisher information.

**Corollary**: Gradient norm weighting approximates optimal weighting.

## Experimental Plan

### Phase 1: Correlation Study
- Generate diverse completions (correct + incorrect)
- Compute entropy at each token position
- Compute counterfactual importance
- Analyze correlation

### Phase 2: Compare Importance Proxies
- Implement all weighting schemes
- Train with same budget
- Compare learning curves and final performance

### Phase 3: Theoretical Validation
- Verify variance reduction from optimal weighting
- Characterize when entropy is good proxy

### Metrics

| Metric | Description |
|--------|-------------|
| Entropy-Pivotality Correlation | ρ(H_i, Pivotality_i) |
| Gradient Variance | Var[∇L] under different weightings |
| Learning Speed | Steps to reach 80% of final performance |
| Final Pass@1 | End-of-training accuracy |
| Token Efficiency | Useful gradient signal per token |

## Theoretical Analysis

### Questions to Answer

1. **When is entropy a good proxy?**
   - Hypothesis: When model is well-calibrated
   
2. **Optimal weighting derivation**
   - Minimize variance of policy gradient
   - Connection to importance sampling

3. **Bias-variance tradeoff**
   - Uniform weighting: unbiased, high variance
   - Entropy weighting: possibly biased, lower variance?

### Key Lemmas

**Lemma 1 (Entropy and Gradient Magnitude)**
Under log-linear policy:
$$\|\nabla_\theta \log \pi_\theta(t|...)\| \approx O(\sqrt{H_t})$$

High entropy → larger gradient magnitude.

**Lemma 2 (Fisher Information Bound)**
$$\text{tr}(F_t) \leq |\mathcal{V}| \cdot H_t$$

Fisher information bounded by entropy (times vocabulary size).

## Implementation Details

### Efficient Computation

**Entropy**: Already computed for sampling (free)

**Gradient Norm**: Extra backward pass per token (expensive)
- Approximation: Use Fisher diagonal

**Perturbation**: Sample perturbations (moderate cost)
- Approximation: Use dropout as perturbation

### Memory Considerations
- Store per-token weights alongside tokens
- Can compute on-the-fly during backward pass

## Key References

1. [GTPO and GRPO-S: Token and Sequence-Level Reward Shaping](https://arxiv.org/abs/2508.04349)
2. [Fisher Information in Deep Learning](https://arxiv.org/abs/1711.04225)
3. [Variance Reduction in Policy Gradient](https://arxiv.org/abs/1506.02438)
4. [GRPO](https://arxiv.org/abs/2402.03300)

## Expected Contributions

1. **Empirical**: First systematic study of entropy-importance correlation in RLVR
2. **Theoretical**: Optimal token weighting for variance reduction
3. **Methodological**: Principled alternatives to entropy weighting
4. **Practical**: Guidelines for when to use token-level credit

## Paper Structure

1. Introduction: Credit assignment in RLVR
2. Background: Entropy-based weighting
3. Is Entropy a Good Proxy?: Correlation study
4. Principled Token Weighting: Theory
5. Alternative Importance Measures
6. Experiments: Comparison of weighting schemes
7. Analysis: When does token weighting help?
8. Conclusion

## Why This is Elegant

- **Theory meets practice**: Principled understanding of heuristics
- **Low compute overhead**: Entropy is essentially free
- **Novel contribution**: Systematic study not done before
- **Clear experiments**: Correlation + comparison
- **Actionable insights**: When to use which weighting
