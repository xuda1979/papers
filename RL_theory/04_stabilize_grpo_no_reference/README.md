# Stabilizing GRPO Without a Reference Model

## Research Focus
Removing the need for a **reference model** and KL regularization tuning in GRPO through stability-focused alternatives.

## Problem Statement

A key compute cost in KL-regularized RL:
- Maintaining a **frozen reference model** (memory overhead)
- Tuning the KL coefficient **β** (hyperparameter sensitivity)
- Computing KL divergence (forward pass overhead)

**Goal**: Achieve stable GRPO training without any reference model dependency.

## Background: GTPO

**GTPO (Group-relative Trajectory-based Policy Optimization)** identifies instability sources in GRPO:

1. **Token-level penalization**: Negative gradients on shared tokens across success/failure
2. **Policy collapse**: Mode collapse to narrow set of solutions
3. **High-entropy noise**: Completions with high entropy are unreliable signals

### GTPO Solutions
- Skip negative updates (only amplify positives)
- Filter high-entropy completions
- **No KL regularization required**

## Research Directions

### 1. Provable Entropy Thresholds

GTPO filters completions with entropy above a threshold. Can we derive this theoretically?

**Questions**:
- What entropy level indicates "unreliable" signal?
- How does optimal threshold depend on:
  - Model size?
  - Task difficulty?
  - Training progress?

**Proposed Analysis**:
$$H_{\text{threshold}} = f(\text{Var}[r], G, |\mathcal{V}|)$$

where G is group size and |V| is vocabulary size.

### 2. Token Conflict Detection

Identify "valuable shared tokens" that appear in both successful and failed completions.

**Formalization**: Multi-sample credit assignment problem

For token t at position i:
$$\text{Conflict}(t, i) = \mathbf{1}[t \in y^+ \cap t \in y^-]$$

where y+ and y- are positive/negative completions.

**Resolution Strategies**:
| Strategy | Description |
|----------|-------------|
| Skip | Don't update conflicting tokens |
| Weight | Weight by P(success|token) |
| Causal | Use counterfactual analysis |
| Aggregate | Majority vote across samples |

### 3. Alternative Regularization

Replace KL with reference-free alternatives:

| Method | Formula | Properties |
|--------|---------|------------|
| Entropy floor | H(π) ≥ H_min | Prevents collapse |
| Entropy ceiling | H(π) ≤ H_max | Focuses learning |
| Trust region | ||θ - θ_old|| ≤ δ | Parameter space constraint |
| Gradient penalty | ||∇θ|| ≤ γ | Smooth updates |

## Proposed Algorithm: Stable GRPO (S-GRPO)

```
Algorithm: S-GRPO
Input: Prompts D, initial policy π_θ, group size G

for each prompt x ∈ D:
    # Sample group
    {y_i}_{i=1}^G ~ π_θ(·|x)
    
    # Compute rewards
    r_i = Verifier(x, y_i)
    
    # Filter high-entropy completions
    H_i = -Σ_t π_θ(t|x, y_{<t}) log π_θ(t|x, y_{<t})
    Keep only {y_i : H_i < H_threshold}
    
    # Detect token conflicts
    Shared = tokens appearing in both r=1 and r=0 completions
    
    # Compute masked advantage
    A_i = (r_i - mean(r)) / std(r)
    A_i[Shared] *= conflict_weight  # reduce gradient on conflicts
    
    # Update (positive-only option)
    if positive_only:
        L = -Σ_{i: r_i > 0} A_i log π_θ(y_i|x)
    else:
        L = -Σ_i A_i log π_θ(y_i|x)
    
    # Entropy regularization (reference-free)
    L += λ_H * max(0, H_min - H(π_θ))
    
    θ ← θ - η∇L
```

## Theoretical Analysis

### Questions to Address

1. **When is positive-only sufficient?**
   - Conditions under which skipping negatives doesn't hurt convergence
   
2. **Entropy floor guarantees**
   - Can entropy floor replace KL in preventing collapse?
   - What H_min is sufficient?

3. **Token conflict resolution**
   - Optimal weighting scheme
   - Connection to credit assignment literature

### Proposed Theorems

**Theorem 1 (Positive-Only Convergence)**
Under binary rewards with P(success) = p > 0, positive-only GRPO converges to optimal policy if:
$$\eta \leq \frac{p}{1-p} \cdot \frac{1}{L}$$
where L is the smoothness constant.

**Theorem 2 (Entropy Floor Sufficiency)**
Entropy floor H_min prevents collapse to deterministic policy if:
$$H_{\min} \geq \log k^*$$
where k* is the number of distinct solutions.

## Experimental Comparison

### Methods to Compare

1. **GRPO** (standard with KL regularization)
2. **GTPO** (skip negatives + entropy filter)
3. **S-GRPO** (our proposed stabilization)
4. **GRPO-EF** (entropy floor only)
5. **GRPO-TC** (token conflict handling only)

### Stability Metrics

| Metric | Description |
|--------|-------------|
| Gradient Norm | ||∇L|| over training |
| Policy Entropy | H(π) trajectory |
| Collapse Rate | Frequency of mode collapse |
| Success Stability | Variance of success rate |
| KL from Init | Divergence from initialization |

### Datasets
- GSM8K (math, binary reward)
- MATH (harder math)
- HumanEval (coding)

## Compute Considerations

### Memory Savings
- No reference model: **50% memory reduction** for same model size
- Enables larger batch sizes

### Throughput
- No KL computation: faster per-step
- Can fit 14B on 20× 910B without reference

## Key References

1. [GTPO: Stabilizing Group Relative Policy Optimization](https://arxiv.org/abs/2508.03772)
2. [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
3. [GRPO-S: Token and Sequence-Level Reward Shaping](https://arxiv.org/abs/2508.04349)

## Expected Contributions

1. **Theoretical**: Conditions for reference-free stable RL
2. **Algorithmic**: S-GRPO with provable stability
3. **Practical**: Memory-efficient training recipe
4. **Empirical**: Comprehensive stability comparison

## Paper Outline

1. Introduction: The cost of reference models
2. Background: GRPO and its instabilities
3. Analysis: Sources of instability (formal treatment)
4. S-GRPO: Our proposed stabilization
5. Theory: Convergence and stability guarantees
6. Experiments: Comparison with GTPO and standard GRPO
7. Ablations: Each component's contribution
8. Conclusion: Guidelines for reference-free RLVR

## Why This Matters

- **Directly reduces compute**: No reference model overhead
- **Simplifies training**: Fewer hyperparameters (no β)
- **Enables larger models**: Memory savings
- **Theoretically grounded**: Not just heuristics
