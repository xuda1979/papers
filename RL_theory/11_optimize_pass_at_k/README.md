# Optimizing Pass@K Directly

## Research Focus
Deriving explicit **Pass@K policy gradients** to optimize for the reasoning boundary (solve within K attempts) rather than single-sample performance.

## Motivation

**Current RLVR** optimizes Pass@1 (single-sample success):
$$J(\pi) = \mathbb{E}_{x \sim D, y \sim \pi} [r(x, y)]$$

**But often we care about Pass@K**:
$$J_K(\pi) = \mathbb{E}_{x \sim D} \left[1 - (1 - P_\pi(\text{success}|x))^K\right]$$

The "reasoning boundary" problem: RLVR can improve Pass@1 while hurting Pass@K (mode collapse).

## Key Insight

Recent work frames **Pass@K policy gradients** and shows that common RL algorithms can be viewed as **advantage shaping** variants targeting specific Pass@K objectives.

**Goal**: Make this connection explicit and derive optimal algorithms.

## Background: Pass@K Objective

### Definition
Given K samples from policy π:
$$\text{Pass@K} = P(\text{at least one sample correct})$$

### Exact Formula
$$\text{Pass@K} = 1 - (1 - p)^K$$

where $p = P(\text{single sample correct})$.

### Gradient
$$\nabla_\theta \text{Pass@K} = K(1-p)^{K-1} \nabla_\theta p$$

**Key observation**: Gradient magnitude depends on K and p!

## Theoretical Framework

### Advantage Shaping View

Standard policy gradient:
$$\nabla J = \mathbb{E}[A \cdot \nabla \log \pi]$$

**Advantage shaping**: Replace A with transformed Ã

| Shaping | Formula | Targets |
|---------|---------|---------|
| Identity | Ã = A | Pass@1 |
| Weighted | Ã = w(p) · A | Pass@K for specific K |
| Adaptive | Ã = f(p, K) · A | Pareto-optimal |

### Pass@K Surrogate Reward

**Theorem (Surrogate Reward)**
Optimizing Pass@K is equivalent to optimizing:
$$J_{\text{surrogate}} = \mathbb{E}[w_K(p) \cdot r(x,y)]$$

where:
$$w_K(p) = \frac{K(1-p)^{K-1}}{1 - (1-p)^K}$$

**Intuition**: Weight rewards by their marginal contribution to Pass@K.

### Optimal Advantage Shaping

**Theorem (Optimal Shaping for Pass@K)**
The advantage shaping that minimizes gradient variance while optimizing Pass@K is:
$$\tilde{A}_i = \alpha_K(p) \cdot (r_i - \bar{r})$$

where:
$$\alpha_K(p) = K \cdot (1-p)^{K-1}$$

**Proof sketch**:
1. Pass@K gradient is $K(1-p)^{K-1} \nabla p$
2. Success probability gradient is $\nabla p = \mathbb{E}[r \nabla \log \pi]$
3. Combining gives the α_K weighting

## Proposed Algorithm: Pass@K-GRPO

### Standard GRPO
```
Advantage: A_i = (r_i - mean(r)) / std(r)
Loss: -Σ A_i log π(y_i|x)
```

### Pass@K-GRPO
```
# Estimate current success rate
p_hat = mean(r_1, ..., r_G)

# Compute K-dependent weight
alpha = K * (1 - p_hat)^(K-1)

# Shaped advantage
A_i = alpha * (r_i - mean(r)) / std(r)

# Loss
Loss = -Σ A_i log π(y_i|x)
```

**Key difference**: Advantage is weighted by α_K, which depends on current success rate and target K.

## Research Directions

### 1. Unbiased Gradient Estimators

**Challenge**: p_hat is biased when estimated from finite samples.

**Goal**: Derive low-variance, unbiased estimators for ∇Pass@K.

**Approaches**:
- Jackknife estimation
- Control variates
- Rao-Blackwellization

### 2. Adaptive K Selection

**Observation**: Optimal K changes during training.
- Early: p is low, target small K
- Late: p is high, target large K

**Proposed schedule**:
$$K_t = K_{\min} + (K_{\max} - K_{\min}) \cdot \frac{p_t}{p_{\text{target}}}$$

### 3. Multi-Objective Optimization

**Goal**: Optimize multiple Pass@K simultaneously.

**Pareto frontier**: Find policies that are Pareto-optimal across K values.

**Scalarization**:
$$J = \sum_{k \in \{1, 4, 16, 64\}} w_k \cdot \text{Pass@k}$$

### 4. Theoretical Guarantees

**Questions to answer**:
1. Under what conditions does Pass@K-GRPO converge?
2. What's the sample complexity as a function of K?
3. Can we prove monotonic improvement?

## Experimental Design

### Phase 1: Validate Surrogate Reward Theory

**Experiment**: Train with different advantage shapings:
1. Standard GRPO (no shaping)
2. Pass@K-GRPO (K=1, 4, 16, 64)
3. Verify that each targets its respective K

**Metrics**: Pass@1, Pass@4, Pass@16, Pass@64

**Hypothesis**: Pass@K-GRPO(K=k) achieves best Pass@k.

### Phase 2: Pareto Frontier Analysis

**Experiment**: Train models with different K targets.

**Plot**: Pass@1 vs Pass@64 for each model.

**Questions**:
- Is there a tradeoff?
- Can multi-objective training achieve better Pareto frontier?

### Phase 3: Adaptive K Schedule

**Experiment**: Compare fixed K vs adaptive K.

**Baselines**:
- Fixed K=1 (standard RLVR)
- Fixed K=16
- Fixed K=64
- Adaptive schedule

**Hypothesis**: Adaptive achieves best of both worlds.

### Phase 4: Small Model Validation

**Models**: 4B, 7B on GSM8K and MATH

**K values**: 1, 8, 32, 128

**Show**: Pass@K curves for all methods.

## Theoretical Contributions

### Main Theorems to Prove

**Theorem 1 (Convergence)**
Pass@K-GRPO with learning rate $\eta \leq \eta_{\max}(K)$ converges to a local optimum of $J_K$.

**Theorem 2 (Sample Complexity)**
To achieve $\epsilon$-optimal Pass@K:
$$N_{\text{samples}} = O\left(\frac{K^2 \cdot \text{Var}[r]}{(1-p)^{2K} \epsilon^2}\right)$$

**Theorem 3 (Variance Reduction)**
Optimal advantage shaping reduces gradient variance by factor:
$$\frac{\text{Var}[\tilde{A}]}{\text{Var}[A]} = O\left(\frac{1}{K \cdot (1-p)^{K-1}}\right)$$

### Key Lemmas

**Lemma 1 (Pass@K Gradient Formula)**
$$\nabla_\theta \text{Pass@K} = \mathbb{E}_{x,y \sim \pi} \left[K(1-p_x)^{K-1} \cdot r(x,y) \cdot \nabla_\theta \log \pi(y|x)\right]$$

**Lemma 2 (Optimal Shaping)**
Among all shapings $\tilde{A} = f(p) \cdot A$ that preserve gradient direction, the variance-minimizing choice is $f(p) = K(1-p)^{K-1}$.

## Implementation Details

### Computing p_hat
```python
# Option 1: Group-based (GRPO-style)
p_hat = rewards.mean()  # in current group

# Option 2: Moving average
p_hat = 0.9 * p_hat_old + 0.1 * rewards.mean()

# Option 3: Global estimate
p_hat = success_count / total_samples
```

### Numerical Stability
For large K and p near 1:
$$(1-p)^{K-1}$$ can underflow.

**Solution**: Work in log-space:
```python
log_alpha = log(K) + (K-1) * log1p(-p)
alpha = exp(log_alpha)
```

## Key References

1. [Advantage Shaping as Surrogate Reward Maximization](https://arxiv.org/abs/2510.23049v1)
2. [RLVR Success Amplification](https://arxiv.org/abs/2503.06639)
3. [Limit of RLVR](https://limit-of-rlvr.github.io/)
4. [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)

## Expected Contributions

1. **Theoretical Framework**: Rigorous Pass@K policy gradient derivation
2. **Pass@K-GRPO**: Practical algorithm with provable properties
3. **Empirical Validation**: Pass@K curves on math/coding benchmarks
4. **Practical Guidelines**: Which K to target for different scenarios

## Paper Outline

1. Introduction: The Pass@K objective
2. Background: Policy gradients and advantage shaping
3. Theory: Pass@K gradients and optimal shaping
4. Pass@K-GRPO: Practical algorithm
5. Theoretical Analysis: Convergence and sample complexity
6. Experiments: Validation on GSM8K/MATH/HumanEval
7. Discussion: When to optimize for which K
8. Conclusion

## Why This is a Great Math-Heavy Project

- **Pure theory possible**: Derive results without massive experiments
- **Small models sufficient**: 7B on 20× 910B can test up to K=128
- **Novel contribution**: Direct Pass@K optimization not fully explored
- **Practical value**: Addresses the "reasoning boundary shrinkage" problem
- **Clean evaluation**: Pass@K curves are straightforward to measure
- **Connects to existing work**: Builds on advantage shaping literature

## Practical Impact

Understanding Pass@K optimization helps with:
- **Test-time compute**: How many samples to use at inference?
- **Training objectives**: What to optimize for during RLVR?
- **Exploration-exploitation**: Large K encourages diversity
- **Compute budgets**: Optimize performance per sample or per wall-clock time?
