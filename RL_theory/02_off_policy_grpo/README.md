# Off-Policy GRPO: Sample Reuse for Compute Efficiency

## Research Focus
Extending GRPO to **off-policy** settings to dramatically reduce rollout costs through sample reuse.

## Motivation

With limited compute, **sample efficiency is the whole game**:
- Rollout generation is the bottleneck on Ascend clusters
- On-policy methods discard samples after one update
- Off-policy methods can reuse samples multiple times

## Background

Recent work shows GRPO can be adapted to off-policy training:
- Off-policy GRPO performs **on par or better** than on-policy
- Reduces serving/communication burden
- Maintains GRPO's key advantage: **no critic network**

## Proposed Methods

### 1. Replay-Buffer GRPO

Maintain a buffer of experience tuples:
```
Buffer = {(prompt, completion, reward, logprob_behavior)}
```

Do multiple epochs of updates with importance weighting:

$$L(\theta) = \mathbb{E}_{(x,y,r,\mu) \sim \text{Buffer}} \left[ \min\left(\rho \cdot A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) \cdot A\right) \right]$$

where $\rho = \frac{\pi_\theta(y|x)}{\mu(y|x)}$ is the importance ratio.

### 2. Importance Weighting Schemes

| Scheme | Formula | Properties |
|--------|---------|------------|
| Standard IS | ρ = π/μ | Unbiased, high variance |
| Clipped IS | clip(ρ, 1-ε, 1+ε) | Biased, low variance |
| Self-normalized IS | ρ / Σρ | Lower variance |
| Retrace-style | λ-weighted returns | Bias-variance tradeoff |

### 3. Trust Region for Off-Policy GRPO

Constrain policy updates to stay close to behavior policy:

$$D_{KL}(\pi_\theta || \mu) \leq \delta$$

or use adaptive clipping based on divergence.

## Theoretical Analysis

### Key Questions

1. **Policy Improvement Guarantee**: Under what conditions does off-policy GRPO improve the policy?

2. **Variance Analysis**: How does importance sampling variance grow with off-policyness?

3. **Optimal Staleness**: How far off-policy can you go before variance kills learning?

### Proposed Theorems

**Theorem 1 (Policy Improvement Bound)**
Under binary verifiable rewards and bounded KL divergence $D_{KL}(\pi || \mu) \leq \delta$:

$$J(\pi_{\text{new}}) \geq J(\pi_{\text{old}}) - O(\sqrt{\delta})$$

**Theorem 2 (Optimal Clipping)**
The optimal clipping threshold ε* that minimizes MSE of the gradient estimator is:

$$\epsilon^* = \Theta\left(\sqrt{\frac{\text{Var}[r]}{\text{Var}[\rho]}}\right)$$

**Theorem 3 (Sample Complexity)**
Off-policy GRPO with replay buffer size B achieves sample complexity:

$$O\left(\frac{1}{B \cdot (1 - \delta)}\right)$$

where δ bounds the KL divergence.

## Practical Considerations for Ascend Clusters

### Memory Management
- Buffer size vs GPU memory tradeoff
- Priority-based experience replay
- Compression of stored completions

### Distributed Training
- Async experience collection
- Centralized vs distributed buffers
- Staleness in multi-worker settings

### Optimal Replay Ratio
- How many gradient steps per new sample?
- Adaptive replay based on policy divergence

## Experimental Plan

### Phase 1: Baseline Implementation
- Implement on-policy GRPO baseline
- Implement replay buffer infrastructure
- Test on GSM8K with 4B model

### Phase 2: Off-Policy Methods
- Standard importance-weighted GRPO
- Clipped importance weighting
- Trust-region constrained GRPO

### Phase 3: Efficiency Analysis
- Sample efficiency curves (performance vs samples)
- Compute efficiency (performance vs FLOPs)
- Wall-clock time comparisons

### Phase 4: Theoretical Validation
- Verify bounds hold empirically
- Characterize variance growth
- Identify optimal operating points

## Metrics

| Metric | Description |
|--------|-------------|
| Sample Efficiency | Pass@1 per unique sample |
| Replay Ratio | Gradient steps per new sample |
| Effective Sample Size | 1/Σρ² (for IS) |
| Policy Divergence | KL(π_current || μ_buffer) |
| Gradient Variance | Variance of policy gradient |

## Key References

1. [Revisiting GRPO: On-Policy and Off-Policy Training](https://arxiv.org/html/2505.22257v1)
2. [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
3. [Off-Policy Actor-Critic](https://arxiv.org/abs/1801.01290)

## Expected Contributions

1. **Theoretical**: First rigorous analysis of off-policy GRPO under verifiable rewards
2. **Algorithmic**: Optimal clipping and trust-region schemes for RLVR
3. **Practical**: Guidelines for replay buffer management on Ascend clusters
4. **Empirical**: Comprehensive efficiency comparisons (on-policy vs off-policy)

## Why This Matters

- **Direct compute savings**: 2-10x sample reuse
- **Infrastructure friendly**: Reduces inference serving load
- **Theoretically clean**: Binary rewards enable tight analysis
- **Practical impact**: Essential for compute-constrained settings
