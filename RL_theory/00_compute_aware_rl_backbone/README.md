# Compute-Aware RL Backbone Selection

## Research Focus
Establishing the optimal **critic-free** RL algorithm infrastructure for compute-constrained settings (20× Ascend 910B).

## Key Insight
Use a **critic-free** RL algorithm to avoid training and synchronizing a large value network.

## Primary Algorithm: GRPO (Group Relative Policy Optimization)

GRPO removes the critic and estimates advantage from a *group* of sampled outputs:
- Introduced as a PPO variant
- Improves reasoning while reducing PPO's memory overhead
- No separate value network required

### Why GRPO for Limited Compute?
- **Rollout generation dominates** compute on 20 chips
- Reduces training-side overhead
- Stabilizes learning at smaller batch sizes
- Memory-efficient: no critic network to maintain

## GRPO Variants to Study

1. **Standard GRPO** (DeepSeekMath)
2. **GTPO** (Group-relative Trajectory-based Policy Optimization)
   - Addresses token-level conflicts
   - Prevents policy collapse
3. **Off-policy GRPO**
   - Sample reuse for better efficiency

## Research Questions

1. What is the optimal group size for advantage estimation under compute constraints?
2. How does batch size affect GRPO stability on distributed Ascend clusters?
3. Can we derive theoretical bounds on GRPO's sample complexity?

## Mathematical Framework

### GRPO Objective
```
L_GRPO(θ) = E_{x~D, y_i~π_old}[ Σ_i A_i · log π_θ(y_i|x) ]
```

Where advantage A_i is computed from group-normalized rewards:
```
A_i = (r_i - mean(r)) / std(r)
```

### Key Theoretical Questions
- Convergence guarantees under group normalization
- Variance analysis of group-based advantage estimation
- Optimal group size selection

## Experimental Plan

### Phase 1: Baseline Establishment
- Implement GRPO on 4B model
- Benchmark on GSM8K and MATH
- Measure: throughput, memory, convergence speed

### Phase 2: Scalability Analysis
- Scale to 8B and 14B models
- Distributed training efficiency on 20× 910B
- Communication overhead analysis

### Phase 3: Variant Comparison
- GRPO vs GTPO vs Off-policy GRPO
- Stability metrics across batch sizes
- Pass@k analysis

## Key References

1. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)
2. [GTPO: Stabilizing Group Relative Policy Optimization](https://arxiv.org/abs/2508.03772)
3. [Revisiting GRPO: On-Policy and Off-Policy Training](https://arxiv.org/html/2505.22257v1)

## Expected Contributions

- Comprehensive benchmark of critic-free RL methods on Ascend hardware
- Practical guidelines for compute-constrained RLVR
- Theoretical analysis of GRPO under limited batch sizes
