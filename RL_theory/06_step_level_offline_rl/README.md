# Step-Level Offline RL: DAPO-Style Two-Stage Training

## Research Focus
**Decoupled training** of critic and actor in two stages to get step-level dense signals while avoiding actor-critic instability.

## Problem Statement

Two pain points in RLVR:
1. **Terminal reward sparsity**: Binary pass/fail gives little learning signal
2. **Actor-critic instability**: Joint training of policy and value network is unstable

**DAPO (Direct Advantage Policy Optimization)** addresses both by:
- Training critic FIRST (offline, on rollouts)
- Training actor SECOND (using frozen critic)
- Two stages → no instability from joint training

## DAPO Background

### Two-Stage Training

**Stage 1: Critic Training**
- Generate rollouts with current/old policy
- Train critic to predict step-level values
- Objective: minimize TD error or Monte Carlo MSE

**Stage 2: Actor Training**
- Freeze critic
- Use critic for dense advantage estimation
- Standard policy gradient with step-level advantages

### Why Two Stages Help
- No gradient interference between actor and critic
- Critic can be trained to convergence
- Actor sees stable advantage estimates

## Specialization to Verifiable Domains

### Math Domains

**Critic predicts step correctness / "distance to verified solution"**:
- Input: problem + partial solution
- Output: P(eventually correct | current state)

**Training signal from verifier**:
```
For each rollout:
    For each step i:
        y_i = critic target based on:
            - Final correctness (0/1)
            - Step position (early vs late)
            - Partial verifier checks
```

### Coding Domains

**Critic predicts code quality / test pass likelihood**:
- Input: problem + partial code
- Output: P(passes all tests | current code)

**Intermediate signals**:
- Syntax validity
- Type checking
- Partial test execution
- Code coverage estimates

## Proposed Method: V-DAPO (Verifier-Enhanced DAPO)

### Stage 1: Step-Level Critic Training

```
Algorithm: Train V-Critic

1. Generate diverse rollouts: {(x, y, r)}
2. For each rollout, compute step targets:
   - If r = 1 (success): all steps get positive credit
   - If r = 0 (failure): steps before error get partial credit
3. Use verifier to identify error location (if possible)
4. Train critic V_φ to minimize:
   L_critic = Σ_{x,y,i} (V_φ(x, y_{≤i}) - target_i)^2
```

**Key Innovation**: Use verifier feedback to localize errors:
- Math: CAS checks equation validity at each step
- Code: Compiler + partial tests identify error location

### Stage 2: Actor Training with Frozen Critic

```
Algorithm: Train Actor with V-Critic

1. Generate new rollouts with current policy
2. Compute step-level advantages:
   A_i = r_i + γV_φ(s_{i+1}) - V_φ(s_i)  # TD-style
   OR
   A_i = G_i - V_φ(s_i)  # Monte Carlo style
3. Policy gradient update:
   L_actor = -Σ_i A_i log π_θ(a_i|s_i)
```

## Theoretical Analysis

### When Does Step-Level Critic Help?

**Theorem (Sample Complexity Reduction)**
Under step-level critic with estimation error ε_V:
$$\text{Sample Complexity} = O\left(\frac{1}{(1-\gamma)^2 \epsilon^2}\right)$$

vs terminal-only:
$$\text{Sample Complexity} = O\left(\frac{H^2}{(1-\gamma)^2 \epsilon^2}\right)$$

where H is horizon length. **Improvement factor: O(H^2)**.

### Conditions for Monotonic Improvement

**Theorem (Policy Improvement)**
DAPO achieves monotonic policy improvement if:
1. Critic error bounded: ||V_φ - V^π|| ≤ ε
2. Policy update is conservative: KL(π_new || π_old) ≤ δ

Then:
$$J(π_{\text{new}}) \geq J(π_{\text{old}}) - O(\epsilon + \sqrt{\delta})$$

### Comparison to Online Actor-Critic

| Aspect | Online A-C | DAPO |
|--------|------------|------|
| Stability | Joint training issues | Decoupled, stable |
| Critic quality | May lag actor | Trained to convergence |
| Sample efficiency | Poor critic reuse | Full critic training |
| Compute pattern | Alternating | Sequential |

## Experimental Design

### Sample Efficiency Comparison

Compare on **same compute budget**:
1. GRPO (terminal reward)
2. TreeRPO (tree-based step credit)
3. V-DAPO (critic-based step credit)
4. Online A-C (joint training baseline)

### Metrics

| Metric | Description |
|--------|-------------|
| Pass@1 | Final accuracy |
| Sample Efficiency | Accuracy per rollout |
| Critic Error | ||V_φ - V^π|| on held-out data |
| Advantage Quality | Correlation(A_estimated, A_true) |
| Training Stability | Variance of returns over training |

### Ablations

1. **Critic architecture**: Same as actor vs smaller
2. **Training ratio**: Critic epochs vs actor epochs
3. **Verifier granularity**: Binary vs step-level vs detailed
4. **Advantage type**: TD(0) vs TD(λ) vs Monte Carlo

## Implementation Plan

### Phase 1: Infrastructure
- Implement step-level data collection
- Build critic training pipeline
- Integrate verifiers (math CAS, code sandbox)

### Phase 2: V-DAPO Implementation
- Stage 1 critic training
- Stage 2 actor training
- Full pipeline

### Phase 3: Experiments
- Sample efficiency comparison
- Ablation studies
- Scaling analysis

## Compute Considerations

### Cost Breakdown
- **Stage 1 (Critic)**: ~30% of compute
  - Generate rollouts (expensive)
  - Train critic (cheaper than actor)
- **Stage 2 (Actor)**: ~70% of compute
  - Generate rollouts with advantage computation
  - Actor updates

### Memory
- Need to store critic (but can be smaller than actor)
- Rollout buffer for critic training

### Ascend 910B Specifics
- Critic can be 1B-4B (smaller than actor)
- Actor: 7B-14B
- Fits in 20× 910B with careful memory management

## Key References

1. [DAPO: Direct Advantage Policy Optimization](https://arxiv.org/html/2412.18279v1)
2. [Process Reward Models](https://arxiv.org/abs/2305.20050)
3. [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
4. [GRPO](https://arxiv.org/abs/2402.03300)

## Expected Contributions

1. **V-DAPO**: Verifier-enhanced two-stage training for RLVR
2. **Theoretical Analysis**: Sample complexity bounds for step-level credit
3. **Empirical Comparison**: When DAPO beats GRPO/TreeRPO
4. **Practical Guidelines**: Critic architecture and training recipe

## Paper Outline

1. Introduction: Dense signals in RLVR
2. Background: DAPO and step-level credit
3. V-DAPO: Verifier-enhanced DAPO
4. Theory: Sample complexity and convergence
5. Experiments: Math and coding domains
6. Analysis: When does two-stage help?
7. Conclusion

## Why This Project?

- **High relevance**: Rollout generation is the bottleneck
- **Theoretical depth**: Sample complexity analysis
- **Practical value**: Better use of verifier information
- **Clear comparison**: DAPO vs GRPO vs TreeRPO
- **Compute feasible**: Two-stage fits constrained settings
