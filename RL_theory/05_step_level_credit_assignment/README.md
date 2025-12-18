# Step-Level Credit Assignment Without Training a Reward Model

## Research Focus
Using **tree sampling** to estimate step-level rewards and provide **dense training signals** without training a separate reward model.

## Problem Statement

RLVR rewards are typically **terminal/binary**:
- ✓ Answer is correct OR ✗ Answer is wrong
- No information about which steps were good/bad
- **Coarse credit assignment** hurts learning efficiency

**Solution**: TreeRPO uses tree sampling to estimate intermediate step rewards.

## TreeRPO Background

**TreeRPO (Tree Relative Policy Optimization)**:
- Uses **tree sampling** to estimate expected rewards at intermediate steps
- Forms **step-level groups** for advantage estimation
- Produces **dense signals** without a separate reward model
- Reports improved Pass@1 and reduced response length vs GRPO

### How It Works

```
For each step i in reasoning:
    1. Sample multiple continuations from step i
    2. Get terminal rewards for each branch
    3. Estimate step-i value as mean reward of branches
    4. Compute step-level advantage
```

## Research Directions

### 1. Compute-Aware TreeRPO

Make TreeRPO efficient for limited compute:

**Adaptive Branching Factor**
- Branch more when verifier uncertainty is high
- Branch less for "obvious" steps

$$B_i = f(\text{Entropy}(\pi(\cdot|x, y_{<i})), \text{Var}[\hat{V}_i])$$

**Early Cutoff**
- Use partial verifiers to prune bad branches early
- Math: Check equation validity at each step
- Code: Run partial tests or type checks

**Subtree Reuse**
- Cache subtrees across epochs (off-policy tree replay)
- Importance weight reused branches

### 2. Theoretical Analysis

**Main Question**: Does TreeRPO approximate a stepwise advantage estimator? What are bias/variance tradeoffs?

#### Bias Analysis

Let V*(s_i) be true value of state at step i.

TreeRPO estimate: $\hat{V}(s_i) = \frac{1}{B} \sum_{j=1}^B r(\text{branch}_j)$

**Theorem (Bias Bound)**
$$|\mathbb{E}[\hat{V}(s_i)] - V^*(s_i)| \leq O\left(\frac{1}{B}\right) + \epsilon_{\text{policy}}$$

where ε_policy accounts for policy shift.

#### Variance Analysis

$$\text{Var}[\hat{V}(s_i)] = \frac{\sigma_r^2}{B} + \text{Var}_{\text{tree}}$$

where Var_tree depends on tree structure.

**Theorem (Variance Reduction)**
TreeRPO with B branches reduces variance by factor O(B) compared to terminal-only advantage.

### 3. Comparison to Terminal-Only

When does step-level credit help most?

| Condition | Terminal-Only | TreeRPO |
|-----------|---------------|---------|
| Short problems | Fine | Overkill |
| Long reasoning | Poor | Essential |
| High reward variance | Struggling | Better |
| Low success rate | Very noisy | More signal |

## Proposed Algorithm: Efficient TreeRPO

```
Algorithm: E-TreeRPO
Input: Prompt x, policy π, branching schedule B(·)

# Generate tree
tree = {}
tree[root] = x

for depth d = 1 to max_depth:
    for each node n at depth d-1:
        if should_branch(n):
            # Adaptive branching
            b = B(entropy(π(·|n)), depth=d)
            children = sample(π(·|n), num=b)
            tree[n].children = children
        else:
            # Single continuation
            tree[n].children = [sample(π(·|n), num=1)]

# Evaluate leaves
for each leaf l in tree:
    l.reward = Verifier(x, path_to_root(l))

# Backpropagate values
for depth d = max_depth-1 to 0:
    for each node n at depth d:
        n.value = mean([c.value for c in n.children])

# Compute step-level advantages
for each node n:
    n.advantage = n.value - parent(n).value  # if exists

# Update policy with step-level loss
L = -Σ_n A_n · log π(token_n | context_n)
```

### Key Innovations

1. **Adaptive Branching**: Branch factor depends on uncertainty
2. **Partial Verification**: Early pruning of bad branches
3. **Subtree Caching**: Reuse across epochs

## Experimental Plan

### Phase 1: TreeRPO Reproduction
- Implement basic TreeRPO
- Compare to GRPO on GSM8K
- Measure compute overhead

### Phase 2: Compute-Aware Variants
- Implement adaptive branching
- Implement early cutoff
- Measure efficiency gains

### Phase 3: Theoretical Validation
- Empirically verify bias/variance bounds
- Characterize when TreeRPO helps most

### Metrics

| Metric | Description |
|--------|-------------|
| Pass@1 | Single-sample accuracy |
| Sample Efficiency | Accuracy per sample |
| Compute Efficiency | Accuracy per FLOP |
| Tree Overhead | Extra compute for tree sampling |
| Response Length | Average completion length |
| Step Accuracy | Accuracy at intermediate steps |

## Datasets

### Math
- GSM8K (8-step average)
- MATH (longer reasoning)
- Custom multi-step algebra

### Coding
- HumanEval (short)
- MBPP (medium)
- APPS (long, multi-function)

## Key References

1. [TreeRPO: Tree Relative Policy Optimization](https://arxiv.org/abs/2506.05183)
2. [GRPO](https://arxiv.org/abs/2402.03300)
3. [Process Reward Models](https://arxiv.org/abs/2305.20050)

## Expected Contributions

1. **Algorithmic**: E-TreeRPO with adaptive branching and early cutoff
2. **Theoretical**: Bias/variance analysis of tree-based value estimation
3. **Empirical**: When TreeRPO beats GRPO (problem characteristics)
4. **Practical**: Compute-aware guidelines for tree depth and branching

## Paper Structure

1. Introduction: Credit assignment in RLVR
2. Background: GRPO and terminal rewards
3. TreeRPO: Tree-based step-level credit
4. Theory: TreeRPO as stepwise advantage estimator
5. E-TreeRPO: Compute-efficient extensions
6. Experiments: Comparison and ablations
7. Analysis: When does step-level credit help?
8. Conclusion

## Why This is a Great Project

- **Math-heavy**: Bias/variance analysis, advantage estimation theory
- **Practical impact**: Better training signal without reward models
- **Manageable compute**: Tree overhead is controllable
- **Novel extensions**: Adaptive branching not explored
- **Clear baselines**: TreeRPO paper provides starting point
