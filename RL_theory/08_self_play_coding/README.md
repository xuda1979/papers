# Self-Play for Coding Without Ground-Truth Solutions

## Research Focus
**Self-play paradigms** for code generation where the model acts as both solver and verifier, learning without human annotations or ground-truth solutions.

## Why Coding is Perfect for RLVR

- **Unit tests** provide automatic verification
- **Compilers** catch syntax/type errors
- **No human labels needed**
- **Infinite problem generation** possible

## Two Self-Play Paradigms

### Paradigm A: Solver-Verifier Self-Play (Sol-Ver)

**Single model** alternates between two roles:
1. **Solver**: Generate code to solve problem
2. **Verifier**: Generate tests to check code

**Training loop**:
```
for iteration:
    1. Solver generates code for problems
    2. Verifier generates tests for code
    3. Run tests → get rewards
    4. Update model to improve both roles
```

**Key paper**: Sol-Ver improves code and test generation iteratively without human annotations or bigger teacher models.

### Paradigm B: Co-Evolving Coder + Tester (CURE)

**Two-role RL** with interaction outcomes:
1. **Coder role**: Generates code solutions
2. **Tester role**: Generates unit tests

**Training**:
- Coder rewarded when code passes tests
- Tester rewarded when tests discriminate good/bad code
- Co-evolution through interaction

**Key paper**: CURE co-evolves coding and unit-test generation without ground-truth code supervision.

## Research Direction A: Verifier Bottleneck Analysis

### Problem Statement
When does the model's self-generated test suite become:
- **Too weak**: Passes bad code (false positives)
- **Too strong**: Fails good code (false negatives)

### Research Questions

1. **Characterizing weakness**
   - What kinds of bugs slip through self-generated tests?
   - How does test coverage evolve with training?

2. **Characterizing over-specificity**
   - When do tests become too tied to specific implementations?
   - How to maintain test generality?

3. **Optimal test difficulty**
   - Tests too easy → no learning signal
   - Tests too hard → sparse rewards
   - What's the "Goldilocks zone"?

### Proposed Analysis Framework

**Test Quality Metrics**:
| Metric | Formula | Meaning |
|--------|---------|---------|
| Bug Detection Rate | P(test fails \| code buggy) | Sensitivity |
| False Positive Rate | P(test fails \| code correct) | Specificity |
| Discrimination | BDR - FPR | Overall quality |
| Coverage | % of code paths exercised | Thoroughness |

**Curriculum Analysis**:
- Track test quality metrics over training
- Identify "verifier collapse" points
- Propose early stopping or regularization

## Research Direction B: Adversarial Test Generation

### Goal
Generate tests with **formal guarantees** on coverage/quality.

### Approach 1: Property-Based Testing

Instead of specific test cases, generate **properties**:
```python
# Property: sorted list should have same elements
def test_sort_preserves_elements(xs):
    assert sorted(xs) == sorted(sorted(xs))
    assert len(sorted(xs)) == len(xs)
```

**Train model to generate properties**, not just examples.

### Approach 2: Coverage-Guided Generation

Use **code coverage** as reward for test generation:
$$r_{\text{test}} = \text{CoverageIncrease}(\text{new\_test}, \text{existing\_tests})$$

Encourages diverse, comprehensive tests.

### Approach 3: Mutation Testing

Generate tests that **kill mutants**:
1. Create code mutants (small changes)
2. Good test: passes original, fails mutant
3. Reward test generation for mutant-killing ability

### Formal Guarantees

**Theorem (Coverage Completeness)**
Under property-based testing with properties P:
$$\text{BugDetectionRate} \geq 1 - \prod_{p \in P} (1 - \text{Coverage}(p))$$

## Research Direction C: Game-Theoretic Analysis

### Coder-Tester as Two-Player Game

**Players**:
- Coder: Wants to pass tests
- Tester: Wants to discriminate good/bad code

**Game Types**:

| Game Type | Objective | Equilibrium |
|-----------|-----------|-------------|
| Zero-sum | Tester adversarial | Minimax |
| Cooperative | Both improve together | Nash |
| Stackelberg | One leads, one follows | Stackelberg |

### Analysis Questions

1. **What equilibria exist?**
   - Degenerate: trivial tests, trivial code
   - Good: challenging tests, robust code
   
2. **How to ensure good equilibria?**
   - Constraints on test/code complexity
   - Diversity requirements
   - External validation

3. **Failure modes**
   - Collusion: coder and tester "agree" on easy problems
   - Arms race: increasingly complex but not useful

### Formal Framework

**Coder utility**: $U_C(\pi_C, \pi_T) = \mathbb{E}[\text{PassRate}(\pi_C, \pi_T)]$

**Tester utility**: $U_T(\pi_C, \pi_T) = \mathbb{E}[\text{Discrimination}(\pi_T, \pi_C)]$

**Equilibrium analysis**:
- Find $(\pi_C^*, \pi_T^*)$ such that neither can unilaterally improve
- Characterize conditions for "good" equilibria

## Experimental Design

### Datasets
- HumanEval (164 problems)
- MBPP (974 problems)
- APPS (10,000 problems, varying difficulty)
- Custom generated problems

### Baselines
1. **Supervised**: Train on ground-truth solutions
2. **RLVR**: Train with ground-truth tests only
3. **Sol-Ver**: Self-play solver-verifier
4. **CURE**: Co-evolving coder-tester

### Metrics

| Metric | Description |
|--------|-------------|
| Pass@1 | Single-sample pass rate on held-out tests |
| Test Quality | Discrimination on held-out code |
| Bug Detection | Rate of catching injected bugs |
| Training Efficiency | Samples to reach threshold performance |

### Ablations
- Model size: 4B vs 7B vs 14B
- Self-play iterations
- Reward design variants
- Test generation strategies

## Implementation Plan

### Phase 1: Infrastructure
- Code execution sandbox
- Test harness with coverage measurement
- Mutation testing framework

### Phase 2: Baseline Implementation
- Sol-Ver reproduction
- CURE reproduction
- Establish baselines

### Phase 3: Novel Contributions
- Verifier bottleneck analysis
- Adversarial test generation
- Game-theoretic analysis

### Phase 4: Integration
- Best practices synthesis
- Scaling analysis
- Paper writing

## Compute Considerations

### Why This Fits 20× 910B

- **Models**: 4B-7B sufficient for coding
- **No teacher**: Self-play doesn't need bigger models
- **Parallelizable**: Code execution can be distributed
- **Fast feedback**: Tests run in milliseconds

### Cost Breakdown
- Rollout generation: 60%
- Code execution: 20%
- Model updates: 20%

## Key References

1. [Sol-Ver: Self-Play for Code and Test Generation](https://arxiv.org/abs/2502.14948)
2. [CURE: Co-Evolving LLM Coder and Unit Tester](https://arxiv.org/abs/2506.03136)
3. [Self-Play Fine-Tuning](https://arxiv.org/abs/2401.01335)
4. [AlphaCode](https://arxiv.org/abs/2203.07814)

## Expected Contributions

1. **Verifier Bottleneck Analysis**: When self-play fails
2. **Adversarial Test Generation**: Property-based tests with guarantees
3. **Game-Theoretic Framework**: Understanding coder-tester dynamics
4. **Practical Guidelines**: When to use self-play for coding

## Paper Outline

1. Introduction: Self-play for code generation
2. Background: Sol-Ver and CURE
3. Verifier Bottleneck Analysis
4. Adversarial Test Generation
5. Game-Theoretic Analysis
6. Experiments
7. Discussion: Limitations and future work
8. Conclusion

## Why This is High-ROI

- **No human labels**: Perfect for RLVR
- **Infinite data**: Can generate problems
- **Rich theory**: Game theory + formal methods
- **Practical impact**: Code generation is high-value
- **Compute efficient**: Small models work well
