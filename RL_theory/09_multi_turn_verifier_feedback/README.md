# Multi-Turn Verifier Feedback RL

## Research Focus
Using **rich verifier feedback** (error messages, compiler output, failing tests) instead of binary pass/fail to provide much denser learning signals.

## Problem Statement

Binary rewards waste information that verifiers provide:
- ✓ Pass OR ✗ Fail
- But verifiers give much more:
  - **Compiler errors**: syntax errors, type errors, undefined variables
  - **Test failures**: which tests failed, what the diff is
  - **Lean/Coq errors**: tactic failures, type mismatches, unsolved goals
  - **Math CAS**: which step broke, what the error is

**Goal**: Design RL methods that exploit this rich feedback.

## Background: Leanabell-Prover-V2

**Leanabell-Prover-V2** integrates Lean 4 verifier feedback:
- Multi-turn interactions with the verifier
- Feedback token masking
- Simple reward strategy combining:
  - Proof completion
  - Verifier feedback signals
- Achieves strong results with 7B model

### Key Ideas
1. **Multi-turn**: Model can revise based on feedback
2. **Structured feedback**: Not just pass/fail, but detailed errors
3. **Feedback integration**: Model learns to interpret and use feedback

## Extending Beyond Theorem Proving

### Application: Coding with Compiler Feedback

**Setup**:
- Model generates code
- Compiler/test runner returns errors
- Model revises code based on errors

**Multi-turn protocol**:
```
Turn 1: Generate initial code
    → Compiler: "TypeError: expected int, got str at line 23"

Turn 2: Fix type error
    → Tests: "AssertionError: expected [1,2,3], got [1,2]"

Turn 3: Fix logic bug
    → Tests: "All tests passed"
```

### Application: Math with CAS Feedback

**Setup**:
- Model solves math problem step-by-step
- CAS (e.g., SymPy, Mathematica) checks each step
- Model revises based on feedback

**Multi-turn protocol**:
```
Turn 1: x + 3 = 5 → x = 5 - 3 = 2
    → CAS: "Correct"

Turn 2: Simplify (x^2 - 4)/(x-2) = x + 2
    → CAS: "Error: division by zero at x=2"

Turn 3: (x^2-4)/(x-2) = (x+2)(x-2)/(x-2) = x+2 for x≠2
    → CAS: "Correct with caveat"
```

## Proposed Methods

### 1. Feedback-Conditioned Policy

**Model sees feedback as part of context**:
$$\pi_\theta(y_t | x, y_{<t}, \text{feedback}_{<t})$$

where feedback_{<t} is all verifier feedback so far.

**Training**:
- Collect multi-turn trajectories
- Reward based on final success + intermediate progress
- Backprop through all turns

### 2. Feedback Token Masking

**Idea**: Don't compute loss on feedback tokens (model didn't generate them)

**Implementation**:
```python
for turn in trajectory:
    # Model generation
    y_gen = model.generate(context)
    loss_gen = -log p(y_gen | context)
    
    # Verifier feedback (no loss)
    feedback = verifier(y_gen)
    # feedback is appended to context but not in loss
    
    context = context + y_gen + feedback
```

### 3. Reward Shaping from Feedback

**Rich rewards** from feedback signals:

| Feedback Type | Reward Component |
|---------------|------------------|
| Syntax error | -0.5 |
| Type error | -0.3 |
| Runtime error | -0.1 |
| Wrong output | -0.05 + similarity |
| Correct | +1.0 |

**Intermediate progress rewards**:
- Each error fixed: +0.2
- Each test passed: +0.1
- Reduced error count: +0.1

### 4. Feedback-Guided Exploration

Use feedback to **guide search**:
- Parse error messages to identify problem location
- Focus revisions on problematic regions
- Reject revisions that don't address feedback

## Theoretical Analysis

### Questions

1. **When does multi-turn help?**
   - Hypothesis: When feedback localizes errors

2. **Optimal number of turns?**
   - Too few: Can't fix complex errors
   - Too many: Inefficient

3. **Feedback quality**
   - What makes feedback "useful"?
   - Can we quantify feedback information content?

### Proposed Metrics

**Feedback Usefulness**:
$$U(\text{feedback}) = I(Y_{\text{correct}}; \text{feedback} | X, Y_{\text{wrong}})$$

Mutual information between correct solution and feedback.

**Turn Efficiency**:
$$E = \frac{P(\text{success after } t \text{ turns})}{t}$$

Success rate per turn (diminishing returns?).

## Experimental Design

### Domain 1: Coding

**Datasets**:
- HumanEval (function-level)
- MBPP (simple programs)
- APPS (complex programs)

**Verifiers**:
- Python interpreter (syntax, runtime errors)
- Unit tests (functional correctness)
- Type checkers (MyPy)

**Baselines**:
1. Single-turn RLVR (binary reward)
2. Multi-sample (generate k, take best)
3. Multi-turn with feedback (proposed)

### Domain 2: Math

**Datasets**:
- GSM8K (word problems)
- MATH (competition problems)
- Custom algebra/calculus

**Verifiers**:
- Exact match
- SymPy verification
- Step-by-step CAS checking

### Domain 3: Theorem Proving

**Datasets**:
- MiniF2F
- Lean 4 mathlib
- Custom lemmas

**Verifier**: Lean 4

### Metrics

| Metric | Description |
|--------|-------------|
| Final Pass@1 | Success after all turns |
| Turn Distribution | Which turn typically succeeds |
| Feedback Usage | Does model use feedback effectively |
| Error Reduction | % of errors fixed per turn |

## Implementation Plan

### Phase 1: Infrastructure
- Multi-turn rollout collection
- Verifier integration for all domains
- Feedback parsing and formatting

### Phase 2: Baseline Models
- Single-turn RLVR
- Multi-sample baselines
- Reproduce Leanabell results

### Phase 3: Multi-Turn Methods
- Feedback-conditioned policy
- Feedback token masking
- Reward shaping variants

### Phase 4: Analysis
- Turn efficiency analysis
- Feedback usefulness measurement
- Scaling studies

## Compute Considerations

### Memory
- Need to store multi-turn contexts (longer)
- Solution: Context compression or truncation

### Throughput
- Multiple verifier calls per sample
- Solution: Parallelize verifier execution
- Early stopping if success achieved

### Cost-Benefit
- More turns = more compute per sample
- But higher success rate per sample
- **Break-even analysis needed**

## Key References

1. [Leanabell-Prover-V2: Verifier-integrated Reasoning](https://arxiv.org/abs/2507.08649)
2. [Self-Debugging via Multi-Agent Debate](https://arxiv.org/abs/2304.05128)
3. [Self-Repair for Code Generation](https://arxiv.org/abs/2301.13246)
4. [Interactive Theorem Proving with LLMs](https://arxiv.org/abs/2205.12615)

## Expected Contributions

1. **Multi-Domain Extension**: Coding and math, not just theorem proving
2. **Feedback Taxonomy**: Characterizing useful vs useless feedback
3. **Optimal Turn Analysis**: How many turns to use
4. **Reward Shaping Principles**: How to design rewards from rich feedback

## Paper Outline

1. Introduction: Beyond binary rewards
2. Background: Verifier feedback in RL
3. Multi-Turn Protocol: General framework
4. Methods: Feedback integration strategies
5. Theory: When does multi-turn help?
6. Experiments: Coding, math, theorem proving
7. Analysis: Feedback usefulness and turn efficiency
8. Conclusion

## Why This is Valuable

- **Maximizes verifier information**: Don't throw away feedback
- **Multi-domain applicable**: Coding, math, proofs all have rich verifiers
- **Still "no human feedback"**: All signals from verifiers
- **Practical impact**: Higher success rates, better debugging
- **Theoretically interesting**: Information-theoretic analysis of feedback
