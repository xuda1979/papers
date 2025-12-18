# Training-Free RL: Inference-Time Improvements Without Finetuning

## Research Focus
Achieving **RLVR-like improvements** at inference time through clever priors and policies, **without parameter updates**.

## Motivation

**Problem**: Full RL finetuning is expensive.
- Requires many rollouts
- Needs compute for training
- Risks catastrophic forgetting

**Question**: Can we get significant gains with **zero training**?

## Background: Training-Free GRPO

Recent work proposes improving performance without parameter updates by:
- **Distilling experiential knowledge** into a token prior
- **Applying prior during inference**
- Claims improvements with only a few dozen samples in some settings

### Key Idea
Learn a **prior distribution** over tokens from successful vs unsuccessful completions, then use this prior to bias generation.

## Core Question

**What parts of RLVR gains come from distribution shaping that can be approximated at inference time?**

If we can identify these, we can get:
- 80% of RLVR improvements
- 5% of the training compute
- No risk to base model

## Proposed Methods

### Method 1: Success-Weighted Token Prior

**Training phase** (on small sample):
```
1. Generate N completions per problem (N ~ 50-100)
2. Verify each completion
3. Compute token frequencies:
   p_success(token | context)
   p_failure(token | context)
4. Build prior:
   prior(token | context) = p_success / p_failure
```

**Inference phase**:
```
Modified sampling:
p_new(token | context) ∝ p_base(token | context) * prior(token | context)^α
```

where α controls prior strength.

### Method 2: Contrastive Decoding

**Idea**: Decode to maximize likelihood under success model while minimizing under failure model.

**Formulation**:
```
token = argmax_t [log p_success(t | context) - β log p_failure(t | context)]
```

**How to get p_success and p_failure?**
- Collect successful completions → fine-tune or adapt base model (small)
- Collect failed completions → fine-tune or adapt base model (small)

**Training-free approximation**:
Use **in-context examples** instead:
```
p_success ≈ p_base(t | [success examples] + context)
p_failure ≈ p_base(t | [failure examples] + context)
```

### Method 3: Verification-Guided Beam Search

**Standard beam search**: Keep top-k by likelihood

**Verification-guided**: Incorporate partial verification

**Algorithm**:
```
for each beam step:
    1. Generate K candidate continuations
    2. For each candidate:
       - Run partial verifier (if possible)
       - Compute: score = log p + λ * partial_correctness
    3. Keep top-K by score
```

**For math**:
- Partial verifier: Check equation validity, dimensional analysis

**For code**:
- Partial verifier: Type checking, lint, partial tests

### Method 4: Best-of-N with Learned Verifier

**Idea**: Sample N completions, use a lightweight verifier to select best.

**Training-free verifier options**:
1. **Few-shot verifier**: Prompt base model to judge correctness
2. **Heuristic verifier**: Code length, entropy, repetition detection
3. **Ensemble**: Multiple fast verifiers, majority vote

**Cost analysis**:
- Generate N samples: N × generation cost
- Verify each: N × (cheap verification cost)
- Total: ~N × generation cost

**Break-even**: Need N < (RLVR training cost / generation cost)

## Theoretical Analysis

### What Can Be Approximated?

**RLVR learns**:
$$\pi_{\text{RLVR}} = \arg\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta \cdot D_{KL}(\pi || \pi_{\text{base}})$$

**Training-free approximates**:
$$\pi_{\text{approx}}(y|x) \propto \pi_{\text{base}}(y|x) \cdot \exp(\alpha \cdot \text{prior}(y|x))$$

**Question**: Under what conditions are these close?

### Approximation Bound

**Theorem (Informal)**
If the prior captures most of the success patterns:
$$D_{KL}(\pi_{\text{RLVR}} || \pi_{\text{approx}}) \leq \epsilon$$

Then performance gap is:
$$J(\pi_{\text{RLVR}}) - J(\pi_{\text{approx}}) \leq O(\sqrt{\epsilon})$$

### Sample Complexity of Prior Learning

**How many samples needed** to learn a good prior?

**Conjecture**: For token-level prior:
$$N = O\left(\frac{|\mathcal{V}| \cdot H}{\epsilon^2 \cdot p}\right)$$

where:
- |V| = vocabulary size
- H = sequence length
- p = success rate
- ε = desired accuracy

**Comparison to RLVR training**:
RLVR needs ~10K-100K samples.
Training-free might work with 100-1000 samples.

## Experimental Design

### Phase 1: Upper Bound Analysis

**Question**: What's the best possible training-free method?

**Oracle experiment**:
1. Train full RLVR model
2. Extract token-level statistics
3. Use as prior for base model
4. Measure performance

This gives **upper bound** on training-free potential.

### Phase 2: Method Comparison

Compare all methods on GSM8K/HumanEval:

| Method | Training Samples | Inference Cost | Pass@1 |
|--------|------------------|----------------|--------|
| Base | 0 | 1× | baseline |
| Success Prior | 100 | 1.1× | ? |
| Contrastive | 100 | 2× | ? |
| Verification-Guided | 0 | 3× (beam) | ? |
| Best-of-N (N=16) | 0 | 16× | ? |
| RLVR (full) | 10,000 | 1× | target |

### Phase 3: Hybrid Methods

**Observation**: Training-free can be combined with minimal training.

**Hybrid**:
1. Collect 100-1000 samples
2. Learn light adapter (LoRA, 1-10M params)
3. Use adapter to bias base model
4. Apply training-free techniques on top

### Metrics

| Metric | Description |
|--------|-------------|
| Pass@1 | Accuracy |
| Sample Efficiency | Performance per training sample |
| Inference Cost | Compute multiplier at inference |
| Pareto Frontier | Cost-performance tradeoff |

## Research Questions

1. **How much of RLVR is distribution shaping?**
   - Measure: ∆ from base to RLVR in KL, token statistics

2. **What's the minimum training data?**
   - Plot: Performance vs number of samples

3. **When is training-free better?**
   - When you have limited training compute
   - When you want to preserve base model
   - When task distribution shifts (adaptive prior)

4. **Can prior be reused across tasks?**
   - Train prior on GSM8K, test on MATH

## Implementation Plan

### Phase 1: Infrastructure (Weeks 1-2)
- Implement sampling and verification
- Build prior extraction pipeline
- Set up evaluation harness

### Phase 2: Method Implementation (Weeks 3-4)
- Success-weighted prior
- Contrastive decoding
- Verification-guided beam search
- Best-of-N baselines

### Phase 3: Experiments (Weeks 5-6)
- All methods on GSM8K, MATH, HumanEval
- Sample efficiency curves
- Inference cost analysis

### Phase 4: Analysis (Weeks 7-8)
- What's approximated well vs poorly?
- Theoretical analysis
- Practical guidelines

## Compute Considerations

### Why This Fits Small Compute

- **No training**: Save the expensive part
- **Small prior**: Learn from 100-1000 samples only
- **Inference overhead**: Acceptable for many applications

### Cost Breakdown

| Component | Cost | Fraction |
|-----------|------|----------|
| Collect samples | N × gen | Small |
| Build prior | Negligible | Tiny |
| Inference | 1-3× gen | Moderate |
| **Total** | ~1% of RLVR | **Huge savings** |

## Key References

1. [Training-Free Group Relative Policy Optimization](https://arxiv.org/abs/2510.08191)
2. [Contrastive Decoding](https://arxiv.org/abs/2210.15097)
3. [Best-of-N Baselines](https://arxiv.org/abs/2211.11597)
4. [In-Context Learning](https://arxiv.org/abs/2301.00234)

## Expected Contributions

1. **Theoretical Framework**: What aspects of RLVR can be approximated inference-time?
2. **Method Comparison**: Comprehensive comparison of training-free techniques
3. **Sample Efficiency**: How few samples suffice for good priors?
4. **Practical Guidelines**: When to use training-free vs full RLVR

## Paper Outline

1. Introduction: The cost of RL finetuning
2. Background: RLVR and distribution shaping
3. Training-Free Methods: Four approaches
4. Theory: What can be approximated?
5. Experiments: Comparison on math and coding
6. Analysis: Cost-performance tradeoffs
7. Discussion: When to use each method
8. Conclusion

## Why This is Valuable

- **Extremely practical**: Most people can't afford full RLVR
- **Novel question**: "What's essential in RLVR?" not well studied
- **Low barrier**: Easy to try and iterate
- **Immediate impact**: Can deploy right away
- **Theoretical interest**: Fundamental question about RL

## Extensions

### Dynamic Priors
Update prior as you see new problems (online learning without training).

### Task-Specific Priors
Learn separate priors for different problem types.

### Hybrid Training
Minimal training + inference-time techniques.

### Transferable Priors
Learn prior on one domain, transfer to another.
