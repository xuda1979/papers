# GRPO as Contrastive Learning: Theoretical Foundations

## Research Focus
Understanding GRPO through the lens of **contrastive learning** to design new losses with theoretical guarantees.

## Key Insight

GRPO's reward normalization induces a **weighted contrastive loss** over samples from the previous policy:
- Positive samples: high-reward completions
- Negative samples: low-reward completions
- Normalization creates implicit contrast

This view enables:
- Principled loss design
- Variance analysis
- Theoretical guarantees on success amplification

## Contrastive Interpretation of GRPO

### Standard GRPO Loss
$$L_{\text{GRPO}}(\theta) = -\mathbb{E}_{x, \{y_i\}_{i=1}^G} \left[ \sum_{i=1}^G \hat{A}_i \log \pi_\theta(y_i|x) \right]$$

where $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$ is the normalized advantage.

### Contrastive View
This is equivalent to:
$$L_{\text{contrast}}(\theta) = -\mathbb{E}_x \left[ \sum_{i: r_i > \bar{r}} w_i^+ \log \pi_\theta(y_i|x) + \sum_{j: r_j < \bar{r}} w_j^- \log \pi_\theta(y_j|x) \right]$$

where $w^+$ and $w^-$ are derived from the normalization.

## Research Questions

### 1. Optimal Normalization

Which normalization yields best stability under binary rewards?

| Normalization | Formula | Properties |
|---------------|---------|------------|
| Mean-only | A_i = r_i - mean(r) | Simple, may have large variance |
| Mean+Std | A_i = (r_i - mean(r))/std(r) | Unit variance, undefined if all same |
| Percentile | A_i based on rank | Robust to outliers |
| Softmax | exp(r_i/τ) / Σexp(r_j/τ) | Temperature-controlled |

### 2. Optimal Whitening

Can we derive an **optimal whitening rule** that minimizes gradient variance?

For binary rewards r ∈ {0, 1}:
- Let p = fraction of successful samples in group
- Variance of gradient depends on p

**Conjecture**: Optimal normalization adapts to p:
$$\hat{A}_i = \begin{cases} \frac{1-p}{p} & \text{if } r_i = 1 \\ -1 & \text{if } r_i = 0 \end{cases}$$

### 3. Success Amplification Bounds

Recent analysis shows GRPO induces explicit recurrences for success probability:

$$p_{t+1} = f(p_t, \eta, \beta)$$

**Goal**: Derive closed-form bounds on:
- Convergence rate to high success probability
- Dependence on learning rate η and regularization β
- Conditions for guaranteed improvement

## Proposed New Loss: Explicit Contrastive RLVR

### Design Principles
1. Keep GRPO's critic-free efficiency
2. Have clearer theoretical objective
3. Provide bounds on success amplification

### Proposed Loss
$$L_{\text{EC-RLVR}}(\theta) = -\mathbb{E}_x \left[ \log \frac{\sum_{i: r_i=1} \pi_\theta(y_i|x)^{1/\tau}}{\sum_{j=1}^G \pi_\theta(y_j|x)^{1/\tau}} \right]$$

This is an InfoNCE-style loss where:
- Positives: correct completions
- Negatives: all completions
- Temperature τ controls sharpness

### Theoretical Properties

**Theorem (Success Amplification)**
Under EC-RLVR with learning rate η and temperature τ:

$$p_{t+1} \geq p_t + \eta \cdot p_t(1-p_t) \cdot \frac{1}{\tau}$$

until reaching stationary point.

**Theorem (Gradient Variance)**
The variance of the EC-RLVR gradient is:

$$\text{Var}[\nabla L] = O\left(\frac{1}{G \cdot p(1-p)}\right)$$

minimized when p = 0.5 (balanced groups).

## Experimental Validation

### Experiments to Run

1. **Normalization Comparison**
   - Compare mean-only, mean+std, percentile, softmax
   - Measure: gradient variance, convergence speed, final performance

2. **EC-RLVR vs GRPO**
   - Same compute budget
   - Compare Pass@1, Pass@k curves
   - Stability across seeds

3. **Temperature Sweep**
   - τ ∈ {0.1, 0.5, 1.0, 2.0}
   - Effect on exploration vs exploitation

### Metrics
| Metric | Description |
|--------|-------------|
| Gradient Variance | Variance of policy gradient estimates |
| Effective Sample Size | 1/Σw² for weighted updates |
| Success Rate Trajectory | p_t over training |
| KL from Base | Divergence from reference policy |

## Mathematical Tools

### Required Analysis
1. **Gradient computation** for contrastive losses
2. **Variance bounds** under importance weighting
3. **Convergence analysis** of success probability dynamics
4. **Comparison to InfoNCE** theory

### Key Lemmas to Prove
- Lemma 1: EC-RLVR gradient is unbiased for success probability gradient
- Lemma 2: Optimal temperature τ* depends on group size G
- Lemma 3: Variance reduction from larger groups

## Key References

1. [RLVR Success Amplification and Dynamics](https://arxiv.org/abs/2503.06639)
2. [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
3. [InfoNCE and Mutual Information](https://arxiv.org/abs/1807.03748)
4. [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)

## Expected Contributions

1. **Theoretical Framework**: Unifying view of GRPO as contrastive learning
2. **New Algorithm**: EC-RLVR with provable guarantees
3. **Variance Analysis**: Optimal normalization for binary rewards
4. **Practical Insights**: When to use which normalization

## Paper Structure (Proposed)

1. Introduction: RLVR and the need for theoretical understanding
2. Background: GRPO and contrastive learning
3. GRPO as Contrastive Learning: The connection
4. Variance Analysis: Optimal normalization
5. EC-RLVR: A principled alternative
6. Theoretical Guarantees: Success amplification bounds
7. Experiments: Validation and comparison
8. Discussion: Implications and limitations

## Why This is a Good Project

- **Math-heavy, compute-light**: Theory is the main contribution
- **Novel perspective**: Contrastive view not fully explored
- **Practical value**: Better loss designs for RLVR
- **Clear evaluation**: Gradient variance is measurable
