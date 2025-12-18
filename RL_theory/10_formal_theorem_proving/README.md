# Formal Theorem Proving as a Reasoning Gym

## Research Focus
Using **formal theorem proving** (Lean 4, Coq, Isabelle) as a domain for studying **verifiable reasoning** with perfect reward signals and no possibility of "answer hacking".

## Why Theorem Proving is Ideal

### Perfect Verification
- **Deterministic verifier**: Proof is correct or not, no ambiguity
- **Structured errors**: Verifier returns exactly what's wrong
- **No shortcuts**: Can't "game" the verifier
- **Incremental feedback**: Tactic-by-tactic verification

### Reasoning Gym Benefits
- **Clean domain** for studying reasoning
- **Scalable difficulty**: From simple to Olympiad-level
- **Rich structure**: Types, tactics, proof strategies
- **Transfer potential**: Does proof-planning help other reasoning tasks?

### Compute Feasibility
- **Small models work**: 7B sufficient (Leanabell-Prover-V2 shows this)
- **Fast verification**: Lean/Coq checking is quick
- **Deterministic**: No need for multiple expensive verifier calls

## Background: Leanabell-Prover-V2

Achieved strong results on MiniF2F with:
- **7B model** (DeepSeek-Coder-7B-Base)
- **Lean 4** verifier feedback
- **Multi-turn** interaction
- **Simple reward strategy**

### Key Results
- Improved pass@ metrics on MiniF2F
- Effective use of structured error messages
- Demonstrates viability at 7B scale

## Research Directions

### Direction 1: Search vs Learning Decomposition

**Question**: How much of theorem proving success comes from:
- **Search**: Exploring proof space at inference time
- **Learning**: Training-time internalization of tactics/strategies

**Experiments**:
| Method | Search | Learning | Pass@1 | Pass@32 |
|--------|--------|----------|---------|---------|
| Baseline | Beam=1 | None | X | X |
| + Search | Beam=32 | None | ? | ? |
| + RL | Beam=1 | RLVR | ? | ? |
| + Both | Beam=32 | RLVR | ? | ? |

**Analysis**:
- Is search + learning complementary or redundant?
- Where is the compute better spent?

### Direction 2: Credit Assignment Over Proof Steps

**Challenge**: Proofs can be dozens of tactics long.

**Questions**:
1. Which tactics are "pivotal"?
2. How to assign credit to early tactics that enable later success?
3. Can we use proof structure (subgoals) for credit?

**Proposed Methods**:

**A) Subgoal-Based Credit**
```
For each subgoal:
    reward_subgoal = 1 if subgoal proved, 0 otherwise
    credit tactics that contribute to this subgoal
```

**B) Counterfactual Tactics**
```
For each tactic t:
    Try removing t
    If proof breaks → t is important
    Credit ∝ importance
```

**C) Value Functions on Proof States**
```
V(proof_state) = P(eventually prove | current_state)
Advantage_tactic = V(state_after) - V(state_before)
```

### Direction 3: Curriculum Learning

**Start easy, grow hard**:

**Curriculum stages**:
1. **Stage 1**: Trivial lemmas (1-2 tactics)
2. **Stage 2**: Simple theorems (3-5 tactics)
3. **Stage 3**: MiniF2F test level
4. **Stage 4**: Olympiad-level

**Research questions**:
- What's the optimal curriculum schedule?
- Can we automatically sequence by difficulty?
- Does curriculum help vs flat training?

**Difficulty metrics**:
- Proof length
- Number of subgoals
- Tactic diversity needed
- Human difficulty ratings

### Direction 4: Tactic Learning and Abstraction

**Observation**: Human mathematicians develop "macro tactics" or "proof patterns".

**Goal**: Can the model learn reusable proof strategies?

**Approaches**:

**A) Hierarchical RL**
- High-level: Choose proof strategy
- Low-level: Execute tactics

**B) Tactic Synthesis**
- Learn to propose new tactics
- Compose from primitives

**C) Proof Sketch → Detailed Proof**
- First generate high-level plan
- Then fill in details

### Direction 5: Cross-Domain Transfer

**Key Question**: Does improving proof-step planning help with:
- Math word problems?
- Code generation?
- Multi-step reasoning?

**Experimental Protocol**:
1. Pre-train on theorem proving with RLVR
2. Fine-tune on target domain
3. Compare to baseline (no theorem proving)

**Hypothesis**: Theorem proving teaches:
- Systematic decomposition
- Rigorous step-by-step reasoning
- Verification discipline

## Experimental Design

### Datasets

| Dataset | Size | Difficulty | Source |
|---------|------|------------|--------|
| MiniF2F | 488 | Olympiad | GitHub |
| ProofNet | 371 | Undergraduate | Lean 4 |
| Mathlib | ~100k | Varies | Lean 4 stdlib |
| Custom | ∞ | Controlled | Generated |

### Baselines

1. **Few-shot prompting**: GPT-4 style
2. **Supervised fine-tuning**: Train on human proofs
3. **RLVR (terminal)**: Binary reward at end
4. **RLVR (step-level)**: Credit per tactic
5. **Best-of-k**: Sample many proofs, take first success

### Metrics

| Metric | Description |
|--------|-------------|
| Pass@1 | Single-sample success rate |
| Pass@k | Best-of-k success rate |
| Proof Length | Average tactics per proof |
| Search Efficiency | Success rate per sampled proof |
| Tactic Diversity | Unique tactics used |
| Transfer Performance | Performance on math/code after training |

## Theoretical Analysis

### Research Questions

1. **Sample complexity**: How many theorem-proving examples needed?
2. **Generalization**: From seen to unseen theorems
3. **Compositionality**: Combining learned tactics
4. **Proof search complexity**: Branching factor, depth, etc.

### Proposed Analysis

**Theorem (Sample Complexity Bound)**
Under assumptions:
- Proof length ≤ H
- Tactic space size |T|
- Verifier noise 0 (perfect)

Sample complexity for ε-optimal policy:
$$N = O\left(\frac{|T|^H}{\epsilon^2}\right)$$

But with structured tactics and compositional learning:
$$N = O\left(\frac{|T| \cdot H}{\epsilon^2}\right)$$

**Key**: Structure helps dramatically.

## Implementation Plan

### Phase 1: Infrastructure (Weeks 1-2)
- Set up Lean 4 environment
- Build verifier integration
- Collect datasets (MiniF2F, ProofNet)

### Phase 2: Baselines (Weeks 3-4)
- Reproduce Leanabell results
- Implement search baselines
- Implement RLVR baseline

### Phase 3: Novel Methods (Weeks 5-7)
- Step-level credit assignment
- Curriculum learning
- Hierarchical tactics

### Phase 4: Transfer Study (Weeks 8-9)
- Train on proofs
- Test on math/code
- Analyze transfer

### Phase 5: Analysis & Writing (Weeks 10-12)
- Theoretical analysis
- Comprehensive experiments
- Paper writing

## Compute Budget

### Model Size: 7B
- Fits easily on 20× 910B
- Proven sufficient by Leanabell

### Training Costs
- Rollout generation: 70% (proof search can branch)
- Verification: 10% (Lean is fast)
- Model updates: 20%

### Optimization
- Parallelize proof search across problems
- Cache verified subproofs
- Early stopping on success

## Key References

1. [Leanabell-Prover-V2](https://arxiv.org/abs/2507.08649)
2. [LEGO-Prover](https://arxiv.org/abs/2310.00656)
3. [Draft, Sketch, and Prove](https://arxiv.org/abs/2210.12283)
4. [HyperTree Proof Search](https://arxiv.org/abs/2205.11491)
5. [MiniF2F Dataset](https://arxiv.org/abs/2109.00110)

## Expected Contributions

1. **Search-Learning Decomposition**: Understanding the roles of each
2. **Step-Level Credit**: Better credit assignment for proofs
3. **Curriculum Analysis**: Optimal difficulty progression
4. **Cross-Domain Transfer**: Does proof training help reasoning?
5. **Practical Guidelines**: How to train theorem provers efficiently

## Paper Outline

1. Introduction: Theorem proving as reasoning gym
2. Background: RLVR and formal verification
3. Experimental Setup: Datasets, models, metrics
4. Search vs Learning: Decomposition study
5. Credit Assignment: Step-level methods
6. Curriculum Learning: Difficulty progression
7. Cross-Domain Transfer: Generalization analysis
8. Discussion: Insights for reasoning RL
9. Conclusion

## Why This is Excellent for Your Setting

- **Perfect verifier**: No reward ambiguity
- **7B models**: Compute-efficient
- **Rich structure**: Lots of research questions
- **Clean analysis**: Deterministic setting
- **High impact**: Theorem proving + reasoning transfer
- **Growing field**: Many open problems
