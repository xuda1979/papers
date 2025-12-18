# RL Theory Research Projects

This directory contains research projects on **Reinforcement Learning for Language Models** with a focus on **verifiable rewards** (RLVR) and **compute-efficient training** methods.

## Target Compute Setup
- **20× Ascend 910B** accelerators
- Focus on **post-training / fine-tuning** of **~4B–14B** models
- Using **verifiable, programmatic rewards** (unit tests, exact answers, formal verifiers)

## Research Directions

| ID | Folder | Topic |
|----|--------|-------|
| 0 | `00_compute_aware_rl_backbone` | Critic-free RL algorithms (GRPO) as infrastructure choice |
| 1 | `01_exploration_preserving_rlvr` | Fix RLVR's tendency to shrink reasoning boundary |
| 2 | `02_off_policy_grpo` | Sample reuse and replay buffers for GRPO |
| 3 | `03_grpo_contrastive_learning` | GRPO as contrastive learning with theoretical guarantees |
| 4 | `04_stabilize_grpo_no_reference` | GTPO-style stabilization without reference models |
| 5 | `05_step_level_credit_assignment` | TreeRPO and dense signals without reward models |
| 6 | `06_step_level_offline_rl` | DAPO-style two-stage critic-actor training |
| 7 | `07_per_token_entropy_shaping` | Dynamic entropy weighting for fine-grained credit |
| 8 | `08_self_play_coding` | Solver-Verifier and CURE-style co-evolution |
| 9 | `09_multi_turn_verifier_feedback` | Richer signals from compiler/verifier error messages |
| 10 | `10_formal_theorem_proving` | Lean/Coq as reasoning gyms with perfect verification |
| 11 | `11_optimize_pass_at_k` | Direct Pass@K optimization with policy gradients |
| 12 | `12_training_free_rl` | Inference-time methods without full finetuning |

## Recommended Experimental Setup

### Model Sizes
- Start with **4B–8B** models
- Move to **14B** only after algorithm stability

### Domains
- **Math**: Tasks with exact-answer verifiers (fast feedback)
- **Coding**: Unit tests + sandboxed execution

### Metrics to Always Report
- Pass@1 and Pass@k (k = 8, 32, 128)
- Output length and entropy (collapse indicators)
- "Hacking rate" (correct answer but wrong reasoning)

## Key References

- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)
- [Limit of RLVR](https://limit-of-rlvr.github.io/)
- [GTPO](https://arxiv.org/abs/2508.03772)
- [TreeRPO](https://arxiv.org/abs/2506.05183)
- [DAPO](https://arxiv.org/html/2412.18279v1)
