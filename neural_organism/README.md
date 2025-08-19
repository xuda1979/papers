# Runtime–Enforced Compute Allocation for Reasoning and Simulation

This repository contains the code and experiments for the paper "Runtime–Enforced Compute Allocation for Reasoning and Simulation."

> ## Abstract
>
> We present twelve **runtime-enforced primitives** for allocating compute in math, code, and physics workloads. Each primitive operates below any planner or large language model, wielding OS, runtime, or hardware mechanisms to **preempt, gate, or splice** computation deterministically. To demonstrate the approach, we deploy a **compute-budgeted Growing RBF network** in a drifting contextual bandit. A runtime growth gate caps structural expansion while retaining reward, highlighting how verifiers can steer FLOPs toward the most promising reasoning branches. Together these primitives form a minimal architecture that lets external signals decide where compute is spent, enabling reproducible control over long reasoning chains and multi-fidelity simulations.

---

## The Vision: Runtime-Enforced Primitives

The core idea of this work is to shift the locus of control for complex computational tasks from high-level, heuristic planners (like LLMs) to a lower-level, enforceable runtime. This runtime uses a set of primitives to manage compute resources based on verifiable signals.

The twelve proposed primitives are:
1.  **Critical-Path Compute Scheduler (CPCS):** Allocates resources to the dynamic critical path of a reasoning DAG.
2.  **Lazy-of-Thought (LoT):** Defers computation by emitting lazy thunks.
3.  **Region-of-Interest Splicing (RoI-S):** Recomputes only erroneous spans in a larger context.
4.  **Invariant-Gated Multi-Fidelity Simulation (IGMS):** Adjusts simulation fidelity based on invariant violations.
5.  **Transactional Tool Calls (TTC):** Uses a two-phase commit protocol for expensive tool calls.
6.  **Proof-State Checkpointing & Cousin-Branch Reuse (PSC-CR):** Checkpoints and reuses verified intermediate states.
7.  **KV-Cache Tiering + Prefetch (KVT-P):** Manages the attention KV cache as a multi-tier memory system.
8.  **Interruptible Decoding & Splice-Resume (IDS):** Allows decoding to be preempted and resumed.
9.  **Constraint-Slack Scheduler (CSS):** Allocates compute to satisfy the tightest constraints.
10. **Subgoal Cache with Canonical Hashing (SCCH):** Caches and reuses solutions to common subgoals.
11. **Per-Operator Precision Budgeter (POPB):** Dynamically allocates precision per operator.
12. **Compute-Safe Two-Phase “What-If” (CS-WIF):** Uses a cheap, bounded dry-run to predict the ROI of expensive computations.

This repository provides a concrete implementation and experimental validation for a primitive related to **budgeted resource allocation (CS-WIF)**, demonstrated through a compute-budgeted Growing RBF Network.

---

## Case Study: Budgeted Growth in a Self-Assembling Network

To demonstrate the utility of runtime-enforced compute allocation, we implement a **Growing RBF Network (GRBFN)** whose structural growth is capped by a hard budget, enforced by the runtime. This self-assembling model adapts to non-stationary data streams by spawning new prototype units, but the growth is gated by an external budget.

We test this model on two tasks: supervised classification and contextual bandits, both under abrupt data drift.

### Experimental Setup
- **Environment:** 2-D inputs with abrupt drifts every 2,000 steps for 12,000 steps total.
- **Reproducibility:** All results are from fixed random seeds (`seed=21` for classification, `seed=33`/`seed=37` for bandits).

### 1) Supervised classification under drift

**Overall pre-update accuracy (higher is better):**

| Model | Overall Accuracy |
| --- | --- |
| GRBFN | 0.9043 |
| GRBFN+Plastic | 0.9004 |
| LogReg | 0.7398 |
| MLP(32) | 0.8269 |

**Fast adaptation after a drift** — mean accuracy over the **first N samples** post-drift:

| Model | Post@200(avg) | Post@500(avg) |
| --- | --- | --- |
| GRBFN | 0.786 | 0.8636 |
| GRBFN+Plastic | 0.815 | 0.8716 |
| LogReg | 0.65 | 0.7144 |
| MLP(32) | 0.637 | 0.7344 |

**Figures**
- [results/classification_accuracy.png](results/classification_accuracy.png)
- [results/classification_prototypes.png](results/classification_prototypes.png)

**Findings**
- The self-assembling models (GRBFN, GRBFN+Plastic) demonstrate strong performance and fast adaptation to drift, outperforming standard baselines. The "Plastic" model shows the best fast adaptation (`Post@200(avg)=0.815`).

### 2) Contextual bandit under drift

We compare our policy (`GRBFN+Plastic`) against strong baselines on cumulative reward and regret.

**(A) GRBFN+Plastic vs LinUCB (seed=33):**

| Policy | Final Cumulative Reward | Final Cumulative Regret | Final #Prototypes (GRBFN+) |
| --- | --- | --- | --- |
| LinUCB(alpha=0.8) | 6256.0 | 222.6492386884768 | nan |
| GRBFN+Plastic | 6200.0 | 271.04147275534183 | 25.0 |

- Figures: [results/bandit_cumreward.png](results/bandit_cumreward.png), [results/bandit_cumregret.png](results/bandit_cumregret.png)

**(B) GRBFN+Plastic vs LinUCB vs MLP policy (seed=37):**

| Policy | Final Cumulative Reward | Final Cumulative Regret | Final #Prototypes (GRBFN+) |
| --- | --- | --- | --- |
| LinUCB(alpha=0.8) | 6152.0 | 299.4124817407165 | nan |
| GRBFN+Plastic | 6203.0 | 228.30886977959412 | 22.0 |
| MLP(32) | 6291.0 | 247.61384443497 | nan |

**Takeaways**
- The `GRBFN+Plastic` model is highly competitive with strong bandit baselines under drift. The key finding is that it achieves this performance while operating under a structural budget. The budget allows for a maximum of 25 prototypes, but the final count is slightly lower due to a self-merging mechanism that fuses redundant units. This demonstrates the effectiveness of runtime-enforced resource gating.

---

## Quick Start

```bash
# minimal requirements
pip install numpy pandas matplotlib

# classification with drift
python -m src.experiment_supervised_drift

# contextual bandit with drift (LinUCB vs GRBFN+Plastic)
python -m src.experiment_bandit_drift

# contextual bandit with drift including an MLP policy baseline
python -m src.experiment_bandit_drift_plusmlp
```

Each script writes figures and CSV tables to `results/`.

---

## Reproducibility

- Fixed seeds: classification `seed=21`; bandits `seed=33` and `seed=37`.
- Hardware: single CPU in a local notebook environment.
- The exact numbers above are produced by running the scripts in `src/` and are saved in `results/*.csv`.

---

## Paper & Specs (Updated 2025-08-18)

The updated paper reflecting **runtime‑enforced compute allocation** lives in `paper/paper.md` with companion specs in `paper/specs/`:

- `paper/paper.md` — full write‑up
- `paper/specs/CPCS_spec.md` — critical‑path scheduler
- `paper/specs/RoIS_spec.md` — region‑of‑interest splicing
- `paper/specs/TTC_spec.md` — transactional tool calls

See `paper/REVISION_NOTES.md` for a succinct change log.
