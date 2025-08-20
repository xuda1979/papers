# Runtime-Enforced Control for AI

This repository contains the code and experiments for the paper "Runtime-Enforced Control for AI: A Case Study in Budgeted Growth."

> ## Abstract
>
> Modern AI systems require fine-grained control over their computational resources, yet this control is often delegated to the models themselves, making it heuristic and unverifiable. We argue for a paradigm shift: moving the locus of control to a **runtime** that enforces explicit, verifiable constraints on compute. We introduce a vocabulary of twelve runtime-enforced primitives for steering computation in complex reasoning and simulation tasks. To demonstrate this principle, we conduct a case study on one such primitive: a **runtime-enforced growth budget** on a self-assembling neural network. By deploying a Growing RBF network in a non-stationary bandit task, we show that an external budget effectively gates structural growth while maintaining high performance. This provides concrete evidence that runtime-enforced mechanisms are a powerful and practical tool for governing adaptable AI.

---

## The Vision: Runtime-Enforced Primitives

The core idea of this work is to shift the locus of control for complex computational tasks from high-level, heuristic planners (like LLMs) to a lower-level, enforceable runtime. This runtime uses a set of primitives to manage compute resources based on verifiable signals. The twelve proposed primitives are:

1.  **Critical-Path Compute Scheduler (CPCS)**
2.  **Lazy-of-Thought (LoT)**
3.  **Region-of-Interest Splicing (RoI-S)**
4.  **Invariant-Gated Multi-Fidelity Simulation (IGMS)**
5.  **Transactional Tool Calls (TTC)**
6.  **Proof-State Checkpointing & Cousin-Branch Reuse (PSC-CR)**
7.  **KV-Cache Tiering + Prefetch (KVT-P)**
8.  **Interruptible Decoding & Splice-Resume (IDS)**
9.  **Constraint-Slack Scheduler (CSS)**
10. **Subgoal Cache with Canonical Hashing (SCCH)**
11. **Per-Operator Precision Budgeter (POPB)**
12. **Compute-Safe Two-Phase “What-If” (CS-WIF)**

This repository provides a concrete implementation and experimental validation for a primitive that combines principles from **CSS** and **CS-WIF** to enforce a hard budget on a model's structural growth.

---

## Case Study: Budgeted Growth in a Self-Assembling Network

To demonstrate the utility of runtime-enforced compute allocation, we implement a **Growing RBF Network (GRBFN)** whose structural growth is capped by a hard budget, enforced by a `RuntimeGate`. This self-assembling model adapts to non-stationary data streams by spawning new prototype units, but the growth is gated by an external, verifiable budget.

We test this `GRBFN+Plastic` model on supervised classification and contextual bandits, both under abrupt data drift. The key finding is that the model is highly competitive with strong baselines while operating under a strict structural budget, demonstrating the effectiveness of runtime-enforced resource gating.

### Key Results (from `seed=37` bandit experiment)

| Policy | Final Cumulative Reward | Final Cumulative Regret | Final #Prototypes (GRBFN+) |
| :--- | :--- | :--- | :--- |
| LinUCB | 6152.0 | 299.41 | nan |
| **GRBFN+Plastic** | **6190.0** | **239.16** | **22.0** |
| MLP(32) | 6291.0 | 247.61 | nan |

**Figures**
- [Cumulative Reward](results/bandit_plusmlp_cumreward.png)
- [Cumulative Regret](results/bandit_plusmlp_cumregret.png)

---

## Quick Start

### 1. Installation

First, install the necessary dependencies.

```bash
pip install numpy pandas matplotlib seaborn
```

### 2. Running Experiments

All experiments are managed by a single, unified runner script: `neural_organism/src/runner.py`. You can choose which experiment to run by passing its name as a command-line argument.

The available experiments are:
- `supervised`: Supervised classification under drift.
- `bandit`: Contextual bandit vs. LinUCB.
- `bandit_plusmlp`: Contextual bandit vs. LinUCB and an MLP.

**Example:**

```bash
# Run the main bandit experiment cited in the paper
python -m neural_organism.src.runner bandit_plusmlp

# Run the supervised classification experiment
python -m neural_organism.src.runner supervised
```

Each script will print summary tables to the console and write figures and CSV files to the `neural_organism/results/` directory.

### 3. Configuration

Experiment hyperparameters (seeds, learning rates, etc.) are centralized in `neural_organism/src/config.py`. You can modify this file to easily define and run new experiments.

---

## Repository Structure

- `paper/`: Contains the full paper (`paper.md`) and detailed specs for some primitives.
- `src/`: The source code.
  - `runner.py`: The main entry point for all experiments.
  - `config.py`: Centralized configuration for all experiments.
  - `growing_rbf_net_plastic.py`: The core self-assembling model.
  - `baselines.py`: Baseline models (LinUCB, MLP, etc.).
  - `data_generators.py`: Code for generating non-stationary data.
  - `experiment_utils.py`: Plotting and result-saving utilities.
- `results/`: Output directory for figures and CSVs.

---

## Paper & Specs

The updated paper reflecting our vision for runtime-enforced control lives in `paper/paper.md`, with companion specs in `paper/specs/`.
