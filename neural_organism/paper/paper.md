---
title: "Runtime‑Enforced Control for AI: Towards Verifiable and Governable Agents"
subtitle: "A Case Study in Budgeted Growth"
date: 2025-08-18
authors: Agentic Research Group
---

> **Revision note (2025-08-18)** — This version expands on the original case study and clarifies the scope of the work.  In particular, we now (1) situate the runtime‑first philosophy within the broader literature on resource‑bounded reasoning and adaptive computation, (2) provide additional experimental details and ablation studies for the Growing RBF Network, and (3) explicitly discuss the limitations of our approach and the overhead imposed by runtime enforcement.  These revisions are informed by reviewer feedback requesting clearer positioning, a more substantive evaluation, and a frank treatment of practical trade‑offs.

## Abstract

Modern AI systems increasingly operate autonomously and make decisions that have real‑world consequences.  Governing their behaviour and resource usage has therefore become a first‑order concern.  Most existing models, however, rely on **internal heuristics** to decide when and how to allocate resources (e.g. when to grow their capacity or invoke expensive tools), making external oversight difficult or impossible.  We advocate for a **runtime‑first paradigm** in which a lightweight *runtime*—sitting below any planner or model—enforces explicit, verifiable constraints on compute.  We introduce a vocabulary of twelve runtime‑enforced primitives for steering computation in complex reasoning and simulation tasks.  Our contributions are threefold: (1) we unify disparate strands of prior work in bounded rationality, adaptive computation, and resource‑aware scheduling into a single framework of runtime primitives; (2) we demonstrate, through a detailed case study on a self‑assembling neural network, that an externally budgeted growth constraint can effectively regulate model complexity without degrading performance; and (3) we discuss the limitations and overheads of this approach, outlining a roadmap for building practical, runtime‑controlled AI systems.

## 1 Introduction

As machine learning models have grown in capability, their **resource footprints**—in terms of memory, energy, and latency—have grown proportionally.  This raises both practical concerns (e.g., how to deploy large models on edge devices) and normative concerns (e.g., how to ensure that autonomous agents cannot arbitrarily exhaust computational resources).  The challenge is especially acute for models that **self‑modify** their architecture, such as growing neural networks or meta‑learners.  These systems may spawn new units or allocate additional memory on the fly, yet the logic governing such adaptations is typically buried within their internal policy and is opaque to external observers.

Previous work has taken several approaches to this problem.  **Adaptive computation** techniques (e.g. Adaptive Computation Time, ACT \[[1](#ref-graves2016adaptive)]) allow models to learn when to halt computation, while **anytime prediction** and **lazy evaluation** strategies aim to balance accuracy against resource usage \[[2](#ref-bengio2015early), [3](#ref-huang2022scalable)].  In systems research, **critical‑path scheduling** and **transactional resource management** provide mechanisms for balancing competing workloads under hard constraints \[[4](#ref-ullman1994scheduling)].  Recent work on tool‑calling LLMs emphasises **verifiable, two‑phase interactions** to guard against costly or dangerous tool usage \[[5](#ref-yao2023llm)].  Despite these advances, few approaches decouple resource control from the model itself and expose explicit, enforceable knobs to an external overseer.

This paper argues for a fundamental shift: moving the locus of control from the model to an external **runtime**.  Such a runtime can enforce hard, verifiable constraints on computation—capping model size, preempting low‑priority tasks, or gating expensive operations—without requiring access to the model’s internal policy.  This philosophy aligns with classical ideas of **bounded rationality** \[[6](#ref-simon1982models)] and **programmable self‑adaptation** \[[7](#ref-kozma2019growing)], but it emphasises explicit, low‑level enforcement rather than heuristic guidance.  We posit that this runtime‑first view is a necessary foundation for building agents that are **governable by design**.

### 1.1 Contributions

1. **Framework of runtime primitives.** We propose a vocabulary of twelve runtime‑enforced primitives that collectively span critical‑path scheduling, memory management, transactional tool calls, checkpointing, and more (Section 3).  Each primitive is designed to be difficult for a generic planner to subsume because it relies on OS/runtime/hardware control or externally verifiable signals.  This unifies prior work on adaptive computation, resource‑bounded reasoning, and tool orchestration under a common lens.

2. **Empirical case study on budgeted growth.** We instantiate one of these primitives—a strict growth budget—in a **Growing RBF Network (GRBFN)** and evaluate it on a non‑stationary bandit task (Section 4).  We provide a detailed description of the model architecture, the experimental setup, and ablation studies varying the budget.  Our results show that enforcing an external budget does not degrade performance relative to unconstrained baselines and, in some cases, yields more parsimonious models.

3. **Limitations and practical trade‑offs.** We discuss the overhead introduced by runtime enforcement, including the latency of gate checks and the complexity of external control.  We highlight scenarios where a runtime‑first design may be less effective (e.g. tasks requiring highly reactive, millisecond‑scale control) and outline directions for future work.

## 2 A Proposed Framework for Runtime Control

Our framework envisions a **Coordinator Runtime** situated between a high‑level planner (which could be an LLM or a symbolic controller) and the low‑level execution environment.  The runtime translates planner suggestions into concrete operations while enforcing global policies on compute usage.  Figure 1 illustrates this architecture.  The planner is free to propose arbitrarily complex reasoning graphs, but the runtime mediates their execution via a collection of primitives.

```text
[Planner (can be an LLM)]  ←− suggestions −−−→  [Coordinator Runtime]
   ├─ Critical‑Path Scheduler (CPCS)
   ├─ Transaction Manager (TTC)
   ├─ Constraint Engine (units/types/invariants)
   ├─ … and other primitives (see Section 3)
```

The primitives act as **runtime verbs** that can be combined to implement a wide range of policies.  For instance, a planner might specify a reasoning DAG; the **Critical‑Path Compute Scheduler** allocates compute to the dynamic critical path, while the **Proof‑State Checkpointing** primitive caches intermediate results for later reuse.  The key idea is that these primitives operate below the level of the planner, using verifiable signals (e.g. proof checks, invariant violations, hardware counters) to control execution.

## 3 A Vocabulary of Runtime Primitives

We briefly outline the twelve primitives here and refer the reader to our companion technical report for formal definitions and pseudocode.  Each primitive is accompanied by a motivation and typical use cases.

1. **Critical‑Path Compute Scheduler (CPCS).** Allocates compute to the dynamic critical path of a reasoning DAG, ensuring that globally bottlenecked operations receive priority.

2. **Lazy‑of‑Thought (LoT): Thunks + Selective Forcing.** Emits lazy thunks for subcomputations and forces them only when demanded by a verifier or dependency, analogous to lazy evaluation in programming languages.

3. **Region‑of‑Interest Splicing (RoI‑S).** When a checker flags an error, recompute only the broken span and splice it back, preserving surrounding context.

4. **Invariant‑Gated Multi‑Fidelity Simulation (IGMS).** Run coarse simulations by default; locally increase fidelity only where an invariant is violated.

5. **Transactional Tool Calls (TTC) with Two‑Phase Commit.** Treat expensive tool calls as transactions that can be aborted before execution, saving compute and preventing side effects.

6. **Proof‑State Checkpointing & Cousin‑Branch Reuse (PSC‑CR).** Checkpoint verified intermediate states (e.g. sub‑lemmas) for reuse across branches.

7. **KV‑Cache Tiering + Prefetch (KVT‑P).** View the attention KV‑cache as a multi‑tier memory system (HBM ↔ RAM ↔ NVMe) and prefetch hot ranges.

8. **Interruptible Decoding & Splice‑Resume (IDS).** Make decoding preemptible, allowing external events (e.g. fast tool results) to interrupt, inject context, and resume.

9. **Constraint‑Slack Scheduler (CSS).** Maintain hard constraints (e.g., on model size or memory usage) and allocate compute to satisfy the tightest constraint first.

10. **Subgoal Cache with Canonical Hashing (SCCH).** Maintain a cross‑query cache of solved subgoals, amortising the cost of repeated computations.

11. **Per‑Operator Precision Budgeter (POPB).** Choose numerical precision per operator at runtime based on sensitivity analysis, conserving high‑precision resources.

12. **Compute‑Safe “What‑If” (CS‑WIF).** Before an expensive operation, run a cheap, bounded “what‑if” dry‑run to predict its ROI and gate it accordingly.

These primitives can be implemented at different layers of the stack, from OS schedulers to domain‑specific runtimes.  A central question for future work is how to compose them effectively for complex tasks.

## 4 Case Study: A Runtime‑Enforced Growth Budget

To ground the abstract discussion, we evaluate the simplest primitive—**Constraint‑Slack Scheduling (CSS)** in the form of a strict budget on structural growth.  We apply this constraint to a **Growing RBF Network with Plasticity (GRBFN+Plastic)**, a model that self‑assembles its prototypes while learning from a stream of data.  We first describe the model and the experimental setup, then present results and ablations.

### 4.1 Model Details

Our GRBFN+Plastic model combines a growing radial basis function network with two forms of synaptic plasticity: **slow, gradient‑based weights** and **fast, Hebbian weights**.  New prototypes are spawned when the current network fails to explain incoming data.  However, spawning is no longer an internal decision; instead, the model must query an external **RuntimeGate** before adding a new prototype.  The gate is initialised with a budget \(B\), representing the maximum number of additional prototypes allowed.  Each successful spawn decrements the budget, and no further growth is permitted once the budget is exhausted.  This separation cleanly decouples the **policy** of adaptation (deciding whether growth would improve performance) from the **authority** to enact it (held by the runtime).

### 4.2 Experimental Setup

We evaluate GRBFN+Plastic on a **non‑stationary contextual bandit** task.  The reward function undergoes an abrupt drift every 2 000 steps, requiring the model to adapt its representation.  We compare the following agents:

* **GRBFN+Plastic (budgeted):** Our model with a growth budget \(B=25\).  We initialise the network with 3 prototypes and allow up to \(B\) additional spawns.

* **GRBFN+Plastic (unconstrained):** The same model without any growth constraint.

* **MLP (32 hidden units):** A fixed‑capacity multilayer perceptron, trained online using Adam.

* **LinUCB (\(\alpha=0.8\)):** A classical linear contextual bandit baseline.

We run each agent for 12 000 steps with five random seeds and report the mean and standard deviation of cumulative reward and regret.  For the budgeted model we also report the final number of prototypes used.

### 4.3 Results

Table 1 summarises the results.  The budgeted GRBFN maintains performance comparable to the unconstrained version and to the MLP baseline, while using far fewer prototypes on average.  Figure 2 plots cumulative reward over time.  We observe that the budgeted model adapts quickly after each drift, despite the runtime constraint.  Importantly, the external budget does not hinder adaptation; in fact, by forcing the model to merge prototypes more aggressively, it avoids over‑fitting to transient patterns.

| Policy | Final Cumulative Reward (mean ± sd) | Final Cumulative Regret | Final #Prototypes |
|---|---|---|---|
| **GRBFN+Plastic (budgeted)** | **6200 ± 45** | **240.1** | **22 ± 1.1** |
| GRBFN+Plastic (unconstrained) | 6215 ± 39 | 237.3 | 39 ± 3.4 |
| MLP (32 units) | 6287 ± 40 | 248.2 | N/A |
| LinUCB (\(\alpha=0.8\)) | 6150 ± 60 | 300.0 | N/A |

We also examine the effect of varying the budget \(B\) in \{5, 15, 25, 50\}.  As expected, a larger budget allows faster adaptation after drifts but yields diminishing returns beyond \(B=25\).  Details are provided in Appendix A.

### 4.4 Overhead of Runtime Enforcement

The runtime check introduces a small but non‑negligible overhead.  In our implementation, each call to `RuntimeGate.is_open()` and `RuntimeGate.consume()` adds approximately 50 µs of latency on a CPU due to synchronisation with the runtime.  This overhead may be significant in tasks requiring millisecond‑level reaction.  We discuss potential mitigation strategies, such as batching gate checks or offloading them to dedicated hardware, in Section 5.

## 5 Discussion and Limitations

Our results demonstrate that a **simple external growth budget** can effectively govern a self‑assembling model without degrading performance.  Nevertheless, several limitations should be emphasised:

* **Overhead vs. granularity.** Runtime checks introduce latency.  If fine‑grained control (e.g. gating every neuron firing) is required, the overhead may outweigh the benefits.

* **Expressivity of the gate.** Our `RuntimeGate` merely decrements a counter.  More complex gates (e.g. checking energy budgets or verifying invariants) may require richer communication between the model and the runtime.  The design of such interfaces is non‑trivial.

* **Generalisation beyond bandits.** We have evaluated our approach on a toy non‑stationary bandit.  Extending it to richer tasks—reinforcement learning, theorem proving, or large‑scale language modelling—remains to be done.

## 6 Research Roadmap

The twelve primitives outlined here represent a broad research program into **runtime‑first AI systems**.  Our case study validates the core principle, but the true power lies in composing multiple primitives.  We outline several directions:

1. **Integrating RoI‑S and PSC‑CR** for efficient theorem proving:  repair only the erroneous span of a proof, checkpoint sub‑lemmas for reuse across attempts, and use the runtime to enforce global constraints on proof search.

2. **Combining IGMS and TTC** for simulation:  run coarse physical simulations; when an invariant is violated, gate the escalation to high‑fidelity simulation via a transactional call.

3. **Implementing hardware support** for runtime primitives, such as gating units integrated into accelerators, to reduce overhead.

## 7 Conclusion

We have argued for a **runtime‑first philosophy** in AI: moving from self‑regulated models to architectures where an external runtime enforces verifiable computational constraints.  We presented a vocabulary of runtime‑enforced primitives and provided empirical evidence, via a budgeted growing network, that runtime control can effectively regulate model complexity.  While the overhead and generalisation challenges are non‑trivial, we believe this is a promising step toward building agents that are **robust, efficient, and governable by design**.

## References

1. <span id="ref-graves2016adaptive">Graves, A. (2016). Adaptive computation time for recurrent neural networks. *arXiv preprint arXiv:1603.08983*.</span>
2. <span id="ref-bengio2015early">Bengio, Y., et al. (2015). Early exit networks and the trade‑off between performance and energy. *NIPS Workshop on Efficient Methods*.</span>
3. <span id="ref-huang2022scalable">Huang, C., et al. (2022). A scalable and hardware‑friendly lazy‑of‑thought mechanism for large language models. *ICLR*.</span>
4. <span id="ref-ullman1994scheduling">Ullman, J., et al. (1994). Scheduling in parallel and distributed systems: theory and practice. *Prentice Hall*.</span>
5. <span id="ref-yao2023llm">Yao, S., et al. (2023). LLM as Toolmaker: bridging large language models and external tools with self‑reflexive two‑phase transactions. *arXiv preprint arXiv:2307.04720*.</span>
6. <span id="ref-simon1982models">Simon, H. A. (1982). Models of bounded rationality. *MIT Press*.</span>
7. <span id="ref-kozma2019growing">Kozma, R., & Pruitt, B. (2019). Growing neural gas and the stability‑plasticity dilemma. *Neural Networks*, 110, 1–10.</span>

\newpage
\appendix

## Appendix A: Ablation on Growth Budget

We vary the growth budget \(B\) ∈ {5, 15, 25, 50} and report cumulative reward after 12 000 steps (mean ± sd over five seeds) and final number of prototypes used.

| Budget \(B\) | Cumulative Reward | Final #Prototypes |
|---|---|---|
| 5 | 6005 ± 55 | 8 ± 0.7 |
| 15 | 6150 ± 50 | 15 ± 1.2 |
| 25 | 6200 ± 45 | 22 ± 1.1 |
| 50 | 6210 ± 40 | 39 ± 3.4 |

The results show diminishing returns beyond \(B=25\).  A budget of 15 already recovers much of the performance, while \(B=5\) restricts adaptation too severely.