---
title: Runtime–Enforced Compute Allocation for Reasoning and Simulation
subtitle: Below-Orchestrator Primitives that an LLM Cannot Subsume
date: 2025-08-18
authors: Your Team
---

> **Revision note (2025-08-18)** — This version replaces any hand‑wavy “let an LLM decide compute” policy with **runtime primitives** and **enforceable mechanisms** that sit **below** any planner/orchestrator (LLM or not). These mechanisms use OS/runtime/hardware controls and externally verified signals. They are practical to build now and directly useful for **math/algorithms/theoretical physics with simulations**.

## Abstract

We present twelve **runtime-enforced primitives** for allocating compute in math, code, and physics workloads. Each primitive operates below any planner or large language model, wielding OS, runtime, or hardware mechanisms to **preempt, gate, or splice** computation deterministically. To demonstrate the approach, we deploy a **compute-budgeted Growing RBF network** in a drifting contextual bandit. A runtime growth gate caps structural expansion while retaining reward, highlighting how verifiers can steer FLOPs toward the most promising reasoning branches. Together these primitives form a minimal architecture that lets external signals decide where compute is spent, enabling reproducible control over long reasoning chains and multi-fidelity simulations.

## 1. Motivation

Heuristic, policy‑only approaches (e.g., “smart LLM allocators”) cannot *enforce* where compute goes, nor preempt live kernels, nor gate expensive calls against verifiers. We therefore shift the locus of control to the **runtime**: the place that owns **GPU streams, memory, processes, and verifiers**.

## 2. Minimal Glue Architecture

```
[Planner (can be an LLM)]  <--suggestions-->
[Coordinator Runtime]
   ├─ Critical-Path Scheduler (CPCS)
   ├─ Transaction Manager (TTC)
   ├─ Constraint Engine (units/types/invariants)
   ├─ Cache Services (PSC-CR, SCCH, KVT-P)
   └─ Decode Engine (IDS, RoI-S, POPB)

Enforcement path: Coordinator owns GPU streams, KV memory, and tool processes.
Planner cannot overrule preemption or hard gates once set by the runtime.
```

## 3. Runtime Primitives (Build‑Now)

Each primitive below is (i) practical today, (ii) hard for a generic LLM allocator to subsume because it relies on **OS/runtime/hardware** or **verifier‑level** control, and (iii) immediately useful for math/algorithms/physics.

### 3.1 Critical‑Path Compute Scheduler (CPCS) for Reasoning DAGs

**Idea.** Convert the evolving proof/code/sim plan into a DAG with predicted durations and success probabilities; allocate GPU/CPU/tool time to the **dynamic critical path** (maximize expected makespan reduction), not merely the next step the planner suggests.

**Mechanism (enforceable).**
- Each subgoal \(g\) carries ETA \(\hat{t}(g)\) and success probability \(p_\text{succ}(g)\) from *cheap probes* (verifier hints, compile/sanity checks, tiny sim pilots).
- Recompute the critical path periodically and **preempt** off‑path decoding/sim jobs via CUDA stream priorities and OS cgroups.
- Budgets are wall‑clock + FLOPs; scheduler ticks every \(~50–200\) ms.

**Prototype.** Tasks as JSON; a central async scheduler sets stream priorities and CPU affinities; predictors update \(\hat{t}, p_\text{succ}\).

**Why policy alone can’t replace it.** Preemption & stream priorities are **physical runtime controls**; a planner can suggest, but can’t seize GPU streams mid‑kernel.

### 3.2 Lazy‑of‑Thought (LoT): Thunks + Selective Forcing

**Idea.** The model emits **lazy thunks** (e.g., `⟦proveLemma(name, preconds)⟧`) instead of expanding everything. A demand‑driven executor **forces** only thunks whose values are required by a verifier or downstream dependency.

**Mechanism.**
- Decoder supports special thunk tokens with cost estimates.
- A demand executor calls tools/LLMs **only when demanded** (laziness for reasoning).
- **Splicing:** when a thunk is forced, its expansion replaces the placeholder in KV/context via targeted edit—no full re‑generation.

**Prototype.** Regex pass that turns “(details omitted)” into a thunk; simple demand rule: if a verifier can’t proceed, force the relevant thunk.

**Why not an LLM allocator.** Laziness is a **runtime evaluation strategy** requiring KV surgery and dependency tracking.

### 3.3 Region‑of‑Interest Splicing (RoI‑S): Recompute Only the Broken Span

**Idea.** When a checker flags an error, **regenerate only that span** with higher beams/model and splice it back, preserving caches for the rest.

**Mechanism.**
- RoI detector from verifier deltas (failed algebra step, invariant breach, unit mismatch).
- KV‑cache **windowed rewind** to span start; keep prefix/suffix; re‑decode RoI; **stitch** KV blocks back deterministically.

**Prototype.** Track token offsets per step; failure → map to `(start,end)`; controlled partial decode with stop‑at token; join.

**Why policy can’t replace it.** Needs **cache‑level rollback** and deterministic splice.

### 3.4 Invariant‑Gated Multi‑Fidelity Simulation (IGMS)

**Idea.** Maintain symbolic invariants/units (mass/energy, dimensionality). Run coarse sims by default; **escalate fidelity locally** only where an invariant is violated.

**Mechanism.**
- Extract invariants/units from the prompt or auto‑generated PDE/ODE forms.
- Watchdog computes local residuals (conservation error, CFL breach) → triggers **local mesh/time‑step refinement** or higher‑precision kernels for just that region/time window.

**Prototype.** Wrap sim with hooks: `residual(x,t)`, `refine(cell)` and a simple invariant checker.

**Why policy can’t replace it.** This is **numerical runtime control** tied to solver kernels and grid management.

### 3.5 Transactional Tool Calls (TTC) with Two‑Phase Commit

**Idea.** Treat big external ops (hour‑long sim, SAT solve) as **transactions**: *prepare* (check inputs, bounds), then only *commit* if a consumer will use it; otherwise **abort** to save compute.

**Mechanism.**
- Call metadata includes **preconditions** and a **consumer list**.
- If the plan changes, coordinator sends `ABORT` before the heavy run; no orphaned results.

**Prototype.** Thin coordinator library (`prepare()`, `commit()`, `abort()`) + per‑tool adapters. Decisions are logged for audit.

**Why policy can’t replace it.** Requires **protocol & shared state** across processes.

### 3.6 Proof‑State Checkpointing & Cousin‑Branch Reuse (PSC‑CR)

**Idea.** Canonicalize intermediate math states (normalized expressions, solved sub‑lemmas) and **checkpoint** them. On a new attempt, **reuse** nearest checkpoints (cousin branches) rather than recompute.

**Mechanism.**
- Hash‑cons canonical forms (sorted monomials, α‑equivalent proof terms).
- On failure: restart from last **verified** checkpoint; on new branches: **graft** reusable subproofs.

**Prototype.** SymPy/Lean‑style normal forms; small store mapping canonical key → proof artifact.

**Why policy can’t replace it.** **Content‑addressable storage + verified grafting** is a runtime/data‑plane capability.

### 3.7 KV‑Cache Tiering + Prefetch (KVT‑P)

**Idea.** Treat attention KV as a multi‑tier memory (HBM ↔ RAM ↔ NVMe). **Predict hot ranges** and prefetch them; demote cold ranges.

**Mechanism.**
- Lightweight predictor (n‑gram/structure) anticipates future attention spans.
- Async DMA moves KV blocks; decoder blocks only on misses.

**Prototype.** Block KV into 32–64‑token chunks; ring‑buffer with async copy threads; LRU+structure eviction.

**Why policy can’t replace it.** This is a **memory system design**.

### 3.8 Interruptible Decoding & Splice‑Resume (IDS)

**Idea.** Make decoding **preemptible** across agents. If a fast tool answers a subgoal, **interrupt**, inject the result, and **resume** without restarting.

**Mechanism.**
- Run decoding in CUDA streams with priorities.
- Maintain resumable checkpoints (KV + RNG state). On interrupt: pause kernels, edit context, resume.

**Prototype.** Torch with CUDA Graphs + cooperative kernels; `pause()`, `resume(ctx_delta)` API.

**Why policy can’t replace it.** Requires **kernel‑level preemption** and state capture.

### 3.9 Constraint‑Slack Scheduler (CSS)

**Idea.** Maintain a set of **hard constraints** (types, units, side‑conditions). Compute **slack** per constraint and spend compute where slack is smallest (lexicographic max‑min).

**Mechanism.**
- Constraint engine yields \(\text{slack}_i\).
- Scheduler funds actions that increase minimum slack: “tighten step k,” “re‑derive lemma L,” or “increase sim precision in cell c.”

**Prototype.** Units/type checker returns per‑span slack; candidates are prioritized by slack gain.

**Why policy can’t replace it.** It optimizes **external constraint metrics**, not token probabilities.

### 3.10 Subgoal Cache with Canonical Hashing (SCCH)

**Idea.** Live **cross‑query** cache of subgoals (integrals, transforms, lemmas). Allocate compute to **reusable** subgoals; hit‑ratio drives priority.

**Mechanism.**
- Canonicalize subgoals (e.g., \(\int f(ax+b)\,dx\) normalized).
- Admission policy favors subgoals observed across projects; eviction by reuse half‑life.

**Prototype.** Redis key = canonical form; store proof/code + verifier hash; planner consults cache first.

**Why policy can’t replace it.** This is **workload‑level amortization** with persistence and grafting.

### 3.11 Per‑Operator Precision Budgeter (POPB)

**Idea.** Choose precision **per operator** (attention/MLP/layernorm) and **per token group** at runtime based on sensitivity probes; reserve fp16/bf16 (or higher) *only where needed*.

**Mechanism.**
- Run a one‑step low‑cost probe (Jacobian norm / entropy change) to tag hot spots.
- Launch mixed‑precision kernels with a per‑block mask.

**Prototype.** bitsandbytes/int4 + custom kernels; measure micro‑loss change to classify blocks.

**Why policy can’t replace it.** Needs **kernel choices and per‑block masks**.

### 3.12 Compute‑Safe Two‑Phase “What‑If” (CS‑WIF)

**Idea.** Before expensive escalations, run a **shadow, bounded “what‑if”** dry‑run with stubs (fast mock sim) to **predict ROI**; escalate only if ROI clears a threshold.

**Mechanism.**
- Plug‑in stubs: symbolic bound calculators, coarse sim surrogates, unit checkers.
- The coordinator **refuses** escalation unless the predicted delta passes a gate.

**Prototype.** Micro‑surrogates for PDEs (tiny grids), stub SAT counterexample samplers; simple ROI rule.

**Why policy can’t replace it.** A **hard gate** enforced by the runtime.

## 4. How These Win in Practice

- **Math / proofs.** PSC‑CR + RoI‑S slash retries by repairing only broken spans and reusing verified lemmas. CSS drives compute to the tightest contradictions first.
- **Algorithms / coding.** TTC prevents long solvers you won’t consume; IDS splices quick unit tests/results mid‑decode; our budgeted GRBF network keeps reward steady under prototype caps.
- **Physics sims.** IGMS localizes fidelity upgrades; CPCS funnels GPUs to time‑critical cell blocks; KVT‑P makes ultra‑long contexts feasible (derivations + logs).

## 5. Interfaces & Snippets

### 5.1 RoI Splice (partial re‑decode)

```python
def repair_span(model, kv, tokens, start, end, beam=8, max_new=128):
    prefix = tokens[:start]
    suffix = tokens[end:]
    kv_prefix = kv.slice(:start)             # keep
    kv_repair  = kv_prefix.clone()           # rollback state
    repair = decode(model, kv_repair, prefix, stop_at=end, beam=beam, max_new=max_new)
    new_tokens = prefix + repair + suffix
    kv_new = kv_prefix.extend(repair).attach_suffix(suffix)  # stitch
    return new_tokens, kv_new
```

### 5.2 Two‑Phase Tool Call

```python
call_id = tool.prepare(inputs, est_cost, consumers=[step42])
if coordinator.will_consume(call_id):
    result = tool.commit(call_id)
    coordinator.attach(result, step42)
else:
    tool.abort(call_id)  # nothing heavy executed
```

### 5.3 Budgeted Prototype Growth

```python
net = GrowingRBFNet(d=2, k=3, growth_budget=32)
for x, y in stream:
    net.update(x, y)
    if net.growth_budget is not None and net.growth_budget <= 0:
        break  # runtime gate stops further structural growth
```

## 6. Fast Validation Plan (2–3 days per item)

- **RoI‑S + PSC‑CR.** Long algebra proofs (200–800 tokens). Metrics: tokens regenerated per fix ↓, end‑to‑end latency ↓, accuracy ≥ baseline.
- **IGMS.** 2D diffusion / shallow‑water PDE mini‑cases. Metrics: same error with ≤40% sim time; localized refinement count vs. AMR baseline.
- **TTC.** Mix of SAT solves and long sims. Metrics: orphaned heavy runs → ~0; wall time ↓ when plans change.
- **CPCS.** Multi‑subgoal proofs with varied ETAs. Metrics: makespan and GPU utilization vs. FIFO/LLM‑driven sequencing.
- **KVT‑P.** 32k–128k contexts. Metrics: HBM footprint ↓ with similar throughput; stall time bounded.
- **IDS.** Mixed agents on one GPU. Metrics: preempt latency <50 ms; saved recompute on early tool completion.

## 7. Conclusion

All 12 mechanisms are **runtime‑enforced**. A planner (LLM or otherwise) can still *advise*, but cannot replace controls that manipulate **streams, memory, transactions, constraints, and verifiers**.
