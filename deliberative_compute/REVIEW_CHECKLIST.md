# Review Checklist — Deliberative / Test‑Time Compute Papers

This checklist helps ensure your manuscript is crystal‑clear, reproducible, and fairly compared.
It is organized by what reviewers typically look for.

## 1) Problem & Contribution
- [ ] One‑sentence problem statement (what capability gap does TTC address here?).
- [ ] Define the TTC *operator(s)* used: generate (N), verify, revise, tool‑use, planner, etc.
- [ ] State the *allocation policy* (fixed N, adaptive early‑exit, tournament/knockout, etc.).
- [ ] Claim(s) are specific, measurable, and tied to concrete deltas over strong baselines.

## 2) Method
- [ ] Exact compute budget per example (tokens, passes, verifier calls, tool invocations).
- [ ] Clear pseudo‑code or algorithm box with stopping criteria and failure modes.
- [ ] Verifier description (training data, calibration, thresholding, length bias controls).
- [ ] Revision loop details (when to revise; max depth; acceptance checks).
- [ ] Any external tools or retrievers documented (APIs, latency, caches).

## 3) Baselines & Fairness
- [ ] Greedy CoT, self‑consistency / best‑of‑N, verifier‑guided selection included.
- [ ] Compute‑matched comparisons (same model family, prompt, temperature, tokens).
- [ ] Parameter‑matched comparisons (when claiming TTC vs model scaling tradeoffs).
- [ ] Strong domain baselines (math, code, QA, planning, agents, etc., as appropriate).

## 4) Evaluation Protocol
- [ ] Datasets/splits fixed and versioned; license/usage clarified.
- [ ] Metrics include *quality* and *efficiency* (quality per unit compute / wall‑clock).
- [ ] At least one *scaling curve* vs TTC (samples/passes/tokens); include diminishing returns.
- [ ] Confidence intervals or bootstrapped CIs; number of seeds; variance reported.
- [ ] Robustness tests: perturbations, distribution shift, distractors or adversarial cases.

## 5) Ablations & Analysis
- [ ] Ablate N (samples), verifier strength, revision depth, and allocation policy.
- [ ] Report failure taxonomy (where TTC helps; where it hurts / inverse scaling cases).
- [ ] Error analysis with representative examples and short qualitative diagnosis.

## 6) Reproducibility
- [ ] Release prompts, decoding configs, and seeds.
- [ ] Provide scripts to regenerate figures/logs; pin package versions and CUDA/cuDNN.
- [ ] Include a **Compute Budget Table** (see template below) and hardware notes.
- [ ] If applicable: carbon/energy estimate and caching policy disclosure.

## 7) Writing & Presentation
- [ ] Figures are legible at 100%; include units and axis labels; use `booktabs` for tables.
- [ ] Consistent notation and cross‑references (`\cref` for theorems/figures/tables/algs).
- [ ] Limitations & societal impacts discussed; safety implications if behavior changes under TTC.

---

### Compute Budget Table (template)

| Task | Model | Prompt | N (samples) | Verifier? | Revise Depth | Tokens (gen+ctx) | Calls | Wall‑clock (s/ex) | GPU(s) | Notes |
|------|-------|--------|-------------|-----------|--------------|------------------|-------|-------------------|--------|-------|
| ex: GSM8K | 8B‑Instruct | CoT‑v2 | 16 | PRM‑L | 1 | 3.8k | 17 | 2.3 | 1×A100 | temp=0.7, top‑p=0.9 |

**Notes**
- *Tokens*: count all generated tokens, including drafts and revisions; include verifier input.
- *Calls*: total LLM invocations (generation + verifier + tool calls).
- Use the same prompt template and temperature for compute‑matched baselines.

---

### Figure Checklist for TTC Scaling Plots
- [ ] X‑axis = TTC (e.g., N, tokens, passes); Y‑axis = quality metric.
- [ ] Shaded CIs; annotate key regimes (diminishing returns / inverse scaling).
- [ ] Overlay compute‑matched baseline curves (e.g., bigger model with same budget).

---

### Releasing Artifacts
- [ ] `requirements.txt` / `environment.yml` with exact versions.
- [ ] `scripts/run_eval.sh` with all flags; `scripts/plot_scaling.py` to reproduce figures.
- [ ] `README` with end‑to‑end commands (download → eval → plot).

---

### Common Pitfalls
- Reporting only accuracy without compute; unclear verifier training; moving goalposts across baselines.
- Not separating *generation diversity* gains from *verification* gains.
- Unreported caching or inconsistent stopping rules across methods.

---

If you adopt this checklist, consider including it in your repo for coauthors to self‑audit before submission.
