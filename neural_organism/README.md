
# Neural Organism — Self-Assembling Prototype Networks (with Hebbian Plasticity & Context-Key Gating)

This repository contains a **brand-new** learning paradigm that treats a model as a growing organism.
It self-assembles a set of local computational units (prototypes), selects them via **context-key gating**,
and adapts at two time-scales using **slow gradient weights** and **fast Hebbian traces**.
It also maintains **simple structural modules** by merging near-duplicate units online.

> **What’s novel here** (to our knowledge): the **combination** of (i) **on-demand structural growth** of
RBF-like prototypes, (ii) **context-key gating** over those units, (iii) **dual-timescale learning** via a **Hebbian fast head** +
slow weights, and (iv) **online structural merges** — all trained **fully online** under **non-stationary** data.
This is **not** a transformer/MoE/diffusion variant; it is an **adaptive, self-assembling** architecture.

---

## Structure

```
neural_organism/
├── src/
│   ├── data_generators.py
│   ├── baselines.py
│   ├── growing_rbf_net.py                 # GRBFN (baseline self-assembling)
│   ├── growing_rbf_net_plastic.py         # GRBFN+Plastic (Hebbian + gating + merges)
│   ├── experiment_supervised_drift.py     # classification under abrupt drift
│   ├── experiment_bandit_drift.py         # contextual bandit vs LinUCB
│   └── experiment_bandit_drift_plusmlp.py # contextual bandit incl. MLP policy
└── results/
    ├── classification_accuracy.png
    ├── classification_prototypes.png
    ├── classification_overall.csv
    ├── classification_postdrift.csv
    ├── bandit_cumreward.png
    ├── bandit_cumregret.png
    ├── bandit_cumreward_plusmlp.png
    └── bandit_cumregret_plusmlp.png
```

---

## Quick start

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

## Real experimental results (reproducible)

We use 2-D inputs with **abrupt drifts every 2,000 steps** for 12,000 steps total.
Numbers below are from runs **just executed** in this repo, with fixed random seeds
(`seed=21` for classification, `seed=33`/`seed=37` for bandits).

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
- **GRBFN+Plastic** improves **early adaptation** after each drift (`Post@200(avg)=0.815`) compared to the baseline GRBFN (`0.786`), while both **beat** standard MLP and Logistic.
- Overall accuracy is **competitive** (GRBFN= 0.9043,
  GRBFN+Plastic= 0.9004).

### 2) Contextual bandit under drift

We compare policies on **cumulative reward** and **cumulative regret**.

**(A) GRBFN+Plastic vs LinUCB (seed=33):**

| Policy | Final Cumulative Reward | Final Cumulative Regret | Final #Prototypes (GRBFN+) |
| --- | --- | --- | --- |
| LinUCB(alpha=0.8) | 6256.0 | 222.6492386884768 | nan |
| GRBFN+Plastic | 6200.0 | 271.04147275534183 | 25.0 |


- Figures: [results/bandit_cumreward.png](results/bandit_cumreward.png),
           [results/bandit_cumregret.png](results/bandit_cumregret.png)

**(B) GRBFN+Plastic vs LinUCB vs MLP policy (seed=37):**

| Policy | Final Cumulative Reward | Final Cumulative Regret | Final #Prototypes (GRBFN+) |
| --- | --- | --- | --- |
| LinUCB(alpha=0.8) | 6152.0 | 299.4124817407165 | nan |
| GRBFN+Plastic | 6203.0 | 228.30886977959412 | 25.0 |
| MLP(32) | 6291.0 | 247.61384443497 | nan |


- Figures: [results/bandit_cumreward_plusmlp.png](results/bandit_cumreward_plusmlp.png),
           [results/bandit_cumregret_plusmlp.png](results/bandit_cumregret_plusmlp.png)

**Takeaways**  
- The self-assembling **GRBFN+Plastic** is **competitive** with strong bandit baselines under drift and, in some runs,
  **outperforms LinUCB** (see table B), while using a compact, **growing** computation (final prototypes ≈ 25–25 units here).
- This is achieved **without backprop through depth** and with **local** (Hebbian + gradient) updates.

---

## Method details

- **Self-assembly:** Start with one prototype per class; **spawn** a new prototype when no unit adequately explains the input.
- **Context-key gating:** Each prototype has a key vector; only the top-G prototypes (by key·context × RBF response) participate per step.
- **Dual timescales:** Scores use **slow weights** + **fast Hebbian traces**. The fast head decays and is refreshed online, enabling one-shot
  adaptation immediately after shifts.
- **Structural modules:** We **merge** near-duplicate prototypes online (distance < *r_merge*) to form stable modules and prevent bloat.
- **Maintenance:** Usage-weighted pruning when at capacity.

All updates are **local** to active units — no end-to-end backprop through deep stacks.

---

## Reproducibility

- Fixed seeds: classification `seed=21`; bandits `seed=33` and `seed=37`.
- Hardware: single CPU in a local notebook environment.
- The exact numbers above are produced by running the scripts in `src/` and are saved in `results/*.csv`.

---

## License

Apache-2.0 (or replace with your preferred license).

---

## Paper & Specs (Updated 2025-08-18)

The updated paper reflecting **runtime‑enforced compute allocation** lives in `paper/paper.md` with companion specs in `paper/specs/`:

- `paper/paper.md` — full write‑up
- `paper/specs/CPCS_spec.md` — critical‑path scheduler
- `paper/specs/RoIS_spec.md` — region‑of‑interest splicing
- `paper/specs/TTC_spec.md` — transactional tool calls

See `paper/REVISION_NOTES.md` for a succinct change log.
