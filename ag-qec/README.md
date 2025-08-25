# AG-QEC (revised): Reproducible pipeline

This revision fixes construction consistency and adds a fully reproducible
simulation stack and figures, per the review feedback:

**What changed**
- Replaced ambiguous AG/Hermitian claim at length 255 with an *explicit,
  algorithmically dual-contained CSS construction* that **guarantees**
  \(H_X H_Z^\top = 0\) and reports the actual `[[n,k]]` via rank checks.
- Added code to **generate** `H_X, H_Z` from a seed, verify CSS commutation,
  and compute `k = n - rank(H_X) - rank(H_Z)`.
- Implemented an **asymmetric, first-order Markov Pauli channel** (parameters
  `pX, pZ, pY, eta`).
- Decoder: **damped min-sum BP** (+ optional **OSD-2** rescue) with
  iteration and damping controls; syndrome‑consistent outputs.
- Evaluation: Monte‑Carlo with **Wilson 95% CIs**, random seeds, CSV logging,
  and a script to regenerate the **logical error rate** figure.

> **Note on naming**  
> The repo/folder name *ag-qec* is preserved. The revised paper now states the
> construction method precisely (dual-contained CSS generated algorithmically at
> \(n=255\)) and avoids incorrect field/length associations.

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make all
```

This will:
1) Generate `matrices/HX.csv` and `matrices/HZ.csv` (seeded),
2) Verify `H_X H_Z^T = 0`,
3) Run Monte‑Carlo and write `data/logical_error_rate.csv`,
4) Plot `figs/logical_error_rate.png`,
5) Compile the paper `ag-qec.pdf` (if you have LaTeX installed).

Targets:
```bash
make matrices     # (re)generate HX/HZ from a fixed seed
make simulate     # run Monte‑Carlo experiments -> data/*.csv
make plot         # regenerate figs/logical_error_rate.png
make pdf          # compile the LaTeX paper
make all          # matrices + simulate + plot + pdf
```

---

## Parameters you can tune

Scripts expose CLI args; defaults are chosen to reproduce the paper figure:
- Code length: `n=255`
- Target rate: `k≈33` (achieved by selecting `rank(HX)+rank(HZ)=222`)
- Channel: asymmetric, first‑order Markov
  - `pX=0.010`, `pZ=0.030`, `pY=0.000`, persistence `eta=0.60`
- Decoder:
  - BP iterations: `--iters 60`
  - Damping: `--damp 0.5`
  - OSD order‑2 rescue: `--osd True --osd_topk 8`
- Trials: `--trials 50000` (adjust for tighter CIs)

Example:
```bash
python scripts/run_experiment.py \
  --seed 20250825 --n 255 --rx 111 --rz 111 \
  --pX 0.01 --pZ 0.03 --pY 0.0 --eta 0.6 \
  --trials 20000 --iters 60 --damp 0.5 --osd True --osd_topk 8 \
  --out data/logical_error_rate.csv
```

---

## Files

- `ag-qec.tex` – revised paper with corrected construction, channel, decoder, CI reporting, and a clear reproducibility section.
- `requirements.txt` – minimal Python deps.
- `Makefile` – convenience targets.
- `scripts/generate_css.py` – generates `HX.csv`, `HZ.csv` reproducibly; prints `n,k,rank(HX),rank(HZ)`, and checks `H_X H_Z^T=0`.
- `scripts/verify_css.py` – sanity checks the matrices.
- `scripts/run_experiment.py` – samples the **asymmetric Markov Pauli** channel, runs **BP + OSD** decoding, logs Wilson CIs.
- `scripts/plot_logical_error_rate.py` – regenerates the figure from CSV.
- `matrices/` – generated parity‑check matrices (`.csv`).
- `data/` – experiment output (`.csv`).
- `figs/` – plots.

---

## Interpreting results

We call a **logical error** when either residual \(Z\) (decoded with \(H_X\)) or
residual \(X\) (decoded with \(H_Z\)) lands in the normalizer but **outside** its
corresponding stabilizer row space. We detect this by checking membership in the
row spaces of \(H_X\) and \(H_Z\) via GF(2) linear solves. We report Monte‑Carlo
block‑logical error rates with **Wilson 95% CIs**.

---

## Limitations (honest)

- The matrices are *algorithmically* dual‑contained and sparse, but not tied to
  a specific AG/Hermitian family at \(n=255\). This avoids the previous field/length
  inconsistency while keeping the practical rate/length.
- The BP uses a **min‑sum** (sum‑product approximation) with damping; we add a
  small **OSD‑2** rescue. Full degeneracy‑aware BP (e.g., supernodes) and factor
  graphs with explicit temporal correlation edges are future work.
- Hardware numbers are not claimed; the paper now describes a pipeline design
  and leaves utilization/latency to a dedicated RTL follow‑up.

---

## Reproduce the figure only
```bash
make matrices
make simulate
make plot
```

---

## License
MIT (source code). The paper text © the authors.
