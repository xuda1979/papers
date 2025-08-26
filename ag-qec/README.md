# ag-qec artifacts (reproducibility update)

This folder adds a minimal, dependency-light reference implementation for the paper. It constructs dual-contained CSS codes at `n=255`, simulates an **asymmetric first-order Markov Pauli channel**, decodes with **damped min-sum BP** and optional **OSD-2**, and logs Wilson 95% CIs, per-block runtime, and OSD usage.

## Quickstart
```bash
cd ag-qec
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all     # codes + sweeps + plots
```

Outputs:
```
data/css_n255_seed1337.npz     # Hx, Hz matrices
data/stats_seed1337.json       # degrees, 4-cycles, realized k, rank checks
data/results.csv               # trials, BER, Wilson CIs, runtimes (BP/OSD)
figs/ablation.pdf, figs/complexity.pdf
```

## Key CLI toggles
- `--eta-list 0,0.3,0.6,0.9` sweeps temporal correlation.
- `--pX, --pZ, --pY` set asymmetric Pauli rates; CSS uses `pX* = pX + pY`, `pZ* = pZ + pY`.
- `--iters`, `--damping` tune BP; `--osd 2` enables OSD-2 rescue.
- `--baseline-lp` runs a tiny lifted-product toy baseline (not part of `make all`).

## CSV columns
See Appendix A or open the header row in `data/results.csv`.

## Recompile paper
The TeX file auto-includes three appendices if present:
```
appendix_methods.tex
appendix_runtime.tex
appendix_reproducibility.tex
```
Figures are conditionally included if `figs/*.pdf` exist.

## License
Research code; no warranty. Use at your own risk.

