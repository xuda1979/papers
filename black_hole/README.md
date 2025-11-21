# HMC Artifacts Bundle

This bundle contains:
- `fig/comb_code_embedding.pdf` — schematic figure for the code-subspace comb.
- `fig/kk_re_vs_hilbert.pdf` and `fig/kk_residuals.pdf` — Kramers–Kronig checks.
- `kk_checks.json` — machine-readable KK diagnostics (frequencies, real/imag, reconstructions, residuals, metrics).
- `fig/identifiability_true_vs_est_kernel.pdf` — identification of a finite-memory kernel (true vs estimated).
- `identifiability_report.json` — identifiability diagnostics (condition numbers, MSE/NMSE, coefficients).

## Reproduce locally

```bash
# Create figures and reports in the current folder
python3 code/make_comb_figure.py
python3 code/simulate_kk_identifiability.py
python3 code/generate_reports.py
```

All plots are generated with matplotlib (no seaborn), each is a single chart (no subplots), and no explicit colors are set.
Random seeds are fixed for reproducibility.
