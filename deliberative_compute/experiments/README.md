# Minimal synthetic harness for TTC
This folder provides a tiny simulator to visualize **Budget–Performance Frontiers (BPF)** and compare IGD / UCB‑IGD / RS‑MCTT / CSC on toy tasks.

**Why synthetic?** The paper is theory‑first; these scripts *illustrate* concepts with controlled improvement processes and verifiers without depending on external APIs.

## Quick start
```bash
python run_simulated_frontiers.py --seed 1 --budgets 16 32 64 128
```
This prints FA/MVC estimates and writes `bpf_plot.png`.

## Files
- `simulators.py`: stochastic generators for per-thread improvements, rollouts, and verifiers.
- `algorithms.py`: IGD, UCB-IGD, RS-MCTT (risk-averse), CSC filtering.
- `metrics.py`: BPF estimation, frontier area (FA), marginal value of compute (MVC) with bootstrap CIs.
- `run_simulated_frontiers.py`: command-line demo.

## Repro philosophy
All results are random-seeded; budgets count *micro-action* costs; utilities are in [0,1].
