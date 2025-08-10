# SafeRL-Window
A simulation-only research prototype for **Safe Reinforcement Learning** control of **sliding-window decoding**
in **time-varying quantum channels** with **CSS-QLDPC** codes.

This repository contains:
- `paper/` LaTeX source of the research paper (IEEEtran).
- `code/` Python skeleton for experiments: baseline sliding-window decoding and Safe RL controller.

> Note: This is a minimal, research-grade skeleton to help you get started. You will need to install the
listed dependencies and plug in real QLDPC matrices/decoders to run large-scale experiments.

## Quick Start (code)
```bash
cd code
python -m experiments.run_baseline --config configs/baseline.yaml
python -m experiments.run_saferl --config configs/saferl.yaml
python -m experiments.plot_results --logdir ./data/logs
```

## Environments / Dependencies
See `code/requirements.txt`. You may need GPU for training (PyTorch).

## License
MIT
