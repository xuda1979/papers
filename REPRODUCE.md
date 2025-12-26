# Reproducibility Guide

This repository contains the source code and data to reproduce the results and figures in the paper "Horizon Memory Combs: A Finite-Memory Framework for Black Hole Evaporation and Information Flow".

## Prerequisites

You need a Python 3 environment with the following packages installed:

```bash
pip install numpy scipy matplotlib pandas tqdm
```

## Reproducing Figures and Data

To generate all main datasets and figures (Page curves, $g^{(2)}$ correlations, ablation studies), run the master script:

```bash
python generate_all.py
```

This script will:
1. Run the PT-MPO simulations (or load cached data if available).
2. Generate the data tables used in the LaTeX manuscript.
3. Produce the plots in the `fig/` directory.

### Individual Scripts

- `generate_page_curve.py`: Simulates the evaporation and computes the Page curve entropy $S(R)$.
- `generate_g2.py`: Computes the second-order correlation function $g^{(2)}(\tau)$ to show sidebands.
- `run_ablation_suite.py`: Runs the ablation studies for different memory depths and scrambling parameters.

## Running the PT-MPO Simulation

To run a standalone PT-MPO simulation with custom parameters:

```python
from simulation import run_simulation

# Example: Run a small simulation
results = run_simulation(
    L_mem=4,          # Memory depth
    chi=16,           # Bond dimension
    n_steps=50,       # Number of time steps
    scramble_type='Haar' # Scrambling model
)
print(results['entropy'])
```

## Verifying Data Integrity

The repository includes checksums for the key data files to ensure integrity. To verify:

```bash
python check_ricci.py
```

(Note: `check_ricci.py` performs integrity checks on the generated data).

## LaTeX Compilation

To compile the paper:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
