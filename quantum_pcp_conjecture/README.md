# Quantum PCP Conjecture Review

This directory contains a review paper on the Quantum PCP Conjecture and a Python simulation demonstrating the computational hardness of the Local Hamiltonian problem.

## Contents

*   `paper.tex`: The review paper in LaTeX format.
*   `simulation.py`: A Python script that calculates the ground state energy of a 1D Transverse Field Ising Model (TFIM) for increasing system sizes.
*   `references.bib`: Bibliography for the paper.

## Running the Simulation

The simulation requires `numpy`, `scipy`, and `matplotlib`.

```bash
python3 simulation.py
```

This will:
1.  Compute the ground state energy for TFIM chains of length $N=2$ to $N=10$.
2.  Print the results and execution time.
3.  Generate `complexity_plot.png` showing the exponential time scaling.
