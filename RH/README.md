# Symmetry Restoration in Arithmetic Quantum Field Theory

## Overview

This project investigates the **Riemann Hypothesis (RH)** through the lens of **Chrono-Topological Field Theory (CTFT)**. We propose that the location of the Riemann zeta zeros on the critical line is a consequence of **symmetry restoration** in the infrared limit of an arithmetic Renormalisation Group (RG) flow.

The repository contains:
1.  **Theory**: A research paper (`rh_ctft.tex`) outlining the framework and presenting numerical evidence.
2.  **Simulation**: Python code (`code/rg_flow_simulation.py`) that models the spectral flow of a Hamiltonian under RG, demonstrating eigenvalue condensation.
3.  **Data**: Generated plots (`data/`) visualizing the symmetry restoration process.

## Structure

-   `rh_ctft.tex`: The main LaTeX research paper.
-   `code/`:
    -   `rg_flow_simulation.py`: Simulates the RG flow using Random Matrix Theory (GUE + Ginibre perturbation).
    -   `prime_density.py`: (Legacy) Basic prime density analysis.
-   `data/`:
    -   `eigenvalue_flow.png`: Visualization of eigenvalues moving to the real axis.
    -   `symmetry_restoration.png`: Plot of the order parameter decay.
-   `references.bib`: Bibliography for the paper.

## Reproduction

To reproduce the results and figures:

1.  Install dependencies:
    ```bash
    pip install numpy matplotlib
    ```

2.  Run the simulation:
    ```bash
    python3 code/rg_flow_simulation.py
    ```

3.  The plots will be generated in the `data/` directory.

## Abstract

We introduce CTFT to unify spectral and arithmetic perspectives on RH. We interpret the zeta zeros as eigenvalues of a limiting Hamiltonian $H_{CTFT}$. Our numerical simulations of a toy model show that under "arithmetic time" evolution, symmetry-breaking terms become irrelevant, forcing the spectrum onto the critical line.
