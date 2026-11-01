# Symmetry Restoration in Arithmetic Quantum Field Theory

## Overview

This project investigates the **Riemann Hypothesis (RH)** through the lens of **Chrono-Topological Field Theory (CTFT)**. We propose that the location of the Riemann zeta zeros on the critical line is a consequence of **symmetry restoration** in the infrared limit of an arithmetic Renormalisation Group (RG) flow.

We are currently pursuing four interconnected research tracks:

1.  **General Framework**: `rh_ctft.tex` - The core theory of symmetry restoration.
2.  **Quantum Hamiltonian**: `rh_berry_keating.tex` - Discrete realizations of the Berry-Keating $xp+px$ operator.
3.  **Adelic Geometry**: `rh_adele_space.tex` - The geometric foundation on the Adele class space.
4.  **Flow Dynamics**: `rh_dynamic_flow.tex` - Detailed analysis of the symplectic RG flow.

## Structure

-   `rh_ctft.tex`, `rh_berry_keating.tex`, `rh_adele_space.tex`, `rh_dynamic_flow.tex`: Research papers.
-   `code/`:
    -   `rg_flow_simulation.py`: Simulates the RG flow using Random Matrix Theory.
    -   `berry_keating_sim.py`: (Planned) Simulation of discrete $xp+px$.
-   `data/`: Generated plots and data.
-   `references.bib`: Shared bibliography.

## Reproduction

To reproduce the results and figures:

1.  Install dependencies:
    ```bash
    pip install numpy matplotlib scipy
    ```

2.  Run the simulations:
    ```bash
    python3 code/rg_flow_simulation.py
    # python3 code/berry_keating_sim.py (Coming soon)
    ```

## Abstract

We introduce CTFT to unify spectral and arithmetic perspectives on RH. We interpret the zeta zeros as eigenvalues of a limiting Hamiltonian $H_{CTFT}$. Our numerical simulations of a toy model show that under "arithmetic time" evolution, symmetry-breaking terms become irrelevant, forcing the spectrum onto the critical line.
