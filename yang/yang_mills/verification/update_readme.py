import os

readme_content = r'''# Yang-Mills Existence and Mass Gap: Computer-Assisted Verification Suite

**Author:** Da Xu  
**Affiliation:** China Mobile Research Institute  
**Date:** January 14, 2026 (Updated for Audit)

## Overview

This repository contains the **Computer-Assisted Proof (CAP)** artifacts accompanying the manuscript *"On the Existence and Mass Gap of Four-Dimensional Yang-Mills Theory"*. 

The suite performs a rigorous, interval-arithmetic-based verification of the Renormalization Group (RG) flow contraction, bridging the gap between the analytic Strong Coupling regime and the Asymptotic Freedom regime.

## Verified Regimes (Unified)

1.  **Strong Coupling (Analytic):** $\beta \in (0, 0.40]$.
    *   Handled by **Cluster Expansion** (Phase 1) and Dobrushin Finite-Size Criterion.
    *   Verified analytically; code performs handshake check at $\beta=0.40$.
2.  **Intermediate (CAP Verification):** $\beta \in (0.40, 6.0]$.
    *   Handled by **Interval Arithmetic Tube Tracking** (Phase 2).
    *   Verified by `full_verifier_phase2.py`.
3.  **Weak Coupling (Asymptotic Freedom):** $\beta > 6.0$.
    *   Handled by perturbative scaling and Balaban bounds.

## Key Components

The verification logic is partitioned into the following modules:

1.  **`rigorous_constants_derivation.py`**  
    *   **Role:** The geometric foundation. Derives the "Pollution Constants" and bounding norms ab initio.
    *   **Method:** Uses interval arithmetic to strictly bound operator mixing.

2.  **`full_verifier_phase2.py`**  
    *   **Role:** The core engine. Iterates the RG map on the "Tube".
    *   **Method:** Checks $R(T_k) \subset T_{k+1}$ for $\beta \in [0.40, 6.0]$.

3.  **`ab_initio_jacobian.py`**
    *   **Role:** Computes rigorous bounds for the Jacobian of the RG map.
    *   **Method:** Uses **Rigorous Remainder Perturbation Theory** (Ab Initio) to bound mass gap scaling across the crossover.

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   No external heavy dependencies.

### Running the Proof Audit

1. **Generate Constants:**
   ```bash
   python rigorous_constants_derivation.py
   ```
2. **Run Verification:**
   ```bash
   python full_verifier_phase2.py
   ```

## Reviewer Notes

*   **Consistency Fix (Jan 14):** The coupling ranges have been unified. The verification establishes a direct handshake at $\beta=0.40$, matching the extended radius of convergence of the Strong Coupling phase.
*   **Ab Initio Jacobian:** The Jacobian estimator now explicitly uses rigorous remainder bounds for the perturbative expansion, ensuring validity across the crossover regime.
'''

with open(r"c:\Users\Lenovo\papers\yang\yang_mills\verification\README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)
