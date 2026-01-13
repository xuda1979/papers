# Yang-Mills Existence and Mass Gap: Computer-Assisted Verification Suite

**Author:** Da Xu  
**Affiliation:** China Mobile Research Institute  
**Date:** January 13, 2026

## Overview

This repository contains the **Computer-Assisted Proof (CAP)** artifacts accompanying the manuscript *"On the Existence and Mass Gap of Four-Dimensional Yang-Mills Theory"*. 

The suite performs a rigorous, interval-arithmetic-based verification of the Renormalization Group (RG) flow contraction, bridging the gap between the analytic Strong Coupling regime and the Asymptotic Freedom regime.

## Key Components

The verification logic is partitioned into the following modules:

1.  **`rigorous_constants_derivation.py`**  
    *   **Role:** The geometric foundation. Derives the "Pollution Constants" and bounding norms ab initio from the definition of the lattice action and gauge group geometry.
    *   **Method:** Uses interval arithmetic to strictly bound operator mixing, ensuring that "relevant" operators do not destabilize the "irrelevant" tail.
    *   **Status:** Hardened against circularity via geometric scaling arguments (Jan 2026).

2.  **`full_verifier_phase2.py`**  
    *   **Role:** The core engine. Iterates the RG map step-by-step ($k=0$ to $k=N$) on the "Tube" of effective actions.
    *   **Method:** Checks the condition $R(T_k) \subset T_{k+1}$ using the constants derived above.

3.  **`ab_initio_jacobian.py`** & **`rigorous_character_expansion.py`**
    *   **Role:** Provides spectral data for the transition matrix. Computes the eigenvalues $\lambda_{rel}$ and $\lambda_{irr}$ of the linearized RG map.

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   No external heavy dependencies (Standard Library + Local Modules). 
*   `numpy` is used for some intermediate representations but the core logic relies on the custom `Interval` class for rigor.

### Running the Proof Audit

To verify the analytic bounds and generate the `rigorous_constants.json` certificate:

```bash
python rigorous_constants_derivation.py
```

To run the full flow verification (Phase 2):

```bash
python full_verifier_phase2.py
```

## Reviewer Notes

*   **Circularity Check:** The `rigorous_constants_derivation.py` script now includes an explicit routine `verify_analytic_tail_bound()` which demonstrates that the contraction factor holds without assuming a mass gap a priori.
*   **Gap Closure:** The verification explicitly checks for continuity at $\beta=0.4$, matching the radius of convergence of the Strong Coupling Expansion.

## License

MIT License. See `LICENSE` file.
