# Quantum Learning Theory: Entanglement and Generalization

## Project Overview

This project investigates the intersection of Quantum Machine Learning (QML) and Statistical Field Theory. Specifically, it explores the hypothesis that the generalization capability of a Variational Quantum Classifier (VQC) is intrinsicly linked to the entanglement spectrum of its ansatz.

We propose a novel theoretical framework where the "training" of a QNN is viewed as a renormalization group (RG) flow, and the point of optimal generalization corresponds to a critical phase transition in the effective Hamiltonian of the quantum circuit.

## Contents

*   **paper.tex**: The main research paper titled "Entanglement-Induced Generalization in Quantum Neural Networks: A Field-Theoretic Perspective".
*   **simulation.py**: A Python script that performs numerical experiments to validate the theoretical conjectures. It simulates a toy QNN model and analyzes the relationship between entanglement entropy and test accuracy.
*   **references.bib**: Bibliography for the paper.

## Instructions

1.  **Run the Simulation**:
    To generate the experimental data and figures, run the python script:
    ```bash
    python simulation.py
    ```
    This will produce `entanglement_vs_generalization.png` and `effective_dimension.png`.

2.  **Compile the Paper**:
    The paper is written in standard LaTeX. Compile it using:
    ```bash
    pdflatex paper.tex
    bibtex paper
    pdflatex paper.tex
    pdflatex paper.tex
    ```

## Dependencies

*   Python 3
*   `numpy`
*   `matplotlib`
*   `scipy`
