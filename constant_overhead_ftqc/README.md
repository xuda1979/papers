# Constant-Overhead Fault-Tolerant Quantum Computation

This directory contains a research paper and associated simulation scripts for establishing a fault-tolerance threshold for asymptotically good qLDPC codes (specifically Lifted Product codes).

## Overview

The paper "Constant-Overhead Fault-Tolerant Quantum Computation: Establishing a Threshold Theorem for Asymptotically Good qLDPC Codes" addresses the scalability bottleneck of the Surface Code by leveraging the constant encoding rate and linear distance of qLDPC codes.

Key contributions:
1.  **Single-Shot Error Correction (SSEC):** Utilizing the soundness property of expander graphs to correct errors in $O(1)$ time.
2.  **Teleportation-Based Logical Fabric:** A protocol for executing logical gates without destroying sparsity.
3.  **Threshold Theorem:** A rigorous proof of a non-zero threshold $p_{th}$ under circuit-level noise.

## Structure

*   `paper.tex`: The LaTeX source code for the paper.
*   `references.bib`: Bibliography file.
*   `simulation.py`: (Planned) Python script for verifying single-shot stability and spectral properties of the codes.

## Usage

To compile the paper:
```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```
