Yang-Mills Mass Gap Verification Module
=======================================

This folder contains computer-assisted verification scripts for the Yang-Mills existence and mass gap problem.

Files:
------
interval_gap_check.py:
    Implements a rigorous Interval Arithmetic check for the spectral gap of the Transfer Matrix.
    Run this script to verify Condition C0 for the Hierarchical Log-Sobolev Inequality.
    
    Usage:
    python interval_gap_check.py

Dependencies:
    - numpy
    - scipy (optional, for comparison)
    
Note: This is a Proof-of-Concept implementation of the Interval Arithmetic framework.
For the full mathematical proof, the 'Interval' class must be replaced with a library
that handles IEEE 754 rounding modes correctly (e.g., mpmath or specialized interval libraries).
