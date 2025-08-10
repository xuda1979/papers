"""
Code definitions and simple helpers.
This is a minimal placeholder: provide small CSS-QLDPC-like Hx, Hz for demonstration.
Replace with real SC-QLDPC matrices in practice.
"""
from __future__ import annotations
import numpy as np

def tiny_css_example():
    # Small  (not a practical code) demo matrices Hx, Hz satisfying Hx * Hz^T = 0 mod 2.
    Hx = np.array([
        [1,0,1,0,1,0],
        [0,1,1,0,0,1],
        [1,1,0,1,0,0]], dtype=int)
    Hz = np.array([
        [1,1,1,0,0,0],
        [1,0,0,1,1,0],
        [0,1,0,1,0,1]], dtype=int)
    assert ((Hx @ Hz.T) % 2 == 0).all()
    return Hx, Hz

def random_band_binary(m, n, band=5, seed=0):
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=int)
    for i in range(m):
        for _ in range(3):
            j = (i + rng.integers(-band, band+1)) % n
            H[i, j] = 1
    return H

def sc_qldpc_placeholder(n=6000, rate=0.5, seed=0):
    # Placeholder "banded" Hx, Hz pair; not a real SC-QLDPC, but keeps interfaces consistent.
    m = int(n * (1-rate))
    Hx = random_band_binary(m//2, n, band=7, seed=seed)
    Hz = random_band_binary(m//2, n, band=7, seed=seed+1)
    # Force orthogonality loosely by zeroing conflicts (toy, for demo only)
    prod = (Hx @ Hz.T) % 2
    rows, cols = np.where(prod == 1)
    for r, c in zip(rows, cols):
        # Flip a random 1 in Hx[r] to 0 to reduce conflict
        ones = np.where(Hx[r] == 1)[0]
        if len(ones) > 0:
            Hx[r, ones[0]] = 0
    return Hx, Hz
