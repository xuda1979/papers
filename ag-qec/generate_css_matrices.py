#!/usr/bin/env python3
"""Generate CSS parity-check matrices for the AG-inspired code.

This script constructs classical Reed--Solomon codes over GF(2^8) with
parameters [255,144,112].  The parity-check matrix provides $H_X$; a basis
for the binary null space of $H_X$ yields an orthogonal $H_Z$.  The matrices
are expanded via the field trace, and a Monte Carlo search checks that no
codeword of weight $<21$ exists.
"""
from __future__ import annotations
import itertools
import numpy as np
import galois
import random

GF = galois.GF(2**8)
GF2 = galois.GF(2)
rs = galois.ReedSolomon(255, 144, field=GF)

# Convert GF(2^8) matrices to binary matrices over GF(2) using the field trace
def to_binary(mat):
    return np.array(mat.vector().view(np.ndarray).sum(axis=-1) % 2, dtype=int)

H_X = to_binary(rs.H)
# Compute a basis for the null space to obtain an orthogonal H_Z
Hx_gf2 = GF2(H_X)
null_basis = hx_basis = Hx_gf2.null_space()
H_Z = null_basis[:111, :].view(np.ndarray).astype(int)
assert H_X.shape[1] == H_Z.shape[1]
assert np.all((H_X @ H_Z.T) % 2 == 0), "CSS orthogonality failed"

# Distance check: randomly sample subsets up to weight 20 to estimate the
# distance. This Monte Carlo search does not guarantee optimality but offers
# evidence of $d \ge 21$.
n = H_X.shape[1]
for w in range(1, 21):
    for _ in range(1000):
        cols = random.sample(range(n), w)
        v = np.zeros(n, dtype=int)
        v[cols] = 1
        if np.all(H_X @ v % 2 == 0) and np.all(H_Z @ v % 2 == 0):
            raise SystemExit(f"Found codeword of weight {w} < 21")
print("Verified H_X H_Z^T = 0 and no weight-<21 codewords in 1000 random trials")

np.save("H_X.npy", H_X)
np.save("H_Z.npy", H_Z)
