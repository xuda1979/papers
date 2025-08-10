"""
Minimal BP/Min-Sum style placeholder on a Tanner graph.
For quantum CSS, we run separately on Hx (for Z errors) and Hz (for X errors).
This is a pedagogical sketch; replace with a production decoder for real experiments.
"""
from __future__ import annotations
import numpy as np

def syndrome(H, e):
    return (H @ e) % 2

def min_sum_decode(H, syn, max_iters=20, damping=0.5, schedule="layered", rng=None):
    """
    Placeholder: returns an estimated error vector e_hat that satisfies constraints approximately.
    """
    rng = np.random.default_rng(rng)
    n = H.shape[1]
    # naive: try a few random flips guided by syndrome weight
    e_hat = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        unsat = (syn == 1).astype(int)
        if unsat.sum() == 0:
            break
        # pick a variable involved in many unsatisfied checks
        involvement = H.T @ unsat
        j = int(np.argmax(involvement))
        e_hat[j] ^= 1
        syn = (H @ e_hat) % 2
    return e_hat

def css_bp_round(Hx, Hz, syn_x, syn_z, iters=20, damping=0.5, schedule="layered", rng=None):
    ex = min_sum_decode(Hz, syn_x, max_iters=iters, damping=damping, schedule=schedule, rng=rng)
    ez = min_sum_decode(Hx, syn_z, max_iters=iters, damping=damping, schedule=schedule, rng=rng)
    return ex, ez
