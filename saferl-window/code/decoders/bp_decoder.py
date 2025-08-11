"""
Minimal BP/Min-Sum style placeholder on a Tanner graph.
For quantum CSS, we run separately on Hx (for Z errors) and Hz (for X errors).
This is a pedagogical sketch; replace with a production decoder for real experiments.
"""
from __future__ import annotations
import numpy as np

def syndrome(H, e):
    return (H @ e) % 2

def greedy_decode_placeholder(H, syn, max_iters=20, rng=None):
    """
    Placeholder decoder using a simple greedy bit-flip strategy.
    This is NOT a belief propagation or min-sum decoder.
    It serves as a functional placeholder in the simulation skeleton.
    """
    rng = np.random.default_rng(rng)
    n = H.shape[1]
    e_hat = np.zeros(n, dtype=int)
    current_syn = np.copy(syn)

    for _ in range(max_iters):
        if np.sum(current_syn) == 0:
            break
        # Find unsatisfied checks
        unsat_checks = np.where(current_syn == 1)[0]
        # Pick a random unsatisfied check
        check_idx = rng.choice(unsat_checks)
        # Find variables involved in this check
        var_indices = np.where(H[check_idx, :] == 1)[0]
        if len(var_indices) > 0:
            # Flip a random variable in that check
            var_to_flip = rng.choice(var_indices)
            e_hat[var_to_flip] ^= 1
        # Recalculate syndrome
        current_syn = (H @ e_hat) % 2
    return e_hat

def css_bp_round(Hx, Hz, syn_x, syn_z, iters=20, damping=0.5, schedule="layered", rng=None):
    # Note: damping and schedule are unused by the placeholder decoder
    ex = greedy_decode_placeholder(Hz, syn_x, max_iters=iters, rng=rng)
    ez = greedy_decode_placeholder(Hx, syn_z, max_iters=iters, rng=rng)
    return ex, ez
