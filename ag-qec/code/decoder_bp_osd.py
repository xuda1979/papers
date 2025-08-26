from __future__ import annotations
import time
import numpy as np
from typing import Dict, Tuple

def minsum_bp_decode(H: np.ndarray, llr: np.ndarray, iters: int = 100, damping: float = 0.5) -> Tuple[np.ndarray, bool, int]:
    """
    Damped min-sum BP for a binary linear code defined by parity-check H (m x n).
    llr: log-likelihood ratios for each variable (positive favors 0).
    Returns (hard_decision, success, iters_used).
    """
    m, n = H.shape
    # Build neighbor lists
    var2chk = [np.where(H[:, j])[0] for j in range(n)]
    chk2var = [np.where(H[i, :])[0] for i in range(m)]
    # Initialize messages
    msg_vc = {(i, j): 0.0 for i in range(m) for j in chk2var[i]}
    msg_cv = {(i, j): 0.0 for i in range(m) for j in chk2var[i]}
    for t in range(1, iters+1):
        # Variable to check
        for j in range(n):
            neigh = var2chk[j]
            for i in neigh:
                s = llr[j] + sum(msg_cv[(ii, j)] for ii in neigh if ii != i)
                # damping on v->c?
                msg_vc[(i, j)] = s
        # Check to variable (min-sum)
        for i in range(m):
            vs = chk2var[i]
            if len(vs) == 0: 
                continue
            signs = np.ones(len(vs))
            mags = np.zeros(len(vs))
            for idx, j in enumerate(vs):
                x = msg_vc[(i, j)]
                signs[idx] = 1.0 if x >= 0 else -1.0
                mags[idx] = abs(x)
            total_sign = np.prod(signs)
            min1_idx = np.argmin(mags)
            min1 = mags[min1_idx]
            mags2 = mags.copy()
            mags2[min1_idx] = np.inf
            min2 = mags2.min()
            for idx, j in enumerate(vs):
                sgn = total_sign * signs[idx]
                mag = min2 if idx == min1_idx else min1
                new = sgn * mag
                # damping on c->v
                msg_cv[(i, j)] = (1 - damping) * new + damping * msg_cv[(i, j)]
        # Posterior and decision
        post = np.zeros(n)
        for j in range(n):
            post[j] = llr[j] + sum(msg_cv[(i, j)] for i in var2chk[j])
        hard = (post < 0).astype(np.uint8)
        # Syndrome check
        syn = (H @ hard) % 2
        if not syn.any():
            return hard, True, t
    return hard, False, iters

def osd2_rescue(H: np.ndarray, llr: np.ndarray, hard: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Lightweight OSD-2 around a tentative hard decision.
    We flip up to order-2 bits among the least reliable positions and pick the lowest syndrome-weight fix.
    """
    reliab = np.abs(llr)
    idx = np.argsort(reliab)  # least reliable first
    cand = [hard.copy()]
    # order-1
    for a in idx[:32]:
        v = hard.copy(); v[a] ^= 1; cand.append(v)
    # order-2 (limit search width)
    L = min(32, len(idx))
    for i in range(L):
        for j in range(i+1, L):
            v = hard.copy(); v[idx[i]] ^= 1; v[idx[j]] ^= 1; cand.append(v)
    # choose candidate with valid syndrome; tie-break by |H@v| (should be zero ideally)
    best = hard
    best_syn_w = H.shape[0] + 1
    for v in cand:
        syn_w = int(((H @ v) % 2).sum())
        if syn_w < best_syn_w:
            best_syn_w = syn_w
            best = v
            if syn_w == 0:
                break
    return best

def css_decode(Hx: np.ndarray, Hz: np.ndarray, ex_llr: np.ndarray, ez_llr: np.ndarray,
               iters: int, damping: float, osd_order: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Decode X and Z error patterns separately (CSS).
    Returns (ex_hat, ez_hat, metrics)
    """
    t0 = time.time()
    hx, sx, itx = minsum_bp_decode(Hx, ex_llr, iters=iters, damping=damping)
    bp_time = (time.time() - t0)
    osd_calls = 0
    if sx is False and osd_order > 0:
        t1 = time.time()
        hx = osd2_rescue(Hx, ex_llr, hx, order=osd_order)
        osd_time_x = (time.time() - t1)
        osd_calls += 1
    else:
        osd_time_x = 0.0
    t2 = time.time()
    hz, sz, itz = minsum_bp_decode(Hz, ez_llr, iters=iters, damping=damping)
    bp_time += (time.time() - t2)
    if sz is False and osd_order > 0:
        t3 = time.time()
        hz = osd2_rescue(Hz, ez_llr, hz, order=osd_order)
        osd_time_z = (time.time() - t3)
        osd_calls += 1
    else:
        osd_time_z = 0.0
    metrics = {
        "bp_time_ms": (bp_time * 1000.0),
        "osd_time_ms": ((osd_time_x + osd_time_z) * 1000.0),
        "iters_x": itx, "iters_z": itz, "osd_calls": osd_calls
    }
    return hx, hz, metrics
