#!/usr/bin/env python3
"""
Monte-Carlo for CSS under asymmetric first-order Markov Pauli channel.
Decoders: damped min-sum BP (+ optional OSD-2).
Writes CSV with Wilson 95% CIs and params.
"""
from pathlib import Path
import argparse, csv
import numpy as np

# ----------------------- Utils (GF(2) linear algebra) -----------------------
def set_seed(s: int): np.random.seed(s % (2**32-1))

def mod2(A): return (A.astype(np.uint8) & 1)

def row_echelon_mod2(M):
    M = mod2(M.copy()); r=c=0; m,n=M.shape; piv=[]
    while r<m and c<n:
        rows = np.where(M[r:,c]==1)[0]
        if rows.size==0: c+=1; continue
        i=r+rows[0]
        if i!=r: M[[r,i]]=M[[i,r]]
        for j in range(m):
            if j!=r and M[j,c]==1: M[j]^=M[r]
        piv.append(c); r+=1; c+=1
    return M, piv

def rank_mod2(M): return len(row_echelon_mod2(M)[1])

def solve_membership_rowspace(B, v):
    """Return True iff v is in rowspan(B). Solve B^T y = v^T over GF(2)."""
    BT = B.T.copy().astype(np.uint8) & 1
    b = v.copy().astype(np.uint8) & 1
    m, n = BT.shape  # m = n_cols_of_B
    r = c = 0
    while r<m and c<n:
        rows = np.where(BT[r:,c]==1)[0]
        if rows.size==0: c+=1; continue
        i=r+rows[0]
        if i!=r: BT[[r,i]]=BT[[i,r]]
        if b[r]!=0 and BT[r,c]==0: pass
        for j in range(m):
            if j!=r and BT[j,c]==1:
                BT[j]^=BT[r]; b[j]^=b[r]
        r+=1; c+=1
    # System is consistent iff remaining rows have 0 = b
    for i in range(r, m):
        if b[i]!=0 and not BT[i].any(): return False
    # If underdetermined, still consistent -> in row space
    return True

# ----------------------- Tanner graph for BP -----------------------
def tanner_from_H(H):
    H = mod2(H); m, n = H.shape
    checks = [np.where(H[i]==1)[0] for i in range(m)]
    vars_ = [np.where(H[:,j]==1)[0] for j in range(n)]
    return checks, vars_

def bp_decode_min_sum(H, syndrome, llr0, iters=50, damp=0.5, norm=0.9):
    """Damped normalized min-sum BP; returns hard decision and final LLRs."""
    H = mod2(H); m, n = H.shape
    checks, vars_ = tanner_from_H(H)
    # Messages: m_c2v and m_v2c as dicts (c->v) and (v->c)
    m_c2v = { (ci, v): 0.0 for ci in range(m) for v in checks[ci] }
    m_v2c = { (v, ci): llr0[v] for v in range(n) for ci in vars_[v] }
    L = llr0.copy()
    for _ in range(iters):
        # Check update
        new_c2v = {}
        for ci in range(m):
            neigh = checks[ci]
            sgn = 1.0
            mags = []
            msgs = []
            for v in neigh:
                x = m_v2c[(v,ci)]
                sgn *= np.sign(x) if x!=0 else 1.0
                mags.append(abs(x)); msgs.append(x)
            for idx, v in enumerate(neigh):
                others = mags[:idx]+mags[idx+1:]
                if len(others)==0: val=0.0
                else: val = norm * np.prod([np.sign(msgs[k]) if k!=idx else 1 for k in range(len(neigh))]) * min(others)
                old = m_c2v[(ci,v)]
                new_c2v[(ci,v)] = (1-damp)*val + damp*old
        m_c2v = new_c2v
        # Variable update
        new_v2c = {}
        for v in range(n):
            total = llr0[v] + sum(m_c2v[(ci,v)] for ci in vars_[v])
            for ci in vars_[v]:
                val = total - m_c2v[(ci,v)]
                old = m_v2c[(v,ci)]
                new_v2c[(v,ci)] = (1-damp)*val + damp*old
        m_v2c = new_v2c
        # Posterior and hard decision
        L = np.array([ llr0[v] + sum(m_c2v[(ci,v)] for ci in vars_[v]) for v in range(n) ])
        hard = (L < 0).astype(np.uint8)  # 1 if negative LLR
        if np.all(((H @ hard) & 1) == syndrome): 
            return hard, L
    return hard, L

# ----------------------- OSD-2 (lightweight rescue) -----------------------
def osd2_rescue(H, syndrome, L_post, hard, topk=8):
    """Try flipping up to 2 of the K least-reliable bits to satisfy syndrome."""
    H = mod2(H); s = syndrome
    order = np.argsort(np.abs(L_post))[:topk]
    best = hard.copy(); best_cost = np.inf
    # 0th order: check hard
    if np.all(((H @ best) & 1)==s): return best
    # 1st order
    for i in range(len(order)):
        cand = hard.copy(); v = order[i]; cand[v]^=1
        if np.all(((H @ cand)&1)==s):
            cost = np.abs(L_post[v])
            if cost < best_cost: best, best_cost = cand, cost
    if best_cost < np.inf: return best
    # 2nd order
    for i in range(len(order)):
        for j in range(i+1, len(order)):
            cand = hard.copy(); v1, v2 = order[i], order[j]
            cand[v1]^=1; cand[v2]^=1
            if np.all(((H @ cand)&1)==s):
                cost = np.abs(L_post[v1]) + np.abs(L_post[v2])
                if cost < best_cost: best, best_cost = cand, cost
    return best

# ----------------------- Channel: asymmetric Markov Pauli -----------------------
def sample_markov_binary(n, p, eta):
    """Return 0/1 vector with P(1)=p and persistence eta along index."""
    x = np.zeros(n, dtype=np.uint8)
    x[0] = (np.random.rand() < p)
    for i in range(1, n):
        if np.random.rand() < eta: x[i] = x[i-1]
        else: x[i] = (np.random.rand() < p)
    return x

def sample_pauli_errors(n, pX, pZ, pY, eta):
    eX = sample_markov_binary(n, pX, eta)
    eZ = sample_markov_binary(n, pZ, eta)
    eY = sample_markov_binary(n, pY, eta) if pY>0 else np.zeros(n, dtype=np.uint8)
    eX_tot = eX ^ eY
    eZ_tot = eZ ^ eY
    return eX_tot, eZ_tot

def llr_from_p(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return np.log((1-p)/p)

# ----------------------- Trial logic -----------------------
def logical_error_flags(HX, HZ, eX_tot, eZ_tot, iters, damp, osd_topk, use_osd):
    n = HX.shape[1]
    # Z-decoding with HX
    pZstar = 0.5  # we use constant priors; set 0.5->uninformative, or use channel p if desired
    L0z = np.full(n, llr_from_p(pZstar), dtype=float)
    sz = (HX @ eZ_tot) & 1
    cz, Lz = bp_decode_min_sum(HX, sz, L0z, iters=iters, damp=damp)
    if use_osd: cz = osd2_rescue(HX, sz, Lz, cz, topk=osd_topk)
    rz = (eZ_tot ^ cz)
    z_ok = solve_membership_rowspace(HX, rz)  # no logical Z iff residual in rowspan(HX)
    # X-decoding with HZ
    pXstar = 0.5
    L0x = np.full(n, llr_from_p(pXstar), dtype=float)
    sx = (HZ @ eX_tot) & 1
    cx, Lx = bp_decode_min_sum(HZ, sx, L0x, iters=iters, damp=damp)
    if use_osd: cx = osd2_rescue(HZ, sx, Lx, cx, topk=osd_topk)
    rx = (eX_tot ^ cx)
    x_ok = solve_membership_rowspace(HZ, rx)
    return (not z_ok), (not x_ok)

def wilson_interval(k, n, z=1.96):
    if n==0: return (0.0,1.0,0.0)
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = (z*np.sqrt((p*(1-p)/n) + (z*z/(4*n*n))))/denom
    return (max(0.0, center-half), min(1.0, center+half), p)

# ----------------------- Main experiment runner -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--rx", type=int, default=111)
    ap.add_argument("--rz", type=int, default=111)
    ap.add_argument("--HX", type=str, default="matrices/HX.csv")
    ap.add_argument("--HZ", type=str, default="matrices/HZ.csv")
    ap.add_argument("--pX", type=float, default=0.01)
    ap.add_argument("--pZ", type=float, default=0.03)
    ap.add_argument("--pY", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--damp", type=float, default=0.5)
    ap.add_argument("--osd", type=lambda s: s.lower() in {"1","true","t","yes"}, default=True)
    ap.add_argument("--osd_topk", type=int, default=8)
    ap.add_argument("--out", type=str, default="data/logical_error_rate.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    HX = mod2(np.loadtxt(args.HX, dtype=np.uint8, delimiter=","))
    HZ = mod2(np.loadtxt(args.HZ, dtype=np.uint8, delimiter=","))
    assert HX.shape[1]==HZ.shape[1]==args.n

    Path("data").mkdir(parents=True, exist_ok=True)
    fields = ["seed","n","rx","rz","k","pX","pZ","pY","eta","trials","iters","damp","osd","osd_topk","block_logical_rate","ci_lo","ci_hi","x_only","z_only"]
    rkx, rkz = rank_mod2(HX), rank_mod2(HZ)
    k = args.n - rkx - rkz

    blk_err = 0; x_only=0; z_only=0
    for _ in range(args.trials):
        eX, eZ = sample_pauli_errors(args.n, args.pX, args.pZ, args.pY, args.eta)
        zbad, xbad = logical_error_flags(HX, HZ, eX, eZ, args.iters, args.damp, args.osd_topk, args.osd)
        if xbad or zbad: blk_err += 1
        if xbad and not zbad: x_only += 1
        if zbad and not xbad: z_only += 1

    lo, hi, p = wilson_interval(blk_err, args.trials)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(seed=args.seed, n=args.n, rx=args.rx, rz=args.rz, k=k,
                        pX=args.pX, pZ=args.pZ, pY=args.pY, eta=args.eta,
                        trials=args.trials, iters=args.iters, damp=args.damp,
                        osd=args.osd, osd_topk=args.osd_topk,
                        block_logical_rate=p, ci_lo=lo, ci_hi=hi,
                        x_only=x_only/args.trials, z_only=z_only/args.trials))
    print(f"Block logical error rate: {p:.4e} (95% CI {lo:.4e}–{hi:.4e})")

if __name__ == "__main__":
    main()
