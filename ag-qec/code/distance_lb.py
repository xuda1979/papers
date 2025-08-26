from __future__ import annotations
import argparse, numpy as np

def gf2_rank(A: np.ndarray) -> int:
    A = A.copy().astype(np.uint8)
    m, n = A.shape
    r = 0
    col = 0
    for row in range(m):
        while col < n and not A[row:, col].any():
            col += 1
        if col == n:
            break
        pivot = row + np.argmax(A[row:, col])
        if A[pivot, col] == 0:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        below = np.where(A[(row+1):, col] == 1)[0] + row + 1
        for rr in below:
            A[rr, :] ^= A[row, :]
        r += 1
        col += 1
    return r

def lower_bound_distance(H: np.ndarray, t: int = 4, samples: int = 5000, seed: int = 7) -> int:
    """
    Randomized search for codewords up to weight t.
    If none found, return t+1 as a conservative lower bound.
    """
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    # weight-1..t trials
    for w in range(1, t+1):
        for _ in range(max(1, samples // t)):
            pos = rng.choice(n, size=w, replace=False)
            v = np.zeros(n, dtype=np.uint8)
            v[pos] = 1
            if ((H @ v) % 2).sum() == 0:
                return w
    return t + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--css", required=True)
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()
    D = np.load(args.css)
    Hx, Hz = D["Hx"].astype(np.uint8), D["Hz"].astype(np.uint8)
    dx = lower_bound_distance(Hx, t=args.t, samples=args.samples)
    dz = lower_bound_distance(Hz, t=args.t, samples=args.samples)
    print(f"[distance-lb] dX>{dx-1}, dZ>{dz-1} (conservative lower bounds)")

if __name__ == "__main__":
    main()

