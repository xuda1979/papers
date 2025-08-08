import numpy as np, math

# ---------- generic helpers ----------
def idx(x, y, R):          # (1…R,1…R) → [0 … R²-1]
    return (y-1)*R + (x-1)

def trace_inverse_dense(L):
    eigvals = np.linalg.eigvals(L)
    return np.sum(1.0/np.real(eigvals))   # all e‑vals are >0

def estimate_pi(R, weight, neighbours):
    """L = I + w * adjacency  (w negative), return prefactor * R² log(R²)/Tr(L⁻¹)."""
    N = R*R
    rows = np.arange(N)
    L = np.eye(N)
    for y in range(1, R+1):
        for x in range(1, R+1):
            i = idx(x, y, R)
            for dx, dy in neighbours:
                xx, yy = x+dx, y+dy
                if 1 <= xx <= R and 1 <= yy <= R:
                    j = idx(xx, yy, R)
                    L[i, j] = weight
    return L, trace_inverse_dense(L)

# ---------- three walks ----------
R_small = 10                       # dense eig OK up to about 13×13 for laptop

# King walk (8 neighbours, weight = -1/8, prefactor 2/3)
king_nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
L, tr = estimate_pi(R_small, -0.125, king_nbrs)
king_pi = (2/3) * (R_small**2 * math.log(R_small**2)) / tr
print(f'King π (R={R_small}):  {king_pi:.6f}')

# Triangular walk (6 neighbours, weight = -1/6, prefactor 2/√3)
tri_nbrs = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
L, tr = estimate_pi(R_small, -1/6, tri_nbrs)
tri_pi = (2/math.sqrt(3)) * (R_small**2 * math.log(R_small**2)) / tr
print(f'Tri  π (R={R_small}):  {tri_pi:.6f}')

# Knight walk (8 L‑moves, weight = -1/8, prefactor 1/5)
knight_nbrs = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
L, tr = estimate_pi(R_small, -0.125, knight_nbrs)
knight_pi = (1/5) * (R_small**2 * math.log(R_small**2)) / tr
print(f'Knight π (R={R_small}): {knight_pi:.6f}')

print(f'\nTrue π = {math.pi:.6f}')
