import sympy
from sympy import symbols, diag, sin, cos, exp, diff, simplify, Matrix, log

# Coordinates
t, theta, varphi = symbols('t theta varphi', real=True)
c, epsilon, delta = symbols('c epsilon delta', real=True, positive=True)

# Cylinder metric g_bar
# g_bar = dt^2 + g_S2
# g_S2 = dtheta^2 + sin(theta)^2 dvarphi^2
g_bar = diag(1, 1, sin(theta)**2)

# Conformal factor
# phi = c * exp(-t/2) + epsilon * exp(-t*(0.5 + delta))
# Use leading order to check the zero
phi_0 = exp(-t/2)
omega_0 = 2 * log(phi_0) # -t
# d(omega_0) = (-1, 0, 0)
# |d(omega_0)|^2 = 1
# Hess(omega_0) = 0
# Lap(omega_0) = 0
# Ric_tilde = Ric_bar - (n-2)(Hess - d x d) - (Lap + (n-2)|d|^2)g
# For n=3
# Ric_tilde = Ric_bar - (0 - d x d) - (0 + 1)g
#           = Ric_bar + d x d - g_bar
# Ric_bar = diag(0, 1, sin^2)
# d x d = diag(1, 0, 0)
# g_bar = diag(1, 1, sin^2)
# Ric_tilde = diag(0,1,sin^2) + diag(1,0,0) - diag(1,1,sin^2)
#           = diag(1, 1, sin^2) - diag(1, 1, sin^2) = 0
# So for the EXACT cone, Ricci is zero.

# Now check perturbation
phi = c * exp(-t/2) + epsilon * exp(-t*(0.5 + delta))
omega = 2 * log(phi)
factor = exp(2*omega) # = phi^4

# g_tilde
g_tilde = factor * g_bar
g_inv = diag(1/factor, 1/factor, 1/(factor * sin(theta)**2))

# Compute Christoffel symbols of g_tilde
coords = [t, theta, varphi]

def get_gamma(k, i, j):
    res = 0
    for l in range(3):
        if l == k:
            term = diff(g_tilde[j,l], coords[i]) + diff(g_tilde[i,l], coords[j]) - diff(g_tilde[i,j], coords[l])
            res += 0.5 * g_inv[k,k] * term
    return res

def get_ricci(i, k):
    res = 0
    for l in range(3):
        # d_l Gamma^l_ik
        d_l_gamma = diff(get_gamma(l, i, k), coords[l])

        # d_k Gamma^l_il
        d_k_gamma = diff(get_gamma(l, i, l), coords[k])

        term1 = d_l_gamma - d_k_gamma

        term2 = 0
        for m in range(3):
            term2 += get_gamma(l, l, m) * get_gamma(m, i, k)
            term2 -= get_gamma(l, k, m) * get_gamma(m, i, l)

        res += term1 + term2
    return simplify(res)

print("Computing Ricci components...")
r_tt = get_ricci(0, 0)
print("Ric_tt:", r_tt)

r_th_th = get_ricci(1, 1)
print("Ric_theta_theta:", r_th_th)
