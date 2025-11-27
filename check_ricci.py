import sympy
from sympy import symbols, Function, Matrix, diff, simplify, exp, log, sin, diag

# Define coordinates
t, theta, varphi = symbols('t theta varphi', real=True)
s = symbols('s', real=True, positive=True)

# Cylinder metric g_bar
# g_bar = dt^2 + g_S2
# g_S2 = dtheta^2 + sin(theta)^2 dvarphi^2
g_bar = diag(1, 1, sin(theta)**2)
inv_g_bar = diag(1, 1, 1/sin(theta)**2)

# Coordinate list
coords = [t, theta, varphi]

# Conformal factor phi
# phi ~ c * s^{1/2} + v
# s = exp(-t)
# phi = c * exp(-t/2) + epsilon * exp(-t*(1/2 + delta))
c, epsilon, delta = symbols('c epsilon delta', real=True, positive=True)
phi = c * exp(-t/2) + epsilon * exp(-t*(0.5 + delta))

# Conformal metric g_tilde = phi^4 * g_bar
omega = 2 * log(phi)
# g_tilde = exp(2*omega) * g_bar

# We want to check scaling of Ricci tensor of g_tilde
# Formula: Ric_tilde = Ric_bar - (n-2)(Hess(omega) - d(omega) x d(omega)) - (Lap(omega) + (n-2)|d(omega)|^2) g_bar
# n = 3

# Compute derivatives of omega
# omega depends only on t
d_omega = [diff(omega, x) for x in coords]
# d_omega = [omega', 0, 0]

# Hess(omega)
# Hess_ij = partial_i partial_j omega - Gamma^k_ij partial_k omega
# Gamma for g_bar
# Since g_bar is product metric, Gamma^t_ij = 0 except maybe Gamma^t_tt? No, dt^2 is flat.
# Gamma^k_tt = 0.
# So Hess_tt = omega''.
# Hess_theta_theta = - Gamma^t_theta_theta omega' ? No.
# g_bar = diag(1, 1, sin^2)
# Christoffel symbols involving t are all zero because g_tt=1 is constant and g_ij doesn't depend on t.
# So Hess(omega) = diag(omega'', 0, 0)
hess_omega = diag(diff(omega, t, t), 0, 0)

# |d(omega)|^2_bar
# g^{tt} (omega')^2 = (omega')^2
norm_d_omega_sq = diff(omega, t)**2

# Lap(omega) = tr(Hess) = g^{ij} Hess_ij = omega''
lap_omega = diff(omega, t, t)

# Ric_bar
# For cylinder dt^2 + g_S2:
# Ric_tt = 0
# Ric_theta_theta = 1 (Ricci of unit sphere)
# Ric_varphi_varphi = sin^2(theta)
ric_bar = diag(0, 1, sin(theta)**2)

# Ric_tilde components (covariant)
# R_ij_tilde = R_ij_bar - (hess_ij - d_i omega d_j omega) - (lap_omega + |d_omega|^2) g_ij_bar
# Note: formula in paper: Ric_tilde = Ric_bar - (n-2)(Hess - d x d) - (Lap + (n-2)|d|^2)g
# n=3
# Ric_tilde = Ric_bar - (Hess - d x d) - (Lap + |d|^2)g

term1 = ric_bar
term2 = Matrix(3, 3, lambda i, j: hess_omega[i,j] - d_omega[i]*d_omega[j])
term3 = (lap_omega + norm_d_omega_sq) * g_bar

ric_tilde = term1 - term2 - term3

# Evaluate scaling for large t (small s)
# phi = c * e^{-t/2} (1 + eps/c e^{-delta t})
# omega = 2 log phi = 2 (-t/2 + log(c) + log(1 + ...)) = -t + 2 log c + 2(eps/c) e^{-delta t} ...
# omega' = -1 - 2 delta (eps/c) e^{-delta t}
# omega'' = 2 delta^2 (eps/c) e^{-delta t}

# Check cancellation
# Lap + |d|^2 = omega'' + (omega')^2
# (omega')^2 = (-1 - pert)^2 = 1 + 2 pert
# omega'' ~ pert
# So Lap + |d|^2 = 1 + O(e^{-delta t})

# Components:
# RR_tt: 0 - (omega'' - (omega')^2) - (omega'' + (omega')^2)*1
# = -omega'' + (omega')^2 - omega'' - (omega')^2 = -2 omega''
# ~ e^{-delta t}

# RR_theta_theta: 1 - (0 - 0) - (omega'' + (omega')^2)*1
# = 1 - (omega'' + (omega')^2)
# = 1 - (1 + O(e^{-delta t})) ~ e^{-delta t}

# So Ric_tilde components in g_bar basis scale like e^{-delta t}.

# In s coordinates: s = e^{-t}, e^{-delta t} = s^delta.
# So components scale like s^delta.

# Norm |Ric_tilde|_tilde
# |Ric|^2 = g_tilde^{ac} g_tilde^{bd} R_ac R_bd
# = phi^{-4} g_bar^{ac} phi^{-4} g_bar^{bd} R_ac R_bd
# = phi^{-8} |Ric_tilde|_bar^2
# |Ric_tilde|_bar^2 = g^{tt}g^{tt} R_tt^2 + ...
# ~ (s^delta)^2
# So |Ric_tilde|_tilde ~ phi^{-4} s^delta ~ (s^{1/2})^{-4} s^delta = s^{-2} s^delta = s^{-2+delta}

# Volume element dVol_tilde
# dVol_tilde = phi^6 dVol_bar
# dVol_bar = dt dsigma = (ds/s) dsigma
# phi^6 ~ (s^{1/2})^6 = s^3
# dVol_tilde ~ s^3 (ds/s) = s^2 ds
# Note: Paper said s^5, I suspected s^2. This confirms s^2.

# Integrability:
# int |Ric| dVol ~ int s^{-2+delta} s^2 ds = int s^delta ds
# Converges for delta > -1.
# Since delta > 0, it always converges!

# Wait, let me double check the matrix algebra.
print("Running sympy check...")

t_val = symbols('t_val', real=True)
def get_order(expr):
    # Substitute phi
    expr_t = expr.subs(theta, 0) # theta doesn't affect scaling in t usually
    # We want to see the leading order term in e^{-t} expansion
    # actually let's just print the simplified expression
    return simplify(expr)

# Print components
print("Ric_tt:")
print(simplify(ric_tilde[0,0]))
print("Ric_theta_theta:")
print(simplify(ric_tilde[1,1]))
