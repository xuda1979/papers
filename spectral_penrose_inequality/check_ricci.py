import numpy as np
from scipy.interpolate import interp1d

def mollifier(t, epsilon):
    """Standard smooth mollifier supported on [-epsilon, epsilon]."""
    if abs(t) >= epsilon:
        return 0.0
    x = t / epsilon
    # Normalized standard mollifier
    # We ignore the exact constant C because we normalize numerically
    return np.exp(-1 / (1 - x**2))

def get_normalized_mollifier(epsilon, num_points=2000):
    t = np.linspace(-epsilon, epsilon, num_points)
    y = np.array([mollifier(ti, epsilon) for ti in t])
    integral = np.trapz(y, t)
    return t, y / integral

def convolve(f_vals, s_grid, epsilon):
    """
    Compute convolution (eta_epsilon * f) numerically.
    f_vals is defined on s_grid.
    We assume s_grid is uniform.
    """
    dt = s_grid[1] - s_grid[0]
    taus, eta_vals = get_normalized_mollifier(epsilon, num_points=1000)

    # Interpolate f to evaluate at s - tau
    f_interp = interp1d(s_grid, f_vals, kind='linear', fill_value="extrapolate")

    res = []
    # We only evaluate where convolution is valid without boundary effects
    # but here we rely on extrapolate or just evaluate in the middle

    for t in s_grid:
        # eta is non-zero in [t-eps, t+eps] ??
        # (f * g)(t) = int f(t-tau) g(tau) dtau
        # We need f at t - taus
        f_shifted = f_interp(t - taus)
        val = np.trapz(f_shifted * eta_vals, taus)
        res.append(val)

    return np.array(res)

def test_cancellation_structure():
    """
    Test E = (g_eps)^(-1) * g_dot_eps - (eta * (g^(-1) * g_dot))
    for g(s) = 1 + |s|.
    """

    def g_func(s):
        return 2.0 + np.abs(s) # Make sure it's bounded away from 0

    def g_dot_func(s):
        return 1.0 if s > 0 else -1.0

    epsilons = [0.1, 0.05, 0.02, 0.01, 0.005]

    print(f"{'Epsilon':<10} | {'max|dE/ds|':<15} | {'Scaling':<10}")
    print("-" * 40)

    prev_max_deriv = None

    for eps in epsilons:
        # Grid
        L = 5 * eps
        s_eval = np.linspace(-L, L, 5000)
        ds = s_eval[1] - s_eval[0]

        # Original functions on grid
        g_vals = g_func(s_eval)
        g_dot_vals = np.array([g_dot_func(s) for s in s_eval])

        # 1. Compute smooth g_eps and g_dot_eps
        g_eps = convolve(g_vals, s_eval, eps)
        g_dot_eps = convolve(g_dot_vals, s_eval, eps)

        # 2. Term 1: (g_eps)^(-1) * g_dot_eps
        Term1 = (1.0 / g_eps) * g_dot_eps

        # 3. Term 2: eta * (g^(-1) * g_dot)
        prod = (1.0 / g_vals) * g_dot_vals
        Term2 = convolve(prod, s_eval, eps)

        # 4. Error
        E = Term1 - Term2

        # 5. Derivative dE/ds
        # Only look in the middle to avoid boundary artifacts of convolution
        valid_mask = (s_eval > -2*eps) & (s_eval < 2*eps)
        if np.sum(valid_mask) == 0:
             print("Grid too small")
             continue

        E_valid = E[valid_mask]
        # s_valid = s_eval[valid_mask]

        # Numerical derivative
        dE_ds = np.gradient(E, ds)
        dE_ds_valid = dE_ds[valid_mask]

        max_deriv = np.max(np.abs(dE_ds_valid))

        scaling = "N/A"
        if prev_max_deriv is not None:
            ratio = max_deriv / prev_max_deriv
            expected_ratio_if_linear = epsilons[epsilons.index(eps)-1] / eps
            if abs(ratio - 1) < 0.5:
                scaling = "O(1)"
            elif abs(ratio - expected_ratio_if_linear) < 1.0:
                scaling = "O(1/eps)"
            else:
                scaling = f"Ratio {ratio:.2f}"

        print(f"{eps:<10} | {max_deriv:<15.4f} | {scaling:<10}")
        prev_max_deriv = max_deriv

if __name__ == "__main__":
    test_cancellation_structure()
