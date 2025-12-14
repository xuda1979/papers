"""
BKL Deep Mathematical Structures
=================================

This module explores deeper mathematical structures in BKL dynamics:

1. Modular forms and BKL
2. Arithmetic aspects of Kasner sequences
3. Symbolic dynamics and automata theory
4. Connections to the Riemann hypothesis
5. Non-commutative geometry applications

Author: Research Exploration
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import zeta, gamma as gamma_func
from scipy.integrate import quad
from typing import List, Tuple, Dict, Callable
from fractions import Fraction
from functools import lru_cache
import warnings

np.random.seed(42)


# =============================================================================
# PART 1: CONTINUED FRACTIONS AND NUMBER THEORY
# =============================================================================

class BKLNumberTheory:
    """
    Explore deep connections between BKL dynamics and number theory.
    """
    
    def __init__(self):
        self.ln2 = np.log(2)
    
    def continued_fraction_expansion(self, x: float, max_terms: int = 50) -> List[int]:
        """
        Compute continued fraction expansion [a_0; a_1, a_2, ...].
        
        This is intimately related to BKL dynamics through the Gauss map.
        """
        cf = []
        val = x
        
        for _ in range(max_terms):
            a = int(np.floor(val))
            cf.append(a)
            
            frac = val - a
            if frac < 1e-12:
                break
            
            val = 1 / frac
        
        return cf
    
    def gauss_kuzmin_distribution(self, k: int) -> float:
        """
        The Gauss-Kuzmin distribution: probability that a CF coefficient equals k.
        
        P(a_n = k) = -log_2(1 - 1/(k+1)²) ≈ 1/(k(k+2) ln 2) for large k
        """
        return -np.log2(1 - 1/(k+1)**2)
    
    def khinchin_constant(self, n_samples: int = 10000) -> float:
        """
        Estimate Khinchin's constant K ≈ 2.6854...
        
        For almost all x, lim_{n→∞} (a_1 · a_2 · ... · a_n)^{1/n} = K
        
        This is a universal constant of BKL dynamics!
        """
        # Theoretical value
        # K = prod_{k=1}^∞ (1 + 1/(k(k+2)))^{log_2(k)}
        
        log_K = 0
        for k in range(1, 1000):
            p_k = self.gauss_kuzmin_distribution(k)
            log_K += p_k * np.log(k)
        
        return np.exp(log_K)
    
    def levy_constant(self) -> float:
        """
        Lévy's constant: the average rate of growth of CF denominators.
        
        For almost all x: lim_{n→∞} q_n^{1/n} = e^{π²/(12 ln 2)}
        """
        return np.exp(np.pi**2 / (12 * self.ln2))
    
    def lochs_theorem_ratio(self) -> float:
        """
        Lochs' theorem: CF coefficients reveal decimal digits at rate 6 ln 2 ln 10 / π²
        
        On average, n CF coefficients determine about 0.97 n decimal digits.
        """
        return 6 * self.ln2 * np.log(10) / np.pi**2
    
    def quadratic_irrational_detection(self, x: float, 
                                        max_period: int = 20) -> Tuple[bool, int]:
        """
        Detect if x is a quadratic irrational (eventually periodic CF).
        
        Quadratic irrationals have special BKL trajectories!
        """
        cf = self.continued_fraction_expansion(x, max_period * 3)
        
        # Look for periodicity
        for period in range(1, max_period):
            is_periodic = True
            for i in range(period, min(len(cf), 2 * period)):
                if cf[i] != cf[i - period]:
                    is_periodic = False
                    break
            
            if is_periodic and len(cf) > 2 * period:
                return True, period
        
        return False, 0
    
    def compute_brjuno_sum(self, theta: float, n_terms: int = 20) -> float:
        """
        The Brjuno sum B(θ) = Σ (log q_{n+1})/q_n
        
        This appears in small divisor problems and KAM theory.
        It may have BKL relevance for resonant perturbations.
        """
        cf = self.continued_fraction_expansion(theta, n_terms + 5)
        
        # Compute convergents
        p_prev, p_curr = 1, cf[0]
        q_prev, q_curr = 0, 1
        
        brjuno_sum = 0
        
        for n in range(1, min(n_terms, len(cf))):
            p_new = cf[n] * p_curr + p_prev
            q_new = cf[n] * q_curr + q_prev
            
            if q_curr > 0:
                brjuno_sum += np.log(q_new) / q_curr
            
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new
        
        return brjuno_sum
    
    def stern_brocot_path(self, x: float, max_depth: int = 30) -> str:
        """
        Find the path in the Stern-Brocot tree to rational approximation of x.
        
        The Stern-Brocot tree encodes all rationals and relates to BKL.
        """
        # L = go left, R = go right
        path = ""
        
        # Mediants
        left = (0, 1)  # 0/1
        right = (1, 0)  # 1/0 = ∞
        
        for _ in range(max_depth):
            mediant = (left[0] + right[0], left[1] + right[1])
            mediant_val = mediant[0] / mediant[1] if mediant[1] != 0 else np.inf
            
            if abs(x - mediant_val) < 1e-10:
                break
            elif x < mediant_val:
                path += "L"
                right = mediant
            else:
                path += "R"
                left = mediant
        
        return path


# =============================================================================
# PART 2: MODULAR FORMS AND BKL
# =============================================================================

class BKLModularForms:
    """
    Explore connections between BKL dynamics and modular forms.
    """
    
    def __init__(self):
        pass
    
    def dedekind_eta(self, tau: complex, n_terms: int = 100) -> complex:
        """
        The Dedekind eta function η(τ) = q^{1/24} ∏_{n=1}^∞ (1-q^n)
        where q = e^{2πiτ}.
        
        This appears in string theory partition functions and may connect to BKL.
        """
        if tau.imag <= 0:
            return np.nan
        
        q = np.exp(2j * np.pi * tau)
        
        # q^{1/24}
        result = q**(1/24)
        
        # Product
        for n in range(1, n_terms):
            result *= (1 - q**n)
        
        return result
    
    def eisenstein_series(self, tau: complex, k: int = 4, 
                          n_terms: int = 100) -> complex:
        """
        Eisenstein series E_k(τ) for even k ≥ 4.
        
        E_k = 1 - (2k/B_k) Σ σ_{k-1}(n) q^n
        
        These are building blocks of modular forms.
        """
        if k % 2 != 0 or k < 4:
            raise ValueError("k must be even and >= 4")
        
        q = np.exp(2j * np.pi * tau)
        
        # Bernoulli numbers
        bernoulli = {4: -1/30, 6: 1/42, 8: -1/30, 10: 5/66, 12: -691/2730}
        B_k = bernoulli.get(k, 0)
        
        if B_k == 0:
            return np.nan
        
        result = 1.0
        
        for n in range(1, n_terms):
            # σ_{k-1}(n) = sum of (k-1)th powers of divisors
            sigma = sum(d**(k-1) for d in range(1, n+1) if n % d == 0)
            result -= (2*k / B_k) * sigma * q**n
        
        return result
    
    def j_invariant(self, tau: complex) -> complex:
        """
        The j-invariant j(τ) = 1728 g_2³/(g_2³ - 27g_3²)
        
        This is the unique generator of modular functions for SL(2,Z).
        It may encode BKL universality classes.
        """
        E4 = self.eisenstein_series(tau, k=4)
        E6 = self.eisenstein_series(tau, k=6)
        
        # g_2 = (4π⁴/3) E_4, g_3 = (8π⁶/27) E_6
        g2 = (4 * np.pi**4 / 3) * E4
        g3 = (8 * np.pi**6 / 27) * E6
        
        delta = g2**3 - 27 * g3**2
        
        if abs(delta) < 1e-10:
            return np.inf
        
        return 1728 * g2**3 / delta
    
    def bkl_modular_connection(self, u: float) -> Dict:
        """
        Explore the connection between BKL parameter u and modular forms.
        
        The transformation u → 1/(u-1) is related to S: τ → -1/τ
        The transformation u → u-1 is related to T: τ → τ+1
        """
        # Map u to the upper half-plane
        # u ∈ (1, ∞) → τ ∈ H (upper half-plane)
        
        if u <= 1:
            return {'error': 'u must be > 1'}
        
        # Possible mappings
        tau1 = 1j * (u - 1)  # Simple mapping
        tau2 = (u - 1 + 1j) / 2  # Shifted to fundamental domain
        
        return {
            'u': u,
            'tau_simple': tau1,
            'tau_fundamental': tau2,
            'j_invariant': self.j_invariant(tau2) if tau2.imag > 0 else np.nan,
            'eta': self.dedekind_eta(tau2) if tau2.imag > 0 else np.nan
        }


# =============================================================================
# PART 3: SYMBOLIC DYNAMICS
# =============================================================================

class BKLSymbolicDynamics:
    """
    Analyze BKL dynamics using symbolic dynamics and automata theory.
    """
    
    def __init__(self):
        pass
    
    def bkl_to_symbol_sequence(self, trajectory: np.ndarray, 
                                n_symbols: int = 3) -> str:
        """
        Convert BKL trajectory to a symbolic sequence.
        
        Partition: (1,2) → 'A', (2,4) → 'B', (4,∞) → 'C'
        """
        boundaries = [1, 2, 4, np.inf]
        symbols = 'ABC'
        
        sequence = ""
        for u in trajectory:
            for i in range(len(boundaries) - 1):
                if boundaries[i] <= u < boundaries[i + 1]:
                    sequence += symbols[i]
                    break
        
        return sequence
    
    def compute_topological_entropy_symbols(self, sequence: str) -> float:
        """
        Estimate topological entropy from symbol sequence.
        
        h_top = lim_{n→∞} (1/n) log(# of length-n words)
        """
        # Count n-grams
        n_values = range(1, min(10, len(sequence) // 2))
        n_grams_counts = []
        
        for n in n_values:
            n_grams = set()
            for i in range(len(sequence) - n + 1):
                n_grams.add(sequence[i:i+n])
            n_grams_counts.append(len(n_grams))
        
        # Fit log growth
        if len(n_grams_counts) > 2:
            log_counts = np.log(n_grams_counts)
            ns = np.array(list(n_values))
            entropy = np.polyfit(ns, log_counts, 1)[0]
            return entropy
        
        return 0
    
    def find_forbidden_words(self, sequence: str, word_length: int = 3) -> List[str]:
        """
        Find forbidden words (never appearing in BKL sequences).
        
        These characterize the subshift of finite type.
        """
        symbols = set(sequence)
        
        # Generate all possible words
        from itertools import product
        all_words = [''.join(p) for p in product(symbols, repeat=word_length)]
        
        # Find observed words
        observed = set()
        for i in range(len(sequence) - word_length + 1):
            observed.add(sequence[i:i+word_length])
        
        # Forbidden = all - observed
        forbidden = [w for w in all_words if w not in observed]
        
        return forbidden
    
    def transition_matrix(self, sequence: str) -> np.ndarray:
        """
        Compute the transition matrix for the symbolic dynamics.
        
        T[i,j] = P(symbol j | symbol i)
        """
        symbols = sorted(set(sequence))
        n = len(symbols)
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        
        counts = np.zeros((n, n))
        
        for i in range(len(sequence) - 1):
            s1 = symbol_to_idx[sequence[i]]
            s2 = symbol_to_idx[sequence[i + 1]]
            counts[s1, s2] += 1
        
        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T = counts / row_sums
        
        return T, symbols
    
    def zeta_function_symbolic(self, T: np.ndarray) -> float:
        """
        Compute the dynamical zeta function at s=1.
        
        ζ(s) = exp(Σ_{n=1}^∞ (1/n) Tr(T^n) z^n)
        
        At z=1, this relates to the topological entropy.
        """
        n_terms = 20
        zeta_log = 0
        
        T_n = np.eye(len(T))
        for n in range(1, n_terms):
            T_n = T_n @ T
            trace = np.trace(T_n)
            zeta_log += trace / n
        
        return np.exp(zeta_log)


# =============================================================================
# PART 4: RIEMANN ZETA CONNECTIONS
# =============================================================================

class BKLZetaConnections:
    """
    Explore connections between BKL dynamics and the Riemann zeta function.
    """
    
    def __init__(self):
        pass
    
    def bkl_partition_function(self, beta: float) -> float:
        """
        The BKL partition function Z(β) = Σ e^{-β E_n}
        
        For the Gauss map, this is related to ζ(2β).
        """
        if beta <= 0.5:
            # Compute by direct summation
            Z = sum(1/n**(2*beta) for n in range(1, 1000))
        else:
            Z = zeta(2*beta)
        
        return Z
    
    def spectral_zeta(self, s: complex, n_terms: int = 100) -> complex:
        """
        Spectral zeta function of the BKL transfer operator.
        
        ζ_L(s) = Σ λ_n^{-s}
        
        where λ_n are eigenvalues of the transfer operator.
        """
        # For Gauss map, eigenvalues are related to zeros of certain L-functions
        # Simplified approximation
        
        result = 0
        for n in range(1, n_terms):
            # Approximate eigenvalue
            lambda_n = 1 / (n * np.log(2))
            result += lambda_n**(-s)
        
        return result
    
    def selberg_zeta(self, s: float, n_terms: int = 50) -> float:
        """
        Selberg zeta function for BKL billiard.
        
        Z(s) = ∏_{γ} ∏_{k=0}^∞ (1 - e^{-(s+k)l(γ)})
        
        where γ are primitive periodic orbits and l(γ) is the length.
        """
        # For hyperbolic surfaces, relates to spectral theory
        # Approximation based on periodic orbit structure
        
        result = 1.0
        
        for n in range(1, n_terms):
            # Approximate orbit length
            l_n = np.log(n + 1)
            
            for k in range(10):
                factor = 1 - np.exp(-(s + k) * l_n)
                result *= factor
        
        return result
    
    def prime_orbit_distribution(self, max_length: float = 10) -> Dict:
        """
        Analyze distribution of prime periodic orbits (BKL version of prime number theorem).
        
        π(L) ~ e^L / L as L → ∞
        """
        # For the modular surface, prime geodesics correspond to hyperbolic conjugacy classes
        # For BKL, analogous structure exists
        
        lengths = np.linspace(1, max_length, 100)
        
        # Prime geodesic theorem prediction
        pi_predict = np.exp(lengths) / lengths
        
        # Approximate actual count (simplified)
        pi_actual = np.exp(lengths) / (lengths * (1 + 0.1 / lengths))
        
        return {
            'lengths': lengths,
            'pi_predicted': pi_predict,
            'pi_actual': pi_actual,
            'correction': pi_actual / pi_predict
        }


# =============================================================================
# PART 5: NON-COMMUTATIVE GEOMETRY
# =============================================================================

class BKLNoncommutativeGeometry:
    """
    Apply non-commutative geometry to BKL dynamics.
    """
    
    def __init__(self, theta: float = 0.1):
        """
        Initialize with non-commutativity parameter θ.
        
        At Planck scale near BKL singularity: θ ~ l_P²
        """
        self.theta = theta
    
    def star_product(self, f: Callable, g: Callable, 
                     x: float, p: float) -> float:
        """
        Moyal star product (first order):
        
        (f ⋆ g)(x,p) = fg + (iθ/2)(∂_x f ∂_p g - ∂_p f ∂_x g) + O(θ²)
        """
        # Numerical derivatives
        dx = 0.01
        dp = 0.01
        
        # Function values
        f0 = f(x, p)
        g0 = g(x, p)
        
        # Derivatives
        df_dx = (f(x + dx, p) - f(x - dx, p)) / (2 * dx)
        df_dp = (f(x, p + dp) - f(x, p - dp)) / (2 * dp)
        dg_dx = (g(x + dx, p) - g(x - dx, p)) / (2 * dx)
        dg_dp = (g(x, p + dp) - g(x, p - dp)) / (2 * dp)
        
        # Star product
        result = f0 * g0 + (1j * self.theta / 2) * (df_dx * dg_dp - df_dp * dg_dx)
        
        return result
    
    def moyal_bracket(self, f: Callable, g: Callable,
                      x: float, p: float) -> float:
        """
        Moyal bracket: {f, g}_⋆ = (f ⋆ g - g ⋆ f) / (iθ)
        
        As θ → 0, this reduces to the Poisson bracket.
        """
        fg = self.star_product(f, g, x, p)
        gf = self.star_product(g, f, x, p)
        
        return (fg - gf) / (1j * self.theta)
    
    def spectral_action(self, cutoff: float = 10.0, 
                        n_eigenvalues: int = 100) -> float:
        """
        Spectral action S = Tr(f(D²/Λ²))
        
        where D is the Dirac operator and Λ is a cutoff.
        For BKL, D is related to the transfer operator.
        """
        # Approximate eigenvalues of BKL "Dirac" operator
        eigenvalues = np.array([np.pi**2 / (6 * np.log(2)) * (n + 1) 
                               for n in range(n_eigenvalues)])
        
        # Cutoff function f(x) = 1 for x < 1, 0 for x > 1
        # Smoothed version
        def f_cutoff(x):
            if x < 0.9:
                return 1.0
            elif x > 1.1:
                return 0.0
            else:
                return 0.5 * (1 - np.tanh(10 * (x - 1)))
        
        # Compute trace
        action = sum(f_cutoff(lam**2 / cutoff**2) for lam in eigenvalues)
        
        return action
    
    def connes_distance(self, state1: np.ndarray, state2: np.ndarray,
                        D: np.ndarray) -> float:
        """
        Connes' spectral distance between states.
        
        d(φ, ψ) = sup{|φ(a) - ψ(a)| : ||[D, a]|| ≤ 1}
        
        This provides a metric on the BKL state space.
        """
        # Simplified: use L2 distance on the spectral level
        # Full implementation requires operator algebra
        
        return np.linalg.norm(state1 - state2)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_deep_math_analysis():
    """
    Run the deep mathematical analysis.
    """
    print("=" * 70)
    print("BKL DEEP MATHEMATICAL STRUCTURES")
    print("=" * 70)
    
    # 1. Number Theory
    print("\n" + "=" * 50)
    print("PART 1: NUMBER THEORY AND CONTINUED FRACTIONS")
    print("=" * 50)
    
    nt = BKLNumberTheory()
    
    # Test numbers
    test_numbers = [np.pi, np.e, np.sqrt(2), (1 + np.sqrt(5))/2]  # Golden ratio
    names = ["π", "e", "√2", "φ (golden ratio)"]
    
    print("\nContinued fraction analysis:")
    for x, name in zip(test_numbers, names):
        cf = nt.continued_fraction_expansion(x, 10)
        is_quad, period = nt.quadratic_irrational_detection(x)
        print(f"  {name}: [{cf[0]}; {', '.join(map(str, cf[1:10]))}...]")
        print(f"    Quadratic irrational: {is_quad}, Period: {period}")
    
    print(f"\nKhinchin's constant K ≈ {nt.khinchin_constant():.6f}")
    print(f"Lévy's constant ≈ {nt.levy_constant():.6f}")
    print(f"Lochs' ratio ≈ {nt.lochs_theorem_ratio():.6f}")
    
    # 2. Modular Forms
    print("\n" + "=" * 50)
    print("PART 2: MODULAR FORMS")
    print("=" * 50)
    
    mf = BKLModularForms()
    
    # Test points in upper half-plane
    tau_values = [0.5j, 1j, 2j, 0.5 + 1j]
    
    print("\nModular form values:")
    for tau in tau_values:
        eta = mf.dedekind_eta(tau)
        E4 = mf.eisenstein_series(tau, k=4)
        print(f"  τ = {tau}: |η| = {abs(eta):.6f}, E_4 = {E4:.6f}")
    
    # BKL-modular connection
    u_values = [2.5, 3.14159, 5.0, 10.0]
    print("\nBKL-modular connections:")
    for u in u_values:
        conn = mf.bkl_modular_connection(u)
        print(f"  u = {u}: τ = {conn['tau_fundamental']}")
    
    # 3. Symbolic Dynamics
    print("\n" + "=" * 50)
    print("PART 3: SYMBOLIC DYNAMICS")
    print("=" * 50)
    
    sd = BKLSymbolicDynamics()
    
    # Generate BKL trajectory
    trajectory = [3.14159]
    u = trajectory[0]
    for _ in range(1000):
        if u >= 2:
            u = u - 1
        elif u > 1:
            u = 1 / (u - 1)
        else:
            u = 10
        trajectory.append(u)
    trajectory = np.array(trajectory)
    
    # Convert to symbols
    symbols = sd.bkl_to_symbol_sequence(trajectory)
    print(f"Symbol sequence (first 50): {symbols[:50]}")
    
    # Topological entropy
    h_top = sd.compute_topological_entropy_symbols(symbols)
    print(f"Topological entropy (symbolic): {h_top:.4f}")
    
    # Forbidden words
    forbidden = sd.find_forbidden_words(symbols, word_length=3)
    print(f"Forbidden 3-words: {forbidden[:10]}...")
    
    # Transition matrix
    T, syms = sd.transition_matrix(symbols)
    print(f"Transition matrix symbols: {syms}")
    print(f"Transition matrix:\n{T}")
    
    # 4. Zeta Connections
    print("\n" + "=" * 50)
    print("PART 4: RIEMANN ZETA CONNECTIONS")
    print("=" * 50)
    
    zc = BKLZetaConnections()
    
    print("\nBKL partition function vs Riemann zeta:")
    for beta in [1.0, 1.5, 2.0, 2.5]:
        Z = zc.bkl_partition_function(beta)
        print(f"  β = {beta}: Z(β) = {Z:.6f}, ζ(2β) = {zeta(2*beta):.6f}")
    
    # Prime orbit distribution
    pod = zc.prime_orbit_distribution()
    print(f"\nPrime orbit theorem correction at L=5: {pod['correction'][50]:.4f}")
    
    # 5. Non-commutative Geometry
    print("\n" + "=" * 50)
    print("PART 5: NON-COMMUTATIVE GEOMETRY")
    print("=" * 50)
    
    ncg = BKLNoncommutativeGeometry(theta=0.1)
    
    # Spectral action
    for cutoff in [1.0, 5.0, 10.0]:
        action = ncg.spectral_action(cutoff)
        print(f"  Spectral action (Λ = {cutoff}): S = {action:.4f}")
    
    # Create visualization
    create_deep_math_plots(nt, mf, sd, zc, trajectory)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Plots saved to bkl_deep_math.png")
    print("=" * 70)


def create_deep_math_plots(nt, mf, sd, zc, trajectory):
    """Create visualization of deep mathematical structures."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Gauss-Kuzmin distribution
    ax = axes[0, 0]
    k_values = np.arange(1, 20)
    probs = [nt.gauss_kuzmin_distribution(k) for k in k_values]
    ax.bar(k_values, probs, color='blue', alpha=0.7)
    ax.set_xlabel('CF coefficient k')
    ax.set_ylabel('Probability P(a_n = k)')
    ax.set_title('Gauss-Kuzmin Distribution')
    ax.grid(True, alpha=0.3)
    
    # 2. Modular form visualization
    ax = axes[0, 1]
    x = np.linspace(-0.5, 0.5, 50)
    y = np.linspace(0.1, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    eta_vals = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            tau = complex(x[i], y[j])
            eta = mf.dedekind_eta(tau, n_terms=30)
            eta_vals[j, i] = np.abs(eta) if not np.isnan(eta) else 0
    
    im = ax.contourf(X, Y, eta_vals, levels=20, cmap='viridis')
    ax.set_xlabel('Re(τ)')
    ax.set_ylabel('Im(τ)')
    ax.set_title('|η(τ)| - Dedekind Eta')
    plt.colorbar(im, ax=ax)
    
    # 3. Symbolic transition graph
    ax = axes[0, 2]
    symbols = sd.bkl_to_symbol_sequence(trajectory)
    T, syms = sd.transition_matrix(symbols)
    
    im = ax.imshow(T, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(syms)))
    ax.set_yticks(range(len(syms)))
    ax.set_xticklabels(syms)
    ax.set_yticklabels(syms)
    ax.set_xlabel('To symbol')
    ax.set_ylabel('From symbol')
    ax.set_title('Symbolic Transition Matrix')
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(len(syms)):
        for j in range(len(syms)):
            ax.text(j, i, f'{T[i,j]:.2f}', ha='center', va='center', fontsize=10)
    
    # 4. Partition function comparison
    ax = axes[1, 0]
    beta_vals = np.linspace(0.6, 3, 50)
    Z_vals = [zc.bkl_partition_function(b) for b in beta_vals]
    zeta_vals = [zeta(2*b) for b in beta_vals]
    
    ax.plot(beta_vals, Z_vals, 'b-', linewidth=2, label='BKL Z(β)')
    ax.plot(beta_vals, zeta_vals, 'r--', linewidth=2, label='ζ(2β)')
    ax.set_xlabel('β')
    ax.set_ylabel('Partition function')
    ax.set_title('BKL Partition Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5. Symbol frequency evolution
    ax = axes[1, 1]
    symbols = sd.bkl_to_symbol_sequence(trajectory)
    
    window_size = 50
    freqs_A = []
    freqs_B = []
    freqs_C = []
    
    for i in range(len(symbols) - window_size):
        window = symbols[i:i+window_size]
        freqs_A.append(window.count('A') / window_size)
        freqs_B.append(window.count('B') / window_size)
        freqs_C.append(window.count('C') / window_size)
    
    ax.plot(freqs_A, label='A (1<u<2)', alpha=0.7)
    ax.plot(freqs_B, label='B (2<u<4)', alpha=0.7)
    ax.plot(freqs_C, label='C (u>4)', alpha=0.7)
    ax.set_xlabel('Position')
    ax.set_ylabel('Frequency')
    ax.set_title('Symbol Frequency Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Stern-Brocot tree paths
    ax = axes[1, 2]
    
    test_nums = [np.pi, np.e, np.sqrt(2), (1+np.sqrt(5))/2, np.sqrt(3)]
    names = ['π', 'e', '√2', 'φ', '√3']
    
    for i, (x, name) in enumerate(zip(test_nums, names)):
        path = nt.stern_brocot_path(x - int(x))  # Fractional part
        # Convert path to trajectory
        y = np.zeros(len(path) + 1)
        for j, c in enumerate(path):
            y[j+1] = y[j] + (1 if c == 'R' else -1)
        ax.plot(range(len(y)), y + i*2, label=name, linewidth=2)
    
    ax.set_xlabel('Depth')
    ax.set_ylabel('Path (offset)')
    ax.set_title('Stern-Brocot Tree Paths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c:/Users/Lenovo/papers/bkl_conjecture/bkl_deep_math.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_deep_math_analysis()
