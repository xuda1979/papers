"""
BKL Conjecture Simulation and Exploration Framework
====================================================

This module implements numerical simulations and visualizations for exploring
the Belinski-Khalatnikov-Lifshitz (BKL) conjecture on cosmological singularities.

The BKL conjecture describes the generic behavior of spacetime near singularities:
1. Locality: Spatial points decouple near the singularity
2. Oscillatory behavior: Infinite sequence of Kasner epochs
3. Chaos: Sensitive dependence on initial conditions (Mixmaster dynamics)

Author: Research Exploration
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

# =============================================================================
# PART 1: KASNER SOLUTIONS
# =============================================================================

class KasnerSolution:
    """
    The Kasner solution: ds² = -dt² + t^{2p₁}dx² + t^{2p₂}dy² + t^{2p₃}dz²
    
    Constraints:
        p₁ + p₂ + p₃ = 1
        p₁² + p₂² + p₃² = 1
    """
    
    @staticmethod
    def exponents_from_u(u: float) -> Tuple[float, float, float]:
        """
        Compute Kasner exponents from the BKL parameter u ≥ 1.
        
        Returns (p₁, p₂, p₃) with p₁ ≤ p₂ ≤ p₃
        """
        denom = 1 + u + u**2
        p1 = -u / denom
        p2 = (1 + u) / denom
        p3 = u * (1 + u) / denom
        return (p1, p2, p3)
    
    @staticmethod
    def u_from_exponents(p1: float, p2: float, p3: float) -> float:
        """Inverse: compute u from Kasner exponents."""
        # From p₁ = -u/(1+u+u²), solve for u
        if p1 >= 0:
            return 1.0  # Edge case
        # Quadratic: p₁(1 + u + u²) = -u => p₁u² + (p₁+1)u + p₁ = 0
        a, b, c = p1, p1 + 1, p1
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return 1.0
        u = (-b + np.sqrt(discriminant)) / (2*a)
        return max(1.0, abs(u))
    
    @staticmethod
    def verify_constraints(p1: float, p2: float, p3: float, tol: float = 1e-10) -> bool:
        """Verify that Kasner constraints are satisfied."""
        sum_constraint = abs(p1 + p2 + p3 - 1)
        sum_sq_constraint = abs(p1**2 + p2**2 + p3**2 - 1)
        return sum_constraint < tol and sum_sq_constraint < tol
    
    @staticmethod
    def kasner_circle_parametrization(theta: float) -> Tuple[float, float, float]:
        """
        Alternative parametrization on the Kasner circle.
        The constraints define a circle in the (p₁, p₂, p₃) plane.
        """
        # Parametrize the intersection of plane and sphere
        # p₁ + p₂ + p₃ = 1 and p₁² + p₂² + p₃² = 1
        # Center of circle: (1/3, 1/3, 1/3)
        # Radius: sqrt(2/3)
        center = np.array([1/3, 1/3, 1/3])
        # Orthonormal basis in the plane
        e1 = np.array([1, -1, 0]) / np.sqrt(2)
        e2 = np.array([1, 1, -2]) / np.sqrt(6)
        radius = np.sqrt(2/3)
        
        p = center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        return tuple(sorted(p))


# =============================================================================
# PART 2: BKL DYNAMICS AND THE BKL MAP
# =============================================================================

class BKLMap:
    """
    The BKL map describing transitions between Kasner epochs.
    
    u_{n+1} = u_n - 1      if u_n ≥ 2
    u_{n+1} = 1/(u_n - 1)  if 1 < u_n < 2
    
    This is related to the Gauss map for continued fractions.
    """
    
    @staticmethod
    def iterate(u: float) -> float:
        """Single iteration of the BKL map."""
        if u >= 2:
            return u - 1
        elif u > 1:
            return 1 / (u - 1)
        else:
            return u  # Fixed point at u = 1
    
    @staticmethod
    def orbit(u0: float, n_iterations: int) -> np.ndarray:
        """Generate an orbit of the BKL map."""
        orbit = np.zeros(n_iterations + 1)
        orbit[0] = u0
        for i in range(n_iterations):
            orbit[i+1] = BKLMap.iterate(orbit[i])
        return orbit
    
    @staticmethod
    def gauss_map(x: float) -> float:
        """
        The Gauss map T(x) = 1/x - floor(1/x) for x ∈ (0, 1].
        The BKL map is conjugate to this.
        """
        if x == 0:
            return 0
        return 1/x - np.floor(1/x)
    
    @staticmethod
    def lyapunov_exponent_estimate(u0: float, n_iterations: int = 10000) -> float:
        """
        Estimate the Lyapunov exponent of the BKL map.
        The theoretical value for the Gauss map is π²/(6 ln 2) ≈ 2.37.
        """
        u = u0
        lyap_sum = 0.0
        for _ in range(n_iterations):
            # Derivative of the map
            if u >= 2:
                deriv = 1.0
            elif u > 1:
                deriv = -1 / (u - 1)**2
            else:
                deriv = 0
            
            if abs(deriv) > 1e-15:
                lyap_sum += np.log(abs(deriv))
            u = BKLMap.iterate(u)
        
        return lyap_sum / n_iterations
    
    @staticmethod
    def era_structure(u0: float, n_eras: int = 10) -> List[Tuple[int, float]]:
        """
        Decompose the orbit into eras.
        An era consists of transitions where u decreases by 1 until u < 2,
        followed by a "bounce" to 1/(u-1).
        
        Returns list of (era_length, final_u_in_era).
        """
        u = u0
        eras = []
        for _ in range(n_eras):
            era_length = 0
            while u >= 2:
                u = u - 1
                era_length += 1
            eras.append((era_length + 1, u))
            if u > 1:
                u = 1 / (u - 1)
            else:
                break
        return eras


# =============================================================================
# PART 3: MIXMASTER (BIANCHI IX) DYNAMICS
# =============================================================================

class MixmasterDynamics:
    """
    The Mixmaster (Bianchi IX) universe: the prototype for BKL oscillations.
    
    Uses the Misner parametrization:
        a = e^{α+β₊+√3β₋}
        b = e^{α+β₊-√3β₋}
        c = e^{α-2β₊}
    
    The dynamics is governed by motion in a triangular potential well.
    """
    
    def __init__(self):
        self.sqrt3 = np.sqrt(3)
    
    def potential(self, beta_plus: float, beta_minus: float) -> float:
        """
        The Mixmaster potential V(β₊, β₋).
        
        V = e^{-8β₊} - 4e^{-2β₊}cosh(2√3 β₋) + 2e^{4β₊}[cosh(4√3 β₋) - 1]
        """
        bp, bm = beta_plus, beta_minus
        term1 = np.exp(-8*bp)
        term2 = -4 * np.exp(-2*bp) * np.cosh(2*self.sqrt3*bm)
        term3 = 2 * np.exp(4*bp) * (np.cosh(4*self.sqrt3*bm) - 1)
        return term1 + term2 + term3
    
    def equations_of_motion(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Equations of motion in the (Ω, β₊, β₋, p₊, p₋) variables.
        
        Using the Hamiltonian formulation with logarithmic time τ = -ln(t).
        """
        # Unpack state
        Omega, bp, bm, pp, pm = y
        
        # Potential derivatives (approximate)
        eps = 1e-8
        V = self.potential(bp, bm)
        dV_dbp = (self.potential(bp + eps, bm) - self.potential(bp - eps, bm)) / (2*eps)
        dV_dbm = (self.potential(bp, bm + eps) - self.potential(bp, bm - eps)) / (2*eps)
        
        # Hamilton's equations
        # In the Misner time gauge
        dOmega = -3 * Omega  # Volume contracts
        dbp = pp
        dbm = pm
        dpp = -dV_dbp * np.exp(-2*Omega)  # Exponential steepening
        dpm = -dV_dbm * np.exp(-2*Omega)
        
        return np.array([dOmega, dbp, dbm, dpp, dpm])
    
    def simplified_billiard_dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Simplified billiard approximation where the potential walls are
        treated as hard reflections.
        
        State: (β₊, β₋, v₊, v₋) in the β-plane
        """
        bp, bm, vp, vm = y
        
        # Free motion between walls
        dbp = vp
        dbm = vm
        dvp = 0
        dvm = 0
        
        return np.array([dbp, dbm, dvp, dvm])
    
    def wall_positions(self, r: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return the three symmetry walls of the Mixmaster potential.
        
        In the β-plane, the walls form an equilateral triangle.
        Returns list of (point_on_wall, normal_vector).
        """
        # The three walls correspond to one scale factor dominating
        walls = []
        
        # Wall 1: β₊ = r
        walls.append((np.array([r, 0]), np.array([-1, 0])))
        
        # Wall 2: rotated by 120°
        angle = 2*np.pi/3
        p2 = r * np.array([np.cos(angle), np.sin(angle)])
        n2 = -np.array([np.cos(angle), np.sin(angle)])
        walls.append((p2, n2))
        
        # Wall 3: rotated by 240°
        angle = 4*np.pi/3
        p3 = r * np.array([np.cos(angle), np.sin(angle)])
        n3 = -np.array([np.cos(angle), np.sin(angle)])
        walls.append((p3, n3))
        
        return walls


# =============================================================================
# PART 4: HYPERBOLIC BILLIARD FORMULATION
# =============================================================================

class HyperbolicBilliard:
    """
    The cosmological billiard in hyperbolic space.
    
    Near the singularity, BKL dynamics is equivalent to geodesic motion
    in a region of hyperbolic space bounded by walls.
    
    The billiard table for vacuum gravity is finite (compact) for D ≤ 10.
    """
    
    def __init__(self, dimension: int = 4):
        """
        Initialize the billiard for spacetime dimension D.
        
        The billiard lives in a (D-1)-dimensional hyperbolic space.
        """
        self.D = dimension
        self.n = dimension - 1  # Spatial dimension
        
    def poincare_metric(self, y: float) -> float:
        """
        The Poincaré half-plane metric: ds² = (dx² + dy²)/y²
        """
        if y <= 0:
            return np.inf
        return 1 / y**2
    
    def geodesic_in_poincare(self, z0: complex, theta: float, 
                             t_span: Tuple[float, float], 
                             n_points: int = 1000) -> np.ndarray:
        """
        Compute a geodesic in the Poincaré upper half-plane.
        
        Geodesics are semicircles centered on the real axis or vertical lines.
        
        Parameters:
            z0: Starting point (complex number with Im(z0) > 0)
            theta: Initial direction angle
            t_span: Time interval
            n_points: Number of points
        """
        x0, y0 = z0.real, z0.imag
        
        # Direction vector
        vx, vy = np.cos(theta), np.sin(theta)
        
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        if abs(vx) < 1e-10:
            # Vertical geodesic
            trajectory = np.column_stack([
                np.full(n_points, x0),
                y0 * np.exp(vy * t)
            ])
        else:
            # Semicircular geodesic
            # Center at (x_c, 0) and radius R
            # Need to find center such that geodesic passes through (x0, y0) with direction (vx, vy)
            R = y0 / abs(np.sin(np.arctan2(vy, vx)))
            x_c = x0 - R * np.cos(np.arctan2(y0, vx))
            
            # Parametric form
            phi0 = np.arctan2(y0, x0 - x_c)
            phi = phi0 + t * np.sign(vx)
            
            trajectory = np.column_stack([
                x_c + R * np.cos(phi),
                R * np.sin(phi)
            ])
            
            # Keep only points in upper half-plane
            mask = trajectory[:, 1] > 0
            trajectory = trajectory[mask]
        
        return trajectory
    
    def billiard_walls_4d(self) -> List[dict]:
        """
        Define the billiard walls for 4D vacuum gravity.
        
        Returns list of wall specifications.
        """
        # In 4D, there are three symmetry walls and three curvature walls
        walls = []
        
        # Symmetry walls (from off-diagonal metric components)
        for i in range(3):
            angle = i * 2 * np.pi / 3
            walls.append({
                'type': 'symmetry',
                'normal': np.array([np.cos(angle), np.sin(angle)]),
                'offset': 0
            })
        
        # Curvature walls (from spatial Ricci tensor)
        # These depend on the specific Bianchi type
        walls.append({
            'type': 'curvature',
            'normal': np.array([1, 0]),
            'offset': 1
        })
        
        return walls
    
    def is_billiard_finite(self, dimension: int) -> bool:
        """
        Check if the cosmological billiard is finite (compact fundamental domain).
        
        For vacuum gravity: finite iff D ≤ 10
        For maximal supergravity: always finite up to D = 11
        """
        return dimension <= 10


# =============================================================================
# PART 5: NUMERICAL SIMULATION OF INHOMOGENEOUS SPACETIMES
# =============================================================================

class InhomogeneousSimulation:
    """
    Simulate the approach to singularity for inhomogeneous spacetimes.
    
    This implements a simplified version of the full Einstein equations
    to explore the BKL locality conjecture.
    """
    
    def __init__(self, n_spatial_points: int = 100):
        """
        Initialize the simulation grid.
        
        Parameters:
            n_spatial_points: Number of spatial grid points
        """
        self.N = n_spatial_points
        self.x = np.linspace(0, 2*np.pi, n_spatial_points, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        
    def initialize_metric_functions(self, 
                                   perturbation_amplitude: float = 0.1,
                                   seed: Optional[int] = None) -> dict:
        """
        Initialize the spatial metric components with small perturbations.
        
        Returns dict with metric functions g_ij(x) and extrinsic curvature K_ij(x).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Background Kasner (u = 2)
        p1, p2, p3 = KasnerSolution.exponents_from_u(2.0)
        
        # Add spatial perturbations (Fourier modes)
        n_modes = 5
        perturb = np.zeros(self.N)
        for k in range(1, n_modes + 1):
            amp = perturbation_amplitude / k
            phase = np.random.uniform(0, 2*np.pi)
            perturb += amp * np.sin(k * self.x + phase)
        
        # Metric diagonal components (logarithmic)
        alpha = np.log(1 + perturbation_amplitude * np.sin(self.x))  # ln(a)
        beta = np.log(1 + perturbation_amplitude * np.cos(2*self.x))  # ln(b)
        gamma = -alpha - beta  # Ensure det(g) = 1 initially
        
        # Extrinsic curvature (velocity of metric)
        K_alpha = p1 * np.ones(self.N) + 0.1 * perturb
        K_beta = p2 * np.ones(self.N) + 0.1 * np.roll(perturb, self.N//4)
        K_gamma = p3 * np.ones(self.N) - 0.1 * perturb - 0.1 * np.roll(perturb, self.N//4)
        
        return {
            'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'K_alpha': K_alpha, 'K_beta': K_beta, 'K_gamma': K_gamma
        }
    
    def spatial_derivative(self, f: np.ndarray) -> np.ndarray:
        """Compute spatial derivative using periodic boundary conditions."""
        return (np.roll(f, -1) - np.roll(f, 1)) / (2 * self.dx)
    
    def spatial_laplacian(self, f: np.ndarray) -> np.ndarray:
        """Compute spatial Laplacian using periodic boundary conditions."""
        return (np.roll(f, -1) - 2*f + np.roll(f, 1)) / self.dx**2
    
    def compute_velocity_dominance_ratio(self, state: dict) -> np.ndarray:
        """
        Compute the ratio of time derivatives to spatial derivatives.
        
        The BKL conjecture predicts this ratio → ∞ near the singularity.
        """
        # Time derivative terms (squared)
        time_terms = (state['K_alpha']**2 + state['K_beta']**2 + 
                     state['K_gamma']**2)
        
        # Spatial derivative terms (squared)
        d_alpha = self.spatial_derivative(state['alpha'])
        d_beta = self.spatial_derivative(state['beta'])
        d_gamma = self.spatial_derivative(state['gamma'])
        spatial_terms = d_alpha**2 + d_beta**2 + d_gamma**2 + 1e-15
        
        return time_terms / spatial_terms
    
    def detect_kasner_transitions(self, K_history: np.ndarray) -> List[int]:
        """
        Detect transitions between Kasner epochs from extrinsic curvature history.
        
        Returns indices where transitions occur.
        """
        # A transition occurs when the ordering of |K_i| changes
        transitions = []
        
        for i in range(1, len(K_history)):
            prev_order = np.argsort(np.abs(K_history[i-1]))
            curr_order = np.argsort(np.abs(K_history[i]))
            if not np.array_equal(prev_order, curr_order):
                transitions.append(i)
        
        return transitions


# =============================================================================
# PART 6: CONNECTIONS TO KAC-MOODY ALGEBRAS
# =============================================================================

class E10Connection:
    """
    Explore the connection between BKL dynamics and the E₁₀ Kac-Moody algebra.
    
    Damour-Henneaux-Nicolai discovered that the billiard walls correspond
    to roots of E₁₀ for 11D supergravity.
    """
    
    def __init__(self):
        # Cartan matrix for E₁₀
        # This is the infinite-dimensional hyperbolic extension of E₈
        self.rank = 10
        
    def cartan_matrix_e8(self) -> np.ndarray:
        """
        Return the Cartan matrix for E₈ (the finite subalgebra).
        """
        # Standard E₈ Cartan matrix
        A = 2 * np.eye(8)
        # Off-diagonal elements from Dynkin diagram
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),  # Main line
            (2, 7)  # Branch to node 8
        ]
        # Correction: E₈ has a specific structure
        A = np.array([
            [ 2, -1,  0,  0,  0,  0,  0,  0],
            [-1,  2, -1,  0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0,  0,  0, -1],
            [ 0,  0, -1,  2, -1,  0,  0,  0],
            [ 0,  0,  0, -1,  2, -1,  0,  0],
            [ 0,  0,  0,  0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0, -1,  2,  0],
            [ 0,  0, -1,  0,  0,  0,  0,  2]
        ])
        return A
    
    def root_system_simple_roots(self) -> List[np.ndarray]:
        """
        Return the simple roots of E₈ in the standard basis.
        """
        # E₈ simple roots in 8D space
        roots = []
        
        # First 7 roots: e_i - e_{i+1}
        for i in range(7):
            r = np.zeros(8)
            r[i] = 1
            r[i+1] = -1
            roots.append(r)
        
        # 8th root: -(e_1 + e_2 + e_3)/2 + (e_4 + ... + e_8)/2
        r8 = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        roots.append(r8)
        
        return roots
    
    def wall_form_from_root(self, root: np.ndarray, beta: np.ndarray) -> float:
        """
        Compute the wall form associated to a root.
        
        The wall is at w(β) = 0 where w is the linear form.
        """
        return np.dot(root, beta)
    
    def weyl_reflection(self, root: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Compute the Weyl reflection of a vector with respect to a root.
        
        r_α(v) = v - 2(v·α)/(α·α) α
        """
        alpha_norm_sq = np.dot(root, root)
        if alpha_norm_sq < 1e-15:
            return vector
        return vector - 2 * np.dot(vector, root) / alpha_norm_sq * root


# =============================================================================
# PART 7: QUANTUM CORRECTIONS AND LOOP QUANTUM COSMOLOGY
# =============================================================================

class QuantumBKL:
    """
    Explore quantum corrections to BKL dynamics.
    
    In Loop Quantum Cosmology (LQC), the singularity is replaced by
    a quantum bounce. We explore how this modifies the classical picture.
    """
    
    def __init__(self, planck_length: float = 1.0):
        """
        Initialize with Planck length scale.
        
        In natural units, ℓ_P = √(ℏG/c³) ≈ 1.6 × 10⁻³⁵ m
        """
        self.l_P = planck_length
        self.rho_crit = 1 / (8 * np.pi * planck_length**2)  # Critical density
        
    def holonomy_correction(self, K: float, mu_bar: float = 1.0) -> float:
        """
        Holonomy correction to the extrinsic curvature in LQC.
        
        The classical K is replaced by sin(μK)/μ.
        """
        mu = mu_bar * self.l_P
        return np.sin(mu * K) / mu
    
    def effective_hamiltonian(self, a: float, p_a: float, 
                             phi: float = 0, p_phi: float = 0) -> float:
        """
        Effective Hamiltonian in Loop Quantum Cosmology.
        
        Includes quantum corrections that resolve the singularity.
        """
        # Gravitational part with holonomy corrections
        mu = self.l_P
        H_grav = -3 * a * np.sin(mu * p_a / a)**2 / (mu**2)
        
        # Matter part (massless scalar field)
        H_matter = p_phi**2 / (2 * a**3)
        
        return H_grav + H_matter
    
    def quantum_bounce_dynamics(self, t_span: Tuple[float, float],
                                y0: np.ndarray) -> dict:
        """
        Simulate the quantum bounce in a Bianchi I model.
        
        Parameters:
            t_span: Time interval
            y0: Initial state [a, p_a, phi, p_phi]
        """
        def eom(t, y):
            a, p_a, phi, p_phi = y
            if a < 1e-10:
                a = 1e-10
            
            mu = self.l_P
            
            # Effective equations with holonomy corrections
            da = -6 * np.sin(mu * p_a / a) * np.cos(mu * p_a / a) / mu
            dp_a = 3 * np.sin(mu * p_a / a)**2 / mu**2 + 3 * p_phi**2 / (2 * a**4)
            dphi = p_phi / a**3
            dp_phi = 0  # Conserved for free scalar
            
            return [da, dp_a, dphi, dp_phi]
        
        sol = solve_ivp(eom, t_span, y0, method='RK45', 
                       dense_output=True, max_step=0.01)
        
        return {
            't': sol.t,
            'a': sol.y[0],
            'p_a': sol.y[1],
            'phi': sol.y[2],
            'p_phi': sol.y[3]
        }


# =============================================================================
# PART 8: VISUALIZATION AND ANALYSIS
# =============================================================================

class BKLVisualizer:
    """
    Comprehensive visualization tools for BKL dynamics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        plt.style.use('default')
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
    def plot_kasner_circle(self, ax: Optional[plt.Axes] = None,
                          n_points: int = 100) -> plt.Axes:
        """Plot the Kasner circle with labeled special points."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Generate points on the Kasner circle
        theta = np.linspace(0, 2*np.pi, n_points)
        points = np.array([KasnerSolution.kasner_circle_parametrization(t) 
                          for t in theta])
        
        # Project to 2D (the circle lies in a plane)
        # Using p₁ and p₂ as coordinates
        ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Kasner circle')
        
        # Mark special points
        special_u = [1, 2, 3, 1.5, (1 + np.sqrt(5))/2]  # Including golden ratio
        for u in special_u:
            p = KasnerSolution.exponents_from_u(u)
            ax.plot(p[0], p[1], 'ro', markersize=8)
            ax.annotate(f'u={u:.2f}', (p[0], p[1]), textcoords='offset points',
                       xytext=(5, 5), fontsize=9)
        
        ax.set_xlabel(r'$p_1$', fontsize=12)
        ax.set_ylabel(r'$p_2$', fontsize=12)
        ax.set_title('Kasner Circle in $(p_1, p_2)$ Projection', fontsize=14)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_bkl_orbit(self, u0: float, n_iterations: int = 50,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the BKL map orbit and its statistical properties."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        orbit = BKLMap.orbit(u0, n_iterations)
        
        ax.plot(range(len(orbit)), orbit, 'b-o', markersize=4, linewidth=1)
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='u = 1')
        ax.axhline(y=2, color='g', linestyle='--', alpha=0.5, label='u = 2')
        
        ax.set_xlabel('Iteration (Kasner epoch)', fontsize=12)
        ax.set_ylabel('u parameter', fontsize=12)
        ax.set_title(f'BKL Map Orbit (u₀ = {u0:.4f})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_mixmaster_potential(self, ax: Optional[plt.Axes] = None,
                                 range_limit: float = 2.0) -> plt.Axes:
        """Plot the Mixmaster potential in the β-plane."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        mixmaster = MixmasterDynamics()
        
        # Create grid
        bp = np.linspace(-range_limit, range_limit, 200)
        bm = np.linspace(-range_limit, range_limit, 200)
        BP, BM = np.meshgrid(bp, bm)
        
        # Compute potential (with clipping for visualization)
        V = np.zeros_like(BP)
        for i in range(BP.shape[0]):
            for j in range(BP.shape[1]):
                V[i, j] = mixmaster.potential(BP[i, j], BM[i, j])
        
        V = np.clip(V, -10, 100)  # Clip for visualization
        
        # Plot contours
        levels = np.linspace(V.min(), min(V.max(), 50), 30)
        contour = ax.contourf(BP, BM, V, levels=levels, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='V(β₊, β₋)')
        
        # Draw the triangular equipotential at V = 1
        ax.contour(BP, BM, V, levels=[1], colors='white', linewidths=2)
        
        ax.set_xlabel(r'$\beta_+$', fontsize=12)
        ax.set_ylabel(r'$\beta_-$', fontsize=12)
        ax.set_title('Mixmaster Potential', fontsize=14)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_billiard_trajectory(self, n_bounces: int = 20,
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot a billiard trajectory in the triangular domain.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw the triangular billiard table
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3, 0])
        r = 1.5
        triangle_x = r * np.cos(angles)
        triangle_y = r * np.sin(angles)
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        
        # Simulate trajectory with random initial conditions
        np.random.seed(42)
        pos = np.array([0.1, 0.05])
        vel = np.array([np.cos(0.7), np.sin(0.7)])
        vel = vel / np.linalg.norm(vel)
        
        trajectory = [pos.copy()]
        
        for _ in range(n_bounces):
            # Find intersection with walls
            min_t = np.inf
            hit_wall = None
            
            for i in range(3):
                # Wall from vertex i to vertex i+1
                v1 = np.array([r * np.cos(angles[i]), r * np.sin(angles[i])])
                v2 = np.array([r * np.cos(angles[i+1]), r * np.sin(angles[i+1])])
                
                # Wall normal (pointing inward)
                wall_dir = v2 - v1
                normal = np.array([-wall_dir[1], wall_dir[0]])
                normal = normal / np.linalg.norm(normal)
                if np.dot(normal, -v1) < 0:
                    normal = -normal
                
                # Ray-wall intersection
                denom = np.dot(vel, normal)
                if denom < -1e-10:  # Moving toward wall
                    t = np.dot(v1 - pos, normal) / denom
                    if 0 < t < min_t:
                        hit_point = pos + t * vel
                        # Check if hit point is on wall segment
                        param = np.dot(hit_point - v1, wall_dir) / np.dot(wall_dir, wall_dir)
                        if -0.01 < param < 1.01:
                            min_t = t
                            hit_wall = (normal, v1, wall_dir)
            
            if hit_wall is None or min_t == np.inf:
                break
            
            # Move to wall and reflect
            pos = pos + min_t * vel * 0.999
            trajectory.append(pos.copy())
            
            normal = hit_wall[0]
            vel = vel - 2 * np.dot(vel, normal) * normal
        
        # Plot trajectory
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1, alpha=0.7)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')
        
        ax.set_xlabel(r'$\beta_+$', fontsize=12)
        ax.set_ylabel(r'$\beta_-$', fontsize=12)
        ax.set_title(f'Billiard Trajectory ({n_bounces} bounces)', fontsize=14)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_velocity_dominance(self, sim: InhomogeneousSimulation,
                                state: dict, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the velocity-to-spatial derivative ratio across space."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        ratio = sim.compute_velocity_dominance_ratio(state)
        
        ax.semilogy(sim.x, ratio, 'b-', linewidth=2)
        ax.axhline(y=1, color='r', linestyle='--', label='Equal contribution')
        
        ax.set_xlabel('Spatial coordinate x', fontsize=12)
        ax.set_ylabel('Velocity / Spatial ratio', fontsize=12)
        ax.set_title('BKL Velocity Dominance Test', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


# =============================================================================
# PART 9: MAIN EXPLORATION SCRIPT
# =============================================================================

def explore_bkl_dynamics():
    """
    Main function to explore BKL dynamics and generate visualizations.
    """
    print("=" * 70)
    print("BKL CONJECTURE EXPLORATION")
    print("Oscillatory Approach to Cosmological Singularities")
    print("=" * 70)
    
    # Initialize visualizer
    viz = BKLVisualizer()
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Kasner circle
    ax1 = fig.add_subplot(2, 3, 1)
    viz.plot_kasner_circle(ax1)
    
    # Panel 2: BKL map orbit
    ax2 = fig.add_subplot(2, 3, 2)
    u0 = 1 + np.pi  # Irrational initial condition for chaos
    viz.plot_bkl_orbit(u0, n_iterations=50, ax=ax2)
    
    # Panel 3: Mixmaster potential
    ax3 = fig.add_subplot(2, 3, 3)
    viz.plot_mixmaster_potential(ax3, range_limit=1.5)
    
    # Panel 4: Billiard trajectory
    ax4 = fig.add_subplot(2, 3, 4)
    viz.plot_billiard_trajectory(n_bounces=30, ax=ax4)
    
    # Panel 5: Era structure analysis
    ax5 = fig.add_subplot(2, 3, 5)
    eras = BKLMap.era_structure(u0, n_eras=20)
    era_lengths = [e[0] for e in eras]
    ax5.bar(range(len(era_lengths)), era_lengths, color='steelblue', alpha=0.7)
    ax5.set_xlabel('Era number', fontsize=12)
    ax5.set_ylabel('Kasner epochs in era', fontsize=12)
    ax5.set_title('BKL Era Structure', fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Lyapunov exponent convergence
    ax6 = fig.add_subplot(2, 3, 6)
    iterations = np.logspace(1, 4, 20).astype(int)
    lyap_estimates = [BKLMap.lyapunov_exponent_estimate(u0, n) for n in iterations]
    theoretical_lyap = np.pi**2 / (6 * np.log(2))  # ≈ 2.37
    ax6.semilogx(iterations, lyap_estimates, 'b-o', label='Numerical')
    ax6.axhline(y=theoretical_lyap, color='r', linestyle='--', 
                label=f'Theoretical: {theoretical_lyap:.3f}')
    ax6.set_xlabel('Number of iterations', fontsize=12)
    ax6.set_ylabel('Lyapunov exponent', fontsize=12)
    ax6.set_title('Chaos in BKL Dynamics', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bkl_dynamics_exploration.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Compute statistics
    print(f"\nBKL Map Analysis (u₀ = {u0:.6f}):")
    print(f"  - Lyapunov exponent: {lyap_estimates[-1]:.4f}")
    print(f"  - Theoretical value: {theoretical_lyap:.4f}")
    print(f"  - Number of eras analyzed: {len(eras)}")
    print(f"  - Mean era length: {np.mean(era_lengths):.2f}")
    
    # Kasner exponents for initial u
    p1, p2, p3 = KasnerSolution.exponents_from_u(u0)
    print(f"\nInitial Kasner exponents:")
    print(f"  p₁ = {p1:.6f}")
    print(f"  p₂ = {p2:.6f}")
    print(f"  p₃ = {p3:.6f}")
    print(f"  Sum: {p1+p2+p3:.10f} (should be 1)")
    print(f"  Sum of squares: {p1**2+p2**2+p3**2:.10f} (should be 1)")
    
    return fig


def demonstrate_quantum_bounce():
    """
    Demonstrate the quantum bounce that replaces the classical singularity.
    """
    print("\n" + "=" * 70)
    print("QUANTUM BOUNCE IN LOOP QUANTUM COSMOLOGY")
    print("=" * 70)
    
    qbkl = QuantumBKL(planck_length=0.1)
    
    # Initial conditions: contracting universe
    y0 = [10.0, -1.0, 0.0, 1.0]  # a, p_a, phi, p_phi
    t_span = (0, 50)
    
    result = qbkl.quantum_bounce_dynamics(t_span, y0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot scale factor evolution
    axes[0].plot(result['t'], result['a'], 'b-', linewidth=2)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Scale factor a(t)', fontsize=12)
    axes[0].set_title('Quantum Bounce: Scale Factor', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot phase space
    axes[1].plot(result['a'], result['p_a'], 'b-', linewidth=1)
    axes[1].plot(result['a'][0], result['p_a'][0], 'go', markersize=10, label='Start')
    axes[1].plot(result['a'][-1], result['p_a'][-1], 'ro', markersize=10, label='End')
    axes[1].set_xlabel('Scale factor a', fontsize=12)
    axes[1].set_ylabel('Momentum p_a', fontsize=12)
    axes[1].set_title('Phase Space Trajectory', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_bounce.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nInitial scale factor: {y0[0]}")
    print(f"Minimum scale factor (bounce): {min(result['a']):.4f}")
    print(f"Final scale factor: {result['a'][-1]:.4f}")
    
    return fig


def analyze_inhomogeneous_singularity():
    """
    Analyze the approach to singularity in an inhomogeneous spacetime.
    """
    print("\n" + "=" * 70)
    print("INHOMOGENEOUS SPACETIME NEAR SINGULARITY")
    print("Testing BKL Locality Conjecture")
    print("=" * 70)
    
    sim = InhomogeneousSimulation(n_spatial_points=200)
    state = sim.initialize_metric_functions(perturbation_amplitude=0.2, seed=42)
    
    viz = BKLVisualizer()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot metric functions
    axes[0, 0].plot(sim.x, state['alpha'], label=r'$\alpha = \ln(a)$')
    axes[0, 0].plot(sim.x, state['beta'], label=r'$\beta = \ln(b)$')
    axes[0, 0].plot(sim.x, state['gamma'], label=r'$\gamma = \ln(c)$')
    axes[0, 0].set_xlabel('Spatial coordinate x', fontsize=12)
    axes[0, 0].set_ylabel('Logarithmic scale factors', fontsize=12)
    axes[0, 0].set_title('Spatial Metric Components', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot extrinsic curvature (velocities)
    axes[0, 1].plot(sim.x, state['K_alpha'], label=r'$K_\alpha$')
    axes[0, 1].plot(sim.x, state['K_beta'], label=r'$K_\beta$')
    axes[0, 1].plot(sim.x, state['K_gamma'], label=r'$K_\gamma$')
    axes[0, 1].set_xlabel('Spatial coordinate x', fontsize=12)
    axes[0, 1].set_ylabel('Extrinsic curvature components', fontsize=12)
    axes[0, 1].set_title('Metric Velocities', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot velocity dominance
    viz.plot_velocity_dominance(sim, state, ax=axes[1, 0])
    
    # Plot local Kasner parameters
    # At each point, compute effective u parameter
    u_local = np.zeros(sim.N)
    for i in range(sim.N):
        K_sorted = np.sort([state['K_alpha'][i], state['K_beta'][i], state['K_gamma'][i]])
        # Approximate Kasner exponents from K_i
        K_sum = K_sorted.sum()
        if abs(K_sum) > 1e-10:
            p_approx = K_sorted / K_sum
            u_local[i] = KasnerSolution.u_from_exponents(*p_approx)
        else:
            u_local[i] = 1.0
    
    axes[1, 1].plot(sim.x, u_local, 'b-', linewidth=2)
    axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Spatial coordinate x', fontsize=12)
    axes[1, 1].set_ylabel('Local u parameter', fontsize=12)
    axes[1, 1].set_title('Local Kasner Parameter Distribution', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inhomogeneous_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSpatial grid points: {sim.N}")
    print(f"Mean velocity dominance ratio: {sim.compute_velocity_dominance_ratio(state).mean():.4f}")
    print(f"Local u parameter range: [{u_local.min():.4f}, {u_local.max():.4f}]")
    
    return fig


# =============================================================================
# PART 10: FUTURE DIRECTIONS AND BIG DREAMS
# =============================================================================

def print_research_directions():
    """
    Print potential research directions and big dreams for BKL research.
    """
    print("\n" + "=" * 70)
    print("RESEARCH DIRECTIONS: THE BIG DREAMS")
    print("=" * 70)
    
    directions = """
    1. RIGOROUS PROOF OF THE BKL CONJECTURE
       - Extend Ringström's results beyond homogeneous cosmologies
       - Develop new techniques for handling inhomogeneous perturbations
       - Understand the role of spikes in generic singularity formation
    
    2. CONNECTIONS TO QUANTUM GRAVITY
       - How does Loop Quantum Cosmology modify BKL oscillations?
       - What is the fate of chaotic dynamics in quantum regime?
       - Can we detect signatures of BKL dynamics in primordial fluctuations?
    
    3. E₁₀ AND HIDDEN SYMMETRIES
       - Understand the full role of E₁₀ Kac-Moody algebra
       - Connect BKL dynamics to M-theory and exceptional geometry
       - Explore if billiard chaos reveals quantum gravity structure
    
    4. NUMERICAL ADVANCES
       - High-precision simulations of generic vacuum spacetimes
       - Machine learning to identify Kasner transitions
       - Automated detection of spike formation
    
    5. OBSERVATIONAL SIGNATURES
       - Can we detect BKL dynamics through gravitational waves?
       - What are the implications for the cosmic microwave background?
       - Loop quantum cosmology predictions for B-mode polarization
    
    6. MATHEMATICAL STRUCTURES
       - Connection to ergodic theory and continued fractions
       - Understanding the measure-theoretic aspects
       - Topology of the space of singular spacetimes
    """
    
    print(directions)
    
    print("\n" + "=" * 70)
    print("KEY OPEN PROBLEMS")
    print("=" * 70)
    
    problems = """
    □ Prove AVTD (Asymptotic Velocity Term Dominance) for generic initial data
    □ Characterize the measure-zero exceptions to BKL behavior  
    □ Understand spike dynamics and their role in singularity formation
    □ Connect classical BKL chaos to quantum unitarity
    □ Extend results to higher dimensions and supergravity
    □ Prove or disprove: "Generic singularities are locally BKL"
    """
    
    print(problems)


if __name__ == "__main__":
    # Run the main exploration
    print("\n🌌 Starting BKL Conjecture Exploration...\n")
    
    try:
        fig1 = explore_bkl_dynamics()
    except Exception as e:
        print(f"Note: Visualization may require display. Error: {e}")
    
    # try:
    #     fig2 = demonstrate_quantum_bounce()
    # except Exception as e:
    #     print(f"Note: Quantum bounce visualization skipped. Error: {e}")
    
    # try:
    #     fig3 = analyze_inhomogeneous_singularity()
    # except Exception as e:
    #     print(f"Note: Inhomogeneous analysis skipped. Error: {e}")
    
    # Print research directions
    print_research_directions()
    
    print("\n" + "=" * 70)
    print("Exploration complete! Check the generated PNG files.")
    print("=" * 70)
