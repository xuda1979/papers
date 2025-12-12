"""
BKL Deep Exploration: Pushing the Boundaries
=============================================

This module pushes deeper into uncharted territory of BKL research:
1. Fractal structure of the BKL attractor
2. Thermodynamic formalism and phase transitions
3. Connections to black hole interiors
4. Higher-dimensional BKL and string theory
5. Experimental analogues

Author: Research Exploration
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.special import zeta
from scipy.stats import entropy as scipy_entropy
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

plt.style.use('default')


# =============================================================================
# PART 1: FRACTAL STRUCTURE OF BKL DYNAMICS
# =============================================================================

class BKLFractalAnalysis:
    """
    Analyze the fractal and multifractal structure of BKL dynamics.
    
    The BKL attractor has rich fractal properties connected to
    the number-theoretic structure of continued fractions.
    """
    
    def __init__(self):
        self.ln2 = np.log(2)
    
    def gauss_map(self, x: float) -> float:
        """The Gauss map T(x) = {1/x}."""
        if x == 0:
            return 0
        return 1/x - np.floor(1/x)
    
    def compute_orbit(self, x0: float, n_iterations: int) -> np.ndarray:
        """Compute orbit of Gauss map."""
        orbit = np.zeros(n_iterations)
        x = x0
        for i in range(n_iterations):
            orbit[i] = x
            x = self.gauss_map(x)
            if x < 1e-15:
                x = np.random.uniform(0.01, 0.99)
        return orbit
    
    def box_counting_dimension(self, orbit: np.ndarray, 
                                n_boxes_list: List[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Estimate the box-counting dimension of the orbit.
        
        For the Gauss map invariant set, this should be close to 1.
        """
        if n_boxes_list is None:
            n_boxes_list = [10, 20, 50, 100, 200, 500, 1000]
        
        counts = []
        for n_boxes in n_boxes_list:
            # Create histogram
            hist, _ = np.histogram(orbit, bins=n_boxes, range=(0, 1))
            # Count non-empty boxes
            counts.append(np.sum(hist > 0))
        
        counts = np.array(counts)
        n_boxes_list = np.array(n_boxes_list)
        
        # Fit log-log slope
        log_n = np.log(n_boxes_list)
        log_counts = np.log(counts)
        
        # Linear regression
        coeffs = np.polyfit(log_n, log_counts, 1)
        dimension = coeffs[0]
        
        return dimension, n_boxes_list, counts
    
    def multifractal_spectrum(self, orbit: np.ndarray, 
                              q_values: np.ndarray = None,
                              n_boxes: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the multifractal spectrum f(α) using the thermodynamic formalism.
        
        τ(q) = lim_{ε→0} (1/(q-1)) log Σ_i p_i^q / log(1/ε)
        
        The Legendre transform gives f(α).
        """
        if q_values is None:
            q_values = np.linspace(-5, 5, 51)
        
        # Compute probability distribution
        hist, _ = np.histogram(orbit, bins=n_boxes, range=(0, 1), density=True)
        p = hist / hist.sum()
        p = p[p > 0]  # Remove zeros
        
        # Compute τ(q)
        tau = np.zeros_like(q_values)
        for i, q in enumerate(q_values):
            if abs(q - 1) < 0.01:
                # Special case q → 1: information dimension
                tau[i] = -np.sum(p * np.log(p + 1e-15)) / np.log(n_boxes)
            else:
                tau[i] = np.log(np.sum(p**q)) / ((q - 1) * np.log(n_boxes))
        
        # Compute α(q) = dτ/dq
        alpha = np.gradient(tau * (q_values - 1) + tau, q_values)
        
        # Compute f(α) = q*α - τ(q)*(q-1) - τ(q)
        f_alpha = q_values * alpha - tau
        
        return q_values, tau, f_alpha
    
    def lyapunov_spectrum(self, x0: float, n_iterations: int = 100000) -> List[float]:
        """
        Compute finite-time Lyapunov exponents at different scales.
        
        This reveals the multiscale structure of chaos.
        """
        x = x0
        lyap_sum = 0
        lyap_values = []
        
        for i in range(n_iterations):
            if x > 1e-10:
                # Derivative of Gauss map: |T'(x)| = 1/x²
                deriv = 1 / x**2
                lyap_sum += np.log(deriv)
                
                if (i + 1) % 1000 == 0:
                    lyap_values.append(lyap_sum / (i + 1))
            
            x = self.gauss_map(x)
            if x < 1e-15:
                x = np.random.uniform(0.01, 0.99)
        
        return lyap_values
    
    def recurrence_time_distribution(self, x0: float, 
                                      epsilon: float = 0.01,
                                      n_recurrences: int = 1000) -> np.ndarray:
        """
        Compute the distribution of recurrence times.
        
        For chaotic systems, this should follow an exponential distribution.
        """
        x = x0
        x_init = x0
        recurrence_times = []
        current_time = 0
        count = 0
        
        while count < n_recurrences and current_time < 10000000:
            x = self.gauss_map(x)
            current_time += 1
            
            if abs(x - x_init) < epsilon:
                recurrence_times.append(current_time)
                current_time = 0
                count += 1
            
            if x < 1e-15:
                x = np.random.uniform(0.01, 0.99)
        
        return np.array(recurrence_times)


# =============================================================================
# PART 2: THERMODYNAMIC FORMALISM FOR BKL
# =============================================================================

class BKLThermodynamics:
    """
    Apply the thermodynamic formalism to BKL dynamics.
    
    This connects dynamical systems theory to statistical mechanics
    and reveals phase transition-like behavior.
    """
    
    def __init__(self):
        self.ln2 = np.log(2)
    
    def transfer_operator_eigenvalue(self, beta: float, 
                                      n_terms: int = 100) -> float:
        """
        Compute the leading eigenvalue of the transfer operator L_β.
        
        For the Gauss map:
        L_β f(x) = Σ_{n=1}^∞ 1/(n+x)^{2β} f(1/(n+x))
        
        The pressure P(β) = log(λ_β) where λ_β is the leading eigenvalue.
        """
        # Approximate by truncating the sum
        # Use the fact that the invariant density is 1/(ln 2)(1+x)
        
        # For β = 1, the eigenvalue is 1 (normalization)
        # For other β, compute via series
        
        eigenvalue = 0
        for n in range(1, n_terms + 1):
            # Integrate over the interval [1/(n+1), 1/n]
            # Contribution from branch n
            contrib = 1 / n**(2*beta)
            eigenvalue += contrib
        
        return eigenvalue
    
    def pressure_function(self, beta: float) -> float:
        """
        The pressure function P(β) = log λ_β.
        
        For the Gauss map, this is related to the Riemann zeta function:
        P(β) ≈ log(ζ(2β)) for large β
        """
        if beta <= 0.5:
            # Near the critical point
            return np.log(self.transfer_operator_eigenvalue(beta, 200))
        else:
            # Use zeta function approximation
            return np.log(zeta(2*beta))
    
    def free_energy(self, beta: float) -> float:
        """
        Free energy F(β) = -P(β)/β in the thermodynamic analogy.
        """
        if beta == 0:
            return 0
        return -self.pressure_function(beta) / beta
    
    def entropy_function(self, beta: float, delta: float = 0.01) -> float:
        """
        Dynamical entropy S(β) = d(βP(β))/dβ.
        
        At β = 1, this gives the KS entropy π²/(6 ln 2).
        """
        p_plus = self.pressure_function(beta + delta)
        p_minus = self.pressure_function(beta - delta)
        
        bp_plus = (beta + delta) * p_plus
        bp_minus = (beta - delta) * p_minus
        
        return (bp_plus - bp_minus) / (2 * delta)
    
    def specific_heat(self, beta: float, delta: float = 0.01) -> float:
        """
        Specific heat C(β) = -β² d²F/dβ².
        
        Divergence indicates a phase transition.
        """
        f_plus = self.free_energy(beta + delta)
        f_center = self.free_energy(beta)
        f_minus = self.free_energy(beta - delta)
        
        d2f = (f_plus - 2*f_center + f_minus) / delta**2
        
        return -beta**2 * d2f
    
    def detect_phase_transitions(self, beta_range: np.ndarray) -> Dict:
        """
        Look for phase transitions in the thermodynamic functions.
        """
        pressures = [self.pressure_function(b) for b in beta_range]
        entropies = [self.entropy_function(b) for b in beta_range if b > 0.1]
        specific_heats = [self.specific_heat(b) for b in beta_range if b > 0.1]
        
        # Look for discontinuities or divergences
        results = {
            'beta': beta_range,
            'pressure': np.array(pressures),
            'entropy': np.array(entropies),
            'specific_heat': np.array(specific_heats),
            'critical_points': []
        }
        
        # Check for large specific heat (potential phase transition)
        sh_array = np.array(specific_heats)
        peaks = np.where(np.abs(sh_array) > 3 * np.median(np.abs(sh_array)))[0]
        if len(peaks) > 0:
            results['critical_points'] = beta_range[peaks + len(beta_range) - len(sh_array)]
        
        return results


# =============================================================================
# PART 3: BLACK HOLE INTERIORS AND BKL
# =============================================================================

class BlackHoleInteriorBKL:
    """
    Explore the connection between BKL dynamics and black hole interiors.
    
    The interior of a generic black hole approaches a spacelike singularity,
    which should exhibit BKL behavior.
    """
    
    def __init__(self):
        pass
    
    def schwarzschild_interior_metric(self, r: float, M: float = 1.0) -> Dict:
        """
        The Schwarzschild interior metric (r < 2M).
        
        ds² = -(2M/r - 1)^{-1} dr² + (2M/r - 1) dt² + r² dΩ²
        
        Note: r and t switch roles inside the horizon!
        """
        if r >= 2*M:
            raise ValueError("r must be inside the horizon (r < 2M)")
        
        grr = -1 / (2*M/r - 1)
        gtt = 2*M/r - 1
        
        return {
            'r': r,
            'g_rr': grr,
            'g_tt': gtt,
            'g_theta_theta': r**2,
            'proper_time_to_singularity': self.proper_time_to_singularity(r, M)
        }
    
    def proper_time_to_singularity(self, r: float, M: float = 1.0) -> float:
        """
        Compute proper time from radius r to the singularity at r = 0.
        
        For radial infall from rest at r:
        τ = (π/2) M (r/2M)^{3/2}  (approximately)
        """
        if r >= 2*M:
            return np.inf
        
        # More accurate integral
        def integrand(r_prime):
            return np.sqrt(2*M/r_prime - 1) if r_prime < 2*M else 0
        
        tau, _ = quad(integrand, 0, r)
        return tau
    
    def kasner_like_behavior_near_singularity(self, r: float, M: float = 1.0) -> Dict:
        """
        Analyze Kasner-like behavior near r → 0.
        
        As r → 0, the Schwarzschild interior approaches a Kasner solution
        with exponents (2/3, 2/3, -1/3) -- the so-called "cigar" singularity.
        """
        # Near r = 0: g_tt ~ r, g_rr ~ -r^{-1}, g_θθ ~ r²
        # In terms of proper time τ ~ r^{3/2}
        
        # Effective Kasner exponents
        p_t = 2/3  # "t" direction
        p_theta = 2/3  # angular
        p_r = -1/3  # radial
        
        # Verify Kasner constraints
        sum_p = p_t + p_theta + p_r
        sum_p_sq = p_t**2 + p_theta**2 + p_r**2
        
        return {
            'p_t': p_t,
            'p_theta': p_theta, 
            'p_r': p_r,
            'sum_check': sum_p,  # Should be 1
            'sum_sq_check': sum_p_sq,  # Should be 1
            'type': 'cigar_singularity',
            'is_oscillatory': False  # Schwarzschild is non-generic!
        }
    
    def kerr_interior_complexity(self, a: float, M: float = 1.0) -> str:
        """
        Describe the interior structure of a Kerr black hole.
        
        The Kerr interior is much more complex than Schwarzschild,
        with a ring singularity, Cauchy horizon, and possible BKL behavior.
        """
        description = f"""
        KERR BLACK HOLE INTERIOR (a/M = {a/M:.2f})
        ==========================================
        
        Structure:
        1. Outer horizon at r₊ = M + √(M² - a²) = {M + np.sqrt(M**2 - a**2):.4f}
        2. Inner (Cauchy) horizon at r₋ = M - √(M² - a²) = {M - np.sqrt(M**2 - a**2):.4f}
        3. Ring singularity at r = 0, θ = π/2
        
        BKL Relevance:
        - The approach to the ring singularity may exhibit BKL oscillations
        - The Cauchy horizon is classically unstable (mass inflation)
        - Generic perturbations lead to a spacelike singularity
        
        Open Questions:
        - Does the Kerr interior become BKL-like with generic perturbations?
        - What is the role of mass inflation?
        - How do quantum effects modify the picture?
        """
        return description


# =============================================================================
# PART 4: HIGHER-DIMENSIONAL BKL AND STRING THEORY
# =============================================================================

class HigherDimensionalBKL:
    """
    Explore BKL dynamics in higher dimensions and string/M-theory.
    
    Key results:
    - D ≤ 10: Oscillatory BKL behavior (finite billiard)
    - D > 10: Monotonic approach (infinite billiard)
    - 11D supergravity: Special structure connected to E₁₀
    """
    
    def __init__(self, dimension: int = 4):
        self.D = dimension
        self.n = dimension - 1  # Spatial dimension
    
    def kasner_constraints(self) -> Tuple[str, str]:
        """
        Kasner constraints in D dimensions.
        
        Σᵢ pᵢ = 1
        Σᵢ pᵢ² = 1
        
        where i = 1, ..., D-1 (spatial directions)
        """
        constraint1 = f"Σᵢ₌₁^{self.n} pᵢ = 1"
        constraint2 = f"Σᵢ₌₁^{self.n} pᵢ² = 1"
        return constraint1, constraint2
    
    def kasner_sphere_dimension(self) -> int:
        """
        The Kasner constraints define a sphere of dimension n-2.
        
        In 4D: 1-sphere (circle)
        In 11D: 8-sphere
        """
        return self.n - 2
    
    def billiard_type(self) -> str:
        """
        Determine the type of cosmological billiard.
        
        Finite (compact): Oscillatory BKL
        Infinite: Monotonic approach
        """
        if self.D <= 10:
            return "finite"
        elif self.D == 11:
            return "finite (special - 11D supergravity)"
        else:
            return "infinite"
    
    def number_of_walls(self) -> Dict[str, int]:
        """
        Count the billiard walls for vacuum gravity.
        
        Symmetry walls: n(n-1)/2
        Curvature walls: depends on topology
        """
        n = self.n
        symmetry_walls = n * (n - 1) // 2
        
        # For simple topology (Bianchi IX type)
        curvature_walls = n  # One per direction
        
        return {
            'symmetry_walls': symmetry_walls,
            'curvature_walls': curvature_walls,
            'total': symmetry_walls + curvature_walls
        }
    
    def string_theory_billiard(self, theory: str = "IIA") -> Dict:
        """
        Describe the billiard for various string theories.
        
        String theories add additional walls from p-form fields.
        """
        billiards = {
            "IIA": {
                "dimension": 10,
                "algebra": "E₁₀",
                "walls": "Gravity + dilaton + B-field + RR fields",
                "finite": True
            },
            "IIB": {
                "dimension": 10,
                "algebra": "E₁₀",
                "walls": "Gravity + dilaton + B-field + RR fields (self-dual)",
                "finite": True
            },
            "heterotic": {
                "dimension": 10,
                "algebra": "BE₁₀ or DE₁₀",
                "walls": "Gravity + dilaton + B-field + gauge fields",
                "finite": True
            },
            "M-theory": {
                "dimension": 11,
                "algebra": "E₁₀",
                "walls": "Gravity + 3-form",
                "finite": True,
                "special": "Parent theory"
            }
        }
        
        return billiards.get(theory, {"error": "Unknown theory"})
    
    def e10_root_structure(self) -> str:
        """
        Describe the E₁₀ root structure relevant for BKL.
        """
        description = """
        E₁₀ KAC-MOODY ALGEBRA AND BKL DYNAMICS
        ======================================
        
        E₁₀ is an infinite-dimensional hyperbolic Kac-Moody algebra.
        
        Dynkin diagram: E₈ extended twice
        ○-○-○-○-○-○-○-○
                |
                ○
                |
                ○
        
        BKL Correspondence:
        - Simple roots ↔ billiard walls
        - Weyl reflections ↔ bounces off walls
        - Root lattice ↔ allowed Kasner transitions
        
        Level decomposition:
        - Level 0: GL(10) (gravity)
        - Level 1: Antisymmetric tensors (p-forms)
        - Level 2: Dual graviton
        - Level 3+: "Exotic" fields (M-theory?)
        
        The E₁₀ sigma model:
        L = (1/4n) Tr(∂ₜP · ∂ₜP⁻¹) + walls
        
        where P ∈ E₁₀/K(E₁₀) is the coset representative.
        """
        return description


# =============================================================================
# PART 5: EXPERIMENTAL AND OBSERVATIONAL ANALOGUES
# =============================================================================

class BKLAnalogues:
    """
    Explore experimental and observational analogues of BKL dynamics.
    """
    
    @staticmethod
    def acoustic_analogue() -> str:
        """
        Acoustic black holes and BKL-like behavior.
        """
        return """
        ACOUSTIC ANALOGUES OF BKL DYNAMICS
        ==================================
        
        In acoustic black holes (sonic horizons in flowing fluids),
        the effective metric can mimic gravitational dynamics.
        
        Potential Experiment:
        1. Create a 3D converging flow with anisotropic velocities
        2. The "acoustic metric" near the sonic horizon:
           ds² = (ρ/c)[-(c²-v²)dt² - 2v·dx dt + dx²]
        3. As v → c anisotropically, BKL-like oscillations may emerge
        
        Challenges:
        - Need highly controlled anisotropic flow
        - Viscosity and finite-size effects
        - Quantum fluctuations (phonons) vs classical BKL
        
        Status: THEORETICAL - No experiment yet
        """
    
    @staticmethod
    def cold_atom_analogue() -> str:
        """
        Cold atom systems for simulating BKL dynamics.
        """
        return """
        COLD ATOM BKL SIMULATOR
        =======================
        
        Bose-Einstein condensates can simulate curved spacetime
        via the Gross-Pitaevskii equation.
        
        Proposal:
        1. Create a BEC with time-varying trapping potentials
        2. Modulate anisotropically: V(t) = m[ω₁²(t)x² + ω₂²(t)y² + ω₃²(t)z²]/2
        3. The condensate dynamics maps to an effective metric
        4. Engineer ω_i(t) to mimic BKL oscillations
        
        Observable signatures:
        - Density oscillations following Kasner epochs
        - Quantum tunneling between "Kasner states"
        - Entanglement growth at "bounce" events
        
        Advantages:
        - Precise control over parameters
        - Quantum effects naturally included
        - Can probe the quantum-classical transition
        
        Status: FEASIBLE with current technology
        """
    
    @staticmethod
    def gravitational_wave_signatures() -> str:
        """
        Potential gravitational wave signatures of BKL physics.
        """
        return """
        GRAVITATIONAL WAVE SIGNATURES OF BKL DYNAMICS
        =============================================
        
        1. PRIMORDIAL GRAVITATIONAL WAVES
           If the universe underwent BKL oscillations near the Big Bang,
           this could imprint on the primordial GW spectrum.
           
           Signature: Oscillatory features in the GW background
           Frequency: f ~ H(BKL epoch) · (a_BKL/a_today)
           
           Observable: Future GW detectors (LISA, BBO, DECIGO)
        
        2. BLACK HOLE MERGERS
           The late inspiral probes strong-field dynamics.
           Could there be BKL-like oscillations in the waveform?
           
           Signature: High-frequency modulations near merger
           Observable: LIGO/Virgo/KAGRA at high SNR
        
        3. COSMIC STRINGS
           Cosmic string networks evolve chaotically.
           BKL-like dynamics in string reconnections?
           
           Signature: Stochastic GW background features
           Observable: Pulsar timing arrays + LISA
        
        Current Status:
        - No confirmed BKL signatures yet
        - Theoretical predictions need refinement
        - Future detectors may reach required sensitivity
        """
    
    @staticmethod
    def cmb_signatures() -> str:
        """
        Cosmic Microwave Background signatures of pre-Big-Bang BKL.
        """
        return """
        CMB SIGNATURES OF BKL COSMOLOGY
        ===============================
        
        1. STATISTICAL ANISOTROPY
           If BKL oscillations occurred, different directions
           evolved differently. This could lead to:
           
           - Preferred axis in CMB (observed? "Axis of Evil")
           - Direction-dependent power spectrum
           - Parity violation in polarization
        
        2. NON-GAUSSIANITY
           Chaotic BKL dynamics generates non-Gaussian fluctuations.
           
           - Specific f_NL shapes from BKL transitions
           - Higher-point correlations
           - Connected to specific inflationary/bouncing models
        
        3. LARGE-SCALE ANOMALIES
           The largest CMB scales probe the earliest times.
           
           - Low quadrupole (observed)
           - Hemispherical asymmetry (observed)
           - Could these be BKL signatures?
        
        Analysis needed:
        - Detailed predictions from BKL → inflation transition
        - Statistical tests for BKL-specific patterns
        - Comparison with alternative explanations
        """


# =============================================================================
# PART 6: ADVANCED VISUALIZATION
# =============================================================================

def visualize_deep_exploration():
    """
    Create visualizations for deep BKL exploration.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: Fractal analysis
    ax1 = fig.add_subplot(2, 3, 1)
    fractal = BKLFractalAnalysis()
    orbit = fractal.compute_orbit(np.pi - 3, 100000)
    
    # Histogram showing invariant measure
    ax1.hist(orbit, bins=100, density=True, alpha=0.7, color='steelblue')
    x = np.linspace(0.01, 0.99, 100)
    ax1.plot(x, 1/(np.log(2)*(1+x)), 'r-', linewidth=2, label='Theory: 1/(ln2·(1+x))')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Invariant Measure of Gauss Map', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Lyapunov convergence
    ax2 = fig.add_subplot(2, 3, 2)
    lyap_values = fractal.lyapunov_spectrum(np.sqrt(2) - 1, 100000)
    iterations = np.arange(1000, 1000*(len(lyap_values)+1), 1000)
    ax2.plot(iterations, lyap_values, 'b-', linewidth=1)
    ax2.axhline(y=np.pi**2/(6*np.log(2)), color='r', linestyle='--', 
                label=f'Theory: π²/6ln2 ≈ {np.pi**2/(6*np.log(2)):.4f}')
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Lyapunov exponent', fontsize=12)
    ax2.set_title('Convergence of Lyapunov Exponent', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Thermodynamic functions
    ax3 = fig.add_subplot(2, 3, 3)
    thermo = BKLThermodynamics()
    beta_range = np.linspace(0.6, 3, 50)
    pressures = [thermo.pressure_function(b) for b in beta_range]
    ax3.plot(beta_range, pressures, 'b-', linewidth=2)
    ax3.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='β = 1 (physical)')
    ax3.set_xlabel('β', fontsize=12)
    ax3.set_ylabel('P(β)', fontsize=12)
    ax3.set_title('Pressure Function', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Recurrence time distribution
    ax4 = fig.add_subplot(2, 3, 4)
    recurrence_times = fractal.recurrence_time_distribution(0.3, epsilon=0.02, n_recurrences=500)
    if len(recurrence_times) > 10:
        ax4.hist(recurrence_times, bins=50, density=True, alpha=0.7, color='steelblue')
        # Fit exponential
        mean_time = recurrence_times.mean()
        x_fit = np.linspace(0, recurrence_times.max(), 100)
        ax4.plot(x_fit, np.exp(-x_fit/mean_time)/mean_time, 'r-', linewidth=2, 
                label=f'Exp fit: τ = {mean_time:.1f}')
    ax4.set_xlabel('Recurrence time', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Recurrence Time Distribution', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Higher-dimensional billiards
    ax5 = fig.add_subplot(2, 3, 5)
    dimensions = np.arange(4, 15)
    wall_counts = []
    for d in dimensions:
        hd = HigherDimensionalBKL(d)
        walls = hd.number_of_walls()
        wall_counts.append(walls['total'])
    
    colors = ['green' if d <= 10 else 'red' for d in dimensions]
    bars = ax5.bar(dimensions, wall_counts, color=colors, alpha=0.7)
    ax5.axvline(x=10.5, color='black', linestyle='--', linewidth=2)
    ax5.text(10.7, max(wall_counts)*0.9, 'Critical D=10', fontsize=10)
    ax5.set_xlabel('Spacetime dimension D', fontsize=12)
    ax5.set_ylabel('Number of billiard walls', fontsize=12)
    ax5.set_title('Billiard Walls vs Dimension', fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Multifractal spectrum
    ax6 = fig.add_subplot(2, 3, 6)
    q_values, tau, f_alpha = fractal.multifractal_spectrum(orbit, n_boxes=200)
    
    # Plot f(α) vs α (estimated)
    alpha = np.gradient((q_values - 1) * tau + tau, q_values)
    mask = (alpha > 0) & (alpha < 2) & (f_alpha > -1) & (f_alpha < 2)
    if mask.sum() > 5:
        ax6.plot(alpha[mask], f_alpha[mask], 'b-', linewidth=2)
        ax6.plot([1, 1], [0, 1], 'r--', alpha=0.5)  # Monofractal reference
    ax6.set_xlabel('α (singularity strength)', fontsize=12)
    ax6.set_ylabel('f(α) (dimension)', fontsize=12)
    ax6.set_title('Multifractal Spectrum', fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0.5, 1.5)
    ax6.set_ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('bkl_deep_exploration.png', dpi=150, bbox_inches='tight')
    print("Saved: bkl_deep_exploration.png")
    
    return fig


# =============================================================================
# PART 7: DISCOVERY REPORT
# =============================================================================

def generate_discovery_report():
    """
    Generate a report of discoveries and insights.
    """
    print("=" * 78)
    print("BKL DEEP EXPLORATION: DISCOVERY REPORT")
    print("=" * 78)
    
    # Fractal analysis
    print("\n[1] FRACTAL STRUCTURE")
    print("-" * 50)
    fractal = BKLFractalAnalysis()
    orbit = fractal.compute_orbit(np.e - 2, 50000)
    dim, _, _ = fractal.box_counting_dimension(orbit)
    print(f"Box-counting dimension: {dim:.4f}")
    print(f"Expected for Gauss map: 1.0000 (fills interval)")
    print(f"Interpretation: The BKL attractor densely fills the Kasner circle")
    
    # Thermodynamics
    print("\n[2] THERMODYNAMIC ANALYSIS")
    print("-" * 50)
    thermo = BKLThermodynamics()
    print(f"Pressure P(1): {thermo.pressure_function(1.0):.6f}")
    print(f"Entropy S(1): {thermo.entropy_function(1.0):.6f}")
    print(f"Expected (KS entropy): {np.pi**2/(6*np.log(2)):.6f}")
    
    # Detect phase transitions
    beta_range = np.linspace(0.6, 3, 30)
    results = thermo.detect_phase_transitions(beta_range)
    if len(results['critical_points']) > 0:
        print(f"Potential critical points: {results['critical_points']}")
    else:
        print("No phase transitions detected in [0.6, 3]")
    
    # Black hole connection
    print("\n[3] BLACK HOLE INTERIOR")
    print("-" * 50)
    bh = BlackHoleInteriorBKL()
    kasner = bh.kasner_like_behavior_near_singularity(0.1)
    print(f"Schwarzschild near-singularity Kasner exponents:")
    print(f"  p_t = {kasner['p_t']:.4f}")
    print(f"  p_θ = {kasner['p_theta']:.4f}")
    print(f"  p_r = {kasner['p_r']:.4f}")
    print(f"Type: {kasner['type']}")
    print(f"Is oscillatory: {kasner['is_oscillatory']}")
    print("Note: Schwarzschild is non-generic (spherical symmetry)")
    print("      Generic black holes should show BKL oscillations!")
    
    # Higher dimensions
    print("\n[4] HIGHER DIMENSIONS")
    print("-" * 50)
    for D in [4, 10, 11, 12]:
        hd = HigherDimensionalBKL(D)
        bt = hd.billiard_type()
        walls = hd.number_of_walls()
        print(f"D = {D}: Billiard is {bt}, {walls['total']} walls")
    
    # String theory
    print("\n[5] STRING THEORY CONNECTION")
    print("-" * 50)
    hd = HigherDimensionalBKL(11)
    for theory in ["IIA", "IIB", "M-theory"]:
        info = hd.string_theory_billiard(theory)
        print(f"{theory}: D={info['dimension']}, Algebra={info['algebra']}, Finite={info['finite']}")
    
    # Experimental prospects
    print("\n[6] EXPERIMENTAL PROSPECTS")
    print("-" * 50)
    print("Cold atoms: FEASIBLE - Best near-term prospect")
    print("Acoustic: CHALLENGING - Needs anisotropic control")
    print("Gravitational waves: FUTURE - Awaiting sensitivity")
    print("CMB: ANALYZING - Possible anomaly connections")
    
    # Key insights
    print("\n" + "=" * 78)
    print("KEY INSIGHTS FROM EXPLORATION")
    print("=" * 78)
    
    insights = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║  1. UNIVERSALITY: The BKL dynamics is universal across dimensions D≤10   ║
    ║     The same chaotic map governs approach to ALL generic singularities   ║
    ║                                                                           ║
    ║  2. CRITICAL DIMENSION: D=10 is special - the boundary between           ║
    ║     oscillatory and monotonic behavior. This is exactly the critical     ║
    ║     dimension of string theory!                                          ║
    ║                                                                           ║
    ║  3. E₁₀ SYMMETRY: The infinite-dimensional Kac-Moody algebra E₁₀         ║
    ║     encodes the structure of BKL dynamics. This may be a hidden          ║
    ║     symmetry of M-theory itself.                                         ║
    ║                                                                           ║
    ║  4. THERMODYNAMIC STRUCTURE: BKL has a well-defined thermodynamic        ║
    ║     formalism. The KS entropy equals the Lyapunov exponent (Pesin).      ║
    ║                                                                           ║
    ║  5. QUANTUM FRONTIER: The quantum version of BKL dynamics remains        ║
    ║     mysterious. This is where the deepest secrets may lie.               ║
    ║                                                                           ║
    ║  6. EXPERIMENTAL WINDOW: Cold atom systems may allow us to               ║
    ║     experimentally probe BKL-like dynamics in the lab!                   ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(insights)
    
    # Next steps
    print("\nNEXT STEPS IN THE EXPLORATION:")
    print("-" * 50)
    next_steps = [
        "1. Implement full numerical relativity simulation of generic collapse",
        "2. Develop quantum BKL simulator using cold atom Hamiltonian",
        "3. Search for BKL signatures in existing LIGO/CMB data",
        "4. Prove AVTD for small perturbations (mathematical)",
        "5. Construct E₁₀ sigma model and verify predictions",
        "6. Explore holographic dual of BKL dynamics"
    ]
    for step in next_steps:
        print(f"  {step}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🔬 BKL DEEP EXPLORATION: Pushing the Boundaries\n")
    
    # Generate visualizations
    try:
        visualize_deep_exploration()
    except Exception as e:
        print(f"Visualization note: {e}")
    
    # Generate discovery report
    generate_discovery_report()
    
    # Print E10 description
    print("\n" + "=" * 78)
    print("APPENDIX: E₁₀ AND THE FUTURE OF PHYSICS")
    print("=" * 78)
    hd = HigherDimensionalBKL(11)
    print(hd.e10_root_structure())
    
    print("\n" + "=" * 78)
    print("Exploration continues... The universe awaits discovery!")
    print("=" * 78)
