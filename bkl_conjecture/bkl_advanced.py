"""
Advanced BKL Research: Towards the Big Dreams
==============================================

This module explores cutting-edge directions in BKL conjecture research:
1. Machine learning for Kasner transition detection
2. Statistical mechanics of BKL dynamics  
3. Connection to modular forms and arithmetic
4. Path towards rigorous proof
5. Quantum-classical correspondence

Author: Research Exploration
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import json
import warnings

# Import from main simulation
from bkl_simulation import KasnerSolution, BKLMap, MixmasterDynamics, HyperbolicBilliard

plt.style.use('default')


# =============================================================================
# PART 1: STATISTICAL MECHANICS OF BKL DYNAMICS
# =============================================================================

class BKLStatisticalMechanics:
    """
    Analyze BKL dynamics through the lens of statistical mechanics.
    
    The BKL map is related to the Gauss map, which has an invariant measure:
        dμ = dx / (ln(2)(1+x))  for x ∈ (0, 1]
    
    This gives rise to thermodynamic-like quantities.
    """
    
    def __init__(self):
        self.ln2 = np.log(2)
        
    def gauss_invariant_measure_density(self, x: np.ndarray) -> np.ndarray:
        """
        The invariant measure for the Gauss map T(x) = {1/x}.
        
        Density: ρ(x) = 1 / (ln(2)(1+x))
        """
        return 1 / (self.ln2 * (1 + x))
    
    def theoretical_lyapunov_exponent(self) -> float:
        """
        The exact Lyapunov exponent for the Gauss map.
        
        λ = π² / (6 ln 2) ≈ 2.373
        """
        return np.pi**2 / (6 * self.ln2)
    
    def continued_fraction_entropy(self) -> float:
        """
        The metric entropy of the Gauss map (Kolmogorov-Sinai entropy).
        
        h = π² / (6 ln 2) = λ (Pesin's identity!)
        """
        return self.theoretical_lyapunov_exponent()
    
    def levy_constant(self) -> float:
        """
        The Lévy constant: almost every real number x has
        
        lim_{n→∞} q_n^{1/n} = exp(π² / (12 ln 2))
        
        where q_n is the n-th continued fraction denominator.
        """
        return np.exp(np.pi**2 / (12 * self.ln2))
    
    def khinchin_constant(self) -> float:
        """
        Khinchin's constant: for almost every x,
        
        lim_{n→∞} (a_1 · a_2 · ... · a_n)^{1/n} = K ≈ 2.685
        
        where a_i are the continued fraction coefficients.
        """
        # Computed from the formula involving Riemann zeta function
        return 2.6854520010653064
    
    def partition_function_bkl(self, beta: float, 
                                n_samples: int = 100000) -> float:
        """
        Define a partition function for BKL dynamics.
        
        Z(β) = Σ exp(-β E_n)
        
        where E_n could be defined as the number of epochs in an era.
        """
        # Sample continued fraction coefficients
        x_samples = np.random.uniform(0.001, 0.999, n_samples)
        
        Z = 0.0
        for x in x_samples:
            # Extract first continued fraction coefficient
            a = int(1/x)
            Z += np.exp(-beta * a)
        
        return Z / n_samples * self.ln2  # Normalize by measure
    
    def compute_era_distribution(self, u0: float, 
                                  n_eras: int = 10000) -> np.ndarray:
        """
        Compute the distribution of era lengths.
        
        For the Gauss map, this should follow:
        P(n bounces) ~ 1/(n(n+1))
        """
        u = u0
        era_lengths = []
        
        for _ in range(n_eras):
            length = 0
            while u >= 2:
                u = u - 1
                length += 1
            era_lengths.append(length + 1)
            if u > 1:
                u = 1 / (u - 1)
            else:
                u = np.random.uniform(1.001, 10)  # Restart with random
        
        return np.array(era_lengths)
    
    def free_energy(self, beta: float, n_samples: int = 50000) -> float:
        """
        Compute the "free energy" F = -ln(Z)/β for BKL dynamics.
        """
        Z = self.partition_function_bkl(beta, n_samples)
        if Z > 0:
            return -np.log(Z) / beta
        return np.inf


# =============================================================================
# PART 2: MODULAR FORMS AND ARITHMETIC CONNECTIONS
# =============================================================================

class ArithmeticBKL:
    """
    Explore the deep connections between BKL dynamics and number theory.
    
    The Gauss map is intimately connected to:
    - Continued fractions
    - The modular group SL(2,Z)
    - Dedekind eta function
    - Hyperbolic geometry
    """
    
    @staticmethod
    def continued_fraction_coefficients(x: float, n_terms: int = 50) -> List[int]:
        """
        Compute the continued fraction expansion of x.
        
        x = a_0 + 1/(a_1 + 1/(a_2 + ...))
        """
        coeffs = []
        for _ in range(n_terms):
            a = int(x)
            coeffs.append(a)
            frac = x - a
            if abs(frac) < 1e-15:
                break
            x = 1 / frac
        return coeffs
    
    @staticmethod
    def convergents(x: float, n_terms: int = 20) -> List[Tuple[int, int]]:
        """
        Compute the convergents p_n/q_n of the continued fraction.
        
        These are the best rational approximations to x.
        """
        coeffs = ArithmeticBKL.continued_fraction_coefficients(x, n_terms)
        
        # Initialize
        p_prev, p_curr = 1, coeffs[0]
        q_prev, q_curr = 0, 1
        
        convergents = [(p_curr, q_curr)]
        
        for i in range(1, len(coeffs)):
            a = coeffs[i]
            p_new = a * p_curr + p_prev
            q_new = a * q_curr + q_prev
            convergents.append((p_new, q_new))
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new
        
        return convergents
    
    @staticmethod
    def modular_group_matrix(a: int) -> np.ndarray:
        """
        The Gauss map iteration corresponds to multiplication by
        modular group elements.
        
        T^a · S where T = [[1,1],[0,1]], S = [[0,-1],[1,0]]
        """
        T = np.array([[1, 1], [0, 1]])
        S = np.array([[0, -1], [1, 0]])
        
        T_power = np.linalg.matrix_power(T, a)
        return T_power @ S
    
    @staticmethod
    def bkl_as_modular_orbit(x0: float, n_iterations: int = 20) -> List[np.ndarray]:
        """
        Represent the BKL orbit as a sequence of modular group elements.
        """
        coeffs = ArithmeticBKL.continued_fraction_coefficients(x0, n_iterations)
        matrices = []
        
        for a in coeffs[1:]:  # Skip a_0
            matrices.append(ArithmeticBKL.modular_group_matrix(a))
        
        return matrices
    
    @staticmethod  
    def dedekind_eta_approximation(tau: complex, n_terms: int = 100) -> complex:
        """
        Approximate the Dedekind eta function:
        
        η(τ) = e^{πiτ/12} ∏_{n=1}^∞ (1 - e^{2πinτ})
        
        This is connected to modular forms and BKL dynamics.
        """
        if tau.imag <= 0:
            raise ValueError("τ must be in upper half-plane")
        
        q = np.exp(2j * np.pi * tau)
        eta = np.exp(1j * np.pi * tau / 12)
        
        for n in range(1, n_terms + 1):
            eta *= (1 - q**n)
        
        return eta


# =============================================================================
# PART 3: PATH TO RIGOROUS PROOF
# =============================================================================

class RigorousBKLAnalysis:
    """
    Tools for analyzing the BKL conjecture rigorously.
    
    Key ingredients for a proof:
    1. AVTD (Asymptotic Velocity Term Dominance)
    2. Stability of Kasner epochs
    3. Control of spatial derivatives
    4. Measure-theoretic genericity
    """
    
    def __init__(self, grid_size: int = 100):
        self.N = grid_size
        self.dx = 2 * np.pi / grid_size
        self.x = np.linspace(0, 2*np.pi, grid_size, endpoint=False)
    
    def avtd_functional(self, state: Dict[str, np.ndarray]) -> float:
        """
        Compute the AVTD functional:
        
        F[g] = ||∂_t g||² / ||∂_x g||²
        
        The BKL conjecture predicts F → ∞ near the singularity.
        """
        # Time derivatives (extrinsic curvature)
        K_sq = (state['K_alpha']**2 + state['K_beta']**2 + 
                state['K_gamma']**2).mean()
        
        # Spatial derivatives
        d_alpha = np.gradient(state['alpha'], self.dx)
        d_beta = np.gradient(state['beta'], self.dx)
        d_gamma = np.gradient(state['gamma'], self.dx)
        
        spatial_sq = (d_alpha**2 + d_beta**2 + d_gamma**2).mean() + 1e-20
        
        return K_sq / spatial_sq
    
    def kasner_stability_eigenvalues(self, p1: float, p2: float, 
                                     p3: float) -> np.ndarray:
        """
        Compute the stability eigenvalues of a Kasner solution.
        
        These determine whether the Kasner state is an attractor
        or repeller in the space of Bianchi IX solutions.
        """
        # For Bianchi IX, the stability is determined by the
        # exponential growth rates of curvature terms
        
        # The three wall collision eigenvalues
        eigenvalues = np.array([
            2 * (p2 + p3) - 2 * p1,  # Wall 1
            2 * (p1 + p3) - 2 * p2,  # Wall 2
            2 * (p1 + p2) - 2 * p3   # Wall 3
        ])
        
        return eigenvalues
    
    def identify_spike_candidates(self, state: Dict[str, np.ndarray],
                                   threshold: float = 5.0) -> List[int]:
        """
        Identify spatial locations that might form spikes.
        
        Spikes are localized regions where AVTD appears to break down.
        """
        # Compute local gradient magnitudes
        grad_alpha = np.gradient(state['alpha'], self.dx)
        grad_beta = np.gradient(state['beta'], self.dx)
        
        gradient_mag = np.sqrt(grad_alpha**2 + grad_beta**2)
        
        # Spikes appear where gradient is much larger than average
        mean_grad = gradient_mag.mean()
        spike_indices = np.where(gradient_mag > threshold * mean_grad)[0]
        
        return list(spike_indices)
    
    def bkl_attractor_measure(self, n_samples: int = 1000,
                               n_iterations: int = 100) -> float:
        """
        Estimate the measure of initial conditions that exhibit
        BKL behavior (oscillatory approach).
        
        For generic initial data, this should be 1.
        """
        bkl_count = 0
        
        for _ in range(n_samples):
            # Random initial u
            u = np.random.uniform(1.001, 100)
            
            # Check if orbit is oscillatory (bounces at least 10 times)
            n_bounces = 0
            for _ in range(n_iterations):
                if u >= 2:
                    u = u - 1
                elif u > 1:
                    u = 1 / (u - 1)
                    n_bounces += 1
                else:
                    break
            
            if n_bounces >= 10:
                bkl_count += 1
        
        return bkl_count / n_samples


# =============================================================================
# PART 4: QUANTUM-CLASSICAL CORRESPONDENCE
# =============================================================================

class QuantumClassicalBKL:
    """
    Explore the quantum-classical correspondence in BKL dynamics.
    
    Key questions:
    1. How does quantum mechanics regularize the singularity?
    2. What is the quantum analog of BKL chaos?
    3. Can we see signatures of E₁₀ in quantum gravity?
    """
    
    def __init__(self, hbar: float = 1.0):
        self.hbar = hbar
    
    def wheeler_dewitt_potential(self, beta_plus: float, 
                                  beta_minus: float,
                                  Omega: float) -> float:
        """
        The potential in the Wheeler-DeWitt equation for Bianchi IX.
        
        The WDW equation is: (-∂²/∂Ω² + ∂²/∂β₊² + ∂²/∂β₋² + V)Ψ = 0
        """
        sqrt3 = np.sqrt(3)
        
        # Exponentially steep walls (classical limit → billiard)
        V = (np.exp(-8*beta_plus) 
             - 4*np.exp(-2*beta_plus)*np.cosh(2*sqrt3*beta_minus)
             + 2*np.exp(4*beta_plus)*(np.cosh(4*sqrt3*beta_minus) - 1))
        
        # Include Omega-dependent prefactor
        V *= np.exp(-2*Omega)
        
        return V
    
    def semiclassical_wavefunction(self, beta_plus: float, beta_minus: float,
                                    Omega: float, 
                                    classical_action: float) -> complex:
        """
        WKB approximation to the wavefunction:
        
        Ψ ≈ A · exp(i S / ℏ)
        
        where S is the classical action.
        """
        return np.exp(1j * classical_action / self.hbar)
    
    def quantum_bounce_scale(self, mass: float = 1.0) -> float:
        """
        Estimate the scale at which quantum effects prevent the singularity.
        
        In LQC: a_bounce ~ ℓ_P · (ρ/ρ_P)^{1/6}
        """
        l_P = np.sqrt(self.hbar)  # Planck length (natural units)
        return l_P
    
    def decoherence_rate(self, energy_gap: float, 
                         temperature: float = 1.0) -> float:
        """
        Estimate the decoherence rate for quantum BKL superpositions.
        
        This determines how quickly quantum coherence is lost.
        """
        k_B = 1.0  # Natural units
        return energy_gap * (1 + 1/(np.exp(energy_gap/(k_B*temperature)) - 1))
    
    def compute_quantum_corrections(self, classical_potential: float,
                                     curvature: float) -> float:
        """
        One-loop quantum corrections to the BKL potential.
        
        V_quantum ≈ V_classical + ℏ²/m · R + O(ℏ⁴)
        
        where R is the scalar curvature.
        """
        return classical_potential + self.hbar**2 * curvature


# =============================================================================
# PART 5: MACHINE LEARNING FOR BKL ANALYSIS
# =============================================================================

class BKLMachineLearning:
    """
    Machine learning approaches to BKL dynamics.
    
    Applications:
    1. Automatic detection of Kasner transitions
    2. Prediction of era structure
    3. Anomaly detection (spikes, non-generic behavior)
    """
    
    def __init__(self):
        self.training_data = None
        self.transition_classifier = None
    
    def generate_training_data(self, n_samples: int = 10000) -> Dict:
        """
        Generate labeled training data for Kasner transition detection.
        """
        # Generate sequences of u values
        sequences = []
        labels = []  # 1 = transition, 0 = no transition
        
        for _ in range(n_samples):
            u0 = np.random.uniform(1.001, 20)
            u = u0
            seq = [u]
            
            for _ in range(50):
                u_new = BKLMap.iterate(u)
                seq.append(u_new)
                
                # Label: transition when 1 < u < 2
                labels.append(1 if 1 < u < 2 else 0)
                u = u_new
            
            sequences.append(seq)
        
        self.training_data = {
            'sequences': np.array(sequences),
            'labels': np.array(labels)
        }
        
        return self.training_data
    
    def extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract features from a u-sequence for classification.
        """
        features = []
        
        # Basic statistics
        features.append(np.mean(sequence))
        features.append(np.std(sequence))
        features.append(np.max(sequence))
        features.append(np.min(sequence))
        
        # Differences
        diffs = np.diff(sequence)
        features.append(np.mean(np.abs(diffs)))
        features.append(np.max(np.abs(diffs)))
        
        # Fraction of time in (1, 2) interval
        features.append(np.mean((sequence > 1) & (sequence < 2)))
        
        # Second differences (acceleration)
        if len(sequence) > 2:
            d2 = np.diff(diffs)
            features.append(np.mean(np.abs(d2)))
        else:
            features.append(0)
        
        return np.array(features)
    
    def simple_classifier(self, u_window: np.ndarray) -> float:
        """
        Simple rule-based classifier for transition detection.
        
        Returns probability of transition in next step.
        """
        u_current = u_window[-1]
        
        # Clear transition if 1 < u < 2
        if 1 < u_current < 2:
            return 0.95
        
        # Approaching transition
        if 2 <= u_current < 3:
            return 0.3
        
        # Far from transition
        return 0.05
    
    def predict_era_length(self, u0: float) -> int:
        """
        Predict the length of the current era from initial u.
        
        An era ends when u drops below 2.
        """
        # Era length is floor(u0) - 1 plus one final transition
        return int(u0) if u0 >= 2 else 1


# =============================================================================
# PART 6: VISUALIZATION OF ADVANCED CONCEPTS
# =============================================================================

class AdvancedVisualization:
    """
    Advanced visualizations for BKL research.
    """
    
    @staticmethod
    def plot_statistical_distributions(save_path: str = None):
        """
        Plot statistical distributions of BKL dynamics.
        """
        stats = BKLStatisticalMechanics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Invariant measure
        x = np.linspace(0.01, 0.99, 1000)
        density = stats.gauss_invariant_measure_density(x)
        axes[0, 0].plot(x, density, 'b-', linewidth=2)
        axes[0, 0].fill_between(x, density, alpha=0.3)
        axes[0, 0].set_xlabel('x', fontsize=12)
        axes[0, 0].set_ylabel('ρ(x)', fontsize=12)
        axes[0, 0].set_title('Gauss Map Invariant Measure', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: Era length distribution
        era_lengths = stats.compute_era_distribution(1 + np.pi, n_eras=5000)
        
        unique, counts = np.unique(era_lengths, return_counts=True)
        axes[0, 1].bar(unique[:20], counts[:20]/counts.sum(), 
                       color='steelblue', alpha=0.7, label='Numerical')
        
        # Theoretical: P(n) ~ 1/(n(n+1))
        n_theory = np.arange(1, 21)
        p_theory = 1/(n_theory * (n_theory + 1))
        p_theory = p_theory / p_theory.sum()
        axes[0, 1].plot(n_theory, p_theory, 'ro-', label='Theory', markersize=5)
        
        axes[0, 1].set_xlabel('Era length (Kasner epochs)', fontsize=12)
        axes[0, 1].set_ylabel('Probability', fontsize=12)
        axes[0, 1].set_title('Distribution of Era Lengths', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Partition function
        betas = np.linspace(0.1, 5, 30)
        Z_values = [stats.partition_function_bkl(b, n_samples=20000) for b in betas]
        axes[1, 0].semilogy(betas, Z_values, 'b-o', markersize=4)
        axes[1, 0].set_xlabel('β (inverse temperature)', fontsize=12)
        axes[1, 0].set_ylabel('Z(β)', fontsize=12)
        axes[1, 0].set_title('BKL "Partition Function"', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 4: Important constants
        constants = {
            'Lyapunov exponent': stats.theoretical_lyapunov_exponent(),
            'KS entropy': stats.continued_fraction_entropy(),
            'Lévy constant': stats.levy_constant(),
            'Khinchin constant': stats.khinchin_constant()
        }
        
        y_pos = np.arange(len(constants))
        values = list(constants.values())
        names = list(constants.keys())
        
        bars = axes[1, 1].barh(y_pos, values, color='steelblue', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(names)
        axes[1, 1].set_xlabel('Value', fontsize=12)
        axes[1, 1].set_title('Fundamental Constants of BKL/Gauss Dynamics', fontsize=14)
        
        # Add value labels
        for bar, val in zip(bars, values):
            axes[1, 1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                           f'{val:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_modular_connection(save_path: str = None):
        """
        Visualize the connection to modular forms.
        """
        arith = ArithmeticBKL()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Continued fraction convergents approaching π
        x = np.pi
        convergents = arith.convergents(x, n_terms=15)
        
        ratios = [p/q for p, q in convergents]
        errors = [abs(r - x) for r in ratios]
        
        axes[0].semilogy(range(len(errors)), errors, 'bo-', markersize=8)
        axes[0].set_xlabel('Convergent index n', fontsize=12)
        axes[0].set_ylabel('|p_n/q_n - π|', fontsize=12)
        axes[0].set_title('Convergents to π: Best Rational Approximations', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Add convergent labels
        for i, (p, q) in enumerate(convergents[:8]):
            axes[0].annotate(f'{p}/{q}', (i, errors[i]), 
                           textcoords='offset points', xytext=(5, 5), fontsize=8)
        
        # Panel 2: Dedekind eta function on fundamental domain
        # Sample points in the fundamental domain
        n_points = 100
        x_range = np.linspace(-0.5, 0.5, n_points)
        y_min = np.sqrt(1 - x_range**2)  # Arc of unit circle
        y_min = np.maximum(y_min, np.sqrt(3)/2)  # Cut off by horizontal line
        
        # Create grid in upper half-plane
        X = []
        Y = []
        Z = []
        
        for x_val in np.linspace(-0.5, 0.5, 50):
            y_start = max(np.sqrt(3)/2, np.sqrt(max(0, 1 - x_val**2)))
            for y_val in np.linspace(y_start, 2.5, 40):
                tau = complex(x_val, y_val)
                try:
                    eta = arith.dedekind_eta_approximation(tau, n_terms=50)
                    X.append(x_val)
                    Y.append(y_val)
                    Z.append(np.abs(eta))
                except:
                    pass
        
        if len(X) > 10:
            scatter = axes[1].scatter(X, Y, c=Z, cmap='viridis', s=20, alpha=0.7)
            plt.colorbar(scatter, ax=axes[1], label='|η(τ)|')
        
        # Draw fundamental domain boundary
        theta = np.linspace(np.pi/3, 2*np.pi/3, 100)
        arc_x = np.cos(theta)
        arc_y = np.sin(theta)
        axes[1].plot(arc_x, arc_y, 'r-', linewidth=2, label='|τ| = 1')
        axes[1].axvline(x=-0.5, color='r', linewidth=2, linestyle='--')
        axes[1].axvline(x=0.5, color='r', linewidth=2, linestyle='--')
        
        axes[1].set_xlabel('Re(τ)', fontsize=12)
        axes[1].set_ylabel('Im(τ)', fontsize=12)
        axes[1].set_title('Dedekind η Function on Fundamental Domain', fontsize=14)
        axes[1].set_xlim(-0.7, 0.7)
        axes[1].set_ylim(0.4, 2.5)
        axes[1].set_aspect('equal')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_quantum_classical_transition(save_path: str = None):
        """
        Visualize the quantum-classical transition in BKL dynamics.
        """
        qc = QuantumClassicalBKL()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Wheeler-DeWitt potential
        bp = np.linspace(-1.5, 1.5, 100)
        bm = np.linspace(-1.5, 1.5, 100)
        BP, BM = np.meshgrid(bp, bm)
        
        V = np.zeros_like(BP)
        for i in range(BP.shape[0]):
            for j in range(BP.shape[1]):
                V[i, j] = qc.wheeler_dewitt_potential(BP[i, j], BM[i, j], Omega=0)
        
        V = np.clip(V, -5, 50)
        
        contour = axes[0].contourf(BP, BM, V, levels=30, cmap='plasma')
        plt.colorbar(contour, ax=axes[0], label='V(β₊, β₋)')
        axes[0].contour(BP, BM, V, levels=[0], colors='white', linewidths=2)
        
        axes[0].set_xlabel('β₊', fontsize=12)
        axes[0].set_ylabel('β₋', fontsize=12)
        axes[0].set_title('Wheeler-DeWitt Potential (Ω=0)', fontsize=14)
        axes[0].set_aspect('equal')
        
        # Panel 2: Quantum bounce scales
        hbar_values = np.logspace(-3, 0, 50)
        bounce_scales = [QuantumClassicalBKL(h).quantum_bounce_scale() for h in hbar_values]
        
        axes[1].loglog(hbar_values, bounce_scales, 'b-', linewidth=2)
        axes[1].axhline(y=1, color='r', linestyle='--', label='Planck scale')
        axes[1].set_xlabel('ℏ (natural units)', fontsize=12)
        axes[1].set_ylabel('Quantum bounce scale', fontsize=12)
        axes[1].set_title('Scale of Quantum Effects', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# =============================================================================
# PART 7: MAIN EXPLORATION FUNCTIONS
# =============================================================================

def run_advanced_exploration():
    """
    Run comprehensive advanced exploration of BKL dynamics.
    """
    print("=" * 70)
    print("ADVANCED BKL RESEARCH EXPLORATION")
    print("Towards the Big Dreams in Cosmological Singularities")
    print("=" * 70)
    
    # 1. Statistical Mechanics
    print("\n[1] STATISTICAL MECHANICS OF BKL DYNAMICS")
    print("-" * 50)
    
    stats = BKLStatisticalMechanics()
    print(f"Theoretical Lyapunov exponent: {stats.theoretical_lyapunov_exponent():.6f}")
    print(f"Kolmogorov-Sinai entropy: {stats.continued_fraction_entropy():.6f}")
    print(f"Lévy constant: {stats.levy_constant():.6f}")
    print(f"Khinchin constant: {stats.khinchin_constant():.6f}")
    
    # 2. Arithmetic Connections
    print("\n[2] ARITHMETIC AND MODULAR CONNECTIONS")
    print("-" * 50)
    
    arith = ArithmeticBKL()
    
    # Continued fraction of golden ratio
    phi = (1 + np.sqrt(5)) / 2
    cf_golden = arith.continued_fraction_coefficients(phi, 10)
    print(f"Golden ratio CF: {cf_golden}")
    print("  (All 1s - the 'most irrational' number!)")
    
    cf_pi = arith.continued_fraction_coefficients(np.pi, 10)
    print(f"π continued fraction: {cf_pi}")
    
    # 3. Rigorous Analysis
    print("\n[3] RIGOROUS ANALYSIS TOOLS")
    print("-" * 50)
    
    rigorous = RigorousBKLAnalysis()
    bkl_measure = rigorous.bkl_attractor_measure(n_samples=5000)
    print(f"Measure of BKL-like orbits: {bkl_measure:.4f}")
    print("  (Should be 1.0 according to BKL conjecture)")
    
    # Kasner stability
    p1, p2, p3 = KasnerSolution.exponents_from_u(2.0)
    eigenvalues = rigorous.kasner_stability_eigenvalues(p1, p2, p3)
    print(f"\nKasner stability eigenvalues (u=2):")
    print(f"  λ₁ = {eigenvalues[0]:.4f}")
    print(f"  λ₂ = {eigenvalues[1]:.4f}")
    print(f"  λ₃ = {eigenvalues[2]:.4f}")
    
    # 4. Quantum-Classical
    print("\n[4] QUANTUM-CLASSICAL CORRESPONDENCE")
    print("-" * 50)
    
    qc = QuantumClassicalBKL(hbar=1.0)
    bounce_scale = qc.quantum_bounce_scale()
    print(f"Quantum bounce scale: {bounce_scale:.6f} (Planck units)")
    
    # 5. Machine Learning
    print("\n[5] MACHINE LEARNING INSIGHTS")
    print("-" * 50)
    
    ml = BKLMachineLearning()
    data = ml.generate_training_data(n_samples=1000)
    print(f"Training data generated: {len(data['labels'])} samples")
    print(f"Transition fraction: {data['labels'].mean():.4f}")
    
    # 6. Generate visualizations
    print("\n[6] GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    try:
        AdvancedVisualization.plot_statistical_distributions('bkl_statistics.png')
        print("✓ Saved: bkl_statistics.png")
    except Exception as e:
        print(f"× Statistics plot: {e}")
    
    try:
        AdvancedVisualization.plot_modular_connection('bkl_modular.png')
        print("✓ Saved: bkl_modular.png")
    except Exception as e:
        print(f"× Modular plot: {e}")
    
    try:
        AdvancedVisualization.plot_quantum_classical_transition('bkl_quantum.png')
        print("✓ Saved: bkl_quantum.png")
    except Exception as e:
        print(f"× Quantum plot: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY: BIG DREAMS STATUS")
    print("=" * 70)
    
    dreams = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                    THE BIG DREAMS OF BKL                          ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  [◐] 1. RIGOROUS PROOF OF BKL CONJECTURE                          ║
    ║       - AVTD shown for homogeneous cases (Ringström)              ║
    ║       - Generic inhomogeneous case: OPEN                          ║
    ║                                                                   ║
    ║  [◐] 2. UNDERSTAND THE ROLE OF SPIKES                             ║
    ║       - Numerical evidence: spikes exist                          ║
    ║       - Measure-theoretic treatment: OPEN                         ║
    ║                                                                   ║
    ║  [◐] 3. E₁₀ AND HIDDEN SYMMETRIES                                 ║
    ║       - DHN billiard description: ESTABLISHED                     ║
    ║       - Full E₁₀ sigma model: OPEN                                ║
    ║                                                                   ║
    ║  [◐] 4. QUANTUM GRAVITY SIGNATURES                                ║
    ║       - LQC bounce: ESTABLISHED in models                         ║
    ║       - BKL in full quantum gravity: OPEN                         ║
    ║                                                                   ║
    ║  [○] 5. OBSERVATIONAL SIGNATURES                                  ║
    ║       - Pre-Big-Bang cosmology hints                              ║
    ║       - Direct detection: FUTURE                                  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Legend: [●] Solved  [◐] Partial progress  [○] Open
    """
    
    print(dreams)
    
    return {
        'stats': stats,
        'arith': arith,
        'rigorous': rigorous,
        'qc': qc,
        'ml': ml
    }


if __name__ == "__main__":
    results = run_advanced_exploration()
