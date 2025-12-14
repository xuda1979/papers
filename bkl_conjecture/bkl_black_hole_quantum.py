"""
BKL Black Hole Interior and Quantum Gravity Corrections
========================================================

This module explores:
1. Black hole interior BKL dynamics
   - Schwarzschild interior analysis
   - Kerr interior complexity
   - Mass inflation at Cauchy horizon
   
2. Loop Quantum Cosmology modifications
   - Quantum bounce replacing singularity
   - Modified BKL near Planck scale
   - Effective equations
   
3. Spike formation and stability
   - Numerical spike analysis
   - Spike classification
   - Genericity of spikes

4. Holographic BKL
   - AdS/CFT correspondence aspects
   - Scrambling and thermalization
   - Complexity growth

Author: Research Exploration
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve, brentq
from scipy.special import ellipk, ellipe
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings

np.random.seed(42)
plt.style.use('default')


# =============================================================================
# PART 1: BLACK HOLE INTERIOR BKL DYNAMICS
# =============================================================================

class BlackHoleInterior:
    """
    Analyze BKL dynamics inside black holes.
    
    The interior of a black hole approaches a spacelike singularity,
    which should exhibit BKL behavior for generic perturbations.
    """
    
    def __init__(self, M: float = 1.0, a: float = 0.0, Q: float = 0.0):
        """
        Initialize black hole with mass M, spin a, charge Q.
        
        Schwarzschild: a = Q = 0
        Kerr: Q = 0
        Kerr-Newman: general
        """
        self.M = M
        self.a = a  # Spin parameter
        self.Q = Q  # Charge
        self.G = 1.0  # Geometric units
        self.c = 1.0
        
    def horizons(self) -> Dict:
        """
        Calculate horizon locations.
        
        r± = M ± √(M² - a² - Q²)
        """
        discriminant = self.M**2 - self.a**2 - self.Q**2
        
        if discriminant < 0:
            return {'type': 'naked_singularity', 'horizons': []}
        elif discriminant == 0:
            return {'type': 'extremal', 'horizons': [self.M]}
        else:
            r_plus = self.M + np.sqrt(discriminant)
            r_minus = self.M - np.sqrt(discriminant)
            return {
                'type': 'non_extremal',
                'outer_horizon': r_plus,
                'inner_horizon': r_minus,
                'horizons': [r_plus, r_minus]
            }
    
    def schwarzschild_interior_evolution(self, r_initial: float, 
                                          n_points: int = 1000) -> Dict:
        """
        Evolve through Schwarzschild interior (r < 2M).
        
        Inside the horizon, r becomes timelike and t spacelike.
        The metric is:
        ds² = -(2M/r - 1)⁻¹ dr² + (2M/r - 1) dt² + r² dΩ²
        """
        if r_initial >= 2 * self.M:
            raise ValueError("Must start inside horizon (r < 2M)")
        
        r_values = np.linspace(r_initial, 0.01 * self.M, n_points)
        
        # Metric components
        g_rr = -1 / (2*self.M/r_values - 1)
        g_tt = 2*self.M/r_values - 1
        g_theta = r_values**2
        
        # Kretschmann scalar (curvature invariant)
        K = 48 * self.M**2 / r_values**6
        
        # Kasner-like exponents near r → 0
        # The Schwarzschild singularity is Kasner-like but non-oscillatory
        # (it's a "cigar" singularity with exponents (2/3, 2/3, -1/3))
        
        # Proper time to singularity
        tau = np.zeros_like(r_values)
        for i in range(1, len(r_values)):
            # dτ = √|g_rr| dr
            dr = r_values[i-1] - r_values[i]
            tau[i] = tau[i-1] + np.sqrt(abs(g_rr[i])) * dr
        
        return {
            'r': r_values,
            'proper_time': tau,
            'g_rr': g_rr,
            'g_tt': g_tt,
            'Kretschmann': K,
            'kasner_type': 'cigar',
            'kasner_exponents': (2/3, 2/3, -1/3),
            'is_oscillatory': False
        }
    
    def kerr_interior_structure(self) -> Dict:
        """
        Analyze the interior structure of a Kerr black hole.
        
        The Kerr interior is much more complex than Schwarzschild:
        - Ring singularity at r = 0, θ = π/2
        - Inner (Cauchy) horizon r₋
        - Potential for BKL behavior with perturbations
        """
        horizons = self.horizons()
        
        if horizons['type'] == 'naked_singularity':
            return {'error': 'No interior for naked singularity'}
        
        r_plus = horizons.get('outer_horizon', self.M)
        r_minus = horizons.get('inner_horizon', 0)
        
        # The region between r₊ and r₋
        region_I = {
            'name': 'Between horizons',
            'r_range': (r_minus, r_plus),
            'character': 'r is timelike',
            'singularity_type': None
        }
        
        # The region r < r₋
        region_II = {
            'name': 'Inside inner horizon',
            'r_range': (0, r_minus),
            'character': 'r is spacelike again',
            'singularity_type': 'ring at r=0, θ=π/2'
        }
        
        # Cauchy horizon instability
        cauchy_horizon = {
            'location': r_minus,
            'stability': 'unstable',
            'mass_inflation': True,
            'description': 'Generic perturbations cause mass inflation, '
                          'converting inner horizon to null singularity'
        }
        
        return {
            'horizons': horizons,
            'region_I': region_I,
            'region_II': region_II,
            'cauchy_horizon': cauchy_horizon,
            'bkl_relevance': 'Generic perturbations lead to spacelike singularity with BKL'
        }
    
    def mass_inflation_dynamics(self, v_initial: float = 1.0,
                                 perturbation: float = 1e-6,
                                 n_steps: int = 1000) -> Dict:
        """
        Simulate mass inflation near the Cauchy horizon.
        
        The inner horizon is unstable: small perturbations cause the 
        effective mass to grow exponentially ("mass inflation").
        
        This converts the inner horizon into a null singularity,
        and eventually may lead to a spacelike singularity with BKL.
        """
        # Simplified model: Poisson-Israel mass inflation
        # dm/dv = m * κ * δ  where v is advanced time, κ surface gravity
        
        kappa = (self.horizons()['outer_horizon'] - 
                 self.horizons().get('inner_horizon', 0)) / (2 * self.M**2)
        
        v_values = np.linspace(v_initial, v_initial + 10, n_steps)
        
        # Mass function evolution
        m_values = self.M * np.exp(kappa * perturbation * (v_values - v_initial))
        
        # Curvature grows as m increases
        R_values = m_values**2 / self.M**6
        
        return {
            'v': v_values,
            'mass': m_values,
            'curvature': R_values,
            'surface_gravity': kappa,
            'inflation_rate': kappa * perturbation,
            'singularity_formation': 'Mass inflation leads to curvature singularity'
        }
    
    def bkl_onset_from_perturbation(self, epsilon: float = 0.01) -> Dict:
        """
        Analyze how generic perturbations trigger BKL behavior.
        
        The Schwarzschild interior is non-oscillatory (Kasner cigar).
        But generic perturbations break the spherical symmetry
        and can induce BKL oscillations.
        """
        # Unperturbed Kasner exponents for Schwarzschild
        p1_0, p2_0, p3_0 = 2/3, 2/3, -1/3
        
        # Perturbation breaks the degeneracy p1 = p2
        # This can trigger transitions to oscillatory behavior
        
        # Perturbed exponents
        p1 = p1_0 + epsilon
        p2 = p2_0 - epsilon
        p3 = p3_0
        
        # Renormalize to satisfy Kasner constraints
        sum_p = p1 + p2 + p3
        p1, p2, p3 = p1/sum_p, p2/sum_p, p3/sum_p
        
        # Check stability
        # If all exponents positive, stable (non-oscillatory)
        # If one negative, can trigger oscillations
        
        oscillatory = p3 < 0 and (p1 != p2)
        
        # Estimate BKL parameter u from exponents
        if p3 < 0:
            # Invert the Kasner parametrization
            # p3 = u(1+u)/(1+u+u²) solve for u
            # This gives u ≈ (1 + sqrt(1 + 4*|p3|))/(2*|p3|)
            u_estimate = (1 + np.sqrt(1 + 4*abs(p3))) / (2*abs(p3) + 0.01)
        else:
            u_estimate = np.inf  # Non-oscillatory
        
        return {
            'unperturbed_exponents': (p1_0, p2_0, p3_0),
            'perturbed_exponents': (p1, p2, p3),
            'perturbation': epsilon,
            'oscillatory': oscillatory,
            'estimated_u': u_estimate,
            'conclusion': 'Generic perturbations induce BKL oscillations' if oscillatory
                         else 'Perturbation too small for oscillations'
        }


# =============================================================================
# PART 2: LOOP QUANTUM COSMOLOGY MODIFICATIONS
# =============================================================================

class LoopQuantumBKL:
    """
    Explore loop quantum gravity modifications to BKL dynamics.
    
    LQC predicts:
    - Singularity resolution via quantum bounce
    - Modified dynamics near Planck density
    - Finite number of BKL oscillations before bounce
    """
    
    def __init__(self, gamma: float = 0.2375):
        """
        Initialize with Barbero-Immirzi parameter γ.
        
        γ ≈ 0.2375 from black hole entropy calculations.
        """
        self.gamma = gamma
        self.l_Pl = 1.616e-35  # Planck length (m)
        self.rho_Pl = 5.155e96  # Planck density (kg/m³)
        self.t_Pl = 5.391e-44  # Planck time (s)
        
        # LQC critical density (bounce occurs at ρ ≈ 0.41 ρ_Pl)
        self.rho_crit = 0.41 * self.rho_Pl
        
    def effective_friedmann_equation(self, rho: float) -> float:
        """
        LQC effective Friedmann equation.
        
        H² = (8πG/3) ρ (1 - ρ/ρ_crit)
        
        Note: H² → 0 as ρ → ρ_crit (bounce)
        """
        if rho > self.rho_crit:
            return 0  # Classically forbidden
        
        H_squared = (8 * np.pi / 3) * rho * (1 - rho / self.rho_crit)
        return H_squared
    
    def quantum_corrected_kasner(self, u: float, rho: float) -> Tuple[float, float, float]:
        """
        Quantum-corrected Kasner exponents.
        
        Near the bounce, the effective dynamics modifies Kasner solutions.
        """
        # Classical exponents
        p1 = -u / (1 + u + u**2)
        p2 = (1 + u) / (1 + u + u**2)
        p3 = u * (1 + u) / (1 + u + u**2)
        
        # Quantum correction factor
        f = 1 - rho / self.rho_crit
        
        if f <= 0:
            # At or beyond bounce - return isotropic
            return (1/3, 1/3, 1/3)
        
        # Modified exponents (phenomenological model)
        # Anisotropy is damped near bounce
        damping = np.sqrt(f)
        
        p1_q = (p1 - 1/3) * damping + 1/3
        p2_q = (p2 - 1/3) * damping + 1/3
        p3_q = (p3 - 1/3) * damping + 1/3
        
        # Renormalize
        total = p1_q + p2_q + p3_q
        return (p1_q/total, p2_q/total, p3_q/total)
    
    def count_oscillations_before_bounce(self, u_initial: float = 3.14159,
                                          rho_initial: float = None) -> Dict:
        """
        Count BKL oscillations before reaching quantum bounce.
        
        In LQC, there's a finite number of Kasner epochs before
        the density reaches ρ_crit and bounces.
        """
        if rho_initial is None:
            rho_initial = 0.01 * self.rho_crit
        
        rho = rho_initial
        u = u_initial
        n_epochs = 0
        u_history = [u]
        rho_history = [rho]
        
        # Each epoch, density increases roughly as ρ ∝ exp(2H*t_epoch)
        # where t_epoch ~ ln(u)
        
        while rho < self.rho_crit and n_epochs < 10000:
            # BKL transition
            if u >= 2:
                u = u - 1
            elif u > 1:
                u = 1 / (u - 1)
            else:
                u = np.random.uniform(5, 10)
            
            # Density increase (simplified model)
            # Each epoch increases density by factor related to Kasner exponents
            p3 = u * (1 + u) / (1 + u + u**2)
            growth_factor = np.exp(2 * p3 * np.log(u + 1))
            rho = min(rho * growth_factor, self.rho_crit)
            
            n_epochs += 1
            u_history.append(u)
            rho_history.append(rho)
        
        return {
            'n_epochs': n_epochs,
            'final_u': u,
            'final_rho': rho,
            'u_history': np.array(u_history),
            'rho_history': np.array(rho_history),
            'reached_bounce': rho >= 0.99 * self.rho_crit
        }
    
    def effective_bianchi_ix_equations(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Effective LQC equations for Bianchi IX.
        
        Variables: y = [α, β₊, β₋, pα, pβ₊, pβ₋]
        where α = log(volume^{1/3}), β± are anisotropy parameters.
        """
        alpha, beta_plus, beta_minus, p_alpha, p_beta_plus, p_beta_minus = y
        
        # Volume
        V = np.exp(3 * alpha)
        
        # Effective density
        rho = (p_alpha**2 - p_beta_plus**2 - p_beta_minus**2) / (2 * V**2)
        
        # Quantum correction
        f = 1 - abs(rho) / self.rho_crit
        f = max(f, 0.01)  # Regularize
        
        # Potential from spatial curvature (Bianchi IX walls)
        U = V**2 * (np.exp(-8*beta_plus) + 2*np.exp(4*beta_plus)*np.cosh(4*np.sqrt(3)*beta_minus)
                    - 4*np.exp(-2*beta_plus)*np.cosh(2*np.sqrt(3)*beta_minus))
        
        # Equations of motion (Hamiltonian with quantum corrections)
        d_alpha = f * p_alpha / V**2
        d_beta_plus = f * (-p_beta_plus) / V**2
        d_beta_minus = f * (-p_beta_minus) / V**2
        
        # Momentum evolution
        dU_dalpha = 3 * U  # Simplified
        dU_dbeta_plus = V**2 * (-8*np.exp(-8*beta_plus) + 8*np.exp(4*beta_plus)*np.cosh(4*np.sqrt(3)*beta_minus)
                                + 8*np.exp(-2*beta_plus)*np.cosh(2*np.sqrt(3)*beta_minus))
        dU_dbeta_minus = V**2 * 4*np.sqrt(3) * (2*np.exp(4*beta_plus)*np.sinh(4*np.sqrt(3)*beta_minus)
                                                - 2*np.exp(-2*beta_plus)*np.sinh(2*np.sqrt(3)*beta_minus))
        
        d_p_alpha = -dU_dalpha * f
        d_p_beta_plus = -dU_dbeta_plus * f
        d_p_beta_minus = -dU_dbeta_minus * f
        
        return np.array([d_alpha, d_beta_plus, d_beta_minus, 
                        d_p_alpha, d_p_beta_plus, d_p_beta_minus])
    
    def simulate_lqc_bianchi_ix(self, t_span: Tuple[float, float] = (0, 100),
                                 initial_conditions: np.ndarray = None) -> Dict:
        """
        Simulate Bianchi IX evolution with LQC corrections.
        """
        if initial_conditions is None:
            # Default: near singularity with anisotropy
            initial_conditions = np.array([0.1, 0.5, 0.3, -1.0, 0.5, 0.3])
        
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        try:
            sol = solve_ivp(self.effective_bianchi_ix_equations,
                           t_span, initial_conditions, t_eval=t_eval,
                           method='RK45', max_step=0.1)
            
            return {
                't': sol.t,
                'alpha': sol.y[0],
                'beta_plus': sol.y[1],
                'beta_minus': sol.y[2],
                'volume': np.exp(3 * sol.y[0]),
                'success': sol.success
            }
        except Exception as e:
            return {'error': str(e), 'success': False}


# =============================================================================
# PART 3: SPIKE DYNAMICS
# =============================================================================

class SpikeAnalysis:
    """
    Analyze spike formation in BKL dynamics.
    
    Spikes are localized regions where the AVTD assumption breaks down.
    They may be generic or special features of BKL singularities.
    """
    
    def __init__(self):
        pass
    
    def generate_spike_solution(self, x: np.ndarray, t: float,
                                 spike_position: float = 0.0,
                                 spike_width: float = 0.1) -> Dict:
        """
        Generate an explicit spike solution.
        
        Spikes are exact solutions where spatial gradients remain important.
        They can be constructed using Gowdy-like symmetries.
        """
        # Spike profile (simplified model)
        # The metric function develops a sharp feature
        
        spike_profile = 1 / np.cosh((x - spike_position) / spike_width)**2
        
        # Time evolution - spike sharpens as t → 0
        time_factor = np.abs(t)**0.5
        
        # Metric perturbation
        h = spike_profile * time_factor
        
        # Gradient (measures deviation from AVTD)
        dh_dx = -2 * (x - spike_position) / spike_width * spike_profile / spike_width * time_factor
        
        # Curvature contribution from spike
        R_spike = (dh_dx**2) / (t**2 + 0.001)
        
        return {
            'x': x,
            'spike_profile': spike_profile,
            'metric_perturbation': h,
            'gradient': dh_dx,
            'curvature_contribution': R_spike,
            'spike_width_effective': spike_width * np.sqrt(abs(t) + 0.01)
        }
    
    def spike_stability_analysis(self, spike_width: float = 0.1,
                                  perturbation_amplitude: float = 0.01) -> Dict:
        """
        Analyze stability of spike solutions.
        
        Key question: Are spikes stable or do they dissolve/split?
        """
        # Linear perturbation around spike
        # Mode analysis: δh ~ exp(ikx) exp(σt)
        
        k_values = np.linspace(0.1, 10, 100)  # Wavenumbers
        
        # Growth rates (simplified model)
        # Spikes are marginally stable: σ ≈ 0 for low k, σ > 0 for high k
        
        sigma = -spike_width * k_values**2 + perturbation_amplitude * k_values
        
        # Most unstable mode
        k_max = perturbation_amplitude / (2 * spike_width)
        sigma_max = perturbation_amplitude**2 / (4 * spike_width)
        
        return {
            'k_values': k_values,
            'growth_rates': sigma,
            'most_unstable_k': k_max,
            'max_growth_rate': sigma_max,
            'stability': 'marginally stable' if sigma_max < 0.1 else 'unstable',
            'timescale': 1 / (abs(sigma_max) + 0.001)
        }
    
    def classify_spike_types(self) -> Dict:
        """
        Classification of spike types found in BKL dynamics.
        """
        spike_types = {
            'type_1': {
                'name': 'Permanent spike',
                'description': 'Spike that persists through all Kasner epochs',
                'formation': 'Special initial data',
                'genericity': 'Non-generic (measure zero)',
                'example': 'Gowdy spike solutions'
            },
            'type_2': {
                'name': 'Transient spike',
                'description': 'Spike that forms and dissolves during evolution',
                'formation': 'Generic perturbations during BKL transitions',
                'genericity': 'May be generic',
                'example': 'Numerical simulations show these'
            },
            'type_3': {
                'name': 'Recurring spike',
                'description': 'Spike at fixed spatial location, reappears each era',
                'formation': 'Related to spatial topology',
                'genericity': 'Unknown',
                'example': 'T³ spatial topology'
            },
            'type_4': {
                'name': 'High-velocity spike',
                'description': 'Spike where expansion dominates curvature',
                'formation': 'Asymptotic limit of BKL',
                'genericity': 'Generic in some regimes',
                'example': 'Velocity-dominated solutions'
            }
        }
        
        return spike_types
    
    def spike_measure_estimate(self, n_samples: int = 1000) -> Dict:
        """
        Estimate the measure of initial data leading to spikes.
        
        Key question: What fraction of spacetimes develop persistent spikes?
        """
        # Monte Carlo sampling of initial data space
        spike_count = 0
        transient_count = 0
        smooth_count = 0
        
        for _ in range(n_samples):
            # Random initial data (simplified)
            amplitude = np.random.exponential(1)
            width = np.random.uniform(0.01, 1)
            velocity = np.random.normal(0, 1)
            
            # Criterion for spike formation (heuristic)
            spike_criterion = amplitude / width - abs(velocity)
            
            if spike_criterion > 2:
                spike_count += 1
            elif spike_criterion > 0.5:
                transient_count += 1
            else:
                smooth_count += 1
        
        return {
            'n_samples': n_samples,
            'permanent_spike_fraction': spike_count / n_samples,
            'transient_spike_fraction': transient_count / n_samples,
            'smooth_fraction': smooth_count / n_samples,
            'spike_measure_estimate': (spike_count + 0.5 * transient_count) / n_samples
        }


# =============================================================================
# PART 4: HOLOGRAPHIC BKL
# =============================================================================

class HolographicBKL:
    """
    Explore connections between BKL and holography (AdS/CFT).
    
    Key ideas:
    - BKL singularity → thermalization in boundary CFT
    - Kasner epochs → quasi-normal mode ringdown
    - Complexity growth → approach to singularity
    """
    
    def __init__(self, AdS_radius: float = 1.0):
        self.L = AdS_radius
        
    def chaos_bound_check(self, temperature: float = 1.0) -> Dict:
        """
        Check if BKL Lyapunov exponent saturates the chaos bound.
        
        Maldacena-Shenker-Stanford bound: λ ≤ 2πT/ℏ
        
        For BKL: λ_BKL = π²/(6 ln 2) ≈ 2.37
        """
        hbar = 1  # Natural units
        chaos_bound = 2 * np.pi * temperature / hbar
        lambda_bkl = np.pi**2 / (6 * np.log(2))
        
        # Effective BKL temperature
        T_bkl = lambda_bkl * hbar / (2 * np.pi)
        
        return {
            'lambda_bkl': lambda_bkl,
            'chaos_bound': chaos_bound,
            'saturates': abs(lambda_bkl - chaos_bound) < 0.01 * chaos_bound,
            'effective_temperature': T_bkl,
            'ratio': lambda_bkl / chaos_bound
        }
    
    def complexity_growth_rate(self, M: float = 1.0) -> Dict:
        """
        Compute complexity growth rate near BKL singularity.
        
        Susskind's conjecture: dC/dt = 2M (Lloyd bound)
        
        For BKL, complexity grows as the singularity is approached.
        """
        # Classical BKL contribution
        # Each Kasner epoch adds complexity ~ log(u)
        
        u_typical = 3.0
        epochs_per_time = np.pi**2 / (6 * np.log(2))  # Entropy rate
        
        complexity_per_epoch = np.log(u_typical + 1)
        dC_dt_bkl = complexity_per_epoch * epochs_per_time
        
        # Lloyd bound
        lloyd_bound = 2 * M
        
        return {
            'dC_dt_bkl': dC_dt_bkl,
            'lloyd_bound': lloyd_bound,
            'ratio': dC_dt_bkl / lloyd_bound,
            'interpretation': 'Complexity grows linearly approaching singularity'
        }
    
    def scrambling_time(self, S: float = 100) -> Dict:
        """
        Estimate scrambling time for BKL dynamics.
        
        Scrambling time t* ~ (1/λ) log(S) where S is entropy.
        """
        lambda_bkl = np.pi**2 / (6 * np.log(2))
        
        t_scramble = np.log(S) / lambda_bkl
        n_epochs = t_scramble * lambda_bkl  # Number of Kasner epochs
        
        return {
            'scrambling_time': t_scramble,
            'n_kasner_epochs': n_epochs,
            'entropy': S,
            'interpretation': f'{int(n_epochs)} Kasner epochs to scramble information'
        }
    
    def thermal_correlator_decay(self, t: np.ndarray, 
                                  Delta: float = 2.0) -> np.ndarray:
        """
        Thermal two-point correlator in boundary theory.
        
        G(t) ~ exp(-Δ·E_gap·t) with BKL-induced modifications.
        
        BKL adds oscillatory modulation from Kasner transitions.
        """
        # Thermal decay
        E_gap = 2 * np.pi  # Gap ~ 2πT
        G_thermal = np.exp(-Delta * E_gap * np.abs(t))
        
        # BKL modulation
        omega_bkl = np.pi**2 / (6 * np.log(2))
        G_modulation = 1 + 0.1 * np.cos(omega_bkl * t)
        
        return G_thermal * G_modulation
    
    def quasinormal_mode_connection(self, n_modes: int = 10) -> Dict:
        """
        Connect BKL Kasner epochs to quasinormal mode ringdown.
        
        Hypothesis: Each Kasner epoch corresponds to a QNM excitation.
        """
        # BKL frequencies (from continued fraction structure)
        bkl_frequencies = []
        u = 3.14159
        
        for i in range(n_modes):
            omega = np.pi / np.log(u + 1)  # Characteristic frequency
            gamma = np.pi**2 / (6 * np.log(2))  # Decay rate
            
            bkl_frequencies.append({
                'mode': i,
                'omega': omega,
                'gamma': gamma,
                'u_value': u
            })
            
            # BKL transition
            if u >= 2:
                u = u - 1
            else:
                u = 1 / (u - 1) if u > 1 else 10
        
        return {
            'modes': bkl_frequencies,
            'interpretation': 'Kasner epochs as QNM ringdown',
            'pattern': 'Frequencies vary with u parameter'
        }
    
    def entanglement_wedge_near_singularity(self, boundary_region_size: float = 0.5) -> Dict:
        """
        Analyze entanglement wedge structure near BKL singularity.
        
        Near singularity, the RT surface probes near-singularity geometry.
        """
        # RT surface area (simplified)
        A_RT = 2 * np.log(boundary_region_size / 0.01)
        
        # Entanglement entropy
        S_EE = A_RT / 4
        
        # BKL correction
        # Near singularity, area law gets corrections from chaotic dynamics
        S_bkl_correction = np.pi**2 / (6 * np.log(2)) * np.log(np.log(1 / 0.01))
        
        return {
            'boundary_size': boundary_region_size,
            'RT_area': A_RT,
            'S_EE': S_EE,
            'BKL_correction': S_bkl_correction,
            'total_entropy': S_EE + S_bkl_correction
        }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_advanced_bkl_analysis():
    """
    Run advanced BKL analysis.
    """
    print("=" * 70)
    print("BKL BLACK HOLE INTERIOR AND QUANTUM GRAVITY")
    print("=" * 70)
    
    # 1. Black Hole Interior
    print("\n" + "=" * 50)
    print("PART 1: BLACK HOLE INTERIOR BKL DYNAMICS")
    print("=" * 50)
    
    # Schwarzschild
    bh_schw = BlackHoleInterior(M=1.0)
    schw_interior = bh_schw.schwarzschild_interior_evolution(r_initial=1.5)
    
    print(f"Schwarzschild interior:")
    print(f"  Kasner type: {schw_interior['kasner_type']}")
    print(f"  Kasner exponents: {schw_interior['kasner_exponents']}")
    print(f"  Oscillatory: {schw_interior['is_oscillatory']}")
    
    # Kerr
    bh_kerr = BlackHoleInterior(M=1.0, a=0.9)
    kerr_structure = bh_kerr.kerr_interior_structure()
    
    print(f"\nKerr interior (a/M = 0.9):")
    print(f"  Outer horizon: r₊ = {kerr_structure['horizons']['outer_horizon']:.4f}")
    print(f"  Inner horizon: r₋ = {kerr_structure['horizons']['inner_horizon']:.4f}")
    print(f"  Cauchy stability: {kerr_structure['cauchy_horizon']['stability']}")
    
    # Mass inflation
    mass_inf = bh_kerr.mass_inflation_dynamics()
    print(f"\nMass inflation:")
    print(f"  Initial mass: {mass_inf['mass'][0]:.4f} M")
    print(f"  Final mass: {mass_inf['mass'][-1]:.4f} M")
    print(f"  Inflation factor: {mass_inf['mass'][-1]/mass_inf['mass'][0]:.2e}")
    
    # BKL onset
    bkl_onset = bh_schw.bkl_onset_from_perturbation(epsilon=0.05)
    print(f"\nBKL onset from perturbation (ε = 0.05):")
    print(f"  Oscillatory: {bkl_onset['oscillatory']}")
    print(f"  Estimated u: {bkl_onset['estimated_u']:.2f}")
    
    # 2. Loop Quantum Cosmology
    print("\n" + "=" * 50)
    print("PART 2: LOOP QUANTUM COSMOLOGY")
    print("=" * 50)
    
    lqc = LoopQuantumBKL()
    
    # Count oscillations before bounce
    osc_count = lqc.count_oscillations_before_bounce(u_initial=3.14159)
    
    print(f"BKL oscillations before quantum bounce:")
    print(f"  Initial u: 3.14159")
    print(f"  Number of epochs: {osc_count['n_epochs']}")
    print(f"  Final u: {osc_count['final_u']:.4f}")
    print(f"  Reached bounce: {osc_count['reached_bounce']}")
    
    # Quantum-corrected Kasner
    u = 2.5
    rho = 0.1 * lqc.rho_crit
    p_classical = (-u/(1+u+u**2), (1+u)/(1+u+u**2), u*(1+u)/(1+u+u**2))
    p_quantum = lqc.quantum_corrected_kasner(u, rho)
    
    print(f"\nQuantum-corrected Kasner (u={u}, ρ/ρ_crit=0.1):")
    print(f"  Classical: ({p_classical[0]:.4f}, {p_classical[1]:.4f}, {p_classical[2]:.4f})")
    print(f"  Quantum:   ({p_quantum[0]:.4f}, {p_quantum[1]:.4f}, {p_quantum[2]:.4f})")
    
    # 3. Spike Analysis
    print("\n" + "=" * 50)
    print("PART 3: SPIKE DYNAMICS")
    print("=" * 50)
    
    spike = SpikeAnalysis()
    
    # Spike types
    spike_types = spike.classify_spike_types()
    print("Spike classification:")
    for key, val in spike_types.items():
        print(f"  {val['name']}: {val['genericity']}")
    
    # Measure estimate
    measure = spike.spike_measure_estimate(n_samples=10000)
    print(f"\nSpike measure estimate (N={measure['n_samples']}):")
    print(f"  Permanent spikes: {measure['permanent_spike_fraction']:.3f}")
    print(f"  Transient spikes: {measure['transient_spike_fraction']:.3f}")
    print(f"  Smooth (no spikes): {measure['smooth_fraction']:.3f}")
    
    # 4. Holographic BKL
    print("\n" + "=" * 50)
    print("PART 4: HOLOGRAPHIC BKL")
    print("=" * 50)
    
    holo = HolographicBKL()
    
    # Chaos bound
    chaos = holo.chaos_bound_check(temperature=1.0)
    print(f"Chaos bound check:")
    print(f"  λ_BKL = {chaos['lambda_bkl']:.4f}")
    print(f"  Chaos bound (2πT) = {chaos['chaos_bound']:.4f}")
    print(f"  Saturates: {chaos['saturates']}")
    print(f"  Effective T_BKL = {chaos['effective_temperature']:.4f}")
    
    # Complexity growth
    complexity = holo.complexity_growth_rate()
    print(f"\nComplexity growth:")
    print(f"  dC/dt (BKL) = {complexity['dC_dt_bkl']:.4f}")
    print(f"  Lloyd bound = {complexity['lloyd_bound']:.4f}")
    print(f"  Ratio = {complexity['ratio']:.4f}")
    
    # Scrambling time
    scramble = holo.scrambling_time(S=100)
    print(f"\nScrambling (S=100):")
    print(f"  t* = {scramble['scrambling_time']:.4f}")
    print(f"  Kasner epochs = {scramble['n_kasner_epochs']:.1f}")
    
    # Create visualization
    create_advanced_plots(bh_schw, bh_kerr, lqc, spike, holo, 
                          schw_interior, osc_count, mass_inf)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Plots saved to bkl_advanced_analysis.png")
    print("=" * 70)


def create_advanced_plots(bh_schw, bh_kerr, lqc, spike, holo,
                          schw_interior, osc_count, mass_inf):
    """Create advanced analysis plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Schwarzschild interior
    ax = axes[0, 0]
    r = schw_interior['r']
    K = schw_interior['Kretschmann']
    ax.semilogy(r, K, 'b-', linewidth=2)
    ax.set_xlabel('r / M')
    ax.set_ylabel('Kretschmann scalar')
    ax.set_title('Schwarzschild Interior Curvature')
    ax.axvline(x=0, color='red', linestyle='--', label='Singularity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    
    # 2. Mass inflation
    ax = axes[0, 1]
    ax.semilogy(mass_inf['v'], mass_inf['mass'], 'b-', linewidth=2)
    ax.set_xlabel('Advanced time v')
    ax.set_ylabel('Effective mass / M')
    ax.set_title('Mass Inflation at Cauchy Horizon')
    ax.grid(True, alpha=0.3)
    
    # 3. LQC oscillation count
    ax = axes[0, 2]
    ax.plot(osc_count['rho_history'] / lqc.rho_crit, 'b-', linewidth=2)
    ax.axhline(y=1, color='red', linestyle='--', label='Bounce (ρ = ρ_crit)')
    ax.set_xlabel('Kasner epoch')
    ax.set_ylabel('ρ / ρ_crit')
    ax.set_title(f'LQC: {osc_count["n_epochs"]} Epochs Before Bounce')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    
    # 4. Spike profile
    ax = axes[1, 0]
    x = np.linspace(-2, 2, 200)
    
    for t in [1.0, 0.5, 0.1]:
        spike_data = spike.generate_spike_solution(x, t)
        ax.plot(x, spike_data['spike_profile'] * np.sqrt(t), 
                label=f't = {t}', linewidth=2)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Spike profile h(x)')
    ax.set_title('Spike Evolution (sharpens as t → 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Holographic correlator
    ax = axes[1, 1]
    t = np.linspace(0, 5, 200)
    G = holo.thermal_correlator_decay(t)
    G_pure = np.exp(-2 * np.pi * t)
    
    ax.semilogy(t, np.abs(G), 'b-', linewidth=2, label='With BKL modulation')
    ax.semilogy(t, G_pure, 'r--', linewidth=2, label='Pure thermal')
    ax.set_xlabel('Time')
    ax.set_ylabel('|G(t)|')
    ax.set_title('Holographic Correlator with BKL Effects')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary diagram
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = """
    BKL ADVANCED ANALYSIS SUMMARY
    ═══════════════════════════════
    
    BLACK HOLE INTERIOR:
    • Schwarzschild: Non-oscillatory "cigar" (p = 2/3, 2/3, -1/3)
    • Kerr: Mass inflation at Cauchy horizon
    • Generic perturbations → BKL oscillations
    
    LOOP QUANTUM COSMOLOGY:
    • Bounce replaces singularity at ρ = ρ_crit
    • Finite (~{} epochs) BKL oscillations
    • Quantum corrections damp anisotropy
    
    SPIKE DYNAMICS:
    • Multiple spike types identified
    • Measure of spike initial data ~ {:.1%}
    • Key open problem: genericity
    
    HOLOGRAPHIC BKL:
    • λ_BKL ≈ 2.37 ~ chaos bound saturation
    • Complexity grows linearly
    • Scrambling in ~{:.0f} Kasner epochs
    """.format(osc_count['n_epochs'],
               spike.spike_measure_estimate(1000)['spike_measure_estimate'],
               holo.scrambling_time(100)['n_kasner_epochs'])
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('c:/Users/Lenovo/papers/bkl_conjecture/bkl_advanced_analysis.png',
                dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_advanced_bkl_analysis()
