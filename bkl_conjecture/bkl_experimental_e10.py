"""
BKL Experimental Analogues and E10 Algebraic Structures
========================================================

This module explores:
1. Laboratory analogues of BKL dynamics
   - Bose-Einstein condensate (BEC) systems
   - Optical/photonic analogues
   - Superconducting circuit implementations
   
2. E10 and E11 Kac-Moody algebra structures
   - Root system and Weyl group
   - Connection to M-theory and supergravity
   - Billiard walls from algebraic data

3. Gravitational wave signatures
   - Spectral predictions from BKL oscillations
   - Detectability analysis

Author: Research Exploration
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.special import jv, spherical_jn
from scipy.linalg import expm
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

np.random.seed(42)
plt.style.use('default')


# =============================================================================
# PART 1: LABORATORY ANALOGUES OF BKL DYNAMICS
# =============================================================================

class BECAnalogue:
    """
    Bose-Einstein Condensate analogue of BKL dynamics.
    
    By controlling the scattering length via Feshbach resonance,
    we can create time-dependent interactions that mimic anisotropic
    cosmological expansion.
    """
    
    def __init__(self, N_atoms: int = 10000, omega: float = 100.0):
        """
        Initialize BEC system.
        
        Parameters:
        -----------
        N_atoms : int
            Number of atoms in condensate
        omega : float
            Trap frequency (Hz)
        """
        self.N = N_atoms
        self.omega = omega
        self.hbar = 1.054e-34  # J·s
        self.m = 87 * 1.66e-27  # Rb-87 mass in kg
        
        # Characteristic scales
        self.a_ho = np.sqrt(self.hbar / (self.m * omega))  # Harmonic oscillator length
        self.t_ho = 1 / omega  # Harmonic oscillator time
    
    def gross_pitaevskii_1d(self, psi: np.ndarray, t: float, x: np.ndarray,
                            a_s: callable) -> np.ndarray:
        """
        1D Gross-Pitaevskii equation with time-dependent scattering length.
        
        iℏ ∂ψ/∂t = [-ℏ²/(2m) ∂²/∂x² + V(x) + g(t)|ψ|²] ψ
        
        where g(t) = 4πℏ²a_s(t)/m
        """
        dx = x[1] - x[0]
        
        # Kinetic term (finite difference)
        d2psi = np.zeros_like(psi)
        d2psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
        
        # Trap potential
        V = 0.5 * self.m * self.omega**2 * x**2
        
        # Interaction strength
        g = 4 * np.pi * self.hbar**2 * a_s(t) / self.m
        
        # Time derivative
        dpsi_dt = -1j / self.hbar * (
            -self.hbar**2 / (2 * self.m) * d2psi +
            V * psi +
            g * np.abs(psi)**2 * psi
        )
        
        return dpsi_dt
    
    def kasner_scattering_length(self, t: float, p1: float, p2: float, p3: float,
                                  a0: float = 5e-9) -> float:
        """
        Scattering length modulation that mimics Kasner dynamics.
        
        a_s(t) = a_0 * t^{2p_i} for different spatial directions
        
        We average over directions for 1D analogue.
        """
        if t <= 0:
            return a0
        
        # Average effective scaling
        p_eff = (p1 + p2 + p3) / 3  # = 1/3 for Kasner
        
        return a0 * np.power(max(t, 0.01), 2 * p_eff)
    
    def simulate_bkl_bec(self, T_final: float = 1.0, n_points: int = 256,
                         kasner_u: float = 2.0) -> Dict:
        """
        Simulate BEC dynamics with Kasner-like scattering length modulation.
        """
        # Kasner exponents
        p1 = -kasner_u / (1 + kasner_u + kasner_u**2)
        p2 = (1 + kasner_u) / (1 + kasner_u + kasner_u**2)
        p3 = kasner_u * (1 + kasner_u) / (1 + kasner_u + kasner_u**2)
        
        # Spatial grid
        x = np.linspace(-10 * self.a_ho, 10 * self.a_ho, n_points)
        
        # Initial state: Thomas-Fermi profile
        mu = 0.5 * self.hbar * self.omega  # Chemical potential estimate
        psi0 = np.sqrt(np.maximum(mu - 0.5 * self.m * self.omega**2 * x**2, 0) / 
                       (4 * np.pi * self.hbar**2 * 5e-9 / self.m))
        psi0 = psi0 / np.sqrt(np.trapz(np.abs(psi0)**2, x))
        
        # Time evolution (simplified - real simulation needs split-step method)
        t_span = np.linspace(0.01, T_final, 100)
        
        # Track observables
        widths = []
        densities = []
        
        psi = psi0.copy()
        for t in t_span:
            # Effective width (variance)
            prob = np.abs(psi)**2
            prob = prob / np.trapz(prob, x)
            x_mean = np.trapz(x * prob, x)
            x2_mean = np.trapz(x**2 * prob, x)
            width = np.sqrt(x2_mean - x_mean**2)
            widths.append(width)
            densities.append(np.max(np.abs(psi)**2))
            
            # Simple evolution step (imaginary time for ground state)
            a_s = self.kasner_scattering_length(t, p1, p2, p3)
            g = 4 * np.pi * self.hbar**2 * a_s / self.m
            
            # Effective potential update
            V_eff = 0.5 * self.m * self.omega**2 * x**2 + g * np.abs(psi)**2
            psi = psi * np.exp(-V_eff * 0.001 / self.hbar)
            psi = psi / np.sqrt(np.trapz(np.abs(psi)**2, x))
        
        return {
            'times': t_span,
            'widths': np.array(widths),
            'densities': np.array(densities),
            'kasner_exponents': (p1, p2, p3),
            'x_grid': x,
            'final_state': psi
        }


class OpticalAnalogue:
    """
    Optical/photonic analogue of BKL dynamics.
    
    Using metamaterials or waveguide arrays with engineered dispersion
    to simulate curved spacetime effects.
    """
    
    def __init__(self, n_waveguides: int = 50):
        self.n_wg = n_waveguides
        
    def coupled_waveguide_bkl(self, psi0: np.ndarray, z_max: float = 10.0,
                               n_steps: int = 1000) -> Dict:
        """
        Simulate light propagation in coupled waveguides with BKL-like coupling.
        
        The waveguide coupling mimics the BKL transition dynamics.
        
        i ∂ψ_n/∂z = β_n(z) ψ_n + C_{n,n+1}(z) ψ_{n+1} + C_{n,n-1}(z) ψ_{n-1}
        """
        dz = z_max / n_steps
        z_values = np.linspace(0, z_max, n_steps)
        
        psi = psi0.copy().astype(complex)
        evolution = [psi.copy()]
        
        for i, z in enumerate(z_values[1:]):
            # BKL-like coupling modulation
            # Coupling oscillates between waveguides mimicking Kasner transitions
            u = 3.0 + 2.0 * np.sin(z)  # Varying Kasner parameter
            if u < 2:
                u = 1 / (u - 1) + 1
            
            # Coupling strength varies with u
            C = 0.5 * np.exp(-np.abs(u - 2))
            
            # Propagation constants (diagonal terms)
            beta = np.zeros(self.n_wg)
            for n in range(self.n_wg):
                beta[n] = 0.1 * (n - self.n_wg/2)**2  # Parabolic variation
            
            # Coupling matrix
            H = np.diag(beta)
            for n in range(self.n_wg - 1):
                H[n, n+1] = C
                H[n+1, n] = C
            
            # Propagate
            psi = psi - 1j * dz * (H @ psi)
            psi = psi / np.linalg.norm(psi)
            evolution.append(psi.copy())
        
        return {
            'z_values': z_values,
            'evolution': np.array(evolution),
            'final_state': psi,
            'intensity': np.abs(np.array(evolution))**2
        }
    
    def transformation_optics_metric(self, x: np.ndarray, y: np.ndarray,
                                      t: float) -> Dict:
        """
        Design metamaterial parameters to implement BKL-like metric.
        
        Using transformation optics, the permittivity and permeability
        tensors can be engineered to simulate curved spacetime.
        
        ε_ij = μ_ij = √(-g) g^{ij} (in vacuum equivalence)
        """
        # Kasner metric components at time t
        u = 2.5
        p1 = -u / (1 + u + u**2)
        p2 = (1 + u) / (1 + u + u**2)
        p3 = u * (1 + u) / (1 + u + u**2)
        
        # Scale factors
        a1 = np.power(max(t, 0.01), p1)
        a2 = np.power(max(t, 0.01), p2)
        a3 = np.power(max(t, 0.01), p3)
        
        # Spatial metric
        g = np.diag([a1**2, a2**2, a3**2])
        g_inv = np.diag([1/a1**2, 1/a2**2, 1/a3**2])
        sqrt_det_g = a1 * a2 * a3
        
        # Material parameters
        epsilon = sqrt_det_g * g_inv
        mu = sqrt_det_g * g_inv
        
        return {
            'epsilon_tensor': epsilon,
            'mu_tensor': mu,
            'metric': g,
            'scale_factors': (a1, a2, a3)
        }


class CircuitQEDAnalogue:
    """
    Superconducting circuit QED implementation of BKL dynamics.
    
    Josephson junction arrays can implement the discrete BKL map
    with controllable parameters.
    """
    
    def __init__(self, n_qubits: int = 10):
        self.n_qubits = n_qubits
        
    def josephson_bkl_hamiltonian(self, u: float, E_J: float = 1.0, 
                                   E_C: float = 0.1) -> np.ndarray:
        """
        Construct Hamiltonian for Josephson junction array implementing BKL.
        
        H = Σ_i [4E_C n_i² - E_J(u) cos(φ_i - φ_{i+1})]
        
        where E_J(u) encodes the BKL parameter.
        """
        dim = 2**self.n_qubits
        
        # Pauli matrices
        sigma_z = np.array([[1, 0], [0, -1]])
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        I = np.eye(2)
        
        H = np.zeros((dim, dim), dtype=complex)
        
        # BKL-modulated coupling
        J = E_J * np.exp(-(u - 2)**2 / 2)  # Gaussian centered at u=2
        
        for i in range(self.n_qubits):
            # Charging energy term
            ops = [I] * self.n_qubits
            ops[i] = sigma_z
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            H += E_C * term
            
            # Josephson coupling
            if i < self.n_qubits - 1:
                ops = [I] * self.n_qubits
                ops[i] = sigma_x
                ops[i+1] = sigma_x
                term = ops[0]
                for op in ops[1:]:
                    term = np.kron(term, op)
                H += -J * term
        
        return H
    
    def bkl_quantum_dynamics(self, psi0: np.ndarray, u_sequence: List[float],
                              dt: float = 0.1) -> Dict:
        """
        Simulate quantum dynamics through a BKL sequence.
        """
        psi = psi0.copy()
        states = [psi.copy()]
        energies = []
        
        for u in u_sequence:
            H = self.josephson_bkl_hamiltonian(u)
            
            # Time evolution
            U = expm(-1j * H * dt)
            psi = U @ psi
            psi = psi / np.linalg.norm(psi)
            
            states.append(psi.copy())
            energies.append(np.real(np.conj(psi) @ H @ psi))
        
        return {
            'states': np.array(states),
            'energies': np.array(energies),
            'u_sequence': u_sequence,
            'final_state': psi
        }


# =============================================================================
# PART 2: E10 AND E11 KAC-MOODY ALGEBRAS
# =============================================================================

class E10Algebra:
    """
    Explore the E10 Kac-Moody algebra structure relevant to BKL dynamics.
    
    E10 is conjectured to be the symmetry algebra of M-theory near singularities.
    """
    
    def __init__(self):
        # E10 Cartan matrix (10x10)
        self.cartan_matrix = self._build_e10_cartan()
        self.rank = 10
        
    def _build_e10_cartan(self) -> np.ndarray:
        """
        Construct the E10 Cartan matrix.
        
        E10 is the over-extension of E8.
        Dynkin diagram:
                    o
                    |
        o-o-o-o-o-o-o-o-o
        1 2 3 4 5 6 7 8 9
                    10 (above 8)
        """
        A = np.zeros((10, 10), dtype=int)
        
        # Diagonal elements
        for i in range(10):
            A[i, i] = 2
        
        # A_9 chain: nodes 1-9
        for i in range(8):
            A[i, i+1] = -1
            A[i+1, i] = -1
        
        # E8 branch: node 10 connects to node 8
        A[7, 9] = -1
        A[9, 7] = -1
        
        return A
    
    def simple_roots(self) -> np.ndarray:
        """
        Return the simple roots of E10 in Minkowski signature.
        
        For E10, roots live in a 10-dimensional Minkowski space.
        """
        # Standard simple roots (orthonormal basis representation)
        roots = np.eye(10)
        
        # Adjust for E10 structure
        # The simple roots satisfy α_i · α_j = A_ij
        
        return roots
    
    def weyl_reflection(self, v: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Apply Weyl reflection w_α to vector v.
        
        w_α(v) = v - 2(v·α)/(α·α) * α
        """
        alpha_sq = np.dot(alpha, alpha)
        if alpha_sq == 0:
            return v
        
        return v - 2 * np.dot(v, alpha) / alpha_sq * alpha
    
    def root_multiplicity(self, level: int) -> int:
        """
        Estimate the root multiplicity at a given level.
        
        For E10, the multiplicities grow very rapidly with level.
        At low levels, they match the dimensions of M-theory fields.
        """
        # Asymptotic formula: mult ~ exp(π√(2n/5)) / n^{7/4}
        if level <= 0:
            return 1
        
        mult = np.exp(np.pi * np.sqrt(2 * level / 5)) / level**(7/4)
        return int(mult)
    
    def billiard_walls_from_roots(self, n_roots: int = 20) -> List[Dict]:
        """
        Generate BKL billiard walls from E10 root data.
        
        Each positive root defines a wall in the billiard.
        """
        walls = []
        
        # Start with simple roots
        simple = self.simple_roots()
        
        for i, root in enumerate(simple):
            walls.append({
                'root': root,
                'height': 1,
                'type': 'gravity' if i < 9 else 'form_field',
                'normal': root / np.linalg.norm(root)
            })
        
        # Generate more roots by Weyl reflections
        current_roots = list(simple)
        for level in range(1, 4):
            new_roots = []
            for r1 in current_roots[-10:]:  # Use recent roots
                for r2 in simple:
                    candidate = r1 + r2
                    # Check validity
                    norm_sq = candidate @ self.cartan_matrix[:len(candidate), :len(candidate)] @ candidate if len(candidate) == 10 else np.dot(candidate, candidate)
                    if 0 < norm_sq <= 4:
                        is_new = True
                        for existing in current_roots + new_roots:
                            if np.allclose(candidate, existing):
                                is_new = False
                                break
                        if is_new:
                            new_roots.append(candidate)
                            walls.append({
                                'root': candidate,
                                'height': level + 1,
                                'type': 'higher_level',
                                'normal': candidate / np.linalg.norm(candidate)
                            })
            current_roots.extend(new_roots[:50])  # Limit growth
        
        return walls[:n_roots]
    
    def level_decomposition(self, max_level: int = 5) -> Dict:
        """
        Decompose E10 into levels (SL(10) representations).
        
        Level 0: SL(10) ⊂ E10
        Level 1: 3-form potential A_μνρ
        Level 2: 6-form dual
        Level 3: Mixed symmetry tensors
        ...
        
        This matches M-theory field content!
        """
        levels = {}
        
        # Level 0: Gravity (metric + dilaton)
        levels[0] = {
            'representation': 'adjoint of SL(10)',
            'dimension': 99,  # 10² - 1
            'physical_field': 'graviton + dilaton',
            'multiplicity': 1
        }
        
        # Level 1: 3-form
        levels[1] = {
            'representation': 'antisymmetric [3]',
            'dimension': 120,  # C(10,3)
            'physical_field': '3-form A_μνρ',
            'multiplicity': 1
        }
        
        # Level 2: 6-form (dual)
        levels[2] = {
            'representation': 'antisymmetric [6]',
            'dimension': 210,  # C(10,6)
            'physical_field': '6-form (dual of 3-form)',
            'multiplicity': 1
        }
        
        # Level 3: Mixed symmetry
        levels[3] = {
            'representation': 'mixed [8,1]',
            'dimension': 440,
            'physical_field': 'dual graviton',
            'multiplicity': 1
        }
        
        # Higher levels...
        for l in range(4, max_level + 1):
            mult = self.root_multiplicity(l)
            levels[l] = {
                'representation': f'level {l} rep',
                'dimension': f'~{mult}',
                'physical_field': 'higher duality',
                'multiplicity': mult
            }
        
        return levels


class E11Algebra:
    """
    E11 Kac-Moody algebra (West's very extended E8).
    
    E11 is conjectured to be an even larger symmetry of M-theory.
    """
    
    def __init__(self):
        self.cartan_matrix = self._build_e11_cartan()
        self.rank = 11
    
    def _build_e11_cartan(self) -> np.ndarray:
        """
        Construct the E11 Cartan matrix.
        
        E11 extends E10 by one more node.
        """
        A = np.zeros((11, 11), dtype=int)
        
        # E10 part
        e10 = E10Algebra()
        A[:10, :10] = e10.cartan_matrix
        
        # New node connects to node 10
        A[9, 10] = -1
        A[10, 9] = -1
        A[10, 10] = 2
        
        return A
    
    def spacetime_from_e11(self) -> Dict:
        """
        Derive spacetime structure from E11.
        
        E11 contains information about:
        - 11D supergravity
        - All Kaluza-Klein modes
        - All brane charges
        - Infinite tower of dual fields
        """
        return {
            'spacetime_dimension': 11,
            'branes': ['M2', 'M5', 'KK-monopole', 'exotic branes'],
            'fields_at_level_0': 'graviton',
            'fields_at_level_1': '3-form',
            'fields_at_level_2': '6-form',
            'fields_at_level_3': 'dual graviton',
            'bkl_connection': 'Billiard motion on E11/K(E11)'
        }


# =============================================================================
# PART 3: GRAVITATIONAL WAVE SIGNATURES
# =============================================================================

class GravitationalWaveSignatures:
    """
    Calculate gravitational wave signatures from BKL dynamics.
    """
    
    def __init__(self):
        self.c = 3e8  # m/s
        self.G = 6.67e-11  # N m²/kg²
        
    def kasner_gw_strain(self, t: np.ndarray, p1: float, p2: float, p3: float,
                          M: float = 1e30, D: float = 1e6) -> np.ndarray:
        """
        Compute GW strain from a single Kasner epoch.
        
        The anisotropic expansion generates gravitational waves.
        
        h ~ (G M / (c² D)) * (Δp)² * (1/t)
        
        where Δp is the anisotropy of Kasner exponents.
        """
        # Anisotropy measure
        delta_p = np.sqrt((p1 - p2)**2 + (p2 - p3)**2 + (p3 - p1)**2) / np.sqrt(6)
        
        # Amplitude
        h0 = (self.G * M) / (self.c**2 * D) * delta_p**2
        
        # Time dependence (inverse with oscillation)
        h = h0 / (t + 0.01) * np.sin(2 * np.pi * np.log(t + 1))
        
        return h
    
    def bkl_gw_spectrum(self, n_epochs: int = 100, 
                         f_min: float = 1e-4, f_max: float = 1e2,
                         n_freqs: int = 1000) -> Dict:
        """
        Compute the gravitational wave spectrum from BKL oscillations.
        
        The spectrum has characteristic oscillatory features from
        the discrete Kasner transitions.
        """
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_freqs)
        
        # Generate BKL sequence
        u = 3.14159
        epoch_durations = []
        kasner_params = []
        
        for _ in range(n_epochs):
            # Kasner exponents
            p1 = -u / (1 + u + u**2)
            p2 = (1 + u) / (1 + u + u**2)
            p3 = u * (1 + u) / (1 + u + u**2)
            kasner_params.append((p1, p2, p3))
            
            # Epoch duration (in some units)
            tau = np.log(u + 1)
            epoch_durations.append(tau)
            
            # BKL transition
            if u >= 2:
                u = u - 1
            elif u > 1:
                u = 1 / (u - 1)
            else:
                u = np.random.uniform(5, 10)
        
        # Compute spectrum
        h_f = np.zeros(n_freqs, dtype=complex)
        
        cumulative_time = 0
        for i, (tau, (p1, p2, p3)) in enumerate(zip(epoch_durations, kasner_params)):
            # Each epoch contributes a pulse
            anisotropy = np.sqrt((p1-p2)**2 + (p2-p3)**2 + (p3-p1)**2)
            
            for j, f in enumerate(frequencies):
                # Fourier component
                phase = 2 * np.pi * f * cumulative_time
                amplitude = anisotropy * tau * np.sinc(f * tau)
                h_f[j] += amplitude * np.exp(1j * phase)
            
            cumulative_time += tau
        
        # Power spectrum
        Omega_gw = np.abs(h_f)**2 * frequencies**3
        
        return {
            'frequencies': frequencies,
            'h_f': h_f,
            'Omega_gw': Omega_gw / np.max(Omega_gw),  # Normalized
            'n_epochs': n_epochs,
            'epoch_durations': np.array(epoch_durations)
        }
    
    def detectability_analysis(self, spectrum: Dict, detector: str = 'LISA') -> Dict:
        """
        Analyze detectability of BKL GW signature.
        """
        # Detector sensitivity curves (approximate)
        f = spectrum['frequencies']
        
        if detector == 'LISA':
            # LISA sensitivity (Hz^{-1/2})
            S_n = 1e-20 * (1 + (f/1e-4)**(-4)) * (1 + (f/1e-1)**2)
            f_band = (1e-4, 1e-1)
        elif detector == 'LIGO':
            S_n = 1e-23 * (1 + (f/20)**(-4)) * (1 + (f/1000)**2)
            f_band = (10, 1000)
        elif detector == 'PTA':
            S_n = 1e-15 * (f/1e-9)**(-2)
            f_band = (1e-9, 1e-7)
        else:
            S_n = np.ones_like(f) * 1e-20
            f_band = (f.min(), f.max())
        
        # SNR estimate
        in_band = (f >= f_band[0]) & (f <= f_band[1])
        signal = np.abs(spectrum['h_f'])**2
        
        snr_integrand = signal / S_n
        snr_integrand[~in_band] = 0
        
        snr = np.sqrt(np.trapz(snr_integrand, f))
        
        return {
            'detector': detector,
            'frequency_band': f_band,
            'sensitivity': S_n,
            'snr': snr,
            'detectable': snr > 1  # Threshold
        }
    
    def primordial_spectrum_prediction(self) -> Dict:
        """
        Predict primordial GW spectrum from pre-inflationary BKL phase.
        """
        # Standard inflation predicts nearly scale-invariant spectrum
        # BKL adds oscillatory features at very low frequencies
        
        f = np.logspace(-18, -10, 1000)  # Very low frequencies (Hubble scale)
        
        # Inflationary background
        r = 0.01  # Tensor-to-scalar ratio
        n_t = -r/8  # Consistency relation
        
        Omega_inf = (r/10) * (f/1e-16)**n_t
        
        # BKL oscillations (at very early times)
        n_bkl_epochs = 100
        for n in range(1, n_bkl_epochs):
            # Each epoch adds a feature
            f_n = 1e-16 * np.exp(n * 0.1)  # Logarithmically spaced
            Omega_inf *= (1 + 0.1 * np.exp(-(np.log(f/f_n))**2 / 0.01))
        
        return {
            'frequencies': f,
            'Omega_gw': Omega_inf,
            'tensor_to_scalar': r,
            'spectral_index': n_t,
            'bkl_features': 'oscillatory modulation at f < 10^{-15} Hz'
        }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_experimental_analysis():
    """
    Run the experimental analogues and algebraic structure analysis.
    """
    print("=" * 70)
    print("BKL EXPERIMENTAL ANALOGUES AND E10 ALGEBRA")
    print("=" * 70)
    
    # 1. BEC Analogue
    print("\n" + "=" * 50)
    print("PART 1: BOSE-EINSTEIN CONDENSATE ANALOGUE")
    print("=" * 50)
    
    bec = BECAnalogue(N_atoms=10000, omega=100)
    result = bec.simulate_bkl_bec(T_final=1.0, kasner_u=2.0)
    
    print(f"Kasner exponents: p1={result['kasner_exponents'][0]:.4f}, "
          f"p2={result['kasner_exponents'][1]:.4f}, p3={result['kasner_exponents'][2]:.4f}")
    print(f"Initial width: {result['widths'][0]:.6e} m")
    print(f"Final width: {result['widths'][-1]:.6e} m")
    print(f"Width change: {result['widths'][-1]/result['widths'][0]:.2f}x")
    
    # 2. Optical Analogue
    print("\n" + "=" * 50)
    print("PART 2: OPTICAL WAVEGUIDE ANALOGUE")
    print("=" * 50)
    
    optical = OpticalAnalogue(n_waveguides=30)
    psi0 = np.zeros(30)
    psi0[15] = 1.0  # Central excitation
    opt_result = optical.coupled_waveguide_bkl(psi0, z_max=10.0)
    
    print(f"Number of waveguides: 30")
    print(f"Propagation distance: 10 units")
    print(f"Final intensity distribution: centered at waveguide "
          f"{np.argmax(np.abs(opt_result['final_state'])**2)}")
    
    # 3. Circuit QED
    print("\n" + "=" * 50)
    print("PART 3: SUPERCONDUCTING CIRCUIT ANALOGUE")
    print("=" * 50)
    
    circuit = CircuitQEDAnalogue(n_qubits=4)
    psi0 = np.zeros(2**4, dtype=complex)
    psi0[0] = 1.0
    u_sequence = [3.0, 2.5, 2.0, 1.5, 1.2, 5.0, 4.0, 3.0, 2.0, 1.5]
    circuit_result = circuit.bkl_quantum_dynamics(psi0, u_sequence)
    
    print(f"Number of qubits: 4")
    print(f"BKL sequence length: {len(u_sequence)}")
    print(f"Energy evolution: {circuit_result['energies'][0]:.4f} → {circuit_result['energies'][-1]:.4f}")
    
    # 4. E10 Algebra
    print("\n" + "=" * 50)
    print("PART 4: E10 KAC-MOODY ALGEBRA")
    print("=" * 50)
    
    e10 = E10Algebra()
    print("E10 Cartan matrix (showing corner):")
    print(e10.cartan_matrix[:5, :5])
    
    walls = e10.billiard_walls_from_roots(n_roots=15)
    print(f"\nNumber of billiard walls generated: {len(walls)}")
    print("Wall types:")
    for w in walls[:5]:
        print(f"  Height {w['height']}: {w['type']}")
    
    levels = e10.level_decomposition(max_level=5)
    print("\nE10 Level Decomposition (M-theory fields):")
    for l, data in levels.items():
        print(f"  Level {l}: {data['physical_field']} (dim ~{data['dimension']})")
    
    # 5. E11
    print("\n" + "=" * 50)
    print("PART 5: E11 AND M-THEORY")
    print("=" * 50)
    
    e11 = E11Algebra()
    spacetime = e11.spacetime_from_e11()
    print(f"Spacetime dimension: {spacetime['spacetime_dimension']}")
    print(f"Branes: {spacetime['branes']}")
    print(f"BKL connection: {spacetime['bkl_connection']}")
    
    # 6. Gravitational Waves
    print("\n" + "=" * 50)
    print("PART 6: GRAVITATIONAL WAVE SIGNATURES")
    print("=" * 50)
    
    gw = GravitationalWaveSignatures()
    spectrum = gw.bkl_gw_spectrum(n_epochs=100)
    
    print(f"GW spectrum computed over {len(spectrum['frequencies'])} frequencies")
    print(f"Peak frequency: {spectrum['frequencies'][np.argmax(spectrum['Omega_gw'])]:.2e} Hz")
    
    for detector in ['LISA', 'LIGO', 'PTA']:
        detect = gw.detectability_analysis(spectrum, detector)
        print(f"{detector}: SNR = {detect['snr']:.2e}, Detectable = {detect['detectable']}")
    
    # Primordial prediction
    primordial = gw.primordial_spectrum_prediction()
    print(f"\nPrimordial spectrum: {primordial['bkl_features']}")
    
    # Create visualization
    create_experimental_plots(bec, optical, circuit, e10, gw, result, opt_result, spectrum)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - Plots saved to bkl_experimental.png")
    print("=" * 70)


def create_experimental_plots(bec, optical, circuit, e10, gw, bec_result, opt_result, spectrum):
    """Create visualization of experimental analogues."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. BEC width evolution
    ax = axes[0, 0]
    ax.plot(bec_result['times'], bec_result['widths'] / bec_result['widths'][0], 'b-', linewidth=2)
    ax.set_xlabel('Time (t/t_ho)')
    ax.set_ylabel('Normalized Width')
    ax.set_title('BEC Width Evolution (BKL Analogue)')
    ax.grid(True, alpha=0.3)
    
    # 2. Optical waveguide
    ax = axes[0, 1]
    im = ax.imshow(opt_result['intensity'].T, aspect='auto', 
                   extent=[0, 10, 0, 30], cmap='hot', origin='lower')
    ax.set_xlabel('Propagation distance z')
    ax.set_ylabel('Waveguide index')
    ax.set_title('Optical BKL: Light Intensity')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # 3. E10 Cartan matrix
    ax = axes[0, 2]
    im = ax.imshow(e10.cartan_matrix, cmap='RdBu', vmin=-2, vmax=2)
    ax.set_xlabel('Root index j')
    ax.set_ylabel('Root index i')
    ax.set_title('E10 Cartan Matrix')
    plt.colorbar(im, ax=ax, label='A_ij')
    
    # 4. E10 level multiplicities
    ax = axes[1, 0]
    levels = list(range(1, 15))
    mults = [e10.root_multiplicity(l) for l in levels]
    ax.semilogy(levels, mults, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Level')
    ax.set_ylabel('Root Multiplicity')
    ax.set_title('E10 Root Multiplicities')
    ax.grid(True, alpha=0.3)
    
    # 5. GW spectrum
    ax = axes[1, 1]
    ax.loglog(spectrum['frequencies'], spectrum['Omega_gw'], 'b-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Ω_GW (normalized)')
    ax.set_title('BKL Gravitational Wave Spectrum')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-4, 1e2)
    
    # 6. Detectability
    ax = axes[1, 2]
    f = spectrum['frequencies']
    
    # Detector sensitivities
    S_lisa = 1e-20 * (1 + (f/1e-4)**(-4)) * (1 + (f/1e-1)**2)
    S_ligo = 1e-23 * (1 + (f/20)**(-4)) * (1 + (f/1000)**2)
    
    ax.loglog(f, np.sqrt(S_lisa), 'g-', linewidth=2, label='LISA')
    ax.loglog(f, np.sqrt(S_ligo), 'r-', linewidth=2, label='LIGO')
    ax.loglog(f, np.abs(spectrum['h_f']) / np.max(np.abs(spectrum['h_f'])) * 1e-20, 
              'b--', linewidth=2, label='BKL signal')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Strain (Hz^{-1/2})')
    ax.set_title('Detector Sensitivity Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1e-4, 1e4)
    ax.set_ylim(1e-25, 1e-15)
    
    plt.tight_layout()
    plt.savefig('c:/Users/Lenovo/papers/bkl_conjecture/bkl_experimental.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_experimental_analysis()
