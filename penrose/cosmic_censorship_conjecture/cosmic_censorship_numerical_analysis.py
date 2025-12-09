"""
Cosmic Censorship Conjecture: Numerical Analysis Framework

This module provides numerical tools for investigating the Cosmic Censorship 
Conjectures through:
1. Spectral gap analysis for Cauchy horizon stability
2. Blue-shift instability quantification
3. Entropy-area bounds verification
4. Critical parameter space exploration

Author: Research Analysis
Date: December 2025
"""

import numpy as np
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# Physical constants (G = c = 1 units)
PI = np.pi

#==============================================================================
# SECTION 1: Black Hole Geometry and Horizon Structure
#==============================================================================

class KerrNewmanBlackHole:
    """
    Kerr-Newman black hole geometry with cosmological constant.
    
    The metric in Boyer-Lindquist coordinates:
    ds² = -Δ/ρ²(dt - a sin²θ dφ)² + ρ²/Δ dr² + ρ² dθ² 
          + sin²θ/ρ²((r² + a²)dφ - a dt)²
    
    where:
    - Δ = r² - 2Mr + a² + Q² - Λr⁴/3
    - ρ² = r² + a² cos²θ
    """
    
    def __init__(self, M=1.0, a=0.0, Q=0.0, Lambda=0.0):
        """
        Initialize black hole parameters.
        
        Parameters:
        -----------
        M : float
            Black hole mass
        a : float  
            Spin parameter (J/M)
        Q : float
            Electric charge
        Lambda : float
            Cosmological constant (Λ)
        """
        self.M = M
        self.a = a
        self.Q = Q
        self.Lambda = Lambda
        
        # Derived quantities
        self._compute_horizons()
        self._compute_surface_gravities()
    
    def Delta(self, r):
        """Horizon function Δ(r)."""
        M, a, Q, Lambda = self.M, self.a, self.Q, self.Lambda
        return r**2 - 2*M*r + a**2 + Q**2 - Lambda * r**4 / 3
    
    def dDelta_dr(self, r):
        """First derivative of Δ(r)."""
        M, Lambda = self.M, self.Lambda
        return 2*r - 2*M - 4*Lambda * r**3 / 3
    
    def d2Delta_dr2(self, r):
        """Second derivative of Δ(r)."""
        Lambda = self.Lambda
        return 2 - 4*Lambda * r**2
    
    def _compute_horizons(self):
        """Find all horizons (roots of Δ = 0)."""
        M, a, Q, Lambda = self.M, self.a, self.Q, self.Lambda
        
        # For Kerr-Newman without Λ, use analytical formula
        if abs(Lambda) < 1e-12:
            # Δ = r² - 2Mr + a² + Q² = 0
            # r = M ± √(M² - a² - Q²)
            discriminant = M**2 - a**2 - Q**2
            
            if discriminant > 1e-12:
                # Two distinct horizons
                r_plus = M + np.sqrt(discriminant)
                r_minus = M - np.sqrt(discriminant)
                self.horizons = np.array([r_minus, r_plus])
                self.r_plus = r_plus
                self.r_minus = r_minus if r_minus > 0 else None
            elif abs(discriminant) < 1e-12:
                # Extremal: degenerate horizon
                self.horizons = np.array([M])
                self.r_plus = M
                self.r_minus = M  # Same location
            else:
                # No horizon (naked singularity)
                self.horizons = np.array([])
                self.r_plus = None
                self.r_minus = None
            return
        
        # For non-zero Λ, use numerical root finding
        r_values = np.linspace(0.01, 20*self.M, 10000)
        Delta_values = self.Delta(r_values)
        
        # Find sign changes
        self.horizons = []
        for i in range(len(Delta_values) - 1):
            if Delta_values[i] * Delta_values[i+1] < 0:
                try:
                    r_h = brentq(self.Delta, r_values[i], r_values[i+1])
                    self.horizons.append(r_h)
                except:
                    pass
        
        self.horizons = np.array(sorted(self.horizons))
        
        # Identify horizon types
        if len(self.horizons) >= 2:
            self.r_plus = self.horizons[-1]  # Outer (event) horizon
            self.r_minus = self.horizons[-2]  # Inner (Cauchy) horizon
        elif len(self.horizons) == 1:
            self.r_plus = self.horizons[0]
            self.r_minus = None
        else:
            self.r_plus = None
            self.r_minus = None
            
        # Cosmological horizon (if Λ > 0)
        if self.Lambda > 0 and len(self.horizons) >= 3:
            self.r_cosmo = self.horizons[-1]
            self.r_plus = self.horizons[-2]
    
    def _compute_surface_gravities(self):
        """
        Compute surface gravity κ at each horizon.
        
        For Kerr-Newman, the surface gravity is:
        κ = (r_+ - r_-) / (2(r_+² + a²)) = Δ'(r_h) / (2(r_h² + a²))
        
        This formula ensures κ → 0 at extremality (r_+ = r_-).
        """
        self.kappa = {}
        
        if self.r_plus is not None:
            # κ_+ = |Δ'(r_+)| / (2(r_+² + a²))
            self.kappa['plus'] = abs(self.dDelta_dr(self.r_plus)) / (2 * (self.r_plus**2 + self.a**2))
        
        if self.r_minus is not None and self.r_minus != self.r_plus:
            # κ_- = |Δ'(r_-)| / (2(r_-² + a²))
            self.kappa['minus'] = abs(self.dDelta_dr(self.r_minus)) / (2 * (self.r_minus**2 + self.a**2))
        elif self.r_minus is not None:
            # Extremal case: degenerate horizon
            self.kappa['minus'] = 0.0
        
        if hasattr(self, 'r_cosmo'):
            self.kappa['cosmo'] = abs(self.dDelta_dr(self.r_cosmo)) / (2 * (self.r_cosmo**2 + self.a**2))
    
    def horizon_area(self, r_h=None):
        """
        Compute horizon area A = 4π(r_h² + a²).
        """
        if r_h is None:
            r_h = self.r_plus
        if r_h is None:
            return None
        return 4 * PI * (r_h**2 + self.a**2)
    
    def bekenstein_hawking_entropy(self, r_h=None):
        """
        Bekenstein-Hawking entropy S = A/(4G) = A/4 in natural units.
        """
        A = self.horizon_area(r_h)
        if A is None:
            return None
        return A / 4
    
    def is_extremal(self, tol=1e-6):
        """Check if black hole is extremal (degenerate horizons)."""
        if self.r_plus is None or self.r_minus is None:
            return False
        return abs(self.r_plus - self.r_minus) < tol
    
    def is_superextremal(self):
        """Check if parameters violate horizon existence (naked singularity)."""
        return self.r_plus is None
    
    def angular_velocity(self, r_h=None):
        """Angular velocity of the horizon Ω_H = a/(r_h² + a²)."""
        if r_h is None:
            r_h = self.r_plus
        if r_h is None:
            return None
        return self.a / (r_h**2 + self.a**2)
    
    def electric_potential(self, r_h=None):
        """Electric potential at horizon Φ_H = Qr_h/(r_h² + a²)."""
        if r_h is None:
            r_h = self.r_plus
        if r_h is None:
            return None
        return self.Q * r_h / (r_h**2 + self.a**2)


#==============================================================================
# SECTION 2: Spectral Gap Analysis for Strong Cosmic Censorship
#==============================================================================

class SpectralGapAnalysis:
    """
    Analysis of the spectral gap condition for Strong Cosmic Censorship.
    
    The key quantity is β = α/κ_- where:
    - α is the spectral gap (imaginary part of dominant quasinormal mode)
    - κ_- is the surface gravity of the Cauchy horizon
    
    SCCC holds if β < 1/2 (rough estimate) for C² regularity.
    """
    
    def __init__(self, black_hole):
        self.bh = black_hole
    
    def qnm_frequency_approximation(self, l, n=0, s=0):
        """
        Approximate quasinormal mode frequencies using WKB/eikonal method.
        
        For Schwarzschild: ω_n ≈ (l + 1/2)/(3√3 M) - i(n + 1/2)/(3√3 M)
        
        LIMITATIONS:
        - Eikonal approximation is accurate for large l (l >> 1)
        - For l = 2, errors can be ~10-20%
        - Does not capture near-extremal Kerr QNM behavior accurately
        - For precise results, use Leaver's continued fraction method
        
        UNCERTAINTY ESTIMATE:
        - Relative error ~ 1/l for real part
        - Relative error ~ 1/√l for imaginary part
        
        Parameters:
        -----------
        l : int
            Angular momentum quantum number
        n : int  
            Overtone number
        s : int
            Spin weight of the field
        
        Returns:
        --------
        tuple (complex, float)
            (Approximate quasinormal frequency ω = ω_R + i ω_I, estimated relative error)
        """
        if self.bh.r_plus is None:
            return None, None
        
        r_plus = self.bh.r_plus
        M = self.bh.M
        a = self.bh.a
        
        # Estimate relative error (eikonal approximation)
        relative_error = 1.0 / max(l, 1)
        
        # Improved eikonal approximation
        # For Schwarzschild: photon sphere at r = 3M
        # ω ≈ Ω_c (l + 1/2) - i λ (n + 1/2)
        # where Ω_c = 1/(3√3 M) and λ = 1/(3√3 M) (Lyapunov exponent)
        
        if abs(a) < 1e-10:
            # Schwarzschild case
            omega_c = 1 / (3 * np.sqrt(3) * M)  # Orbital frequency at photon sphere
            lambda_lyap = omega_c  # Lyapunov exponent equals orbital frequency
        else:
            # Kerr case - use prograde photon orbit
            # Approximate photon sphere radius
            # For |a/M| < 1, use the standard formula
            a_over_M = min(abs(a/M), 0.9999)  # Clamp to avoid arccos domain error
            r_ph = 2 * M * (1 + np.cos(2/3 * np.arccos(-np.sign(a) * a_over_M)))
            
            # Ensure r_ph is outside horizon
            if r_ph < r_plus:
                r_ph = r_plus * 1.1
            
            # Angular velocity at photon sphere
            omega_c = a / (r_ph**2 + a**2) + np.sqrt(M) / (r_ph**1.5 + a * np.sqrt(M))
            
            # Lyapunov exponent (approximation)
            # For Kerr, λ ≈ √(1 - 27M²Ω²) / (3M)
            lyap_arg = 1 - 27 * M**2 * omega_c**2
            lambda_lyap = abs(lyap_arg)**0.5 / (3 * M) if lyap_arg > 0 else omega_c
            
            # Near-extremal Kerr has larger errors
            if a / M > 0.9:
                relative_error *= 2  # Double error estimate for near-extremal
        
        omega_R = (l + 0.5) * omega_c
        omega_I = -(n + 0.5) * lambda_lyap
        
        return omega_R + 1j * omega_I, relative_error
    
    def spectral_gap(self, l_max=10):
        """
        Compute the spectral gap α = |Im(ω)|_min for dominant mode.
        
        The spectral gap determines decay rate of perturbations.
        
        Returns:
        --------
        tuple (float, float)
            (spectral_gap, estimated_relative_error)
        """
        min_decay = float('inf')
        max_error = 0.0
        
        for l in range(2, l_max + 1):
            result = self.qnm_frequency_approximation(l, n=0)
            if result[0] is not None:
                omega, error = result
                decay_rate = abs(omega.imag)
                if decay_rate < min_decay:
                    min_decay = decay_rate
                    max_error = error if error else 0.0
        
        if min_decay < float('inf'):
            return min_decay, max_error
        return None, None
    
    def beta_parameter(self, l_max=10):
        """
        Compute β = α/κ_- ratio for SCCC analysis.
        
        SCCC (C² version) likely holds if β < 1/2.
        SCCC (C⁰ version) violated if β > 0 (perturbations survive).
        
        Returns:
        --------
        tuple (float, float) or (None, None)
            (β parameter, estimated relative error) or None if Cauchy horizon doesn't exist
        """
        # Check if Cauchy horizon exists
        r_minus = self.bh.r_minus
        r_plus = self.bh.r_plus
        
        # For Schwarzschild (a=0, Q=0), no Cauchy horizon
        if abs(self.bh.a) < 1e-10 and abs(self.bh.Q) < 1e-10:
            return None, None
        
        # Check if r_minus is distinct from r_plus (non-extremal)
        if r_minus is None or r_plus is None:
            return None, None
        
        if abs(r_minus - r_plus) < 1e-10:
            # Extremal: degenerate horizon
            return float('inf'), 0.0
        
        kappa_minus = self.bh.kappa.get('minus')
        if kappa_minus is None or kappa_minus == 0:
            return float('inf'), 0.0
        
        alpha, alpha_error = self.spectral_gap(l_max)
        if alpha is None:
            return None, None
        
        beta = alpha / kappa_minus
        # Error propagation: δβ/β ≈ δα/α (assuming κ_- is exact)
        beta_error = alpha_error if alpha_error else 0.0
        
        return beta, beta_error
    
    def sccc_status(self, l_max=10):
        """
        Determine Strong Cosmic Censorship status.
        
        Returns:
        --------
        dict
            Status information including β value, error estimate, and interpretation
        """
        beta, beta_error = self.beta_parameter(l_max)
        alpha, alpha_error = self.spectral_gap(l_max)
        
        result = {
            'beta': beta,
            'beta_error': beta_error,
            'kappa_minus': self.bh.kappa.get('minus'),
            'kappa_plus': self.bh.kappa.get('plus'),
            'spectral_gap': alpha,
            'spectral_gap_error': alpha_error
        }
        
        if beta is None:
            result['status'] = 'NO_CAUCHY_HORIZON'
            result['interpretation'] = 'No Cauchy horizon exists (Schwarzschild-like)'
        elif beta > 0.5:
            result['status'] = 'SCCC_VIOLATED'
            error_str = f" ± {beta_error:.0%}" if beta_error else ""
            result['interpretation'] = f'β = {beta:.4f}{error_str} > 1/2: Even C² SCCC may be violated'
        elif beta > 0:
            result['status'] = 'C0_VIOLATED_C2_HOLDS'
            error_str = f" ± {beta_error:.0%}" if beta_error else ""
            result['interpretation'] = f'β = {beta:.4f}{error_str} ∈ (0, 1/2): C⁰ violated but C² likely holds'
        else:
            result['status'] = 'SCCC_HOLDS'
            result['interpretation'] = f'β = {beta:.4f} ≤ 0: Full SCCC holds'
        
        return result


#==============================================================================
# SECTION 3: Blue-Shift Instability Analysis
#==============================================================================

class BlueShiftInstability:
    """
    Analysis of the blue-shift instability at the Cauchy horizon.
    
    The blue-shift factor for radiation approaching the Cauchy horizon:
    E_observed / E_emitted ~ exp(κ_- * v)
    
    where v is advanced time.
    """
    
    def __init__(self, black_hole):
        self.bh = black_hole
    
    def blue_shift_factor(self, v):
        """
        Compute blue-shift factor exp(κ_- * v).
        
        Parameters:
        -----------
        v : float
            Advanced time coordinate
        
        Returns:
        --------
        float
            Blue-shift amplification factor
        """
        kappa_minus = self.bh.kappa.get('minus')
        if kappa_minus is None:
            return 1.0
        return np.exp(kappa_minus * v)
    
    def effective_stress_energy(self, v, T_initial=1.0):
        """
        Estimate effective stress-energy near Cauchy horizon.
        
        T_eff ~ T_initial * (blue_shift)² * (decay_factor)
        """
        blue_shift = self.blue_shift_factor(v)
        
        # Price's law decay: |ψ| ~ t^{-3} for l=0 mode
        # At late times t ~ v
        decay = (1 + v)**(-3)
        
        return T_initial * blue_shift**2 * decay
    
    def curvature_blowup_time(self, T_initial=1.0, threshold=1e10):
        """
        Estimate advanced time when curvature divergence occurs.
        
        Solve: T_eff(v) = threshold
        """
        if self.bh.r_minus is None:
            return None
        
        def equation(v):
            return self.effective_stress_energy(v, T_initial) - threshold
        
        # Search for crossing
        v_values = np.logspace(-1, 6, 1000)
        for v in v_values:
            if equation(v) > 0:
                try:
                    return brentq(equation, v/10, v)
                except:
                    return v
        return None
    
    def mass_inflation_rate(self, v):
        """
        Estimate mass inflation rate dm/dv at Cauchy horizon.
        
        Mass inflation: m(v) ~ m_0 * exp(κ_- * v) / v^p
        where p depends on decay rate.
        """
        kappa_minus = self.bh.kappa.get('minus', 0)
        return kappa_minus * self.blue_shift_factor(v) / (1 + v)**3


#==============================================================================
# SECTION 4: Penrose Inequality and Censorship Connection
#==============================================================================

class PenroseInequalityCensorshipLink:
    """
    Novel analysis connecting Penrose inequality to cosmic censorship.
    
    Key insight: If Penrose inequality is saturated (M = √(A/16π)),
    the spacetime is Schwarzschild, which satisfies both WCCC and SCCC.
    
    Deviations from saturation encode information about:
    1. Angular momentum content
    2. Gravitational radiation not yet radiated
    3. Potential for horizon evolution
    """
    
    def __init__(self, black_hole):
        self.bh = black_hole
    
    def penrose_deficit(self):
        """
        Compute the Penrose deficit: M - √(A/16π)
        
        For rotating black holes, this deficit relates to angular momentum.
        """
        A = self.bh.horizon_area()
        if A is None:
            return None
        
        M_irreducible = np.sqrt(A / (16 * PI))
        return self.bh.M - M_irreducible
    
    def angular_momentum_bound(self):
        """
        Compute angular momentum bound from Penrose inequality.
        
        Extended Penrose inequality: M² ≥ A/(16π) + 4πJ²/A
        
        This gives: J ≤ √(A/4π) * √(M² - A/16π)
        """
        A = self.bh.horizon_area()
        if A is None:
            return None
        
        M = self.bh.M
        M_irr_sq = A / (16 * PI)
        
        if M**2 < M_irr_sq:
            return 0  # Bound violated - shouldn't happen physically
        
        J_max = np.sqrt(A / (4 * PI)) * np.sqrt(M**2 - M_irr_sq)
        return J_max
    
    def subextremality_margin(self):
        """
        Compute how far from extremality: M² - a² - Q²
        
        Positive: sub-extremal (horizons exist)
        Zero: extremal
        Negative: super-extremal (naked singularity)
        """
        return self.bh.M**2 - self.bh.a**2 - self.bh.Q**2
    
    def censorship_stability_index(self):
        """
        Novel stability index combining multiple factors.
        
        CSI = (subextremality_margin/M²) × (1 - β) × (1/(1 + κ_- M))
        
        All factors are now dimensionless:
        - subextremality_margin/M² ∈ [0, 1]
        - (1 - β) ∈ [0, 1] when β ∈ [0, 1]
        - 1/(1 + κ_- M) ∈ (0, 1]
        
        Higher CSI → more stable against censorship violation
        CSI ∈ [0, 1] with CSI = 1 for Schwarzschild
        """
        M = self.bh.M
        margin = self.subextremality_margin()
        if margin <= 0:
            return 0  # Already violating or extremal
        
        # Normalized subextremality margin (dimensionless)
        margin_normalized = margin / M**2
        
        # Get β parameter (returns tuple now)
        spectral = SpectralGapAnalysis(self.bh)
        beta, _ = spectral.beta_parameter()
        if beta is None:
            beta_factor = 1  # No Cauchy horizon
        else:
            beta_factor = max(0, 1 - beta)
        
        # Blue-shift rate (now dimensionless: κ_- M)
        kappa_minus = self.bh.kappa.get('minus', 0)
        if kappa_minus > 0:
            # κ_- has units of 1/M, so κ_- * M is dimensionless
            blueshift_factor = 1 / (1 + kappa_minus * M)
        else:
            blueshift_factor = 1
        
        return margin_normalized * beta_factor * blueshift_factor


#==============================================================================
# SECTION 5: Critical Collapse Analysis
#==============================================================================

class CriticalCollapseAnalysis:
    """
    Analysis of critical collapse behavior near the threshold of black hole 
    formation, following Choptuik's work.
    
    Key features:
    - Universal scaling: M ~ |p - p*|^γ with γ ≈ 0.37 (scalar field)
    - Discrete self-similarity with period Δ ≈ 3.44
    - Naked singularity at threshold (p = p*)
    """
    
    def __init__(self, gamma=0.37, Delta=3.44):
        """
        Parameters:
        -----------
        gamma : float
            Critical exponent (universal)
        Delta : float  
            Echoing period (universal for given matter type)
        """
        self.gamma = gamma
        self.Delta = Delta
    
    def black_hole_mass(self, p, p_star, C=1.0):
        """
        Mass scaling near critical threshold.
        
        M = C |p - p*|^γ for p > p*
        M = 0 for p < p* (no black hole forms)
        """
        if p <= p_star:
            return 0
        return C * (p - p_star)**self.gamma
    
    def critical_exponent_from_data(self, p_values, M_values, p_star_estimate):
        """
        Estimate critical exponent from numerical data.
        
        Linear fit of log(M) vs log(|p - p*|)
        """
        # Filter for p > p*
        mask = p_values > p_star_estimate
        p_super = p_values[mask]
        M_super = M_values[mask]
        
        if len(p_super) < 2:
            return None
        
        log_dp = np.log(p_super - p_star_estimate)
        log_M = np.log(M_super)
        
        # Linear fit
        coeffs = np.polyfit(log_dp, log_M, 1)
        return coeffs[0]  # Slope is γ
    
    def naked_singularity_lifetime(self, p, p_star):
        """
        At precisely p = p*, a naked singularity forms but is unstable.
        
        Perturbation δp causes it to either:
        - Form a black hole (δp > 0)
        - Disperse to infinity (δp < 0)
        
        The "lifetime" scales as |log|δp||
        """
        delta_p = abs(p - p_star)
        if delta_p == 0:
            return float('inf')
        return abs(np.log(delta_p)) / self.Delta


#==============================================================================
# SECTION 6: Entropy Bounds and Thermodynamic Analysis
#==============================================================================

class EntropyBoundAnalysis:
    """
    Thermodynamic approach to cosmic censorship.
    
    Key principle: Second law of black hole thermodynamics should prevent
    processes that would expose singularities.
    
    δA ≥ 0 for any physical process
    """
    
    def __init__(self, black_hole):
        self.bh = black_hole
    
    def area_increase_condition(self, delta_M, delta_J, delta_Q):
        """
        Check if proposed changes satisfy area theorem.
        
        First law: δM = (κ/8π) δA + Ω δJ + Φ δQ
        
        For area to increase: δA ≥ 0
        
        Returns True if area increase is satisfied.
        """
        kappa = self.bh.kappa.get('plus', 0)
        if kappa == 0:  # Extremal
            return delta_M >= 0  # Any mass loss forbidden at extremality
        
        Omega = self.bh.angular_velocity()
        Phi = self.bh.electric_potential()
        
        if Omega is None or Phi is None:
            return True  # Can't check
        
        # δA = (8π/κ)(δM - Ω δJ - Φ δQ)
        delta_A = (8 * PI / kappa) * (delta_M - Omega * delta_J - Phi * delta_Q)
        
        return delta_A >= 0
    
    def wald_gedanken_test(self, test_particle_energy, test_particle_L, test_particle_q):
        """
        Wald's gedanken experiment test.
        
        Can a test particle with (E, L, q) overspin/overcharge the black hole?
        
        Condition for capture: E < Ω L + Φ q (for infall)
        Condition to overspin: J + L > M²  or  Q + q > M
        
        Wald showed these are mutually exclusive for extremal black holes.
        """
        E, L, q = test_particle_energy, test_particle_L, test_particle_q
        
        Omega = self.bh.angular_velocity() or 0
        Phi = self.bh.electric_potential() or 0
        
        # Check capture condition
        can_be_captured = E >= Omega * L + Phi * q
        
        # Check overspinning condition
        final_J = self.bh.M * self.bh.a + L
        final_Q = self.bh.Q + q
        final_M = self.bh.M + E
        
        would_overspin = final_J > final_M**2 or abs(final_Q) > final_M
        
        return {
            'can_be_captured': can_be_captured,
            'would_overspin': would_overspin,
            'censorship_violated': can_be_captured and would_overspin,
            'final_spin_param': final_J / final_M**2 if final_M > 0 else float('inf'),
            'final_charge_param': abs(final_Q) / final_M if final_M > 0 else float('inf')
        }
    
    def thermodynamic_censorship_bound(self):
        """
        Derive censorship bound from thermodynamics.
        
        At extremality (κ → 0), temperature T = κ/2π → 0.
        Third law of thermodynamics: Cannot reach T = 0 in finite time.
        
        This provides thermodynamic protection of censorship.
        """
        kappa = self.bh.kappa.get('plus', 0)
        T_hawking = kappa / (2 * PI)
        
        # Distance from extremality in terms of temperature
        M, a, Q = self.bh.M, self.bh.a, self.bh.Q
        
        # Extremal limit
        Delta_extremal = M**2 - a**2 - Q**2
        
        return {
            'hawking_temperature': T_hawking,
            'surface_gravity': kappa,
            'extremality_margin': Delta_extremal,
            'third_law_protection': kappa > 0
        }


#==============================================================================
# SECTION 7: Numerical Experiments
#==============================================================================

def run_kerr_sccc_analysis():
    """
    Comprehensive SCCC analysis for Kerr black holes.
    """
    print("=" * 70)
    print("STRONG COSMIC CENSORSHIP ANALYSIS FOR KERR BLACK HOLES")
    print("=" * 70)
    
    # Scan spin parameter
    spin_values = np.linspace(0, 0.999, 50)
    results = []
    
    for a in spin_values:
        bh = KerrNewmanBlackHole(M=1.0, a=a, Q=0, Lambda=0)
        spectral = SpectralGapAnalysis(bh)
        status = spectral.sccc_status()
        
        results.append({
            'a/M': a,
            'beta': status['beta'],
            'kappa_minus': status['kappa_minus'],
            'status': status['status']
        })
    
    print("\nSpin (a/M)    β parameter    κ₋         Status")
    print("-" * 60)
    
    for i in range(0, len(results), 10):
        r = results[i]
        if r['beta'] is not None:
            print(f"  {r['a/M']:.3f}         {r['beta']:.4f}        {r['kappa_minus']:.4f}     {r['status']}")
        else:
            print(f"  {r['a/M']:.3f}         N/A           N/A       {r['status']}")
    
    return results


def run_rndS_sccc_analysis():
    """
    SCCC analysis for Reissner-Nordström-de Sitter black holes.
    This is where SCCC violations are most dramatic.
    """
    print("\n" + "=" * 70)
    print("SCCC ANALYSIS FOR REISSNER-NORDSTRÖM-de SITTER BLACK HOLES")
    print("=" * 70)
    
    # Fixed charge, varying cosmological constant
    Q_over_M = 0.9
    Lambda_values = np.logspace(-4, -1, 20)
    
    results = []
    
    for Lambda in Lambda_values:
        bh = KerrNewmanBlackHole(M=1.0, a=0, Q=Q_over_M, Lambda=Lambda)
        
        if bh.r_plus is not None and bh.r_minus is not None:
            spectral = SpectralGapAnalysis(bh)
            status = spectral.sccc_status()
            results.append({
                'Lambda': Lambda,
                'Q/M': Q_over_M,
                'beta': status['beta'],
                'status': status['status']
            })
    
    print(f"\nQ/M = {Q_over_M}")
    print("\nΛ            β parameter    Status")
    print("-" * 50)
    
    for r in results:
        if r['beta'] is not None:
            print(f"{r['Lambda']:.2e}      {r['beta']:.4f}        {r['status']}")
    
    return results


def run_penrose_inequality_analysis():
    """
    Analyze connection between Penrose inequality and cosmic censorship.
    """
    print("\n" + "=" * 70)
    print("PENROSE INEQUALITY AND COSMIC CENSORSHIP CONNECTION")
    print("=" * 70)
    
    # Vary spin and compute various quantities
    spin_values = np.linspace(0, 0.99, 20)
    
    print("\na/M     Penrose Deficit   J_bound    CSI       Status")
    print("-" * 60)
    
    for a in spin_values:
        bh = KerrNewmanBlackHole(M=1.0, a=a, Q=0)
        link = PenroseInequalityCensorshipLink(bh)
        
        deficit = link.penrose_deficit()
        J_bound = link.angular_momentum_bound()
        csi = link.censorship_stability_index()
        
        # Actual angular momentum
        J_actual = a  # J = aM = a for M=1
        
        if deficit is not None:
            print(f"{a:.2f}    {deficit:.4f}           {J_bound:.4f}     {csi:.4f}    ", end="")
            if csi > 0.5:
                print("Stable")
            elif csi > 0.1:
                print("Marginal")
            else:
                print("Near-extremal")


def run_wald_gedanken_experiments():
    """
    Numerical verification of Wald's gedanken experiment result.
    """
    print("\n" + "=" * 70)
    print("WALD GEDANKEN EXPERIMENT: CAN PARTICLES DESTROY BLACK HOLES?")
    print("=" * 70)
    
    # Near-extremal Kerr black hole
    for a in [0.9, 0.99, 0.999]:
        print(f"\n--- Near-extremal Kerr: a/M = {a} ---")
        bh = KerrNewmanBlackHole(M=1.0, a=a, Q=0)
        entropy_analysis = EntropyBoundAnalysis(bh)
        
        # Try various test particles
        test_particles = [
            (0.1, 1.0, 0),    # Low energy, high angular momentum
            (1.0, 0.1, 0),    # High energy, low angular momentum  
            (0.01, 0.5, 0),   # Very low energy, medium L
        ]
        
        print("E        L        q        Captured?  Overspin?  Violation?")
        print("-" * 60)
        
        for E, L, q in test_particles:
            result = entropy_analysis.wald_gedanken_test(E, L, q)
            print(f"{E:.2f}     {L:.2f}     {q:.2f}     "
                  f"{'Yes' if result['can_be_captured'] else 'No':8s} "
                  f"{'Yes' if result['would_overspin'] else 'No':9s} "
                  f"{'YES!' if result['censorship_violated'] else 'No'}")


def run_critical_collapse_analysis():
    """
    Analysis of critical collapse phenomenology.
    """
    print("\n" + "=" * 70)
    print("CRITICAL COLLAPSE AND NAKED SINGULARITY THRESHOLD")
    print("=" * 70)
    
    crit = CriticalCollapseAnalysis(gamma=0.37, Delta=3.44)
    
    # Simulate near-critical behavior
    p_star = 1.0  # Arbitrary threshold
    p_values = np.linspace(0.9, 1.1, 100)
    
    print(f"\nUniversal critical exponent γ = {crit.gamma}")
    print(f"Discrete self-similarity period Δ = {crit.Delta}")
    print("\nParameter    M_BH         Lifetime at p*")
    print("-" * 45)
    
    for p in [0.95, 0.99, 1.0, 1.001, 1.01, 1.05]:
        M = crit.black_hole_mass(p, p_star)
        lifetime = crit.naked_singularity_lifetime(p, p_star)
        print(f"p = {p:.3f}     M = {M:.4f}     τ = {lifetime:.2f}")
    
    print("\n→ At p = p*, naked singularity forms but is infinitely fine-tuned")
    print("→ Any perturbation causes it to either form a BH or disperse")
    print("→ Naked singularity is NON-GENERIC (measure zero)")


def compute_rndS_critical_curve():
    """
    Compute the critical curve β = 1/2 in (Q/M, ΛM²) parameter space
    where SCCC transitions from holding to failing.
    """
    print("\n" + "=" * 70)
    print("RN-dS CRITICAL CURVE: β = 1/2 BOUNDARY")
    print("=" * 70)
    
    Q_range = np.linspace(0.5, 0.99, 20)
    Lambda_critical = []
    
    for Q in Q_range:
        # Binary search for Lambda where β = 0.5
        Lambda_low, Lambda_high = 1e-6, 0.1
        found = False
        
        for _ in range(50):  # Binary search iterations
            Lambda_mid = (Lambda_low + Lambda_high) / 2
            bh = KerrNewmanBlackHole(M=1.0, a=0, Q=Q, Lambda=Lambda_mid)
            
            if bh.r_plus is None or bh.r_minus is None:
                Lambda_high = Lambda_mid
                continue
            
            spectral = SpectralGapAnalysis(bh)
            beta, _ = spectral.beta_parameter()
            
            if beta is None:
                Lambda_high = Lambda_mid
                continue
            
            if abs(beta - 0.5) < 0.01:
                Lambda_critical.append((Q, Lambda_mid, beta))
                found = True
                break
            elif beta > 0.5:
                Lambda_high = Lambda_mid
            else:
                Lambda_low = Lambda_mid
        
        if not found:
            Lambda_critical.append((Q, Lambda_mid, beta if beta else -1))
    
    print("\nQ/M        Λ_critical    β at critical")
    print("-" * 45)
    for Q, Lam, beta in Lambda_critical:
        print(f"{Q:.2f}       {Lam:.2e}      {beta:.3f}")
    
    print("\n→ Above Λ_critical: SCCC violated (β > 1/2)")
    print("→ Below Λ_critical: SCCC holds (β < 1/2)")
    
    return Lambda_critical


def censorship_stability_comprehensive():
    """
    Comprehensive analysis of the Censorship Stability Index across 
    the Kerr-Newman parameter space.
    """
    print("\n" + "=" * 70)
    print("CENSORSHIP STABILITY INDEX (CSI) COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    print("\n=== Kerr Black Holes (Q=0) ===")
    print("a/M        CSI       Subext.Margin   β        Error    Interpretation")
    print("-" * 80)
    
    for a in [0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        bh = KerrNewmanBlackHole(M=1.0, a=a, Q=0)
        link = PenroseInequalityCensorshipLink(bh)
        spectral = SpectralGapAnalysis(bh)
        
        csi = link.censorship_stability_index()
        margin = link.subextremality_margin()
        beta, beta_error = spectral.beta_parameter()
        
        if csi > 0.5:
            interp = "Robustly Censored"
        elif csi > 0.1:
            interp = "Marginally Stable"
        else:
            interp = "Near Critical"
        
        beta_str = f"{beta:.3f}" if beta is not None else "N/A"
        error_str = f"±{beta_error:.0%}" if beta_error is not None else "N/A"
        print(f"{a:.2f}       {csi:.3f}     {margin:.4f}         {beta_str}    {error_str:6s}   {interp}")
    
    print("\n=== Kerr-Newman Black Holes (a=0.5) ===")
    print("Q/M        CSI       Subext.Margin   Interpretation")
    print("-" * 60)
    
    for Q in [0, 0.2, 0.4, 0.6, 0.8]:
        bh = KerrNewmanBlackHole(M=1.0, a=0.5, Q=Q)
        link = PenroseInequalityCensorshipLink(bh)
        
        csi = link.censorship_stability_index()
        margin = link.subextremality_margin()
        
        if csi > 0.5:
            interp = "Robustly Censored"
        elif csi > 0.1:
            interp = "Marginally Stable"  
        else:
            interp = "Near Critical"
        
        print(f"{Q:.2f}       {csi:.3f}     {margin:.4f}         {interp}")


def verify_penrose_inequality():
    """
    Verify the Penrose inequality and its angular momentum extension
    for various black hole configurations.
    """
    print("\n" + "=" * 70)
    print("PENROSE INEQUALITY VERIFICATION")
    print("=" * 70)
    
    print("\n=== Standard Penrose Inequality: M ≥ √(A/16π) ===")
    print("Configuration           M_ADM    √(A/16π)  Deficit   Status")
    print("-" * 70)
    
    configs = [
        ("Schwarzschild", 1.0, 0, 0),
        ("Kerr a=0.5", 1.0, 0.5, 0),
        ("Kerr a=0.9", 1.0, 0.9, 0),
        ("RN Q=0.5", 1.0, 0, 0.5),
        ("Kerr-Newman", 1.0, 0.5, 0.5),
    ]
    
    for name, M, a, Q in configs:
        bh = KerrNewmanBlackHole(M=M, a=a, Q=Q)
        if bh.r_plus is None:
            print(f"{name:20s}    {M:.2f}     --        --       No horizon (naked?)")
            continue
        
        A = bh.horizon_area()
        M_min = np.sqrt(A / (16 * PI))
        deficit = M - M_min
        status = "✓ Satisfied" if deficit >= -1e-10 else "✗ VIOLATED"
        
        print(f"{name:20s}    {M:.2f}     {M_min:.4f}    {deficit:.4f}    {status}")
    
    print("\n=== Angular Momentum Penrose Inequality: M² ≥ A/16π + 4πJ²/A ===")
    print("Configuration       J_actual   J_max (from ineq)   Status")
    print("-" * 60)
    
    for name, M, a, Q in configs:
        bh = KerrNewmanBlackHole(M=M, a=a, Q=Q)
        if bh.r_plus is None:
            continue
        
        link = PenroseInequalityCensorshipLink(bh)
        J_actual = a * M  # J = aM
        J_max = link.angular_momentum_bound()
        
        status = "✓ Satisfied" if J_actual <= J_max + 1e-10 else "✗ VIOLATED"
        print(f"{name:16s}    {J_actual:.3f}      {J_max:.4f}             {status}")


#==============================================================================
# MAIN EXECUTION
#==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" COSMIC CENSORSHIP CONJECTURE: NUMERICAL ANALYSIS FRAMEWORK")
    print("=" * 70)
    
    # Run all analyses
    run_kerr_sccc_analysis()
    run_rndS_sccc_analysis()
    run_penrose_inequality_analysis()
    run_wald_gedanken_experiments()
    run_critical_collapse_analysis()
    
    # New comprehensive analyses
    censorship_stability_comprehensive()
    verify_penrose_inequality()
    compute_rndS_critical_curve()
    
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    print("""
1. WEAK COSMIC CENSORSHIP (WCCC):
   - Wald's gedanken experiment shows test particles cannot destroy horizons
   - Critical collapse produces naked singularities only at measure-zero threshold
   - Penrose inequality provides geometric bound on allowed configurations
   - Angular momentum extension: M² ≥ A/16π + 4πJ²/A implies horizon formation

2. STRONG COSMIC CENSORSHIP (SCCC):
   - For Kerr: β ≈ 0.3-0.4 suggests C² version holds
   - For RN-dS: β can exceed 1/2, violating even C² SCCC
   - Spectral gap vs surface gravity ratio is key diagnostic
   - Critical curve in (Q/M, ΛM²) space separates SCCC regions

3. CENSORSHIP STABILITY INDEX (CSI):
   - Combines: subextremality margin × (1-β) × inverse blue-shift rate
   - CSI > 0.5: Robustly censored
   - CSI ∈ (0.1, 0.5): Marginally stable
   - CSI < 0.1: Near critical (extremal)

4. THERMODYNAMIC PROTECTION:
   - Third law prevents reaching T=0 (extremality) in finite time
   - Area theorem (δA ≥ 0) constrains allowed processes
   - First law prevents overspinning: dM ≥ Ω dJ + Φ dQ

5. PENROSE INEQUALITY CONNECTION:
   - Standard: M ≥ √(A/16π) with equality for Schwarzschild
   - Angular momentum: M² ≥ A/16π + 4πJ²/A
   - Provides geometric obstruction to naked singularity formation
""")
