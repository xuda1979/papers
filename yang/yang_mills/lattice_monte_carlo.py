"""
Lattice Yang-Mills Monte Carlo Simulation
==========================================

This code implements a simple Monte Carlo simulation for SU(2) lattice gauge theory
in 4 dimensions using the Wilson action. It measures:
1. Average plaquette (order parameter)
2. Wilson loops (for string tension)
3. Polyakov loop correlators (for mass gap)

Author: [Your name]
Date: December 2025
"""

import numpy as np
from typing import Tuple, List, Optional
import time

# =============================================================================
# SU(2) Matrix Operations
# =============================================================================

def random_su2() -> np.ndarray:
    """Generate a random SU(2) matrix using Haar measure."""
    # Parameterize SU(2) as: U = a0*I + i*a_i*sigma_i with a0^2 + a^2 = 1
    # Use rejection method for uniform distribution on S^3
    while True:
        x = np.random.uniform(-1, 1, 4)
        norm_sq = np.sum(x**2)
        if norm_sq <= 1 and norm_sq > 0:
            x = x / np.sqrt(norm_sq)
            break
    
    a0, a1, a2, a3 = x
    # SU(2) matrix: [[a0 + i*a3, a2 + i*a1], [-a2 + i*a1, a0 - i*a3]]
    return np.array([
        [a0 + 1j*a3, a2 + 1j*a1],
        [-a2 + 1j*a1, a0 - 1j*a3]
    ], dtype=complex)


def random_su2_near_identity(epsilon: float) -> np.ndarray:
    """Generate SU(2) matrix close to identity for Metropolis updates."""
    # Small perturbation: U = exp(i * epsilon * n_i * sigma_i)
    n = np.random.randn(3)
    n = n / np.linalg.norm(n) * epsilon * np.random.uniform(0, 1)
    
    # Pauli matrices
    theta = np.linalg.norm(n)
    if theta < 1e-10:
        return np.eye(2, dtype=complex)
    
    n_hat = n / theta
    
    # exp(i * theta * n_hat . sigma) = cos(theta)*I + i*sin(theta)*(n_hat . sigma)
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex)
    ]
    
    U = np.cos(theta) * np.eye(2, dtype=complex)
    for i in range(3):
        U += 1j * np.sin(theta) * n_hat[i] * sigma[i]
    
    return U


def su2_dagger(U: np.ndarray) -> np.ndarray:
    """Hermitian conjugate of SU(2) matrix."""
    return U.conj().T


def su2_trace(U: np.ndarray) -> complex:
    """Trace of SU(2) matrix."""
    return U[0, 0] + U[1, 1]


# =============================================================================
# Lattice Structure
# =============================================================================

class Lattice:
    """4D hypercubic lattice with periodic boundary conditions."""
    
    def __init__(self, L: int, T: int):
        """
        Initialize lattice.
        
        Args:
            L: Spatial extent (L^3 spatial volume)
            T: Temporal extent
        """
        self.L = L
        self.T = T
        self.dims = (L, L, L, T)
        
        # Links: U[x, y, z, t, mu] where mu = 0,1,2,3 for directions
        # Initialize to identity (cold start) or random (hot start)
        self.links = np.zeros((L, L, L, T, 4, 2, 2), dtype=complex)
        
    def cold_start(self):
        """Initialize all links to identity."""
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for t in range(self.T):
                        for mu in range(4):
                            self.links[x, y, z, t, mu] = np.eye(2, dtype=complex)
    
    def hot_start(self):
        """Initialize all links to random SU(2)."""
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for t in range(self.T):
                        for mu in range(4):
                            self.links[x, y, z, t, mu] = random_su2()
    
    def shift(self, pos: Tuple[int, int, int, int], mu: int, sign: int = 1) -> Tuple[int, int, int, int]:
        """Shift position by one step in direction mu with periodic BC."""
        pos = list(pos)
        pos[mu] = (pos[mu] + sign) % self.dims[mu]
        return tuple(pos)
    
    def get_link(self, pos: Tuple[int, int, int, int], mu: int) -> np.ndarray:
        """Get link U_mu(x)."""
        return self.links[pos[0], pos[1], pos[2], pos[3], mu]
    
    def set_link(self, pos: Tuple[int, int, int, int], mu: int, U: np.ndarray):
        """Set link U_mu(x)."""
        self.links[pos[0], pos[1], pos[2], pos[3], mu] = U
    
    def plaquette(self, pos: Tuple[int, int, int, int], mu: int, nu: int) -> np.ndarray:
        """
        Compute plaquette U_mu(x) * U_nu(x+mu) * U_mu(x+nu)^dag * U_nu(x)^dag
        """
        x = pos
        x_mu = self.shift(x, mu)
        x_nu = self.shift(x, nu)
        
        U1 = self.get_link(x, mu)
        U2 = self.get_link(x_mu, nu)
        U3 = su2_dagger(self.get_link(x_nu, mu))
        U4 = su2_dagger(self.get_link(x, nu))
        
        return U1 @ U2 @ U3 @ U4
    
    def staple(self, pos: Tuple[int, int, int, int], mu: int) -> np.ndarray:
        """
        Compute staple sum for link U_mu(x).
        
        The staple is the sum over nu != mu of:
        U_nu(x+mu) * U_mu(x+nu)^dag * U_nu(x)^dag  (forward staple)
        + U_nu(x+mu-nu)^dag * U_mu(x-nu)^dag * U_nu(x-nu)  (backward staple)
        """
        S = np.zeros((2, 2), dtype=complex)
        x = pos
        x_mu = self.shift(x, mu)
        
        for nu in range(4):
            if nu == mu:
                continue
            
            # Forward staple
            x_nu = self.shift(x, nu)
            x_mu_nu = self.shift(x_mu, nu)
            
            U1 = self.get_link(x_mu, nu)  # U_nu(x+mu)
            U2 = su2_dagger(self.get_link(x_nu, mu))  # U_mu(x+nu)^dag
            U3 = su2_dagger(self.get_link(x, nu))  # U_nu(x)^dag
            S += U1 @ U2 @ U3
            
            # Backward staple
            x_mu_mnu = self.shift(x_mu, nu, -1)  # x + mu - nu
            x_mnu = self.shift(x, nu, -1)  # x - nu
            
            U1 = su2_dagger(self.get_link(x_mu_mnu, nu))  # U_nu(x+mu-nu)^dag
            U2 = su2_dagger(self.get_link(x_mnu, mu))  # U_mu(x-nu)^dag
            U3 = self.get_link(x_mnu, nu)  # U_nu(x-nu)
            S += U1 @ U2 @ U3
        
        return S


# =============================================================================
# Monte Carlo Updates
# =============================================================================

def metropolis_update(lattice: Lattice, pos: Tuple[int, int, int, int], 
                      mu: int, beta: float, epsilon: float) -> bool:
    """
    Metropolis update for single link.
    
    Args:
        lattice: Lattice object
        pos: Site position
        mu: Link direction
        beta: Coupling (beta = 4/g^2 for SU(2))
        epsilon: Size of proposed update
    
    Returns:
        True if update accepted, False otherwise
    """
    U_old = lattice.get_link(pos, mu)
    staple = lattice.staple(pos, mu)
    
    # Old action contribution: -beta/2 * Re(Tr(U_old * staple))
    S_old = -beta/2 * np.real(su2_trace(U_old @ staple))
    
    # Propose new link
    dU = random_su2_near_identity(epsilon)
    U_new = dU @ U_old
    
    # New action contribution
    S_new = -beta/2 * np.real(su2_trace(U_new @ staple))
    
    # Metropolis accept/reject
    dS = S_new - S_old
    if dS < 0 or np.random.random() < np.exp(-dS):
        lattice.set_link(pos, mu, U_new)
        return True
    return False


def heatbath_update(lattice: Lattice, pos: Tuple[int, int, int, int], 
                    mu: int, beta: float):
    """
    Heat bath update for SU(2) (Kennedy-Pendleton algorithm).
    
    For SU(2), we can sample directly from the conditional distribution.
    """
    staple = lattice.staple(pos, mu)
    
    # The conditional distribution for U given staple is proportional to
    # exp(beta/2 * Re(Tr(U * staple)))
    # For SU(2), this is the same as exp(beta * k * cos(theta)) where
    # k = |det(staple)|^{1/2} and theta is an angle
    
    # Compute k = sqrt(det(staple)) (for SU(2), det(staple^dag staple) = |Tr(staple)|^2 / 4)
    k = np.sqrt(np.real(su2_trace(staple @ su2_dagger(staple)))) / 2
    
    if k < 1e-10:
        # Staple is nearly zero, sample uniformly
        U_new = random_su2()
    else:
        # Sample a0 = cos(theta) from distribution ~ exp(beta * k * a0) * sqrt(1 - a0^2)
        # Use rejection method
        a = beta * k
        while True:
            # Sample from proposal distribution
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            r4 = np.random.random()
            
            x = -np.log(r1) / a
            y = -np.log(r2) / a
            z = np.cos(2 * np.pi * r3)**2
            w = x + y * z
            
            a0 = 1 - w
            if a0 < -1:
                continue
            if r4**2 <= 1 - a0**2:
                break
        
        # Sample remaining components uniformly on sphere
        phi = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        r = np.sqrt(1 - a0**2)
        a1 = r * sin_theta * np.cos(phi)
        a2 = r * sin_theta * np.sin(phi)
        a3 = r * cos_theta
        
        # Construct SU(2) matrix
        U_rand = np.array([
            [a0 + 1j*a3, a2 + 1j*a1],
            [-a2 + 1j*a1, a0 - 1j*a3]
        ], dtype=complex)
        
        # The new link is U_rand * (staple^dag / k)
        if k > 1e-10:
            staple_normalized = su2_dagger(staple) / (2 * k)
            U_new = U_rand @ staple_normalized
        else:
            U_new = U_rand
    
    lattice.set_link(pos, mu, U_new)


def sweep(lattice: Lattice, beta: float, method: str = 'metropolis', 
          epsilon: float = 0.3) -> float:
    """
    Perform one sweep over all links.
    
    Returns:
        Acceptance rate (for Metropolis)
    """
    n_accept = 0
    n_total = 0
    
    for x in range(lattice.L):
        for y in range(lattice.L):
            for z in range(lattice.L):
                for t in range(lattice.T):
                    pos = (x, y, z, t)
                    for mu in range(4):
                        if method == 'metropolis':
                            if metropolis_update(lattice, pos, mu, beta, epsilon):
                                n_accept += 1
                            n_total += 1
                        elif method == 'heatbath':
                            heatbath_update(lattice, pos, mu, beta)
    
    if method == 'metropolis':
        return n_accept / n_total
    return 1.0


# =============================================================================
# Observables
# =============================================================================

def average_plaquette(lattice: Lattice) -> float:
    """Compute average plaquette: (1/N) * sum_P Re(Tr(U_P)) / 2."""
    total = 0.0
    count = 0
    
    for x in range(lattice.L):
        for y in range(lattice.L):
            for z in range(lattice.L):
                for t in range(lattice.T):
                    pos = (x, y, z, t)
                    for mu in range(4):
                        for nu in range(mu + 1, 4):
                            P = lattice.plaquette(pos, mu, nu)
                            total += np.real(su2_trace(P)) / 2  # Normalize for SU(2)
                            count += 1
    
    return total / count


def wilson_loop(lattice: Lattice, R: int, T: int) -> float:
    """
    Compute Wilson loop of size R x T (spatial x temporal).
    
    Average over all positions and spatial orientations.
    """
    total = 0.0
    count = 0
    
    # Loop over starting positions
    for x0 in range(lattice.L):
        for y0 in range(lattice.L):
            for z0 in range(lattice.L):
                for t0 in range(lattice.T):
                    # Loop over spatial direction (R is in this direction)
                    for mu in range(3):  # Spatial directions only
                        # Compute Wilson loop in mu-4 plane
                        W = np.eye(2, dtype=complex)
                        
                        # Bottom edge (spatial direction)
                        pos = (x0, y0, z0, t0)
                        for _ in range(R):
                            W = W @ lattice.get_link(pos, mu)
                            pos = lattice.shift(pos, mu)
                        
                        # Right edge (temporal direction)
                        for _ in range(T):
                            W = W @ lattice.get_link(pos, 3)
                            pos = lattice.shift(pos, 3)
                        
                        # Top edge (backwards in spatial)
                        for _ in range(R):
                            pos = lattice.shift(pos, mu, -1)
                            W = W @ su2_dagger(lattice.get_link(pos, mu))
                        
                        # Left edge (backwards in temporal)
                        for _ in range(T):
                            pos = lattice.shift(pos, 3, -1)
                            W = W @ su2_dagger(lattice.get_link(pos, 3))
                        
                        total += np.real(su2_trace(W)) / 2
                        count += 1
    
    return total / count


def polyakov_loop(lattice: Lattice, pos_spatial: Tuple[int, int, int]) -> complex:
    """Compute Polyakov loop at spatial position."""
    P = np.eye(2, dtype=complex)
    x, y, z = pos_spatial
    
    for t in range(lattice.T):
        P = P @ lattice.get_link((x, y, z, t), 3)
    
    return su2_trace(P) / 2


def polyakov_correlator(lattice: Lattice, R: int) -> float:
    """
    Compute Polyakov loop correlator <P(0) P(R)^dag> for separation R.
    
    Average over all positions and directions.
    """
    total = 0.0
    count = 0
    
    for x in range(lattice.L):
        for y in range(lattice.L):
            for z in range(lattice.L):
                pos0 = (x, y, z)
                P0 = polyakov_loop(lattice, pos0)
                
                # Separation in x-direction
                pos_R = ((x + R) % lattice.L, y, z)
                PR = polyakov_loop(lattice, pos_R)
                total += np.real(P0 * np.conj(PR))
                count += 1
                
                # Separation in y-direction
                pos_R = (x, (y + R) % lattice.L, z)
                PR = polyakov_loop(lattice, pos_R)
                total += np.real(P0 * np.conj(PR))
                count += 1
                
                # Separation in z-direction
                pos_R = (x, y, (z + R) % lattice.L)
                PR = polyakov_loop(lattice, pos_R)
                total += np.real(P0 * np.conj(PR))
                count += 1
    
    return total / count


# =============================================================================
# String Tension and Mass Gap Extraction
# =============================================================================

def extract_string_tension(wilson_data: dict) -> Tuple[float, float]:
    """
    Extract string tension from Wilson loop data using Creutz ratio.
    
    chi(R, T) = -log( W(R,T) * W(R-1,T-1) / (W(R,T-1) * W(R-1,T)) )
    
    sigma = lim_{R,T -> infty} chi(R, T)
    """
    sigmas = []
    
    for R in range(2, max(wilson_data.keys()) + 1):
        for T in range(2, max(k[1] for k in wilson_data.keys() if k[0] == R) + 1):
            try:
                W_RT = wilson_data[(R, T)]
                W_R1T1 = wilson_data[(R-1, T-1)]
                W_RT1 = wilson_data[(R, T-1)]
                W_R1T = wilson_data[(R-1, T)]
                
                if W_RT > 0 and W_R1T1 > 0 and W_RT1 > 0 and W_R1T > 0:
                    chi = -np.log(W_RT * W_R1T1 / (W_RT1 * W_R1T))
                    sigmas.append(chi)
            except KeyError:
                pass
    
    if sigmas:
        return np.mean(sigmas), np.std(sigmas)
    return 0.0, 0.0


def extract_mass_gap(correlators: List[float], lattice_T: int) -> Tuple[float, float]:
    """
    Extract mass gap from Polyakov loop correlators.
    
    <P(0) P(R)^dag> ~ exp(-m * R) for large R
    
    m = mass gap
    """
    if len(correlators) < 3:
        return 0.0, 0.0
    
    # Fit to exponential decay
    R_values = np.arange(1, len(correlators) + 1)
    log_C = np.log(np.abs(np.array(correlators)) + 1e-10)
    
    # Linear fit: log(C) = -m * R + const
    # Use only R >= 1 (skip R=0)
    if len(R_values) >= 2:
        coeffs = np.polyfit(R_values, log_C, 1)
        m = -coeffs[0]
        return m, 0.0  # TODO: proper error estimation
    return 0.0, 0.0


# =============================================================================
# Main Simulation
# =============================================================================

def run_simulation(L: int = 4, T: int = 4, beta: float = 2.0,
                   n_thermalize: int = 100, n_measure: int = 100,
                   measure_interval: int = 10):
    """
    Run Monte Carlo simulation and measure observables.
    
    Args:
        L: Spatial lattice size
        T: Temporal lattice size
        beta: Coupling constant (beta = 4/g^2 for SU(2))
        n_thermalize: Number of thermalization sweeps
        n_measure: Number of measurement sweeps
        measure_interval: Sweeps between measurements
    """
    print(f"=" * 60)
    print(f"Lattice Yang-Mills Monte Carlo Simulation")
    print(f"=" * 60)
    print(f"Lattice: {L}^3 x {T}")
    print(f"Beta: {beta}")
    print(f"Thermalization: {n_thermalize} sweeps")
    print(f"Measurements: {n_measure} (every {measure_interval} sweeps)")
    print()
    
    # Initialize lattice
    lattice = Lattice(L, T)
    lattice.hot_start()  # Start from random configuration
    
    # Thermalization
    print("Thermalizing...")
    t0 = time.time()
    for i in range(n_thermalize):
        sweep(lattice, beta, method='heatbath')
        if (i + 1) % 20 == 0:
            plaq = average_plaquette(lattice)
            print(f"  Sweep {i+1}/{n_thermalize}: <P> = {plaq:.6f}")
    print(f"Thermalization complete in {time.time() - t0:.1f}s")
    print()
    
    # Measurements
    print("Measuring observables...")
    plaquettes = []
    wilson_loops = {}
    polyakov_corr = {r: [] for r in range(1, L//2 + 1)}
    
    for i in range(n_measure):
        # Sweep between measurements
        for _ in range(measure_interval):
            sweep(lattice, beta, method='heatbath')
        
        # Measure plaquette
        plaq = average_plaquette(lattice)
        plaquettes.append(plaq)
        
        # Measure Wilson loops
        for R in range(1, min(L, 5)):
            for T_loop in range(1, min(T, 5)):
                W = wilson_loop(lattice, R, T_loop)
                if (R, T_loop) not in wilson_loops:
                    wilson_loops[(R, T_loop)] = []
                wilson_loops[(R, T_loop)].append(W)
        
        # Measure Polyakov correlators
        for r in range(1, L//2 + 1):
            C = polyakov_correlator(lattice, r)
            polyakov_corr[r].append(C)
        
        if (i + 1) % 20 == 0:
            print(f"  Measurement {i+1}/{n_measure}: <P> = {plaq:.6f}")
    
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    
    # Average plaquette
    plaq_mean = np.mean(plaquettes)
    plaq_err = np.std(plaquettes) / np.sqrt(len(plaquettes))
    print(f"Average plaquette: {plaq_mean:.6f} +/- {plaq_err:.6f}")
    
    # Wilson loops and string tension
    print("\nWilson loops W(R, T):")
    wilson_means = {}
    for (R, T_loop), values in sorted(wilson_loops.items()):
        mean = np.mean(values)
        err = np.std(values) / np.sqrt(len(values))
        wilson_means[(R, T_loop)] = mean
        print(f"  W({R}, {T_loop}) = {mean:.6f} +/- {err:.6f}")
    
    sigma, sigma_err = extract_string_tension(wilson_means)
    print(f"\nString tension (from Creutz ratio): sigma = {sigma:.4f} +/- {sigma_err:.4f}")
    
    # Polyakov correlators and mass gap
    print("\nPolyakov correlators <P(0)P(R)^dag>:")
    corr_means = []
    for r in range(1, L//2 + 1):
        if polyakov_corr[r]:
            mean = np.mean(polyakov_corr[r])
            err = np.std(polyakov_corr[r]) / np.sqrt(len(polyakov_corr[r]))
            corr_means.append(mean)
            print(f"  C({r}) = {mean:.6f} +/- {err:.6f}")
    
    m_gap, m_gap_err = extract_mass_gap(corr_means, T)
    print(f"\nMass gap (from Polyakov correlators): m = {m_gap:.4f}")
    
    # Check Giles-Teper bound
    if sigma > 0:
        bound = 2 * np.sqrt(np.pi / 3) * np.sqrt(sigma)
        print(f"\nGiles-Teper bound: Delta >= {bound:.4f}")
        print(f"Measured mass gap: Delta = {m_gap:.4f}")
        if m_gap > 0:
            print(f"Ratio Delta / sqrt(sigma) = {m_gap / np.sqrt(sigma):.4f}")
            print(f"(Expected: >= 2*sqrt(pi/3) = {2*np.sqrt(np.pi/3):.4f})")
    
    return {
        'plaquette': (plaq_mean, plaq_err),
        'wilson_loops': wilson_means,
        'string_tension': (sigma, sigma_err),
        'mass_gap': (m_gap, m_gap_err),
        'polyakov_correlators': corr_means
    }


# =============================================================================
# Run if executed directly
# =============================================================================

if __name__ == "__main__":
    # Test with small lattice
    print("Running test simulation...")
    print()
    
    # Strong coupling test (should show confinement)
    print("Strong coupling (beta = 1.0):")
    results_strong = run_simulation(L=4, T=4, beta=1.0, 
                                    n_thermalize=50, n_measure=50,
                                    measure_interval=5)
    print()
    
    # Intermediate coupling
    print("\n" + "=" * 60)
    print("Intermediate coupling (beta = 2.2):")
    results_inter = run_simulation(L=4, T=4, beta=2.2,
                                   n_thermalize=50, n_measure=50,
                                   measure_interval=5)
    print()
    
    # Weak coupling test
    print("\n" + "=" * 60)
    print("Weak coupling (beta = 4.0):")
    results_weak = run_simulation(L=4, T=4, beta=4.0,
                                  n_thermalize=50, n_measure=50,
                                  measure_interval=5)
