"""
BKL Quantum Dynamics Simulation: Cosmological Billiards as Quantum Scramblers

This simulation computes the Lyapunov Exponent numerically and models the 
Mutual Information Decay based on the geometric stretching derived in:
"Cosmological Billiards as Quantum Scramblers: The BKL Singularity as a 
Breakdown of Holographic Error Correction"

Mathematical Framework:
- DeWitt Supermetric: Configuration space of scale factors β^a
- Lyapunov Connection: Divergence rate of geodesics ↔ Code fidelity decay
- Gauss Map: T(u) = 1/u - floor(1/u) with λ_L = π²/(6 ln 2)

Author: Da Xu
Affiliation: China Mobile Research Institute
"""

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')  # Force non-interactive backend
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib not found. Plotting will be disabled.")

class BKLQuantumDynamics:
    """
    Simulates the BKL chaotic billiard dynamics and its effect on 
    holographic quantum error correcting codes.
    
    The BKL (Belinski-Khalatnikov-Lifshitz) dynamics near a spacelike 
    singularity can be mapped to a chaotic billiard on hyperbolic space
    H²/Γ. This class tracks:
    1. The chaotic Kasner epoch evolution
    2. The Lyapunov exponent of the gravitational billiard
    3. The decay of holographic mutual information I(A:B)
    """
    
    def __init__(self, u_start, iterations):
        """
        Initialize the BKL quantum dynamics simulation.
        
        Args:
            u_start: Initial BKL parameter. Use transcendental numbers 
                     (e.g., π-3) to ensure ergodicity and avoid periodic orbits.
            iterations: Number of Kasner epochs to simulate.
        """
        self.u = u_start
        self.iterations = iterations
        self.history_u = []
        self.history_p = []  # Kasner exponents (p1, p2, p3)
        self.history_L = []  # Geodesic stretching factor (∝ Mutual Information)
        self.lyapunov_sum = 0
        
    def kasner_exponents(self, u):
        """
        Computes the Kasner exponents (p1, p2, p3) for the metric:
        
            ds² = -dt² + t^(2p1)dx² + t^(2p2)dy² + t^(2p3)dz²
        
        Subject to the Kasner constraints: Σp_i = Σp_i² = 1
        
        The exponents are parameterized by u ∈ (0,1) via:
            p1 = -u/(1+u+u²)      [contracting direction]
            p2 = (1+u)/(1+u+u²)   [expanding direction]  
            p3 = u(1+u)/(1+u+u²)  [expanding direction]
        
        Args:
            u: The BKL parameter
            
        Returns:
            Sorted list [p1, p2, p3] with p1 < p2 < p3
        """
        denom = 1 + u + u**2
        p = [-u/denom, (1+u)/denom, (u*(1+u))/denom]
        return sorted(p)

    def gauss_map_step(self, u):
        """
        The chaotic Gauss map: T(u) = 1/u - floor(1/u)
        
        This map corresponds to the reflection off the curvature wall 
        in the minisuperspace billiard. It is conjugate to the continued 
        fraction expansion and possesses a positive Lyapunov exponent:
        
            λ_L = π²/(6 ln 2) ≈ 2.37
        
        This is the key dynamical system underlying BKL chaos.
        
        Args:
            u: Current BKL parameter
            
        Returns:
            Next BKL parameter after wall reflection
        """
        if u == 0: 
            return 0
        val = 1.0 / u
        return val - np.floor(val)

    def run_dynamics(self):
        """
        Execute the full BKL dynamics simulation.
        
        This method:
        1. Evolves the Gauss map for self.iterations steps
        2. Computes Kasner exponents at each epoch
        3. Tracks geodesic stretching (mutual information decay)
        4. Accumulates Lyapunov exponent estimate
        
        The key insight: The negative Kasner exponent p_min causes 
        geodesic divergence in the t→0 limit. This divergence "tears 
        apart" the entanglement wedge, causing I(A:B) → 0.
        """
        u_curr = self.u
        log_stretch = 0  # Logarithm of total stretching factor
        
        for i in range(self.iterations):
            # 1. Physics: Calculate Kasner metric state
            # The Kasner formulas use u in (0, infinity), but the Gauss map produces u in (0,1).
            # For u in (0,1), we use 1/u to get the full parameter for exponent calculation.
            # If u is very small (or zero), it corresponds to u_full -> infinity, giving p=(0,0,1).
            if u_curr < 1e-12:
                # Limit u_full -> infinity implies p = [0, 0, 1]
                # We use a large number to approximate this behavior without overflow
                u_full = 1e12
            else:
                u_full = 1.0 / u_curr
            
            p = self.kasner_exponents(u_full)
            self.history_p.append(p)
            self.history_u.append(u_curr)
            
            # 2. Information Theory: Stretching of Geodesics
            # The paper hypothesizes that the information scrambling is driven 
            # by the chaotic divergence of the billiard trajectories.
            # The local divergence rate is given by the derivative of the Gauss map:
            # ln|T'(u)| = ln(1/u^2) = -2 ln(u)
            # This replaces the naive |p_min| estimate which lacks the proper time measure.
            if u_curr > 1e-10:
                stretch_factor = -2 * np.log(u_curr)
            else:
                stretch_factor = 2 * 10 * np.log(10)  # Cap at u ~ 1e-10
            
            # 3. Lyapunov Calculation from map derivative
            # For f(u) = 1/u mod 1: f'(u) = -1/u²
            # Lyapunov exponent = lim (1/N) Σ log|f'(u_n)|
            if u_curr > 1e-5:
                self.lyapunov_sum += np.log(np.abs(1.0/(u_curr**2)))
            
            log_stretch += stretch_factor
            
            # Mutual Information decay: I(A:B) ~ exp(-L)
            # where L is accumulated geodesic stretching
            self.history_L.append(np.exp(-log_stretch))
            
            # Evolve to next Kasner epoch
            u_curr = self.gauss_map_step(u_curr)
            
    def get_lyapunov_exponent(self):
        """
        Returns the numerically computed Lyapunov exponent.
        
        The theoretical value for the Gauss map is:
            λ_L = π²/(6 ln 2) ≈ 2.373
            
        This quantity controls the scrambling rate of the boundary CFT
        and the fidelity decay rate of the holographic code.
        """
        return self.lyapunov_sum / self.iterations


def run_simulation():
    """
    Main simulation routine with visualization.
    
    Generates three-panel plot:
    1. Kasner exponent oscillations (gravity side)
    2. Poincaré return map (chaotic attractor structure)
    3. Mutual information decay (quantum code failure)
    """
    print("Starting simulation...")
    # Start with transcendental number to ensure ergodicity
    # (avoid periodic orbits of rational starting points)
    sim = BKLQuantumDynamics(u_start=np.pi - 3, iterations=2000)
    sim.run_dynamics()

    # --- Analysis ---
    lyapunov = sim.get_lyapunov_exponent()
    theoretical_lyapunov = (np.pi**2) / (6 * np.log(2))  # ≈ 2.373

    print("=" * 60)
    print("BKL Quantum Dynamics: Cosmological Billiards as Scramblers")
    print("=" * 60)
    print(f"Calculated Lyapunov Exponent: {lyapunov:.4f}")
    print(f"Theoretical BKL limit (pi^2/6ln2): {theoretical_lyapunov:.4f}")
    print(f"Relative Error: {100*abs(lyapunov - theoretical_lyapunov)/theoretical_lyapunov:.2f}%")
    print("=" * 60)

    # --- Plotting ---
    if PLOT_AVAILABLE:
        plt.figure(figsize=(12, 10))
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')

        # Panel 1: Kasner Exponents (The Chaos - Gravity Side)
        p_history = np.array(sim.history_p)
        plt.subplot(3, 1, 1)
        plt.plot(p_history[:100, 0], label=r'$p_1$ (Squeezing)', color='red', linewidth=1.5)
        plt.plot(p_history[:100, 1], label=r'$p_2$', color='green', alpha=0.5)
        plt.plot(p_history[:100, 2], label=r'$p_3$', color='blue', alpha=0.5)
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.title(r'BKL Mixmaster Oscillations: Kasner Exponents (First 100 Epochs)', fontsize=12)
        plt.ylabel('Kasner Exponents', fontsize=11)
        plt.xlabel('Epoch Number', fontsize=11)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        # Panel 2: Poincaré Return Map (The Attractor)
        plt.subplot(3, 1, 2)
        plt.scatter(sim.history_u[:-1], sim.history_u[1:], s=0.5, color='black', alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Identity')
        plt.title(r'Poincaré Return Map: $u_{n+1}$ vs $u_n$ (Chaotic Attractor)', fontsize=12)
        plt.xlabel(r'$u_n$', fontsize=11)
        plt.ylabel(r'$u_{n+1}$', fontsize=11)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Panel 3: Holographic Mutual Information (Code Failure)
        plt.subplot(3, 1, 3)
        plt.semilogy(sim.history_L[:200], linewidth=2, color='purple', label=r'$I(A:B)$')
        plt.axhline(y=1e-5, color='orange', linestyle='--', linewidth=2, 
                    label='Knill-Laflamme Threshold')
        plt.fill_between(range(200), 1e-5, min(sim.history_L[:200]), 
                        alpha=0.2, color='red', label='Code Failure Region')
        plt.title(r'Holographic Mutual Information $I(A:B)$ Decay', fontsize=12)
        plt.ylabel('Recoverable Information (Log Scale)', fontsize=11)
        plt.xlabel('Kasner Epochs (Logarithmic Time towards Singularity)', fontsize=11)
        plt.legend(loc='upper right')
        plt.grid(True, which="both", ls="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig('bkl_qecc_advanced.png', dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: bkl_qecc_advanced.png")
        # plt.show()  # Commented out for non-interactive execution
    else:
        print("\nSkipping plot generation (matplotlib not available).")
    
    return sim, lyapunov


def analyze_scrambling_rate(sim):
    """
    Additional analysis: Compare gravitational scrambling to CFT bound.
    
    The MSS bound on chaos: λ ≤ 2πT/ℏ
    For the BKL billiard, we expect the scrambling rate to saturate
    this bound, indicating maximal chaos.
    """
    p_history = np.array(sim.history_p)
    
    # Compute time-averaged stretching
    p_min_avg = np.mean(np.abs(p_history[:, 0]))
    
    # Kolmogorov-Sinai entropy ≈ Lyapunov exponent for this system
    ks_entropy = sim.get_lyapunov_exponent()
    
    print("\n--- Scrambling Analysis ---")
    print(f"Average |p_min|: {p_min_avg:.4f}")
    print(f"Kolmogorov-Sinai Entropy: {ks_entropy:.4f} bits/epoch")
    print(f"Scrambling time scale: {1/ks_entropy:.4f} epochs")
    
    return ks_entropy


if __name__ == "__main__":
    sim, lyapunov = run_simulation()
    analyze_scrambling_rate(sim)
