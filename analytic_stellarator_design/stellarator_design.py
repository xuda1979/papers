import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

class AnalyticStellarator:
    def __init__(self, R0=1.0, r_axis=0.05, nfp=2, iota=0.42):
        """
        Parameters:
        R0 : float
            Major Radius
        r_axis : float
            Axis excursion amplitude
        nfp : int
            Number of field periods
        iota : float
            Total Rotational Transform (not per period). 
            Note: This implementation assumes iota is total.
        """
        self.R0 = R0          # Major Radius
        self.r_axis = r_axis  # Axis excursion amplitude
        self.nfp = nfp        # Number of field periods
        self.iota = iota      # Total Rotational Transform

    def get_axis_properties(self, phi):
        """
        Returns r, r_prime, r_double_prime, r_triple_prime at given phi (scalar or array).
        Analytic derivatives for higher precision.
        """
        sin = np.sin
        cos = np.cos
        N = self.nfp
        
        # Helper variables for derivatives of the radial envelope
        # R_env = R0 + r_axis * cos(N*phi)
        Rb  = self.R0 + self.r_axis * cos(N * phi)
        Rb1 = -self.r_axis * N * sin(N * phi)
        Rb2 = -self.r_axis * (N**2) * cos(N * phi)
        Rb3 =  self.r_axis * (N**3) * sin(N * phi)

        # X component
        # X = Rb * cos(phi)
        X   = Rb * cos(phi)
        X1  = Rb1 * cos(phi) - Rb * sin(phi)
        X2  = Rb2 * cos(phi) - 2 * Rb1 * sin(phi) - Rb * cos(phi)
        X3  = Rb3 * cos(phi) - 3 * Rb2 * sin(phi) - 3 * Rb1 * cos(phi) + Rb * sin(phi)

        # Y component
        # Y = Rb * sin(phi)
        Y   = Rb * sin(phi)
        Y1  = Rb1 * sin(phi) + Rb * cos(phi)
        Y2  = Rb2 * sin(phi) + 2 * Rb1 * cos(phi) - Rb * sin(phi)
        Y3  = Rb3 * sin(phi) + 3 * Rb2 * cos(phi) - 3 * Rb1 * sin(phi) - Rb * cos(phi)

        # Z component
        # Z = -r_axis * sin(N * phi)
        Z   = -self.r_axis * sin(N * phi)
        Z1  = -self.r_axis * N * cos(N * phi)
        Z2  =  self.r_axis * (N**2) * sin(N * phi)
        Z3  =  self.r_axis * (N**3) * cos(N * phi)

        # Stack results
        # Handle both scalar and array inputs
        if np.isscalar(phi):
            r  = np.array([X, Y, Z])
            r1 = np.array([X1, Y1, Z1])
            r2 = np.array([X2, Y2, Z2])
            r3 = np.array([X3, Y3, Z3])
        else:
            r  = np.stack([X, Y, Z], axis=1)
            r1 = np.stack([X1, Y1, Z1], axis=1)
            r2 = np.stack([X2, Y2, Z2], axis=1)
            r3 = np.stack([X3, Y3, Z3], axis=1)

        return r, r1, r2, r3

    def compute_frenet_frame(self, phi_grid):
        """
        Computes the differential geometry: Tangent, Normal, Binormal, Curvature, Torsion.
        Uses analytic derivatives for maximum precision.
        """
        r, r1, r2, r3 = self.get_axis_properties(phi_grid)
        
        # Arc length derivative (Speed)
        s_prime = np.linalg.norm(r1, axis=1)
        
        # Tangent
        T = r1 / s_prime[:, None]
        
        # Binormal direction vector (unnormalized)
        B_raw = np.cross(r1, r2)
        B_norm = np.linalg.norm(B_raw, axis=1)
        B = B_raw / B_norm[:, None]
        
        # Normal
        N = np.cross(B, T)
        
        # Curvature: kappa = |r' x r''| / |r'|^3
        kappa = B_norm / (s_prime**3)
        
        # Torsion: tau = (r' x r'') . r''' / |r' x r''|^2
        # Note: (r' x r'') is B_raw
        numerator = np.einsum('ij,ij->i', B_raw, r3)
        denominator = B_norm**2
        tau = numerator / denominator

        return r, T, N, B, kappa, tau, s_prime

    def solve_riccati(self, phi_grid):
        """
        THE INNOVATION CORE:
        Solves the Riccati ODE to determine the Flux Surface Shape (sigma).
        d(sigma)/dphi = -i * (iota_eff) * sigma - drive * (1 + sigma^2)
        
        Now uses analytic evaluation of kappa/tau inside the solver.
        """
        
        def get_geom_params(phi):
            # Compute local geometry parameters analytically
            _, r1, r2, r3 = self.get_axis_properties(phi)
            
            sp = np.linalg.norm(r1)
            b_raw = np.cross(r1, r2)
            b_norm = np.linalg.norm(b_raw)
            
            k_val = b_norm / (sp**3)
            
            num = np.dot(b_raw, r3)
            den = b_norm**2
            t_val = num / den
            
            return k_val, t_val, sp

        def riccati_ode(phi, y):
            # y is [real(sigma), imag(sigma)]
            sigma = y[0] + 1j * y[1]
            
            k_val, t_val, sp_val = get_geom_params(phi)
            D_val = 2.0 * k_val * sp_val  # Curvature drive
            
            # The Master Equation for QA Construction
            # Assumption: iota is the total rotational transform. 
            # We approximate d(zeta)/d(phi) ~ 1 for the ODE integration variable.
            detuning = 2 * (self.iota - t_val * sp_val)
            d_sigma = -1j * detuning * sigma - 0.5 * D_val * (1 + sigma**2)
            
            return [d_sigma.real, d_sigma.imag]

        # Integrate ODE
        # Enforce periodicity: sigma(0) == sigma(2*pi)
        y0 = [0.0, 0.0] # Initial guess
        
        # We solve once over 0..2pi to get close, then iterate
        for _ in range(10): 
            sol = solve_ivp(riccati_ode, [0, 2*np.pi], y0, t_eval=phi_grid, rtol=1e-8, atol=1e-10)
            y_end = [sol.y[0][-1], sol.y[1][-1]]
            
            # Check convergence
            dist = np.sqrt((y_end[0]-y0[0])**2 + (y_end[1]-y0[1])**2)
            if dist < 1e-6:
                break
            y0 = y_end
        
        return sol.y[0] + 1j * sol.y[1]

    def generate_surface(self, n_phi=200, n_theta=60, minor_radius=0.15):
        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, 2 * np.pi, n_theta)
        
        # 1. Get Axis Geometry (for plotting mainly)
        r0, T, N, B, kappa, tau, s_prime = self.compute_frenet_frame(phi)
        
        # 2. Solve Inverse Problem (Get Sigma)
        # Note: solve_riccati now self-computes kappa/tau as needed
        sigma = self.solve_riccati(phi)
        
        # 3. Construct 3D Surface
        X_surf, Y_surf, Z_surf = [], [], []

        for i in range(n_phi):
            # Decode sigma into Ellipse Parameters
            # sigma = sinh(eta) * e^(i * 2*delta)
            # This mapping relates the Riccati variable to physical geometry
            s_val = sigma[i]
            
            # Limit for numerical stability in this demo
            if np.abs(s_val) > 0.95:
                warnings.warn(f"Sigma value clipped at phi index {i}. Configuration may be singular.", RuntimeWarning)
            s_abs = min(np.abs(s_val), 0.95) 
            
            # Standard definition relates sigma to aspect ratio of the ellipse
            elongation = (1 + s_abs) / (1 - s_abs)
            
            delta = 0.5 * np.angle(s_val) # Rotation angle

            # Build Ellipse in Normal Plane
            # Note: semi-axes are proportional to 1/sqrt(E) and sqrt(E)
            # resulting in aspect ratio sqrt(E)/ (1/sqrt(E)) = E
            c_d = np.cos(delta)
            s_d = np.sin(delta)
            
            # Parametric ring points
            th = theta
            u_loc = (minor_radius / np.sqrt(elongation)) * np.cos(th)
            v_loc = (minor_radius * np.sqrt(elongation)) * np.sin(th)
            
            # Rotate by delta
            u_rot = u_loc * c_d - v_loc * s_d
            v_rot = u_loc * s_d + v_loc * c_d
            
            # Map to 3D: r = r0 + u*N + v*B
            center = r0[i]
            n_vec = N[i]
            b_vec = B[i]
            
            ring = center[:, None] + np.outer(n_vec, u_rot) + np.outer(b_vec, v_rot)
            
            X_surf.append(ring[0, :])
            Y_surf.append(ring[1, :])
            Z_surf.append(ring[2, :])
            
        return np.array(X_surf), np.array(Y_surf), np.array(Z_surf), r0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize Generator
    gen = AnalyticStellarator(nfp=3, iota=0.48) # 3-period, high twist
    
    print("Solving Inverse Riccati Equation...")
    X, Y, Z, Axis = gen.generate_surface()
    
    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Surface
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, edgecolor='none')
    
    # Plot Magnetic Axis
    ax.plot(Axis[:,0], Axis[:,1], Axis[:,2], 'k-', linewidth=3, label='Magnetic Axis')
    
    # Settings
    ax.set_title("Analytic Quasi-Symmetric Stellarator\n(Generated via Inverse Riemannian Embedding)", fontsize=14)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    
    # Equal Aspect Ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.legend()
    plt.savefig('stellarator_plot.png')
    print("Geometry Generated and saved to stellarator_plot.png.")
