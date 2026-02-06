import numpy as np
import matplotlib.pyplot as plt

class HamiltonianSystem:
    def __init__(self):
        # Simple Harmonic Oscillator as a proxy for particle in potential well
        # H = 0.5 * p^2 + 0.5 * q^2
        self.m = 1.0
        self.k = 1.0

    def V_prime(self, q):
        return self.k * q

    def T_prime(self, p):
        return p / self.m

    def energy(self, q, p):
        return 0.5 * self.k * q**2 + 0.5 * p**2 / self.m

def symplectic_euler(system, q0, p0, dt, steps):
    q = np.zeros(steps)
    p = np.zeros(steps)
    E = np.zeros(steps)
    
    q[0] = q0
    p[0] = p0
    E[0] = system.energy(q0, p0)
    
    for i in range(steps - 1):
        # Update p first (using current q)
        p[i+1] = p[i] - dt * system.V_prime(q[i])
        # Update q using new p
        q[i+1] = q[i] + dt * system.T_prime(p[i+1])
        
        E[i+1] = system.energy(q[i+1], p[i+1])
        
    return q, p, E

def explicit_euler(system, q0, p0, dt, steps):
    q = np.zeros(steps)
    p = np.zeros(steps)
    E = np.zeros(steps)
    
    q[0] = q0
    p[0] = p0
    E[0] = system.energy(q0, p0)
    
    for i in range(steps - 1):
        # Standard Euler: use current q and p for both
        p[i+1] = p[i] - dt * system.V_prime(q[i])
        q[i+1] = q[i] + dt * system.T_prime(p[i])
        
        E[i+1] = system.energy(q[i+1], p[i+1])
        
    return q, p, E

def run_comparison():
    system = HamiltonianSystem()
    q0 = 1.0
    p0 = 0.0
    dt = 0.1 # Relatively large step size to highlight differences
    steps = 1000
    
    print("Running Structure-Preserving Integrator Comparison...")
    
    q_sym, p_sym, E_sym = symplectic_euler(system, q0, p0, dt, steps)
    q_exp, p_exp, E_exp = explicit_euler(system, q0, p0, dt, steps)
    
    time = np.arange(steps) * dt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(q_sym, p_sym, label='Symplectic Euler')
    plt.plot(q_exp, p_exp, label='Explicit Euler', linestyle='--')
    plt.title('Phase Space Trajectory')
    plt.xlabel('Position (q)')
    plt.ylabel('Momentum (p)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(time, E_sym, label='Symplectic Euler')
    plt.plot(time, E_exp, label='Explicit Euler')
    plt.title('Total Energy over Time')
    plt.xlabel('Time')
    plt.ylabel('Energy (H)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('integrator_comparison.png')
    print("Comparison plot saved to integrator_comparison.png")
    print(f"Final Energy Error (Symplectic): {abs(E_sym[-1] - E_sym[0]):.6f}")
    print(f"Final Energy Error (Explicit):   {abs(E_exp[-1] - E_exp[0]):.6f}")

if __name__ == "__main__":
    run_comparison()
