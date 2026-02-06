import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class PlasmaSystem:
    def __init__(self):
        # Simplified linearized MHD instability model
        # dx/dt = Ax + Bu
        # State x: [amplitude of instability, rate of change]
        self.A = np.array([[0, 1],
                           [20, -0.5]]) # Unstable eigenvalue (positive trace/det structure)
        self.B = np.array([[0],
                           [1]])
        self.state = np.array([0.1, 0.0]) # Initial small perturbation
        self.dt = 0.001 # 1ms timestep

    def step(self, u):
        # Euler integration for simulation
        dxdt = self.A @ self.state + self.B.flatten() * u
        self.state += dxdt * self.dt
        # Add some process noise (turbulence)
        self.state += np.random.normal(0, 0.01, size=2) * np.sqrt(self.dt)
        return self.state

class KoopmanController:
    def __init__(self):
        # LQR-like gain designed in the lifted space (simplified here to state feedback)
        # We assume we have identified the unstable mode via Operator Theory
        self.K = np.array([30.0, 10.0]) 

    def compute_control(self, state):
        # In a real system, 'state' would be topological features extracted from sensor data
        # u = -Kx
        return -np.dot(self.K, state)

def run_simulation():
    sim_time = 2.0 # seconds
    steps = int(sim_time / 0.001)
    
    plasma = PlasmaSystem()
    controller = KoopmanController()
    
    history_state = []
    history_control = []
    time = []

    print("Starting Real-Time MHD Stability Control Simulation...")
    
    for i in range(steps):
        t = i * 0.001
        
        # 1. Measure State (Simulating Sensor + Feature Extraction)
        current_state = plasma.state
        
        # 2. Compute Control (Real-time Operator Theoretic Prediction)
        if t > 0.5: # Activate control after 0.5s to show instability first
            u = controller.compute_control(current_state)
        else:
            u = 0.0
            
        # 3. Apply Actuation
        plasma.step(u)
        
        history_state.append(current_state.copy())
        history_control.append(u)
        time.append(t)

    history_state = np.array(history_state)
    history_control = np.array(history_control)

    print("Simulation complete. Generating plot...")

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, history_state[:, 0], label='Instability Amplitude')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Control On')
    plt.title('MHD Instability Suppression')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, history_control, label='Control Current (u)', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Current')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mhd_control_result.png')
    print("Results saved to mhd_control_result.png")

if __name__ == "__main__":
    run_simulation()
