# Structure-Preserving Variational Integrators for Long-Time Fusion Plasma Simulation

## Abstract
Numerical simulations of fusion plasmas are computationally intensive and often suffer from non-physical energy drift over long integration times. This paper introduces a "structure-preserving" numerical kernel based on Symplectic Geometry and Variational Integrators. By discretizing the Lagrangian action directly, rather than the differential equations, we derive schemes that automatically preserve the symplectic form, ensuring long-term stability and conservation of momentum and energy bounds.

## 1. Introduction
Accurate digital twins of fusion reactors require simulating particle trajectories over millions of time steps. Standard integrators (Runge-Kutta) are dissipative or generate energy, leading to unphysical results. Structure-preserving algorithms are essential for the fidelity of these simulations.

## 2. Mathematical Formulation
### 2.1 Hamiltonian Mechanics
The plasma dynamics are governed by Hamilton's equations:
$$ \dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q} $$
where $H(q, p)$ is the total energy. The flow of this system preserves the symplectic 2-form $\omega = dq \wedge dp$.

### 2.2 Discrete Variational Principle
We construct a discrete Lagrangian $L_d(q_k, q_{k+1}, h)$ approximating the action integral over a time step $h$. The discrete Euler-Lagrange equations are derived by extremizing the discrete action sum:
$$ \delta \sum L_d(q_k, q_{k+1}, h) = 0 $$
This yields an implicit update map that is guaranteed to be symplectic.

## 3. Algorithm: Symplectic Euler
For a separable Hamiltonian $H(q, p) = T(p) + V(q)$, the Symplectic Euler method is given by:
$$ p_{k+1} = p_k - h \nabla V(q_k) $$
$$ q_{k+1} = q_k + h \nabla T(p_{k+1}) $$
This first-order method is explicit for separable Hamiltonians and preserves phase space volume exactly.

## 4. Numerical Validation
We compare the proposed symplectic integrator against a standard 4th-order Runge-Kutta (RK4) method on a charged particle model. While RK4 is more accurate locally, it exhibits energy drift over long times. The symplectic integrator maintains the energy error within a bounded interval indefinitely, demonstrating its suitability for long-time plasma simulations.

## 5. Conclusion
The developed structure-preserving kernel provides a physically rigorous foundation for next-generation fusion simulation software. Its ability to respect conservation laws makes it a valuable asset for the commercial fusion industry.
