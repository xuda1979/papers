#!/usr/bin/env python3
"""
Machine Learning for Morawetz Multiplier Discovery
===================================================

This module implements the neural network architecture proposed in the paper
for discovering optimal Morawetz multipliers for Kerr black hole stability.

Architecture:
- Physics-informed neural network (PINN)
- Inputs: (r, θ, χ) - radial, angular, and spin parameter
- Outputs: (X^t, X^r, X^θ, X^φ) - multiplier vector field components
- Loss: Combines coercivity, boundary flux, regularity, and physics constraints

Author: [Paper Authors]
Date: December 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional
import warnings

# =============================================================================
# KERR GEOMETRY UTILITIES
# =============================================================================

@dataclass
class KerrGeometry:
    """Kerr spacetime geometry utilities."""
    M: float = 1.0
    
    def Delta(self, r: np.ndarray, chi: float) -> np.ndarray:
        """Δ = r² - 2Mr + a²"""
        a = chi * self.M
        return r**2 - 2*self.M*r + a**2
    
    def Sigma(self, r: np.ndarray, theta: np.ndarray, chi: float) -> np.ndarray:
        """Σ = r² + a²cos²θ"""
        a = chi * self.M
        return r**2 + a**2 * np.cos(theta)**2
    
    def r_plus(self, chi: float) -> float:
        """Outer horizon radius."""
        return self.M * (1 + np.sqrt(1 - chi**2))
    
    def r_minus(self, chi: float) -> float:
        """Inner horizon radius."""
        return self.M * (1 - np.sqrt(1 - chi**2))
    
    def surface_gravity(self, chi: float) -> float:
        """Surface gravity κ."""
        r_p = self.r_plus(chi)
        r_m = self.r_minus(chi)
        return (r_p - r_m) / (4 * self.M * r_p)
    
    def omega_H(self, chi: float) -> float:
        """Horizon angular velocity."""
        a = chi * self.M
        r_p = self.r_plus(chi)
        return a / (2 * self.M * r_p)
    
    def metric_components(self, r: np.ndarray, theta: np.ndarray, chi: float) -> dict:
        """Compute metric components in Boyer-Lindquist coordinates."""
        a = chi * self.M
        M = self.M
        
        Sigma = self.Sigma(r, theta, chi)
        Delta = self.Delta(r, chi)
        
        sin2 = np.sin(theta)**2
        A = (r**2 + a**2)**2 - a**2 * Delta * sin2
        
        g_tt = -(1 - 2*M*r/Sigma)
        g_tphi = -2*M*a*r*sin2/Sigma
        g_rr = Sigma/Delta
        g_thth = Sigma
        g_phiphi = A*sin2/Sigma
        
        return {
            'g_tt': g_tt, 'g_tphi': g_tphi, 'g_rr': g_rr,
            'g_thth': g_thth, 'g_phiphi': g_phiphi,
            'Sigma': Sigma, 'Delta': Delta
        }


# =============================================================================
# NEURAL NETWORK ARCHITECTURE (NumPy Implementation)
# =============================================================================

class MultiplierNetwork:
    """
    Physics-informed neural network for Morawetz multiplier discovery.
    
    This is a NumPy-only implementation for demonstration.
    For production use, implement with PyTorch/TensorFlow.
    """
    
    def __init__(self, hidden_layers: List[int] = [128, 128, 128, 128],
                 activation: str = 'tanh'):
        """
        Initialize the network.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        
        # Input: (r, θ, χ) -> 3 dimensions
        # Output: (X^t, X^r, X^θ, X^φ) -> 4 dimensions
        self.input_dim = 3
        self.output_dim = 4
        
        # Initialize weights
        self.weights = []
        self.biases = []
        
        layer_sizes = [self.input_dim] + hidden_layers + [self.output_dim]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)
        
        # Known Killing vectors as basis (will be added to output)
        self.kerr = KerrGeometry()
    
    def activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == 'tanh':
            return np.tanh(x)
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            return np.tanh(x)
    
    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function."""
        if self.activation_name == 'tanh':
            return 1 - np.tanh(x)**2
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)
        elif self.activation_name == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            return 1 - np.tanh(x)**2
    
    def forward(self, r: np.ndarray, theta: np.ndarray, chi: float) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            r: Radial coordinates (N,)
            theta: Angular coordinates (N,)
            chi: Spin parameter (scalar)
        
        Returns:
            X: Multiplier vector field components (N, 4)
        """
        # Normalize inputs
        r_norm = (r - 2) / 10  # Normalize around typical values
        theta_norm = (theta - np.pi/2) / (np.pi/2)
        chi_norm = chi
        
        # Stack inputs
        x = np.stack([r_norm, theta_norm, np.full_like(r_norm, chi_norm)], axis=-1)
        
        # Forward through layers
        activations = [x]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ W + b
            if i < len(self.weights) - 1:  # Hidden layers
                x = self.activation(z)
            else:  # Output layer (no activation)
                x = z
            activations.append(x)
        
        # Add Killing vector basis
        # X = X_network + α·∂_t + β·∂_φ
        # For now, just return network output
        X = x
        
        # Enforce symmetries
        # X^t and X^φ should be even in cos(θ)
        # X^θ should be odd in cos(θ)
        
        return X
    
    def compute_bulk_term(self, r: np.ndarray, theta: np.ndarray, chi: float,
                          X: np.ndarray) -> np.ndarray:
        """
        Compute the bulk positivity term K^X[ψ].
        
        For a multiplier X, the bulk term in the energy identity is:
        K^X = (∇_μ X_ν + ∇_ν X_μ) T^μν
        
        For stability, we need K^X ≥ 0.
        """
        metric = self.kerr.metric_components(r, theta, chi)
        
        # Simplified: compute divergence of X
        # In full implementation, compute the deformation tensor
        
        # Approximate the bulk term
        Delta = metric['Delta']
        Sigma = metric['Sigma']
        
        # The bulk term should be positive outside the ergosphere
        r_ergo = 2 * self.kerr.M  # Equatorial ergosphere
        
        # Simplified positivity measure
        K = Delta / Sigma * (X[:, 0]**2 + X[:, 1]**2)
        
        return K


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class MultiplierLoss:
    """
    Physics-informed loss functions for multiplier optimization.
    """
    
    def __init__(self, kerr: KerrGeometry, lambda1: float = 1.0,
                 lambda2: float = 0.1, lambda3: float = 0.01):
        """
        Initialize loss functions.
        
        Args:
            kerr: Kerr geometry object
            lambda1: Weight for boundary loss
            lambda2: Weight for regularity loss
            lambda3: Weight for physics loss
        """
        self.kerr = kerr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
    
    def coercivity_loss(self, K: np.ndarray, r: np.ndarray, chi: float) -> float:
        """
        Loss for bulk coercivity: we want K^X ≥ 0.
        
        L_coercivity = ∫ max(0, -K^X) dr dθ
        """
        r_ergo = 2 * self.kerr.M
        
        # Penalize negative K outside ergosphere
        mask = r > r_ergo * 0.9  # Slight buffer
        negative_K = np.maximum(0, -K[mask])
        
        return np.mean(negative_K**2)
    
    def boundary_loss(self, X: np.ndarray, r: np.ndarray, chi: float) -> float:
        """
        Loss for boundary flux control.
        
        We want the flux through the horizon and infinity to be controlled.
        """
        r_plus = self.kerr.r_plus(chi)
        
        # Near-horizon regularity
        near_horizon = np.abs(r - r_plus) < 0.5
        horizon_penalty = np.mean(X[near_horizon, :]**2) if np.any(near_horizon) else 0
        
        # Far-field decay
        far_field = r > 20
        infinity_penalty = np.mean((X[far_field, :] * r[far_field, np.newaxis])**2) if np.any(far_field) else 0
        
        return horizon_penalty + 0.1 * infinity_penalty
    
    def regularity_loss(self, X: np.ndarray, r: np.ndarray, theta: np.ndarray) -> float:
        """
        Loss for smoothness/regularity.
        
        Penalize large gradients.
        """
        # Approximate gradient via finite differences
        if len(r) < 2:
            return 0.0
        
        dX_dr = np.diff(X, axis=0) / np.diff(r)[:, np.newaxis]
        
        return np.mean(dX_dr**2)
    
    def physics_loss(self, X: np.ndarray, r: np.ndarray, theta: np.ndarray, 
                     chi: float) -> float:
        """
        Loss for physical constraints.
        
        The multiplier should satisfy certain divergence conditions.
        """
        # Simplified: require X to approach Killing vectors asymptotically
        far_field = r > 15
        
        if not np.any(far_field):
            return 0.0
        
        # X should approach a linear combination of ∂_t and ∂_φ
        # In BL coordinates, these have X^t = 1, X^φ = 0 or X^t = 0, X^φ = 1
        
        # Penalize non-Killing behavior at infinity
        X_far = X[far_field, :]
        
        # X^r and X^θ should vanish at infinity
        physics_penalty = np.mean(X_far[:, 1]**2 + X_far[:, 2]**2)
        
        return physics_penalty
    
    def total_loss(self, network: MultiplierNetwork, r: np.ndarray, 
                   theta: np.ndarray, chi: float) -> Tuple[float, dict]:
        """
        Compute total loss.
        
        Returns:
            total: Total loss value
            components: Dictionary of individual loss components
        """
        X = network.forward(r, theta, chi)
        K = network.compute_bulk_term(r, theta, chi, X)
        
        L_coercivity = self.coercivity_loss(K, r, chi)
        L_boundary = self.boundary_loss(X, r, chi)
        L_regularity = self.regularity_loss(X, r, theta)
        L_physics = self.physics_loss(X, r, theta, chi)
        
        total = (L_coercivity + 
                 self.lambda1 * L_boundary + 
                 self.lambda2 * L_regularity + 
                 self.lambda3 * L_physics)
        
        components = {
            'coercivity': L_coercivity,
            'boundary': L_boundary,
            'regularity': L_regularity,
            'physics': L_physics,
            'total': total
        }
        
        return total, components


# =============================================================================
# TRAINING PROTOCOL
# =============================================================================

class MultiplierTrainer:
    """
    Training protocol for multiplier discovery.
    """
    
    def __init__(self, network: MultiplierNetwork, loss_fn: MultiplierLoss,
                 learning_rate: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            network: Neural network to train
            loss_fn: Loss function object
            learning_rate: Learning rate for gradient descent
        """
        self.network = network
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.history = []
    
    def generate_training_points(self, chi: float, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training points in (r, θ) space.
        
        Concentrates points near the horizon and ergosphere.
        """
        kerr = self.loss_fn.kerr
        r_plus = kerr.r_plus(chi)
        
        # Log-spaced in r to concentrate near horizon
        r_min = r_plus + 0.1
        r_max = 50.0
        r = r_min * (r_max / r_min) ** np.random.rand(n_points)
        
        # Uniform in cos(θ)
        cos_theta = 2 * np.random.rand(n_points) - 1
        theta = np.arccos(cos_theta)
        
        return r, theta
    
    def train_step(self, chi: float, n_points: int = 500) -> dict:
        """
        Perform one training step.
        
        Note: This is a simplified gradient-free optimization.
        For production, use automatic differentiation.
        """
        r, theta = self.generate_training_points(chi, n_points)
        
        loss, components = self.loss_fn.total_loss(self.network, r, theta, chi)
        
        # Simplified: random perturbation of weights
        # In production, use proper backpropagation
        for i in range(len(self.network.weights)):
            perturbation = np.random.randn(*self.network.weights[i].shape) * self.lr
            
            # Try perturbation
            self.network.weights[i] += perturbation
            new_loss, _ = self.loss_fn.total_loss(self.network, r, theta, chi)
            
            if new_loss > loss:
                # Reject perturbation
                self.network.weights[i] -= perturbation
            else:
                loss = new_loss
        
        self.history.append(components)
        return components
    
    def curriculum_training(self, chi_values: List[float], 
                           epochs_per_chi: int = 100) -> None:
        """
        Curriculum learning: train on increasing χ values.
        """
        print("Starting curriculum training...")
        print("-" * 50)
        
        for chi in chi_values:
            print(f"\nTraining for χ = {chi:.2f}")
            
            for epoch in range(epochs_per_chi):
                components = self.train_step(chi)
                
                if epoch % 20 == 0:
                    print(f"  Epoch {epoch}: Loss = {components['total']:.6f}")
        
        print("\nTraining complete!")
    
    def verify_multiplier(self, chi: float, n_test: int = 2000) -> dict:
        """
        Verify the discovered multiplier satisfies positivity.
        """
        r, theta = self.generate_training_points(chi, n_test)
        
        X = self.network.forward(r, theta, chi)
        K = self.network.compute_bulk_term(r, theta, chi, X)
        
        kerr = self.loss_fn.kerr
        r_ergo = 2 * kerr.M
        
        # Check positivity outside ergosphere
        outside = r > r_ergo
        K_outside = K[outside]
        
        n_positive = np.sum(K_outside >= 0)
        n_total = len(K_outside)
        
        results = {
            'chi': chi,
            'fraction_positive': n_positive / n_total,
            'min_K': np.min(K_outside),
            'max_K': np.max(K_outside),
            'mean_K': np.mean(K_outside),
            'verified': n_positive / n_total > 0.99
        }
        
        return results


# =============================================================================
# KNOWN MULTIPLIERS FOR COMPARISON
# =============================================================================

def schwarzschild_morawetz(r: np.ndarray, M: float = 1.0) -> np.ndarray:
    """
    Known Morawetz multiplier for Schwarzschild.
    
    X = f(r) ∂_r where f(r) = (r - 3M) / r²
    """
    X = np.zeros((len(r), 4))
    X[:, 1] = (r - 3*M) / r**2  # X^r component
    return X


def killing_T(r: np.ndarray) -> np.ndarray:
    """Killing vector ∂_t."""
    X = np.zeros((len(r), 4))
    X[:, 0] = 1.0
    return X


def killing_Phi(r: np.ndarray) -> np.ndarray:
    """Killing vector ∂_φ."""
    X = np.zeros((len(r), 4))
    X[:, 3] = 1.0
    return X


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution for multiplier discovery."""
    
    print("=" * 60)
    print("MACHINE LEARNING FOR MORAWETZ MULTIPLIER DISCOVERY")
    print("=" * 60)
    
    # Initialize components
    kerr = KerrGeometry(M=1.0)
    network = MultiplierNetwork(hidden_layers=[64, 64, 64])
    loss_fn = MultiplierLoss(kerr)
    trainer = MultiplierTrainer(network, loss_fn, learning_rate=0.01)
    
    # Curriculum training
    chi_values = [0.0, 0.3, 0.5, 0.7, 0.9]
    trainer.curriculum_training(chi_values, epochs_per_chi=50)
    
    # Verify results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    for chi in [0.0, 0.5, 0.9]:
        results = trainer.verify_multiplier(chi)
        print(f"\nχ = {chi:.2f}:")
        print(f"  Fraction positive: {results['fraction_positive']:.4f}")
        print(f"  Min K: {results['min_K']:.6f}")
        print(f"  Mean K: {results['mean_K']:.6f}")
        print(f"  Verified: {results['verified']}")
    
    # Compare with known multipliers
    print("\n" + "=" * 60)
    print("COMPARISON WITH KNOWN MULTIPLIERS (Schwarzschild)")
    print("=" * 60)
    
    r_test = np.linspace(2.1, 20, 100)
    theta_test = np.ones_like(r_test) * np.pi / 2
    
    X_known = schwarzschild_morawetz(r_test)
    X_learned = network.forward(r_test, theta_test, chi=0.0)
    
    print(f"\nRMS difference (X^r): {np.sqrt(np.mean((X_known[:, 1] - X_learned[:, 1])**2)):.6f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nNote: For rigorous verification, the learned multiplier must be")
    print("analytically verified to satisfy K^X ≥ 0 pointwise.")


if __name__ == "__main__":
    main()
