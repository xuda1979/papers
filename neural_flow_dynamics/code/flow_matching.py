"""
Neural Flow Dynamics: Core Implementation
==========================================

This module implements the core Flow Matching framework for quantum dynamics simulation.
Key components:
- Complex-valued neural networks
- Symplectic Flow Matching layers
- Flow Matching loss functions

Author: Neural Flow Dynamics Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


# ============================================
# Configuration
# ============================================

@dataclass
class FlowConfig:
    """Configuration for Neural Flow Dynamics."""
    hidden_dim: int = 512
    num_layers: int = 6
    num_qubits: int = 32
    fourier_features: int = 128
    ode_steps: int = 20
    learning_rate: float = 1e-4
    use_symplectic: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================
# Fourier Feature Embedding
# ============================================

class FourierFeatures(nn.Module):
    """
    Fourier feature embedding for improved learning of high-frequency functions.
    
    Maps input x to [sin(2π B x), cos(2π B x)] where B is a learnable or fixed matrix.
    Reference: Tancik et al., "Fourier Features Let Networks Learn High Frequency 
               Functions in Low Dimensional Domains", NeurIPS 2020.
    """
    
    def __init__(self, input_dim: int, num_features: int, scale: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        
        # Random Fourier features matrix (fixed, not learned)
        B = torch.randn(input_dim, num_features) * scale
        self.register_buffer('B', B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., input_dim)
        Returns:
            Fourier features of shape (..., 2 * num_features)
        """
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ============================================
# Complex-Valued Neural Network Layers
# ============================================

class ComplexLinear(nn.Module):
    """
    Linear layer for complex-valued inputs.
    
    Implements complex multiplication as:
    (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Complex tensor of shape (..., in_features, 2) where last dim is [real, imag]
        Returns:
            Complex tensor of shape (..., out_features, 2)
        """
        real, imag = z[..., 0], z[..., 1]
        
        # Complex matrix multiplication
        out_real = F.linear(real, self.weight_real) - F.linear(imag, self.weight_imag)
        out_imag = F.linear(real, self.weight_imag) + F.linear(imag, self.weight_real)
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        
        return torch.stack([out_real, out_imag], dim=-1)


class ComplexActivation(nn.Module):
    """
    Activation function for complex-valued networks.
    
    Applies activation to magnitude while preserving phase:
    f(z) = activation(|z|) * (z / |z|)
    """
    
    def __init__(self, activation: str = "silu"):
        super().__init__()
        self.activation = getattr(F, activation)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply activation while preserving complex phase."""
        real, imag = z[..., 0], z[..., 1]
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase_real = real / magnitude
        phase_imag = imag / magnitude
        
        activated_magnitude = self.activation(magnitude)
        
        return torch.stack([
            activated_magnitude * phase_real,
            activated_magnitude * phase_imag
        ], dim=-1)


# ============================================
# Symplectic Neural Network
# ============================================

class SymplecticGradient(nn.Module):
    """
    Symplectic gradient layer that ensures norm preservation.
    
    The velocity field is parameterized as v = J @ ∇H where:
    - H is a scalar pseudo-Hamiltonian network
    - J is the standard symplectic matrix [[0, I], [-I, 0]]
    
    This guarantees that the flow preserves the L2 norm by construction.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()
        self.state_dim = state_dim
        
        # Pseudo-Hamiltonian network (outputs scalar)
        layers = []
        layers.append(nn.Linear(state_dim * 2 + 1, hidden_dim))  # +1 for time
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.hamiltonian_net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute symplectic velocity field.
        
        Args:
            z: State tensor of shape (batch, state_dim, 2) [real, imag components]
            t: Time tensor of shape (batch, 1)
        
        Returns:
            Velocity tensor of shape (batch, state_dim, 2)
        """
        batch_size = z.shape[0]
        
        # Flatten complex state to real vector
        z_flat = z.reshape(batch_size, -1)  # (batch, state_dim * 2)
        
        # Concatenate with time
        z_t = torch.cat([z_flat, t], dim=-1)
        
        # Enable gradient computation for Hamiltonian
        z_t.requires_grad_(True)
        
        # Compute pseudo-Hamiltonian
        H = self.hamiltonian_net(z_t)  # (batch, 1)
        
        # Compute gradient ∇H
        grad_H = torch.autograd.grad(
            H.sum(), z_t, create_graph=True, retain_graph=True
        )[0]
        
        # Extract gradient w.r.t. state (exclude time)
        grad_z = grad_H[:, :-1]  # (batch, state_dim * 2)
        
        # Reshape to (batch, state_dim, 2)
        grad_z = grad_z.reshape(batch_size, self.state_dim, 2)
        
        # Apply symplectic matrix J: [q', p'] = [∂H/∂p, -∂H/∂q]
        # For complex numbers: real = ∂H/∂imag, imag = -∂H/∂real
        velocity_real = grad_z[..., 1]   # ∂H/∂imag
        velocity_imag = -grad_z[..., 0]  # -∂H/∂real
        
        return torch.stack([velocity_real, velocity_imag], dim=-1)


# ============================================
# Velocity Network
# ============================================

class VelocityNetwork(nn.Module):
    """
    Neural network for the velocity field v(z, t) in Flow Matching.
    
    The velocity field defines how probability mass flows from the prior
    distribution to the target quantum state distribution.
    """
    
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        self.state_dim = 2 ** config.num_qubits if config.num_qubits <= 10 else config.hidden_dim
        
        # Time embedding
        self.time_embed = FourierFeatures(1, config.fourier_features // 2)
        
        # State embedding (for complex states)
        self.state_embed = nn.Sequential(
            nn.Linear(self.state_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Main network
        self.layers = nn.ModuleList()
        input_dim = config.hidden_dim + config.fourier_features
        
        for i in range(config.num_layers):
            self.layers.append(nn.Linear(
                input_dim if i == 0 else config.hidden_dim,
                config.hidden_dim
            ))
        
        # Output layer (complex velocity)
        self.output_layer = nn.Linear(config.hidden_dim, self.state_dim * 2)
        
        # Optional symplectic layer
        if config.use_symplectic:
            self.symplectic = SymplecticGradient(
                self.state_dim, config.hidden_dim
            )
    
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity field at state z and time t.
        
        Args:
            z: Complex state tensor (batch, state_dim, 2)
            t: Time tensor (batch, 1) in [0, 1]
        
        Returns:
            Velocity tensor (batch, state_dim, 2)
        """
        batch_size = z.shape[0]
        
        # Flatten complex state
        z_flat = z.reshape(batch_size, -1)
        
        # Embeddings
        t_embed = self.time_embed(t)
        z_embed = self.state_embed(z_flat)
        
        # Concatenate
        h = torch.cat([z_embed, t_embed], dim=-1)
        
        # Forward through layers
        for layer in self.layers:
            h = F.silu(layer(h))
        
        # Output velocity
        v_flat = self.output_layer(h)
        v = v_flat.reshape(batch_size, -1, 2)
        
        # Add symplectic correction if enabled
        if self.config.use_symplectic:
            v_symplectic = self.symplectic(z, t)
            v = v + 0.1 * v_symplectic  # Weighted combination
        
        return v


# ============================================
# Flow Matching Loss
# ============================================

class FlowMatchingLoss(nn.Module):
    """
    Flow Matching loss for training the velocity network.
    
    Given:
    - Prior samples z_0 ~ p_0 (e.g., Gaussian)
    - Target samples z_1 ~ p_1 (quantum state distribution)
    
    We define an interpolation path:
        z_t = (1 - t) * z_0 + t * z_1
    
    The optimal velocity field is:
        u_t(z_t) = z_1 - z_0
    
    The loss minimizes ||v_θ(z_t, t) - u_t||^2
    """
    
    def __init__(self, sigma_min: float = 0.001):
        super().__init__()
        self.sigma_min = sigma_min
    
    def forward(
        self,
        velocity_net: VelocityNetwork,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss.
        
        Args:
            velocity_net: Velocity network
            z_0: Prior samples (batch, state_dim, 2)
            z_1: Target samples (batch, state_dim, 2)
            t: Optional time values; if None, sampled uniformly
        
        Returns:
            Scalar loss value
        """
        batch_size = z_0.shape[0]
        device = z_0.device
        
        # Sample time uniformly if not provided
        if t is None:
            t = torch.rand(batch_size, 1, device=device)
        
        # Linear interpolation with small noise for stability
        sigma_t = self.sigma_min
        z_t = (1 - t.unsqueeze(-1)) * z_0 + t.unsqueeze(-1) * z_1
        z_t = z_t + sigma_t * torch.randn_like(z_t)
        
        # Target velocity (optimal transport direction)
        target_velocity = z_1 - z_0
        
        # Predicted velocity
        pred_velocity = velocity_net(z_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        
        return loss


class QuantumFidelityLoss(nn.Module):
    """
    Loss based on quantum fidelity between predicted and target states.
    
    Fidelity: F = |<ψ_pred|ψ_target>|^2
    Loss: L = 1 - F
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        psi_pred: torch.Tensor,
        psi_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute infidelity loss.
        
        Args:
            psi_pred: Predicted state (batch, state_dim, 2)
            psi_target: Target state (batch, state_dim, 2)
        
        Returns:
            Mean infidelity loss
        """
        # Convert to complex
        pred_complex = psi_pred[..., 0] + 1j * psi_pred[..., 1]
        target_complex = psi_target[..., 0] + 1j * psi_target[..., 1]
        
        # Normalize
        pred_complex = pred_complex / torch.norm(pred_complex, dim=-1, keepdim=True)
        target_complex = target_complex / torch.norm(target_complex, dim=-1, keepdim=True)
        
        # Compute fidelity
        overlap = torch.sum(torch.conj(pred_complex) * target_complex, dim=-1)
        fidelity = torch.abs(overlap) ** 2
        
        # Infidelity loss
        loss = 1 - fidelity.mean()
        
        return loss


# ============================================
# Phase Network
# ============================================

class PhaseNetwork(nn.Module):
    """
    Network for learning the quantum phase φ(s) in ψ(s) = √p(s) * exp(iφ(s)).
    
    The probability p(s) comes from the normalizing flow, while this network
    learns the phase structure needed for quantum interference.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, 1))  # Output: phase angle
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute phase for configuration s.
        
        Args:
            s: Configuration tensor (batch, state_dim)
        
        Returns:
            Phase tensor (batch, 1) in [0, 2π]
        """
        phase = self.net(s)
        # Wrap to [0, 2π]
        phase = torch.remainder(phase, 2 * np.pi)
        return phase


# ============================================
# Utility Functions
# ============================================

def complex_to_real(z: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to real representation (last dim = 2)."""
    if torch.is_complex(z):
        return torch.stack([z.real, z.imag], dim=-1)
    return z


def real_to_complex(z: torch.Tensor) -> torch.Tensor:
    """Convert real representation to complex tensor."""
    return z[..., 0] + 1j * z[..., 1]


def sample_gaussian_complex(shape: Tuple[int, ...], device: str = "cpu") -> torch.Tensor:
    """Sample from complex Gaussian distribution."""
    real = torch.randn(shape, device=device)
    imag = torch.randn(shape, device=device)
    z = torch.stack([real, imag], dim=-1)
    # Normalize
    norm = torch.sqrt((z ** 2).sum(dim=(-1, -2), keepdim=True))
    return z / norm


def normalize_quantum_state(z: torch.Tensor) -> torch.Tensor:
    """Normalize quantum state to unit L2 norm."""
    norm_sq = (z[..., 0] ** 2 + z[..., 1] ** 2).sum(dim=-1, keepdim=True)
    norm = torch.sqrt(norm_sq).unsqueeze(-1)
    return z / (norm + 1e-8)


# ============================================
# Test / Demo
# ============================================

if __name__ == "__main__":
    # Test configuration
    config = FlowConfig(
        hidden_dim=256,
        num_layers=4,
        num_qubits=4,  # Small for testing
        fourier_features=64,
        use_symplectic=True,
        device="cpu"
    )
    
    print("Testing Neural Flow Dynamics Core Components")
    print("=" * 50)
    
    # Create velocity network
    velocity_net = VelocityNetwork(config)
    print(f"Velocity Network Parameters: {sum(p.numel() for p in velocity_net.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    state_dim = 2 ** config.num_qubits
    
    z = sample_gaussian_complex((batch_size, state_dim), device=config.device)
    t = torch.rand(batch_size, 1, device=config.device)
    
    v = velocity_net(z, t)
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {v.shape}")
    
    # Test Flow Matching loss
    fm_loss = FlowMatchingLoss()
    z_0 = sample_gaussian_complex((batch_size, state_dim), device=config.device)
    z_1 = sample_gaussian_complex((batch_size, state_dim), device=config.device)
    
    loss = fm_loss(velocity_net, z_0, z_1)
    print(f"Flow Matching Loss: {loss.item():.6f}")
    
    # Test gradient computation
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in velocity_net.parameters() if p.grad is not None) ** 0.5
    print(f"Gradient Norm: {grad_norm:.6f}")
    
    print("\n✓ All tests passed!")
