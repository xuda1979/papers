"""
Neural Flow Dynamics Package
============================

A framework for simulating large-scale quantum dynamics using
continuous normalizing flows on tensor accelerators.

Key Features:
- Symplectic Flow Matching for unitarity preservation
- NPU-optimized Neural ODE solvers
- Sign-problem-free inference for quantum simulation

Author: Neural Flow Dynamics Team
"""

from .flow_matching import (
    FlowConfig,
    VelocityNetwork,
    FlowMatchingLoss,
    QuantumFidelityLoss,
    SymplecticGradient,
    ComplexLinear,
    FourierFeatures,
    sample_gaussian_complex,
    normalize_quantum_state
)

from .neural_ode import (
    ODEConfig,
    ODESolver,
    EulerSolver,
    RK4Solver,
    DOPRI5Solver,
    NeuralODE,
    odeint_adjoint
)

from .quantum_systems import (
    TransverseFieldIsing,
    HeisenbergXXZ,
    FermiHubbard2D,
    computational_basis_state,
    random_state,
    product_state,
    ghz_state,
    neel_state,
    entanglement_entropy
)

from .train import (
    TrainingConfig,
    QuantumDynamicsDataset,
    NeuralFlowDynamics,
    Trainer
)

__version__ = "0.1.0"
__author__ = "Neural Flow Dynamics Team"

__all__ = [
    # Flow Matching
    "FlowConfig",
    "VelocityNetwork", 
    "FlowMatchingLoss",
    "QuantumFidelityLoss",
    "SymplecticGradient",
    "ComplexLinear",
    "FourierFeatures",
    "sample_gaussian_complex",
    "normalize_quantum_state",
    
    # Neural ODE
    "ODEConfig",
    "ODESolver",
    "EulerSolver",
    "RK4Solver",
    "DOPRI5Solver",
    "NeuralODE",
    "odeint_adjoint",
    
    # Quantum Systems
    "TransverseFieldIsing",
    "HeisenbergXXZ",
    "FermiHubbard2D",
    "computational_basis_state",
    "random_state",
    "product_state",
    "ghz_state",
    "neel_state",
    "entanglement_entropy",
    
    # Training
    "TrainingConfig",
    "QuantumDynamicsDataset",
    "NeuralFlowDynamics",
    "Trainer"
]
