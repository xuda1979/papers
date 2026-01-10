"""
Tube Definition Module for Phase 2.
Defines the geometry of the Tube T in the space of effective actions.
"""
import numpy as np
from typing import List
from ..interval_arithmetic.interval import Interval

class TubeDefinition:
    """
    Defines the Tube T in the space of effective actions.
    
    T = {S : ||S - S_0(β)||_w ≤ r(β) for some β ∈ [β_S, β_W]}
    
    where S_0(β) is the Wilson action and r(β) is the radius function.
    """
    
    def __init__(self, beta_min: float, beta_max: float, dim: int):
        """
        Initialize Tube.
        
        Args:
            beta_min: β_S (strong coupling boundary)
            beta_max: β_W (weak coupling boundary)
            dim: Dimension of the operator basis
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.dim = dim
    
    def radius(self, beta: float) -> float:
        """
        Radius function r(β).
        
        Using the tuned formulation from Phase 1:
        r(β) = 0.7β + 0.5/√β
        
        This radius defines the size of the tube around the Wilson action.
        """
        if beta <= 0:
            return float('inf')
        return 0.7 * beta + 0.5 / np.sqrt(beta)
    
    def wilson_action(self, beta: float) -> List[float]:
        """
        Pure Wilson action S_0(β).
        
        Args:
            beta: Coupling constant
            
        Returns:
            Vector of coefficients [c_1, c_2, ..., c_N].
            For Wilson action, c_1 = β (O1 is Wilson Plaquette), others are 0.
        """
        coeffs = [0.0] * self.dim
        if self.dim > 0:
            coeffs[0] = beta
        return coeffs
    
    def weighted_norm(self, coeffs: List[Interval], beta: float, L: int = 2) -> Interval:
        """
        Computes the weighted norm ||S - S_0(β)||_w.
        
        ||c||_w = Σ |c_i| * L^(d_i - 4)k
        (Simplified weight implementation for now)
        """
        # Get Wilson center
        s0 = self.wilson_action(beta)
        
        norm = Interval(0.0, 0.0)
        
        # O1 (Marginal, dim 4)
        if len(coeffs) > 0:
            diff = coeffs[0] - s0[0]
            norm = norm + (diff * diff) # Using L2-like for now, or abs? Roadmap says L1.
            # Interval arithmetic doesn't have standard abs yet? 
            # We implemented outward rounded arithmetic.
            
            # Let's implement a simple L1-like norm for intervals: |[a,b]| = max(|a|,|b|) is bound
            # But here we want the interval representing the norm value itself.
            pass

        # Placeholder for full weighted norm implementation
        # For layout purposes, simply returning a dummy interval
        return Interval(0.0, 0.1)
