"""
Adaptive Ball Covering Module for Phase 2.
"""
from typing import List, Tuple
import numpy as np
from .tube_definition import TubeDefinition
from ..interval_arithmetic.interval import Interval

class Ball:
    """
    Represents a ball B_i in the covering of the Tube.
    Defined by a center point (β_i, c_vec_i) and a radius r_i.
    """
    def __init__(self, beta: float, coeffs: List[float], radius: float):
        self.beta = beta
        self.coeffs = coeffs
        self.radius = radius # Radius of THIS ball, distinct from Tube radius r(beta)

    def __repr__(self):
        return f"Ball(β={self.beta:.4f}, r={self.radius:.4f})"

class BallCovering:
    """
    Manages the adaptive mesh of balls covering the Tube.
    """
    
    def __init__(self, tube: TubeDefinition):
        self.tube = tube
        self.balls: List[Ball] = []

    def generate_flow_based_covering(self, step_size: float = 0.1, L: int = 2):
        """
        Generates balls centered along the approximate RG trajectory.
        
        Algorithm:
        1. Start at β_min
        2. Set center c_vec = S_0(β) (Wilson action)
        3. Create ball
        4. Step β forward (β' = β + step)
        """
        current_beta = self.tube.beta_min
        
        while current_beta <= self.tube.beta_max:
            # Center at Wilson action for now (Phase 1 style)
            # In full Phase 2, this center follows the computed flow
            center_coeffs = self.tube.wilson_action(current_beta)
            
            # Ball radius is fraction of Tube radius to ensure containment
            tube_r = self.tube.radius(current_beta)
            ball_r = tube_r * 0.1 # Heuristic
            
            ball = Ball(current_beta, center_coeffs, ball_r)
            self.balls.append(ball)
            
            # Simple linear step for prototype
            current_beta += step_size
            
    def count(self) -> int:
        return len(self.balls)
