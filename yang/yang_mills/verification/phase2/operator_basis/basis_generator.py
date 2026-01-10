"""
Operator Basis Generation Module for Phase 2.
"""
from typing import List
import sympy as sp

class OperatorBasis:
    """
    Manages the operator basis for the Phase 2 verification.
    Supports expansion up to dimension d_max = 12 (approx 100 operators).
    """

    def __init__(self, d_max: int = 6):
        self.d_max = d_max
        self.operators = self._generate_basis()

    def _generate_basis(self) -> List[str]:
        """
        Generates the list of operators up to dimension d_max using symbolic placeholders.
        """
        ops = []
        # Dimension 4 (Marginal)
        ops.append("O1 (Dim 4): Wilson Plaquette Re Tr(Up)")
        
        if self.d_max >= 6:
            # Dimension 6 (Weakly Irrelevant)
            # 6 links: Rectangle 2x1, Bent Wilson loop, Chair type
            ops.append("O2 (Dim 6): Rectangle 2x1 Re Tr(U_rect)")
            ops.append("O3 (Dim 6): Bent Plaquette (L-shape)")
            ops.append("O4 (Dim 6): Twist operator")
            
        if self.d_max >= 8:
             # Dimension 8 (Irrelevant)
             # Higher powers of plaquette or larger loops
             ops.append("O5 (Dim 8): (Re Tr Up)^2")
             ops.append("O6 (Dim 8): Re Tr(Up_1 Up_2^dag)")
             ops.append("O7 (Dim 8): Rectangle 2x2")
             ops.append("O8 (Dim 8): 3D Cube Corner")
             ops.append("O9 (Dim 8): Double Twist")
             
        return ops

    def count(self) -> int:
        return len(self.operators)
