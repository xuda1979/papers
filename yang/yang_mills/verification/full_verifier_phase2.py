"""
Phase 2 Complete Verifier Script.
Incorporates expanded operator basis and refined tube definitions.
"""
import sys
import os

# Add parent directory to path to allow importing from phase2
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from phase2.interval_arithmetic.interval import Interval
from phase2.operator_basis.basis_generator import OperatorBasis
from phase2.tube_geometry.tube_definition import TubeDefinition
from phase2.tube_geometry.ball_covering import BallCovering

def main():
    print("======================================================================")
    print("YANG-MILLS MASS GAP: PHASE 2 VERIFICATION INITIALIZATION")
    print("======================================================================")
    
    # 1. Initialize Basis
    basis = OperatorBasis(d_max=6)
    print(f"Initialized Operator Basis with {basis.count()} operators (Max Dim {basis.d_max})")
    for op in basis.operators:
        print(f"  - {op}")

    # 2. Initialize Tube and Covering
    print("\n[Tube Geometry]")
    tube = TubeDefinition(beta_min=0.3, beta_max=2.4, dim=basis.count())
    print(f"  Tube defined for β ∈ [{tube.beta_min}, {tube.beta_max}]")
    print(f"  Tube radius at β=2.4: r(2.4) = {tube.radius(2.4):.4f}")

    # 3. Generate Mesh
    print("\n[Mesh Generation]")
    covering = BallCovering(tube)
    covering.generate_flow_based_covering(step_size=0.2)
    print(f"  Generated {covering.count()} balls along the Wilson trajectory.")
    for i, ball in enumerate(covering.balls):
         print(f"  Ball {i+1}: {ball}")

    print("\n[Phase 2 Status] Framework initialized.")
    print("  - Operator Basis: DONE")
    print("  - Tube Geometry: DONE")
    print("  - Adaptive Mesh: STARTED")
    print("  - Symbolic Expansion: PENDING")
    print("======================================================================")

if __name__ == "__main__":
    main()
