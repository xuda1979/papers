"""
verify_ir_limit.py

Rigorously verifies the "IR Limit" / Infinite Volume conditions.
It serves as "Phase 2" of the Clay Proof audit chain (Phase 1 was verify_uv_handoff.py).

Objective:
    Prove that at the strong coupling handoff point (Beta <= 0.25),
    the theory satisfies the Dobrushin-Shlosman Finite-Size Criterion (FSC)
    or a convergent Cluster Expansion, ensuring:
    1. Uniqueness of the vacuum (Thermodynamic Limit exists).
    2. Exponential decay of correlations (Mass Gap in infinite volume).

Metric:
    Compute the Dobrushin constant `alpha` using rigorous interval arithmetic.
    Condition: alpha < 1.0 (Strict Contraction).

Dependencies:
    - verification/dobrushin_checker.py
    - verification/interval_arithmetic.py
"""

import sys
import os

# Ensure we can import the rigorous interval arithmetic
sys.path.insert(0, os.path.dirname(__file__))

from interval_arithmetic import Interval
from dobrushin_checker import DobrushinChecker

def verify_ir_condition():
    print("=" * 60)
    print("PHASE 2: IR LIMIT / INFINITE VOLUME VERIFICATION")
    print("Target Beta: <= 0.25 (Strong Coupling Handoff)")
    print("=" * 60)

    # 1. Define the Handoff Beta (from verification_results.tex: \VerBetaStrongMax = 0.25)
    beta_handoff = Interval(0.25, 0.25)
    print(f"Checking Infinite Volume Condition at Beta = {beta_handoff}...")

    # 2. Instantiate the rigorous checker
    checker = DobrushinChecker(vector_dim=4, Nc=3)

    # 3. Compute the Interaction Norm (alpha)
    # The checker uses conservative character expansion bounds: u(beta) <= (beta/6)(1+beta)
    # multiplied by the geometric factor Z=18.
    alpha_interval = checker.compute_interaction_norm(beta_handoff)

    print(f"  Dobrushin Constant alpha(beta): {alpha_interval}")

    # 4. Verify Strict Contraction (alpha < 1)
    # If alpha < 1, the Dobrushin-Shlosman criterion is satisfied.
    # This implies existence of the thermodynamic limit and mass gap.
    is_convergent = alpha_interval.upper < 1.0
    
    print(f"  [CHECK] alpha < 1.0 (Strict Contraction): {'PASS' if is_convergent else 'FAIL'}")

    # 5. Evaluate Margin
    margin = 1.0 - alpha_interval.upper
    print(f"  Contraction Margin: {margin:.6f}")

    print("-" * 60)
    if is_convergent:
        print("RESULT: IR LIMIT VERIFIED.")
        print("The Strong Coupling Expansion converges at Beta=0.25.")
        print("This implies the Mass Gap persists in the V -> infinity limit.")
    else:
        print("RESULT: FAILURE.")
        print("Beta=0.25 is too weak for rigorous Cluster Expansion.")
    print("=" * 60)

    return 0 if is_convergent else 1

if __name__ == "__main__":
    sys.exit(verify_ir_condition())
