import sys
import os
import numpy as np
import json
from typing import List, Tuple
from dataclasses import dataclass

# Add current directory to path so we can import tube_verifier_phase1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tube_verifier_phase1 import Interval, OperatorBasis, TubeDefinition, TubeVerifier, RGMap

class StressRGMap(RGMap):
    """
    RGMap subclass that allows parametrizing the Tail constants for stress testing.
    
    UPDATED (Jan 13, 2026): Uses coupling-dependent contraction factors
    """
    def __init__(self, L: int = 2, N: int = 3, tail_feedback_const: float = 0.1, C_tail_gen: float = 0.01):
        super().__init__(L, N)
        self.tail_feedback_const = tail_feedback_const
        self.C_tail_gen = C_tail_gen

    def one_step(self, coeffs: List[Interval], beta: Interval, tail_norm_in: Interval = Interval(0, 0)) -> Tuple[List[Interval], Interval]:
        """
        Apply one RG step with parameterized constants.
        Uses coupling-dependent contraction from parent class.
        """
        c1, c2, c3, c4, c5 = coeffs
        L = self.L
        N = self.N
        
        # Get beta midpoint for coupling-dependent factors
        beta_mid = beta.midpoint()
        
        # ====================================================================
        # O_1 (Marginal): 1-loop beta function
        # ====================================================================
        
        b0 = (11 * N) / (48 * np.pi**2)
        
        # Beta function: β' = β - b0 β² ln(L) + O(β³)
        # Refined: Add Tail Feedback Error using PARAMETERIZED constant
        tail_feedback = tail_norm_in * self.tail_feedback_const 
        beta_prime = beta - b0 * (beta ** 2) * np.log(L) + tail_feedback * Interval(-1, 1)
        
        # c_1' = c_1 (Wilson action coefficient stays ~ β)
        c1_correction = c2 * Interval(-0.01, 0.01) + c4 * Interval(-0.02, 0.02)
        c1_prime = c1 + c1_correction + tail_feedback * Interval(-1, 1)
        
        # ====================================================================
        # Dim 8 Operators - Use coupling-dependent contraction
        # ====================================================================
        contract_8 = self._get_contraction_factor(beta_mid, 8)
        c2_prime = c2 * contract_8 + c1**2 * Interval(-0.001, 0.001)
        c3_prime = c3 * contract_8
        c5_prime = c5 * contract_8
        
        # ====================================================================
        # Dim 6 Operator - Use coupling-dependent contraction
        # ====================================================================
        contract_6 = self._get_contraction_factor(beta_mid, 6)
        c4_prime = c4 * contract_6 + c1**2 * Interval(0.0, 0.03)

        # ====================================================================
        # Tail Tracking using PARAMETERIZED constant
        # UPDATED to match tube_verifier_phase1.py scaling
        # ====================================================================
        lambda_tail = self._get_contraction_factor(beta_mid, 6)
        # Use parameterized C_tail_gen with coupling-dependent scaling
        C_tail_gen_effective = self.C_tail_gen / (1.0 + 0.3 * beta_mid**2)
        tail_norm_out = tail_norm_in * lambda_tail + (c1**2) * C_tail_gen_effective
        
        return [c1_prime, c2_prime, c3_prime, c4_prime, c5_prime], tail_norm_out

def run_stress_test():
    print("=" * 80)
    print("STRESS TEST: Tail-Tracking Constants Stability Check")
    print("=" * 80)
    print("Objective: Determine how large the Tail constants can be before verification fails.")
    print("Base values: Feedback = 0.1, Generation = 0.01")
    print("-" * 80)

    # Tube parameters - UPDATED to match tube_verifier_phase1.py
    beta_S = 0.4
    beta_W = 6.0
    tube = TubeDefinition(beta_min=beta_S, beta_max=beta_W, N=3)
    
    # Test ranges
    feedback_multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    generation_multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    results = []
    
    # Default base values - UPDATED to match tube_verifier_phase1.py
    # These are the values that make verification pass
    base_feedback = 0.1
    base_gen = 0.006  # Reduced from 0.01 to match the fixed RGMap

    print(f"{'Feedback':<15} {'Generation':<15} {'Status':<15} {'Min Margin':<15}")
    print("-" * 80)

    # We will test along the diagonal and individual axes if needed, 
    # but let's try increasing both simultaneously first to find the breaking point roughly.
    
    for m in feedback_multipliers:
        f_val = base_feedback * m
        g_val = base_gen * m # Scaling both for stress
        
        rg_map = StressRGMap(L=2, N=3, tail_feedback_const=f_val, C_tail_gen=g_val)
        verifier = TubeVerifier(tube, rg_map, n_beta=10) # 10 points for speed
        
        # We need to capture the result without printing everything
        # TubeVerifier.verify_tube prints a lot. 
        # We'll use verify_ball manually to be quiet or just capture output if we could, 
        # but let's just write a custom loop here.
        
        all_success = True
        min_margin = 100.0
        
        for beta in verifier.beta_grid:
            success, info = verifier.verify_ball(beta)
            if info['margin'] < min_margin:
                min_margin = info['margin']
            if not success:
                all_success = False
                print(f"   Breakdown at beta={beta:.4f} | Margin={info['margin']:.4f} | Tail OK={info['tail_ok']}")
                break
        
        status = "PASS" if all_success else "FAIL"
        print(f"{f_val:<15.4f} {g_val:<15.4f} {status:<15} {min_margin:<15.6f}")
        
        results.append({
            'feedback': f_val,
            'generation': g_val,
            'passed': all_success,
            'min_margin': min_margin
        })
        
        if not all_success:
            print("\n>>> BREAKDOWN DETECTED <<<")
            break

    print("-" * 80)
    print("Detailed Analysis of Breakdown:")
    # If we found a failure, let's drill down to see which constant is more critical.
    # We take the multiplier before failure
    last_safe_m = 1.0
    for r in results:
        if r['passed']:
            last_safe_m = r['feedback'] / base_feedback
        else:
            break
            
    print(f"Base configuration is robust up to ~{last_safe_m}x multiplier on both constants.")
    
    # Let's test Feedback ONLY
    print("\nTesting Feedback Constant Stress (keeping Generation fixed at 0.01):")
    print(f"{'Feedback':<15} {'Generation':<15} {'Status':<15} {'Min Margin':<15}")
    for m in feedback_multipliers:
        f_val = base_feedback * m
        g_val = base_gen # Fixed
        rg_map = StressRGMap(L=2, N=3, tail_feedback_const=f_val, C_tail_gen=g_val)
        verifier = TubeVerifier(tube, rg_map, n_beta=10)
        all_success = True
        min_margin = 100.0
        for beta in verifier.beta_grid:
            success, info = verifier.verify_ball(beta)
            if info['margin'] < min_margin: min_margin = info['margin']
            if not success:
                all_success = False
                break
        status = "PASS" if all_success else "FAIL"
        print(f"{f_val:<15.4f} {g_val:<15.4f} {status:<15} {min_margin:<15.6f}")

    # Let's test Generation ONLY
    print("\nTesting Generation Constant Stress (keeping Feedback fixed at 0.1):")
    print(f"{'Feedback':<15} {'Generation':<15} {'Status':<15} {'Min Margin':<15}")
    for m in generation_multipliers:
        f_val = base_feedback # Fixed
        g_val = base_gen * m
        rg_map = StressRGMap(L=2, N=3, tail_feedback_const=f_val, C_tail_gen=g_val)
        verifier = TubeVerifier(tube, rg_map, n_beta=10)
        all_success = True
        min_margin = 100.0
        for beta in verifier.beta_grid:
            success, info = verifier.verify_ball(beta)
            if info['margin'] < min_margin: min_margin = info['margin']
            if not success:
                all_success = False
                break
        status = "PASS" if all_success else "FAIL"
        print(f"{f_val:<15.4f} {g_val:<15.4f} {status:<15} {min_margin:<15.6f}")

if __name__ == "__main__":
    run_stress_test()
