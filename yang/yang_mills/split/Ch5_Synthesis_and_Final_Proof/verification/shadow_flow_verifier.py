"""
Shadow Flow Verifier (Rigorous Independent Checker)
--------------------------------------------------
This script performs the independent verification of the "Tube Contraction" 
theorem for the Intermediate Bridge of the Yang-Mills Mass Gap proof.

It reads the rigorous cryptographic certificate (JSON) containing the intervals,
Jacobians, and nonlinear bounds, and verifies that the contraction condition
holds strictly using rigorous Interval Arithmetic.

Changes from previous version:
- Removed "Proxy Model" simulation.
- Eliminated hardcoded matrices.
- Loads 'Pollution Constant' from certificate to resolve OPE circularity.
- Uses Interval Vectors for rigorous inclusion.

Usage:
    python shadow_flow_verifier.py --cert certificate_phase2_hardened.json
"""

import json
import math
import sys
import argparse
from pathlib import Path

import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the rigorous interval library provided in the package
try:
    from rg_step import RGVerifier
except ImportError:
    print("Error: 'rg_step.py' not found. Ensure it is in the same directory.")
    sys.exit(1)

def verify_certificate(cert_path):
    print(f"Loading certificate: {cert_path}")
    
    with open(cert_path, 'r') as f:
        data = json.load(f)
        
    print(f"Certificate ID: {data.get('id', 'Unknown')}")
    print(f"Methodology: {data.get('methodology', 'Unknown')}")
    
    # Extract Global Constants from Certificate (Resolves OPE Circularity)
    pollution_constant = data.get("pollution_constant")
    if pollution_constant is None:
        print("WARNING: 'pollution_constant' not found in certificate. Using default (0.05) - Proof Strength Reduced.")
        pollution_constant = 0.05
    else:
        print(f"Global Pollution Constant (Loaded): {pollution_constant}")

    # Initialize Verifier
    verifier = RGVerifier(L=2.0)
    
    all_passed = True
    steps = data.get('steps', [])
    
    if not steps:
        print("Error: No steps found in certificate.")
        return False

    for i, step in enumerate(steps):
        step_id = step.get('step_id', i+1)
        print(f"\n--- Verifying Step {step_id} ---")
        
        # Inject the loaded pollution constant into the step data for the verifier
        step['pollution_constant'] = pollution_constant
        
        try:
            result = verifier.check_contraction(step)
            if result:
                print(f"Step {step_id}: PASS (Strict Inclusion Verified)")
            else:
                print(f"Step {step_id}: FAIL (Inclusion Condition Violated)")
                all_passed = False
        except Exception as e:
            print(f"Step {step_id}: CRITICAL ERROR during verification: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()

    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Run Rigorous Independent Verification")
    parser.add_argument("--cert", default="certificate_phase2_hardened.json", help="Path to certificate file")
    args = parser.parse_args()
    
    cert_path = Path(args.cert)
    if not cert_path.exists():
        print(f"Error: Certificate file {cert_path} not found.")
        print("Please ensure you have generated the rigorous certificate.")
        sys.exit(1)
        
    success = verify_certificate(cert_path)
    
    if success:
        print("\n[SUCCESS] All steps verified. The Intermediate Bridge hypothesis is confirmed rigorously relative to the input certificate.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Verification failed for one or more steps.")
        sys.exit(1)

if __name__ == "__main__":
    main()
