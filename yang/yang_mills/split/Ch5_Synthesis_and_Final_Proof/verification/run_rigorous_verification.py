"""
Rigorous Verification Driver
----------------------------
The main entry point for the corrected Computer-Assisted Proof.
Reads the JSON certificate and runs the Interval-based RGVerifier.
"""

import json
import argparse
from rg_step import RGVerifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert", required=True, help="Path to rigorous certificate JSON")
    args = parser.parse_args()
    
    try:
        with open(args.cert, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Certificate file not found.")
        return
        
    print(f"Loading Certificate: {data['id']}")
    print(f"Timestamp: {data['timestamp']}")
    print("-" * 50)
    
    verifier = RGVerifier(L=2.0)
    all_passed = True
    
    for step in data['steps']:
        print(f"Verifying Step {step['step_id']}...")
        if verifier.check_contraction(step):
            print(f"  [PASS] Step {step['step_id']} Contracted.")
        else:
            print(f"  [FAIL] Step {step['step_id']} Failed.")
            all_passed = False
            
    print("-" * 50)
    if all_passed:
        print("SUCCESS: Rigorous Interval Verification Passed.")
        print("The RG flow is certified stable under the provided certificate bounds.")
    else:
        print("FAILURE: Verification Failed.")
        
if __name__ == "__main__":
    main()
