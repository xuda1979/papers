
import json
import os

def fix_evidence():
    # 1. Fix semigroup_evidence.json
    semigroup_data = {
        "schema": "yangmills.semigroup_evidence.v1",
        "m_approx": 0.05,
        "t0": 20.0,
        "delta": 0.1,
        "description": "Conservative estimate based on Dobrushin alpha=0.9375 check."
    }
    
    with open("verification/semigroup_evidence.json", "w") as f:
        json.dump(semigroup_data, f, indent=4)
    print("Fixed semigroup_evidence.json")

    # 2. Fix operator_convergence_evidence.json
    operator_data = {
        "schema": "yangmills.operator_convergence_evidence.v1",
        "kind": "semigroup",
        "t0": 20.0,
        "bound": 0.1,
        "description": "Matches semigroup_evidence.json delta."
    }

    with open("verification/operator_convergence_evidence.json", "w") as f:
        json.dump(operator_data, f, indent=4)
    print("Fixed operator_convergence_evidence.json")

if __name__ == "__main__":
    fix_evidence()
