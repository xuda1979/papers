"""
Audit Script for Interval Arithmetic Correctness.
Demonstrates that the Interval class performs outward rounding.
"""
import sys
import os
import math

try:
    from interval_arithmetic import Interval
except ImportError:
    # Check current directory
    sys.path.append(os.path.dirname(__file__))
    from interval_arithmetic import Interval

def audit_outward_rounding():
    print("AUDIT: Checking Interval Arithmetic Outward Rounding...")
    
    # 1. Addition
    # 0.1 + 0.2 in float is usually 0.30000000000000004
    # We want to ensure we capture the true real value.
    
    i1 = Interval(1.0, 1.0)
    i3 = i1.div_interval(Interval(3.0, 3.0)) # 1/3
    
    print(f"1/3 Interval: {i3}")
    print(f"Lower < 1/3? {i3.lower < 1.0/3.0}") 
    # Note: 1.0/3.0 in float is already rounded. 
    # True test: 3 * i3 should contain 1.
    
    i_one_recon = i3 * Interval(3.0, 3.0)
    print(f"3 * (1/3): {i_one_recon}")
    
    if i_one_recon.lower <= 1.0 <= i_one_recon.upper:
        print("PASS: Identity containment verified.")
    else:
        print("FAIL: Identity containment lost.")

    # 2. NextAfter checks
    val = 1.0
    val_next = math.nextafter(val, math.inf)
    i_test = Interval(val, val) + Interval(0.0, 0.0) # Should be identity?
    
    # 3. Transcendentals
    # sqrt(2)^2 should contain 2
    sqrt2 = Interval(2.0, 2.0).sqrt()
    sq = sqrt2 * sqrt2
    print(f"sqrt(2)^2: {sq}")
    if sq.lower <= 2.0 <= sq.upper:
        print("PASS: sqrt(2)^2 contains 2.0")
    else:
        print("FAIL: sqrt(2)^2 does not contain 2.0")

    print("Audit Complete.")

if __name__ == "__main__":
    audit_outward_rounding()
