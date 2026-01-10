"""
Quick test of the Phase 1 verifier components
"""

import sys
sys.path.insert(0, r'C:\Users\Lenovo\papers\yang\yang_mills\verification')

try:
    from tube_verifier_phase1 import Interval, OperatorBasis, RGMap, TubeDefinition
    
    print("✓ Successfully imported all classes")
    
    # Test interval arithmetic
    I1 = Interval(1.0, 2.0)
    I2 = Interval(0.5, 1.5)
    I3 = I1 + I2
    print(f"✓ Interval arithmetic: {I1} + {I2} = {I3}")
    
    # Test operator basis
    dims = OperatorBasis.dimensions()
    print(f"✓ Operator dimensions: {dims}")
    
    # Test RG map
    rg = RGMap(L=2, N=3)
    print(f"✓ RG map initialized: L={rg.L}, N={rg.N}")
    
    # Test tube
    tube = TubeDefinition(0.3, 2.4, N=3)
    r = tube.radius(1.0)
    print(f"✓ Tube radius at β=1.0: {r:.4f}")
    
    print("\n" + "="*50)
    print("ALL COMPONENT TESTS PASSED")
    print("="*50)
    print("\nReady to run full verification!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
