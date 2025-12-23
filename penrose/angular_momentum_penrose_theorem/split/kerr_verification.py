"""
Kerr Black Hole AM-Penrose Inequality Verification
===================================================
This script computes exact numerical values for Kerr black holes
to verify the Angular Momentum Penrose Inequality:
    M_ADM >= sqrt(A/(16*pi) + 4*pi*J^2/A)

Output is used in sec11-numerical.tex
"""

import numpy as np

def kerr_params(M, a):
    """
    Compute Kerr horizon quantities from mass M and spin parameter a = J/M.
    
    Parameters:
        M: ADM mass
        a: spin parameter (|a| <= M for sub-extremal)
    
    Returns:
        dict with r_plus, A, J, and derived quantities
    """
    if abs(a) > M:
        raise ValueError(f"Super-extremal: |a|={abs(a)} > M={M}")
    
    r_plus = M + np.sqrt(M**2 - a**2)  # outer horizon radius
    A = 4 * np.pi * (r_plus**2 + a**2)  # horizon area
    J = M * a  # angular momentum
    
    return {
        'M': M,
        'a': a,
        'r_plus': r_plus,
        'A': A,
        'J': J
    }

def am_penrose_bound(A, J):
    """Compute the AM-Penrose bound sqrt(A/(16*pi) + 4*pi*J^2/A)."""
    return np.sqrt(A / (16 * np.pi) + 4 * np.pi * J**2 / A)

def verify_kerr(M, a):
    """
    Complete verification of AM-Penrose inequality for Kerr.
    
    Returns dict with all computed quantities.
    """
    params = kerr_params(M, a)
    A, J = params['A'], params['J']
    
    bound = am_penrose_bound(A, J)
    ratio = M / bound
    sub_ext_ratio = A / (8 * np.pi * abs(J)) if J != 0 else float('inf')
    
    return {
        **params,
        'bound': bound,
        'ratio': ratio,
        'sub_ext_ratio': sub_ext_ratio,
        'saturated': np.isclose(ratio, 1.0, rtol=1e-10)
    }

def print_example(M, a, label=""):
    """Print detailed verification for a specific case."""
    result = verify_kerr(M, a)
    
    print(f"\n{'='*60}")
    print(f"Kerr Black Hole: {label}")
    print(f"{'='*60}")
    print(f"Input parameters:")
    print(f"  M (ADM mass)     = {result['M']}")
    print(f"  a (spin param)   = {result['a']}")
    print(f"  J = M*a          = {result['J']}")
    print(f"\nDerived quantities:")
    print(f"  r_+ (horizon)    = {result['r_plus']:.10f}")
    print(f"  A (area)         = {result['A']:.10f}")
    print(f"  A/(pi)           = {result['A']/np.pi:.10f}")
    print(f"\nSub-extremality check:")
    print(f"  A/(8*pi*|J|)     = {result['sub_ext_ratio']:.10f}")
    print(f"  Sub-extremal?    = {result['sub_ext_ratio'] > 1}")
    print(f"\nAM-Penrose verification:")
    print(f"  Bound B          = {result['bound']:.15f}")
    print(f"  M_ADM            = {result['M']:.15f}")
    print(f"  Ratio M/B        = {result['ratio']:.15f}")
    print(f"  |Ratio - 1|      = {abs(result['ratio'] - 1):.2e}")
    print(f"  Saturated?       = {result['saturated']}")
    
    return result

def generate_table(spin_values):
    """Generate a table of results for multiple spin values."""
    print(f"\n{'='*80}")
    print("Table: Kerr AM-Penrose Verification (M = 1)")
    print(f"{'='*80}")
    print(f"{'a/M':<10} {'r_+':<12} {'A':<14} {'J':<10} {'B':<14} {'M/B':<18} {'A/(8πJ)':<12}")
    print("-"*80)
    
    results = []
    for a in spin_values:
        r = verify_kerr(1.0, a)
        results.append(r)
        sub_ext = f"{r['sub_ext_ratio']:.6f}" if r['J'] != 0 else "∞"
        print(f"{a:<10.4f} {r['r_plus']:<12.8f} {r['A']:<14.8f} {r['J']:<10.6f} "
              f"{r['bound']:<14.10f} {r['ratio']:<18.15f} {sub_ext:<12}")
    
    return results

def main():
    print("AM-Penrose Inequality Verification for Kerr Black Holes")
    print("="*60)
    print("Inequality: M_ADM >= sqrt(A/(16π) + 4πJ²/A)")
    print("For Kerr: equality holds (saturation)")
    
    # Example 1: M=1, a=0.6 (used in paper)
    r1 = print_example(1.0, 0.6, "M=1, a=0.6 (Paper Example 1)")
    
    # Example 2: M=1, a=0.999 (near-extremal, used in paper)
    r2 = print_example(1.0, 0.999, "M=1, a=0.999 (Paper Example 2, Near-Extremal)")
    
    # Example 3: M=1, a=0 (Schwarzschild limit)
    r3 = print_example(1.0, 0.0, "M=1, a=0 (Schwarzschild)")
    
    # Example 4: Extremal limit
    r4 = print_example(1.0, 0.9999, "M=1, a=0.9999 (Very Near-Extremal)")
    
    # Generate comprehensive table
    spin_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    generate_table(spin_values)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("All Kerr black holes satisfy M/B = 1 to machine precision.")
    print("This confirms the AM-Penrose inequality is SATURATED for Kerr.")
    print("\nKey values for paper:")
    print(f"  Example 1 (a=0.6):   r_+ = {r1['r_plus']:.1f}, A = {r1['A']/np.pi:.1f}π, ratio = {r1['ratio']:.10f}")
    print(f"  Example 2 (a=0.999): r_+ = {r2['r_plus']:.6f}, A = {r2['A']:.4f}, ratio = {r2['ratio']:.10f}")

if __name__ == "__main__":
    main()
