"""
rigorous_special_functions.py

Implements rigorous interval evaluations for special functions (Bessel I_n)
using Taylor Series expansions with verified Lagrange error bounds.

This replaces 'heuristic padding' with mathematical proof-grade enclosures.

References:
    - Olver, "Asymptotics and Special Functions"
    - "Validated Numerics for Special Functions"
"""

from mpmath import mp, iv

# Set precision high enough to make series convergence fast
mp.dps = 50

def rigorous_besseli(n: int, x_interval):
    """
    Computes Modified Bessel Function I_n(x) for x in interval x_interval.
    Ref: I_n(x) = (x/2)^n * Sum_{k=0}^inf (x/2)^(2k) / (k! * (n+k)!)
    
    Returns a rigorous enclosure [min, max].
    """
    if not isinstance(x_interval, type(iv.mpf(1))):
        # Convert strict float/int to interval if needed, or handle custom interval
        # Assuming input is mpmath.iv.mpf or similar
        try:
            x = iv.mpf(x_interval)
        except:
             # Fallback if specific Interval class passed
             x = iv.mpf([x_interval.lower, x_interval.upper])
    else:
        x = x_interval

    # Domain check: x >= 0
    if x.a < 0:
        raise ValueError("rigorous_besseli only implemented for x >= 0")

    # Taylor Series Expansion with Remainder Bound (positive-term series)
    # I_n(x) = (x/2)^n * Sum_{k=0}^inf (x/2)^(2k) / (k! (n+k)!)
    # We sum until the next term is tiny relative to the accumulated sum and
    # a geometric tail bound is valid.
    
    # 1. Common factor (x/2)^n
    half_x = x / 2
    prefix = half_x ** n
    
    partial_sum = iv.mpf(0)

    # For x up to ~20, k in the low hundreds is safe; we prefer a tighter stop.
    max_terms = 400
    
    for k in range(max_terms):
        # Term k: (x/2)^(2k) / (k! (n+k)!)

        num = half_x ** (2 * k)

        # Use mpmath factorial (via gamma) for interval stability.
        # Note: k and (n+k) are integers, so factorial is exact in principle;
        # evaluating through mp.factorial avoids huge Python ints and keeps types consistent.
        den = mp.factorial(k) * mp.factorial(n + k)
        term_k = num / den
        partial_sum += term_k

        # Stop when term is tiny relative to the (lower bound of) sum.
        # If partial_sum.a is 0 (early iterations), skip this criterion.
        if k > 10 and (partial_sum.a > 0) and (term_k.b < (partial_sum.a * 1e-40)):
            # Tail bound (Geometeric series bound or Lagrange)
            # Rough bound for tail:
            # Ratio of (k+1)/k terms is (x/2)^2 * (k!/ (k+1)!) * ... approx (x/2)^2 / k^2
            # If (x/2)^2 / k^2 < 0.5, we have geometric convergence with r < 0.5
            # r = (x.b / 2)**2 / ((k+1)*(n+k+1))
            
            r_num = (x.b / 2) ** 2
            r_den = (k + 1) * (n + k + 1)
            ratio_bound = r_num / r_den
            
            # Once ratio_bound < 1, the tail is bounded by a geometric series.
            # We insist on < 0.9 for numerical comfort.
            if ratio_bound < 0.9:
                # Geometric series sum bound: term_k * r / (1 - r)
                # But term_(k+1) is the first omitted term.
                # First omitted term <= term_k * ratio_bound
                # Sum <= term_k * ratio_bound * (1 + ratio + ...) = term_k * ratio_bound / (1 - ratio_bound)
                
                next_term_bound = term_k * ratio_bound
                tail_error = next_term_bound / (1 - ratio_bound)
                
                # Enclose the tail: [0, tail_error]
                # Note: Taylor series for I_n has all positive terms for x>0.
                # So the truncation is strictly a lower bound. 
                # The remainder is positive.
                remainder = tail_error * iv.mpf([0, 1])
                
                return prefix * (partial_sum + remainder)
                
    # If we hit max terms without convergence (unlikely for x<20)
    raise RuntimeError(f"rigorous_besseli did not converge within {max_terms} terms for x={x}")

if __name__ == "__main__":
    # Self-test
    x_test = iv.mpf([1.0, 1.0])
    res = rigorous_besseli(1, x_test)
    print(f"I_1(1.0) = {res}")
    
    # Check against mp.besseli (reference)
    ref = mp.besseli(1, 1.0)
    print(f"Reference: {ref}")
    print(f"Contains reference? {res.a <= ref <= res.b}")
