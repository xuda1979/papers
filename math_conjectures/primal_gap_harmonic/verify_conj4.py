"""
Conjecture 4: Power-of-Two Digit-Sum Squares
A = {n in N : S_10(2^n) = k^2 for some k in Z}
Find sharp bounds for |A ∩ [1,N]| = Theta(sqrt(N))
"""

import time
import sys

# Allow very large integer to string conversion (needed for 2^N with large N)
sys.set_int_max_str_digits(0)

def digit_sum(n):
    """Sum of decimal digits of n using string method (faster for big integers)"""
    return sum(int(d) for d in str(n))

def is_perfect_square(n):
    """Check if n is a perfect square"""
    if n < 0:
        return False
    root = int(n**0.5)
    return root * root == n

print("=" * 70)
print("CONJECTURE 4: Power-of-Two Digit-Sum Squares")
print("A = {n : S_10(2^n) is a perfect square}")
print("Target: Find sharp bounds for |A ∩ [1,N]| / sqrt(N)")
print("=" * 70)

max_N = 100_000_000  # 10^8 - will extend if time permits

# Track members of A and ratios
count = 0
min_ratio = float('inf')
max_ratio = 0.0
min_N_ratio = 0
max_N_ratio = 0

power_of_2 = 1

start_time = time.time()
checkpoint = 1_000_000

# First few members of A for verification
first_members = []

for n in range(1, max_N + 1):
    power_of_2 *= 2
    ds = digit_sum(power_of_2)
    
    if is_perfect_square(ds):
        count += 1
        if len(first_members) < 20:
            first_members.append((n, ds, int(ds**0.5)))
    
    if n >= 100:
        ratio = count / (n**0.5)
        if ratio < min_ratio:
            min_ratio = ratio
            min_N_ratio = n
        if ratio > max_ratio:
            max_ratio = ratio
            max_N_ratio = n
    
    if n % checkpoint == 0:
        ratio = count / (n**0.5)
        elapsed = time.time() - start_time
        print(f"  N = {n:>12,} | |A ∩ [1,N]| = {count:>6} | "
              f"ratio = {ratio:.6f} | min = {min_ratio:.6f} | max = {max_ratio:.6f} | {elapsed:.1f}s")
        sys.stdout.flush()

total_time = time.time() - start_time

print("\n" + "=" * 70)
print(f"RESULTS FOR N <= {max_N:,}:")
print("=" * 70)

print(f"\nFirst 20 members of A:")
for n, ds, k in first_members:
    print(f"  n = {n:>6}: digit_sum(2^{n}) = {ds:>4} = {k}^2")

print(f"\n|A ∩ [1,{max_N:,}]| = {count}")
print(f"sqrt({max_N:,}) = {max_N**0.5:.2f}")
print(f"Final ratio = {count / max_N**0.5:.10f}")

print("\n" + "=" * 70)
print("SHARP BOUND ANALYSIS (for N >= 100):")
print("=" * 70)
print(f"Minimum |A ∩ [1,N]| / sqrt(N) = {min_ratio:.10f} at N = {min_N_ratio:,}")
print(f"Maximum |A ∩ [1,N]| / sqrt(N) = {max_ratio:.10f} at N = {max_N_ratio:,}")

print("\n" + "=" * 70)
print("CONJECTURE (c) VERIFICATION:")
print("=" * 70)
print(f"Claimed bounds: 0.5 * sqrt(N) <= |A ∩ [1,N]| <= 1.2 * sqrt(N)")
print(f"Observed bounds: {min_ratio:.6f} * sqrt(N) <= |A ∩ [1,N]| <= {max_ratio:.6f} * sqrt(N)")

if min_ratio >= 0.5 and max_ratio <= 1.2:
    print("✓ Original bounds HOLD")
else:
    print("✗ Original bounds need adjustment")
    
print(f"\nSHARPEST BOUNDS from numerical evidence:")
print(f"  {min_ratio:.6f} * sqrt(N) <= |A ∩ [1,N]| <= {max_ratio:.6f} * sqrt(N)")

print(f"\nTotal computation time: {total_time:.1f}s ({total_time/60:.1f} min)")
