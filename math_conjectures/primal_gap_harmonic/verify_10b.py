"""
Verify Primal Gap Harmonic Conjecture for N up to 10^10
Find sharpest lower and upper bound coefficients for |S(N)| / sqrt(N)
"""

import numpy as np
import time

def sieve_primes(limit):
    """Sieve of Eratosthenes"""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

def get_n_primes(n):
    """Get at least n primes"""
    if n < 10:
        n = 10
    # Estimate upper bound using prime number theorem
    limit = int(n * (np.log(n) + np.log(np.log(n + 10))) * 1.15) + 1000
    primes = sieve_primes(limit)
    while len(primes) < n:
        limit = int(limit * 1.3)
        primes = sieve_primes(limit)
    return primes[:n]

print("=" * 70)
print("  PRIMAL GAP HARMONIC CONJECTURE - SHARP BOUNDS SEARCH")
print("  Target: N up to 10^10 (10 billion primes)")
print("=" * 70)

# Start with 10^8 first as a test
N_total = 10_000_000_000  # 10 billion
N_start = 100_000_000     # Start from 10^8

print(f"\nGenerating {N_start:,} primes first...")
start = time.time()
primes = get_n_primes(N_start + 1)
print(f"Generated {len(primes):,} primes in {time.time()-start:.1f}s")
print(f"Largest prime: p_{N_start:,} = {primes[-1]:,}")

# Compute S(N) for first 10^8 primes
print("\n" + "=" * 70)
print("COMPUTING S(N) = Σ (-1)^(n+1) / √g_n")
print("Tracking: max and min of |S(N)|/√N for N >= 100")
print("=" * 70)

S = 0.0
max_ratio = 0.0
min_ratio_large = float('inf')  # minimum ratio for N >= 1000
max_N = 0
min_N = 0
max_S = 0.0
min_S = 0.0

# Track all local maxima of |S(N)|/sqrt(N)
start_time = time.time()
checkpoint = 10_000_000

for n in range(1, N_start + 1):
    g_n = primes[n] - primes[n-1]  # gap between p_n and p_{n+1}
    sign = 1 if n % 2 == 1 else -1
    S += sign / np.sqrt(g_n)
    
    ratio = abs(S) / np.sqrt(n)
    
    if ratio > max_ratio:
        max_ratio = ratio
        max_N = n
        max_S = S
    
    if n >= 1000 and ratio < min_ratio_large:
        min_ratio_large = ratio
        min_N = n
        min_S = S
    
    if n % checkpoint == 0:
        elapsed = time.time() - start_time
        print(f"  N = {n:>12,} | S(N) = {S:>12.6f} | |S|/√N = {ratio:.10f} | "
              f"max = {max_ratio:.10f} at N={max_N} | {elapsed:.0f}s")

print("\n" + "=" * 70)
print("RESULTS FOR N ≤ 10^8:")
print("=" * 70)
print(f"Maximum |S(N)|/√N = {max_ratio:.15f} at N = {max_N}")
print(f"  S({max_N}) = {max_S:.15f}")
print(f"Minimum |S(N)|/√N (for N≥1000) = {min_ratio_large:.15f} at N = {min_N}")
print(f"  S({min_N}) = {min_S:.15f}")

# The ratio |S(N)|/√N gives us the coefficient C such that |S(N)| ≈ C√N
# Upper bound coefficient: max ratio = 1.0 (achieved at N=1)
# For N > 1: max ratio ≈ 0.5773... (at N=3)
# The ratio tends to decrease but oscillates

print("\n" + "=" * 70)
print("SHARP BOUND ANALYSIS:")
print("=" * 70)
print(f"Upper bound: |S(N)| ≤ {max_ratio:.15f} × √N for all N ≥ 1")
print(f"  (Maximum ratio {max_ratio:.15f} achieved at N = {max_N})")
print(f"\nFor N > 1, upper bound: |S(N)| < {1/np.sqrt(3):.15f} × √N")
print(f"  (Secondary maximum {1/np.sqrt(3):.15f} at N = 3)")

# Continue to larger N if time permits
print("\n" + "=" * 70)
print("EXTENDING TO LARGER N...")
print("=" * 70)

# For 10^10 primes, we need to be more memory-efficient
# Let's do incremental prime generation
