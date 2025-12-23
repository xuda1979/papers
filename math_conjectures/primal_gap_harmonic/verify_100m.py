"""
Verification of 3 Number Theory Conjectures with 10^8 primes
"""

import numpy as np
import time
import sys

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    print(f"  Sieving up to {limit:,}...")
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

def get_n_primes(n):
    """Get at least n prime numbers."""
    if n < 10:
        n = 10
    # Prime counting function estimate
    limit = int(n * (np.log(n) + np.log(np.log(n + 10))) * 1.15) + 1000
    primes = sieve_primes(limit)
    while len(primes) < n:
        limit = int(limit * 1.3)
        primes = sieve_primes(limit)
    return primes[:n]

def main():
    N = 100_000_000  # 100 million primes
    print(f"=" * 70)
    print(f"  VERIFICATION WITH {N:,} PRIMES")
    print(f"=" * 70)
    
    print(f"\nGenerating {N:,} primes...")
    start = time.time()
    primes = get_n_primes(N + 1)
    gen_time = time.time() - start
    print(f"Generated {len(primes):,} primes in {gen_time:.1f}s")
    print(f"Largest prime: p_{N} = {primes[N-1]:,}")
    print(f"p_{N+1} = {primes[N]:,}")
    
    # =========================================================================
    # CONJECTURE 1: Golden-Phase Prime Spiral
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("CONJECTURE 1: Golden-Phase Prime Spiral")
    print(f"Z_N = Σ (1/p_n) · exp(i·φ·p_n), φ = (1+√5)/2")
    print(f"{'=' * 70}")
    
    phi = (1 + np.sqrt(5)) / 2
    Z = 0+0j
    max_Z = 0.0
    max_N_Z = 0
    
    start = time.time()
    chunk = 10_000_000
    for c in range(0, N, chunk):
        end = min(c + chunk, N)
        for i in range(c, end):
            p = primes[i]
            Z += (1.0/p) * np.exp(1j * phi * p)
            absZ = abs(Z)
            if absZ > max_Z:
                max_Z = absZ
                max_N_Z = i + 1
        elapsed = time.time() - start
        print(f"  {end:,} primes | max |Z_N| = {max_Z:.10f} at N={max_N_Z} | {elapsed:.0f}s")
    
    print(f"\n★ CONJECTURE 1 RESULT:")
    print(f"  Max |Z_N| = {max_Z:.15f}")
    print(f"  Maximum at N = {max_N_Z}")
    if max_N_Z == 2:
        print(f"  ✓ CONFIRMED: Maximum at N=2")
    else:
        print(f"  ✗ CHANGED: Maximum NOT at N=2!")
    
    # =========================================================================
    # CONJECTURE 2: Primal Gap Harmonic
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("CONJECTURE 2: Primal Gap Harmonic")
    print(f"S(N) = Σ (-1)^(n+1) / √g_n, where g_n = p_(n+1) - p_n")
    print(f"{'=' * 70}")
    
    gaps = np.diff(primes[:N+1])
    
    start = time.time()
    S = 0.0
    max_ratio = 0.0
    max_ratio_N = 0
    secondary_max = 0.0
    secondary_N = 0
    
    for n in range(1, N + 1):
        sign = 1 if n % 2 == 1 else -1
        S += sign / np.sqrt(gaps[n-1])
        ratio = abs(S) / np.sqrt(n)
        if ratio > max_ratio:
            max_ratio = ratio
            max_ratio_N = n
        if n > 1 and ratio > secondary_max:
            secondary_max = ratio
            secondary_N = n
        if n % 10_000_000 == 0:
            elapsed = time.time() - start
            print(f"  {n:,} gaps | max ratio = {max_ratio:.10f} at N={max_ratio_N} | {elapsed:.0f}s")
    
    print(f"\n★ CONJECTURE 2 RESULT:")
    print(f"  Max |S(N)|/√N = {max_ratio:.15f} at N={max_ratio_N}")
    print(f"  Secondary max = {secondary_max:.15f} at N={secondary_N}")
    if max_ratio_N == 1 and abs(max_ratio - 1.0) < 1e-10:
        print(f"  ✓ CONFIRMED: Maximum ratio = 1.0 at N=1")
    else:
        print(f"  ✗ CHANGED!")
    if secondary_N == 3:
        print(f"  ✓ CONFIRMED: Secondary max at N=3")
    
    # =========================================================================
    # CONJECTURE 3: Square-Root Phase Boundedness
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("CONJECTURE 3: Square-Root Phase Boundedness")
    print(f"W_N = Σ exp(2πi√p_n) / n")
    print(f"{'=' * 70}")
    
    start = time.time()
    W = 0+0j
    max_W = 0.0
    max_W_N = 0
    
    for n in range(1, N + 1):
        p = primes[n-1]
        W += np.exp(2j * np.pi * np.sqrt(p)) / n
        absW = abs(W)
        if absW > max_W:
            max_W = absW
            max_W_N = n
        if n % 10_000_000 == 0:
            elapsed = time.time() - start
            print(f"  {n:,} terms | max |W_N| = {max_W:.10f} at N={max_W_N} | {elapsed:.0f}s")
    
    W_final = W
    print(f"\n★ CONJECTURE 3 RESULT:")
    print(f"  Max |W_N| = {max_W:.15f} at N={max_W_N}")
    print(f"  W_∞ = {W_final.real:.10f} + {W_final.imag:.10f}i")
    print(f"  |W_∞| = {abs(W_final):.10f}")
    if max_W_N == 168:
        print(f"  ✓ CONFIRMED: Maximum at N=168")
    else:
        print(f"  ✗ CHANGED: Maximum at N={max_W_N}, not N=168!")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"N = {N:,} primes verified")
    print(f"\nConjecture 1 (Golden-Phase): max |Z_N| = {max_Z:.15f} at N={max_N_Z}")
    print(f"Conjecture 2 (Primal Gap):   max ratio = {max_ratio:.15f} at N={max_ratio_N}")
    print(f"Conjecture 3 (Square-Root):  max |W_N| = {max_W:.15f} at N={max_W_N}")

if __name__ == "__main__":
    main()
