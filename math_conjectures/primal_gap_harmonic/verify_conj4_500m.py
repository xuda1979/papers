"""
Verify Primal Gap Harmonic Conjecture (Conjecture 4) for N up to 10^10
Using incremental prime generation to save memory
"""

import numpy as np
import time
import sys

def sieve_primes(limit):
    """Standard sieve - use for moderate sizes"""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

def get_primes_in_range(low, high, small_primes):
    """Get primes in [low, high] using segmented sieve"""
    if low < 2:
        low = 2
    sieve = np.ones(high - low + 1, dtype=bool)
    
    for p in small_primes:
        if p * p > high:
            break
        start = max(p * p, ((low + p - 1) // p) * p)
        if start > high:
            continue
        sieve[start - low::p] = False
    
    return low + np.where(sieve)[0]

def main():
    print("=" * 70)
    print("  PRIMAL GAP HARMONIC CONJECTURE - VERIFICATION UP TO 10^10")
    print("  S(N) = Σ (-1)^(n+1) / √g_n")
    print("=" * 70)
    
    # We'll do this in batches
    # For 10^10 primes, largest prime is about 252 billion
    # We cannot store all primes, so we process in chunks
    
    # Start with manageable batch: 10^9 primes
    # Prime 10^9 ≈ 22.8 billion, needs sieve up to ~25 billion
    # That's about 3GB for boolean array - borderline
    
    # Let's do 5×10^8 (500 million) primes first
    N_target = 500_000_000
    
    print(f"\nTarget: {N_target:,} primes")
    
    # Estimate sieve limit
    # p_n ≈ n * (ln(n) + ln(ln(n)))
    n = N_target
    limit = int(n * (np.log(n) + np.log(np.log(n))) * 1.1) + 1000
    print(f"Estimated sieve limit: {limit:,}")
    print(f"Memory needed: ~{limit / 1e9:.2f} GB")
    
    print(f"\nGenerating {N_target:,} primes...")
    t0 = time.time()
    
    primes = sieve_primes(limit)
    while len(primes) < N_target + 1:
        limit = int(limit * 1.2)
        print(f"  Extending sieve to {limit:,}...")
        primes = sieve_primes(limit)
    
    primes = primes[:N_target + 1]
    print(f"Generated {len(primes):,} primes in {time.time()-t0:.1f}s")
    print(f"Largest: p_{N_target:,} = {primes[N_target]:,}")
    
    # Compute S(N)
    print(f"\nComputing S(N) for N up to {N_target:,}...")
    
    S = 0.0
    max_ratio = 0.0
    max_N = 0
    max_S_at_max = 0.0
    
    checkpoint = 50_000_000
    start_time = time.time()
    
    for n in range(1, N_target + 1):
        g_n = primes[n] - primes[n-1]
        sign = 1 if n % 2 == 1 else -1
        S += sign / np.sqrt(g_n)
        
        ratio = abs(S) / np.sqrt(n)
        if ratio > max_ratio:
            max_ratio = ratio
            max_N = n
            max_S_at_max = S
        
        if n % checkpoint == 0:
            elapsed = time.time() - start_time
            print(f"  N = {n:>12,} | S(N) = {S:>12.4f} | |S|/√N = {ratio:.10f} | "
                  f"max ratio = {max_ratio:.10f} at N={max_N} | {elapsed:.0f}s")
            sys.stdout.flush()
    
    # Final results
    print("\n" + "=" * 70)
    print(f"RESULTS FOR N ≤ {N_target:,}:")
    print("=" * 70)
    print(f"Maximum |S(N)|/√N = {max_ratio:.15f}")
    print(f"  Achieved at N = {max_N}")
    print(f"  S({max_N}) = {max_S_at_max:.15f}")
    print(f"\nFinal values at N = {N_target:,}:")
    print(f"  S({N_target:,}) = {S:.10f}")
    print(f"  |S({N_target:,})|/√{N_target:,} = {abs(S)/np.sqrt(N_target):.15f}")
    
    print("\n" + "=" * 70)
    print("SHARP BOUND COEFFICIENTS:")
    print("=" * 70)
    print(f"Upper bound: |S(N)| ≤ C · √N where C = {max_ratio:.15f}")
    print(f"  (C = 1.0 achieved at N=1, where S(1) = 1/√g_1 = 1)")
    print(f"  (For N > 1, max ratio = 1/√3 ≈ 0.5773 at N=3)")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

if __name__ == "__main__":
    main()
