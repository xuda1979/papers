import numpy as np
import matplotlib.pyplot as plt

# Generate primes
def sieve_primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.where(sieve)[0]

primes = sieve_primes(200000)
print(f'Generated {len(primes)} primes')

# Figure 1: Primal Gap Harmonic Sum (Conjecture 1)
print('Generating Figure 1: Primal Gap Harmonic Sum...')
N_plot = 10000
S_vals = []
S = 0.0
for n in range(1, N_plot + 1):
    g_n = primes[n] - primes[n-1]
    sign = 1 if n % 2 == 1 else -1
    S += sign / np.sqrt(g_n)
    S_vals.append(S)

plt.figure(figsize=(12, 6))
plt.plot(range(1, N_plot + 1), S_vals, 'b-', linewidth=0.5)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
N_arr = np.arange(1, N_plot + 1)
plt.fill_between(N_arr, -np.sqrt(N_arr), np.sqrt(N_arr), alpha=0.2, color='red', label=r'$\pm\sqrt{N}$ bound')
plt.xlabel('N', fontsize=12)
plt.ylabel('S(N)', fontsize=12)
plt.title(r'Primal Gap Harmonic Sum: $S(N) = \sum_{n=1}^{N} (-1)^{n+1}/\sqrt{g_n}$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_conjecture_1.png', dpi=150)
plt.close()
print('  Saved fig_conjecture_1.png')

# Figure 2: Golden-Phase Prime Spiral (Conjecture 2)
print('Generating Figure 2: Golden-Phase Prime Spiral...')
phi = (1 + np.sqrt(5)) / 2
N_plot = 5000
Z_vals = []
Z = 0+0j
for i in range(N_plot):
    p = primes[i]
    Z += (1.0/p) * np.exp(1j * phi * p)
    Z_vals.append(Z)

Z_vals = np.array(Z_vals)
plt.figure(figsize=(8, 8))
plt.plot(Z_vals.real, Z_vals.imag, 'b-', linewidth=0.5, alpha=0.7)
plt.plot(Z_vals[0].real, Z_vals[0].imag, 'go', markersize=10, label='Start (N=1)')
plt.plot(Z_vals[1].real, Z_vals[1].imag, 'r*', markersize=15, label=f'Max at N=2, |Z|={abs(Z_vals[1]):.4f}')
plt.xlabel('Re(Z)', fontsize=12)
plt.ylabel('Im(Z)', fontsize=12)
plt.title(r'Golden-Phase Prime Spiral: $Z_N = \sum_{n=1}^{N} p_n^{-1} e^{i\phi p_n}$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('fig_conjecture_2.png', dpi=150)
plt.close()
print('  Saved fig_conjecture_2.png')

# Figure 3: Square-Root Phase Sum (Conjecture 3 - Square-Root Phase Boundedness)
print('Generating Figure 3: Square-Root Phase Boundedness...')
N_plot = 5000
W_vals = []
W = 0+0j
for n in range(1, N_plot + 1):
    p = primes[n-1]
    W += np.exp(2j * np.pi * np.sqrt(p)) / n
    W_vals.append(W)

W_vals = np.array(W_vals)
plt.figure(figsize=(8, 8))
plt.plot(W_vals.real, W_vals.imag, 'b-', linewidth=0.5, alpha=0.7)
plt.plot(W_vals[0].real, W_vals[0].imag, 'go', markersize=10, label='Start (N=1)')
plt.plot(W_vals[167].real, W_vals[167].imag, 'r*', markersize=15, label=f'Max at N=168, |W|={abs(W_vals[167]):.4f}')
plt.plot(W_vals[-1].real, W_vals[-1].imag, 'ms', markersize=8, label=f'N={N_plot}')
plt.xlabel('Re(W)', fontsize=12)
plt.ylabel('Im(W)', fontsize=12)
plt.title(r'Square-Root Phase Sum: $W_N = \sum_{n=1}^{N} n^{-1} e^{2\pi i\sqrt{p_n}}$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('fig_conjecture_3.png', dpi=150)
plt.close()
print('  Saved fig_conjecture_3.png')

# Figure 4: Power-of-Two Digit-Sum Squares (Conjecture 4)
print('Generating Figure 4: Power-of-Two Digit-Sum Squares...')

def digit_sum(n):
    '''Sum of decimal digits of n'''
    s = 0
    while n > 0:
        s += n % 10
        n //= 10
    return s

def is_perfect_square(n):
    '''Check if n is a perfect square'''
    if n < 0:
        return False
    root = int(n**0.5)
    return root * root == n

# Find values of n where digit_sum(2^n) is a perfect square
N_max = 10000  # Compute up to 2^10000
A = []  # List of n where digit_sum(2^n) is a perfect square
digit_sums = []
perfect_square_flags = []

power_of_2 = 1
for n in range(1, N_max + 1):
    power_of_2 *= 2
    ds = digit_sum(power_of_2)
    digit_sums.append(ds)
    is_square = is_perfect_square(ds)
    perfect_square_flags.append(is_square)
    if is_square:
        A.append(n)

# Create a plot showing digit sums and highlighting perfect squares
plt.figure(figsize=(12, 8))
n_vals = np.arange(1, N_max + 1)
plt.subplot(2, 1, 1)
plt.plot(n_vals, digit_sums, 'b-', linewidth=0.5, alpha=0.7, label='Digit sum of $2^n$')
# Highlight perfect squares
square_ns = np.array(A)
square_vals = [digit_sums[n-1] for n in A]
plt.scatter(square_ns, square_vals, color='red', s=20, alpha=0.8, label=f'Perfect squares ({len(A)} found)')
plt.xlabel('n', fontsize=12)
plt.ylabel('Digit sum of $2^n$', fontsize=12)
plt.title('Digit Sums of Powers of Two', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Lower panel: Show the growth of |A ∩ [1,N]|/sqrt(N)
plt.subplot(2, 1, 2)
count_A = []
ratios = []
for N in range(1, N_max + 1):
    count = sum(1 for a in A if a <= N)
    count_A.append(count)
    ratio = count / np.sqrt(N) if N >= 1 else 0
    ratios.append(ratio)

plt.plot(n_vals, ratios, 'g-', linewidth=1, label=r'$|A \cap [1,N]|/\sqrt{N}$')
plt.axhline(y=np.mean(ratios[1000:]), color='r', linestyle='--', alpha=0.7, 
            label=f'Mean ratio ≈ {np.mean(ratios[1000:]):.3f}')
plt.xlabel('N', fontsize=12)
plt.ylabel(r'$|A \cap [1,N]|/\sqrt{N}$', fontsize=12)
plt.title(r'Growth Rate: $A = \{n : S_{10}(2^n) \text{ is a perfect square}\}$', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_conjecture_4.png', dpi=150)
plt.close()
print('  Saved fig_conjecture_4.png')

print('\nAll figures generated successfully!')
