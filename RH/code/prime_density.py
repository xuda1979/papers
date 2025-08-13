import csv
import math
from pathlib import Path
from sympy import primerange


def prime_density_data(limit, step=100):
    results = []
    primes = list(primerange(2, limit + 1))
    for n in range(step, limit + 1, step):
        count = sum(1 for p in primes if p <= n)
        density = count / n
        prediction = 1 / math.log(n)
        abs_err = abs(density - prediction)
        results.append((n, count, density, prediction, abs_err))
    return results


def main():
    """Generate prime-density samples up to ``limit`` and write CSV."""
    limit = 5000
    data = prime_density_data(limit)
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    out_file = data_dir / 'prime_density.csv'
    with out_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'pi(N)', 'density', '1/log N', 'abs_error'])
        writer.writerows(data)

    print(f"Wrote prime density data to {out_file}")

    # Calculate and print the summary statistics mentioned in the paper
    abs_errors = [row[4] for row in data]
    max_abs_dev = max(abs_errors)
    mean_abs_dev = sum(abs_errors) / len(abs_errors)
    max_dev_n = data[abs_errors.index(max_abs_dev)][0]

    print("\n--- Summary Statistics ---")
    print(f"Maximum absolute deviation: {max_abs_dev:.2e} (attained at x={max_dev_n})")
    print(f"Mean absolute deviation: {mean_abs_dev:.2e}")


if __name__ == '__main__':
    main()
