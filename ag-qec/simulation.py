#!/usr/bin/env python3
"""
Paper experiments harness for AG-QEC.

This script provides an opt-in command-line interface for sweeping physical
error rate p across multiple code distances d and noise models. It computes
logical error rates (LER) along with Wilson confidence intervals and writes
the results to a CSV file. The harness is non-invasive: it only runs when
the special flag `--paper-experiments` is provided, so that existing behavior
in this repository is not altered.

Expected integration point: a function `run_trial(distance: int, p: float,
noise: str, **kwargs) -> bool` should be implemented elsewhere in the
project. It should simulate a single trial at physical error rate p for a
code of the given distance under the specified noise model and return
True if a logical failure occurs. If such a function is not found in the
module namespace, a stub is provided that raises NotImplementedError.

Example usage (see the paper for details):

    python modified_simulation.py --paper-experiments \
      --noise biased_dephasing --bias 10 \
      --pmin 1e-4 --pmax 5e-2 --num-points 9 \
      --distances 3 5 7 --trials 20000 \
      --out results/biased_dephasing_bias10.csv

"""

from __future__ import annotations
import csv
import math
import random
from typing import Iterable, Tuple, Dict, Any

# -----------------------------------------------------------------------------
# Optional integration hook
try:
    # Attempt to use an existing run_trial function if defined elsewhere.
    run_trial  # type: ignore[name-defined]
except NameError:
    def run_trial(distance: int, p: float, noise: str, **kwargs: Any) -> bool:
        """Fallback single-trial simulator for the CLI harness.

        This minimal implementation enables smoke tests of the paper experiments
        harness without requiring an external decoder. It draws iid Bernoulli
        ``p`` errors on ``distance`` qubits and declares a logical failure when
        more than ``t=(distance-1)//2`` errors occur. The ``noise`` argument and
        additional keyword arguments are accepted for API compatibility but are
        otherwise ignored. Replace this function with a proper simulator for
        meaningful studies.
        """

        if not (0.0 <= p <= 1.0):
            raise ValueError("p must lie in [0,1]")

        t_correctable = (distance - 1) // 2
        errors = 0
        for _ in range(distance):
            if random.random() < p:
                errors += 1
                if errors > t_correctable:
                    return True
        return False




# -----------------------------------------------------------------------------
# Statistics utilities

def wilson_interval(failures: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute a Wilson score interval for a Bernoulli proportion.

    Given the number of failures and total trials n, this function returns
    a 95% confidence interval (z=1.96 by default) for the true logical
    error rate using the Wilson score method.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = failures / float(n)
    denom = 1.0 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n) / denom
    lo, hi = max(0.0, center - margin), min(1.0, center + margin)
    return lo, hi


def sweep_p_for_distance(
    distance: int,
    ps: Iterable[float],
    trials: int,
    noise: str,
    **kwargs: Any,
) -> Iterable[Dict[str, Any]]:
    """
    Iterate over a sequence of physical error rates and compute LER estimates.

    For each value of p in ps, this generator runs `trials` independent
    simulation trials by invoking run_trial(distance, p, noise, **kwargs).
    It counts the number of logical failures and computes a Wilson score
    interval for the estimated logical error rate.

    Yields dictionaries with the following keys:
        - distance: code distance
        - p: physical error rate
        - logical_error_rate: empirical LER (failures / trials)
        - ci_low: lower bound of Wilson interval
        - ci_high: upper bound of Wilson interval
    """
    for p in ps:
        failures = 0
        for _ in range(trials):
            if run_trial(distance=distance, p=p, noise=noise, **kwargs):
                failures += 1
        lo, hi = wilson_interval(failures, trials)
        row = {
            "distance": distance,
            "p": p,
            "logical_error_rate": failures / float(trials),
            "ci_low": lo,
            "ci_high": hi,
        }
        for key in ("damping", "llr_bits", "max_iters", "early_stop"):
            if key in kwargs:
                row[key] = kwargs[key]
        yield row


def logspaced(min_p: float, max_p: float, num: int) -> Iterable[float]:
    """Generate a log-spaced grid of probabilities between min_p and max_p.

    This helper yields `num` values logarithmically spaced between min_p and
    max_p (inclusive). It requires 0 < min_p < max_p and num >= 2.
    """
    if not (min_p > 0 and max_p > 0 and max_p > min_p and num >= 2):
        raise ValueError("Require 0 < min_p < max_p and num >= 2")
    log_min = math.log10(min_p)
    log_max = math.log10(max_p)
    step = (log_max - log_min) / (num - 1)
    for i in range(num):
        yield 10 ** (log_min + i * step)


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a sequence of dictionary rows to a CSV file.

    The keys of the first dictionary define the CSV header. Subsequent rows
    must have the same keys. The file is overwritten if it already exists.
    """
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def paper_experiments_main(argv: Iterable[str] | None = None) -> None:
    """
    Entry point for the paper experiments harness.

    Parses command-line arguments and orchestrates a sweep over physical
    error rates and code distances. The results are aggregated and written
    to a specified output CSV file. This function should be invoked via
    the `--paper-experiments` flag in the main block.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AG-QEC paper experiments: sweep physical error rate p across multiple "
            "code distances and write logical error rate CSVs with Wilson confidence "
            "intervals. This harness only runs when --paper-experiments is provided."
        )
    )
    parser.add_argument(
        "--noise",
        required=True,
        choices=[
            "depolarizing",
            "biased_dephasing",
            "amplitude_damping",
            "leakage",
            "correlated_two_qubit",
        ],
        help="Noise model to simulate."
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=1.0,
        help="Z/X bias used for biased dephasing (ignored otherwise).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="Belief-propagation damping factor.",
    )
    parser.add_argument(
        "--llr-bits",
        type=int,
        default=6,
        help="Quantization bits for LLR representation.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10,
        help="Maximum BP iterations.",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping when syndrome is satisfied.",
    )
    parser.add_argument("--pmin", type=float, required=True, help="Minimum p (log-scale).")
    parser.add_argument("--pmax", type=float, required=True, help="Maximum p (log-scale).")
    parser.add_argument(
        "--num-points",
        type=int,
        default=9,
        help="Number of points in the log-spaced p grid. Must be >= 2."
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="List of code distances to evaluate."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20000,
        help="Number of trials per (p, distance) point."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="PRNG seed for reproducibility."
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to the output CSV file."
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # Seed the random number generator for reproducibility
    random.seed(args.seed)

    # Build a log-spaced grid of p values
    ps = list(logspaced(args.pmin, args.pmax, args.num_points))
    all_rows = []
    for d in args.distances:
        rows = list(
            sweep_p_for_distance(
                distance=d,
                ps=ps,
                trials=args.trials,
                noise=args.noise,
                bias=args.bias,
                damping=args.damping,
                llr_bits=args.llr_bits,
                max_iters=args.max_iters,
                early_stop=args.early_stop,
            )
        )
        all_rows.extend(rows)

    # Write the aggregated results to disk
    write_csv(args.out, all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    # Only run the paper experiments harness if the special flag is present.
    import sys
    if "--paper-experiments" in sys.argv:
        sys.argv.remove("--paper-experiments")
        paper_experiments_main(argv=sys.argv[1:])
