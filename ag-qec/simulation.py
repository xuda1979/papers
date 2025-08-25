#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte-Carlo logical-failure estimator for a CSS [[n,k,d]] code using a
Pauli error channel with correct Y handling and dB attenuation reporting.

Failure predicate (t-bound model):
  fail  <=>  (#X_type > t) or (#Z_type > t), where t=floor((d-1)/2)
X_type = {X or Y};  Z_type = {Z or Y}.

This script does NOT implement a decoder; it estimates the t-error
correction bound under IID Pauli noise. Swap the failure predicate if/when
you integrate a real decoder.
"""
from __future__ import annotations
import argparse
from typing import Tuple
import json
import numpy as np

from channel_models import (
    attenuation_linear,
    depolarizing_probs,
    asymmetric_probs,
    wilson_interval,
    sample_pauli_iid,
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Monte-Carlo logical failure rate for CSS code under Pauli noise."
    )
    ap.add_argument("--n", type=int, default=255, help="Block length n.")
    ap.add_argument("--d", type=int, default=21, help="Code distance d.")
    ap.add_argument("--trials", type=int, default=200000,
                    help="Monte-Carlo trials (rows).")
    ap.add_argument("--batch-size", type=int, default=100000,
                    help="Process trials in batches to limit memory.")
    ap.add_argument("--seed", type=int, default=12345, help="PRNG seed.")

    # Channel/attenuation reporting (does not force a mapping into p)
    ap.add_argument("--alpha-db-per-km", type=float, default=0.2,
                    help="Fiber attenuation in dB/km (reported).")
    ap.add_argument("--length-km", type=float, default=100.0,
                    help="Link length in km (reported).")

    # Error models
    ap.add_argument("--model", choices=["depol", "asym"], default="depol",
                    help="Pauli channel model.")
    ap.add_argument("--p", type=float, default=0.03,
                    help="Depolarizing probability p (for --model depol).")
    ap.add_argument("--pX", type=float, default=None,
                    help="P(X) for --model asym.")
    ap.add_argument("--pY", type=float, default=None,
                    help="P(Y) for --model asym.")
    ap.add_argument("--pZ", type=float, default=None,
                    help="P(Z) for --model asym.")

    ap.add_argument("--json", action="store_true",
                    help="Emit results as a single JSON object.")
    return ap.parse_args()

def probs_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.model == "depol":
        return depolarizing_probs(args.p)
    # asymmetric:
    missing = [name for name in ("pX", "pY", "pZ") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"--model asym requires {', '.join(missing)}")
    return asymmetric_probs(args.pX, args.pY, args.pZ)

def t_from_distance(d: int) -> int:
    if d <= 0:
        raise ValueError("distance d must be positive.")
    return (d - 1) // 2

def simulate_failures(
    n: int,
    d: int,
    probs: np.ndarray,
    trials: int,
    batch_size: int,
    seed: int,
) -> Tuple[int, int]:
    """
    Returns: (failures, total_trials)
    """
    t = t_from_distance(d)
    rng = np.random.default_rng(seed)

    failures = 0
    done = 0
    while done < trials:
        m = int(min(batch_size, trials - done))
        outcomes = rng.choice(4, size=(m, n), p=probs)
        x_counts = np.count_nonzero((outcomes == 1) | (outcomes == 2), axis=1)
        z_counts = np.count_nonzero((outcomes == 3) | (outcomes == 2), axis=1)
        failures += np.count_nonzero((x_counts > t) | (z_counts > t))
        done += m
    return failures, trials

def main() -> None:
    args = parse_args()
    probs = probs_from_args(args)  # [PI, PX, PY, PZ]
    A = attenuation_linear(args.alpha_db_per_km, args.length_km)

    failures, total = simulate_failures(
        n=args.n,
        d=args.d,
        probs=probs,
        trials=args.trials,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    p_hat = failures / float(total)
    lo, hi = wilson_interval(failures, total)

    result = {
        "n": args.n,
        "d": args.d,
        "t": t_from_distance(args.d),
        "model": args.model,
        "probs": {
            "I": float(probs[0]),
            "X": float(probs[1]),
            "Y": float(probs[2]),
            "Z": float(probs[3]),
        },
        "trials": total,
        "failures": failures,
        "p_logical_hat": p_hat,
        "wilson95": {"lo": lo, "hi": hi},
        "attenuation": {
            "alpha_db_per_km": args.alpha_db_per_km,
            "length_km": args.length_km,
            "A_linear": A,
            "note": "A is the power ratio after propagation; mapping A->p is experiment/model-specific.",
        },
        "seed": args.seed,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"CSS [[{args.n}, k, {args.d}]]  (t={result['t']})")
        print(f"Model={args.model}  P(I,X,Y,Z)={tuple(round(x,8) for x in probs)}")
        print(f"Trials={total:,}  Failures={failures:,}")
        print(f"p_logical \u2248 {p_hat:.6e}  (Wilson 95%: [{lo:.6e}, {hi:.6e}])")
        print(f"Attenuation: alpha={args.alpha_db_per_km} dB/km, L={args.length_km} km, "
              f"A_linear={A:.6e}")

if __name__ == "__main__":
    main()
