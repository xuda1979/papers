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
    ap.add_argument(
        "--model",
        choices=["depol", "asym", "markov"],
        default="depol",
        help=(
            "Noise model. 'depol' for IID depolarizing, 'asym' for IID asymmetric Pauli, "
            "'markov' for the two-state Markov-correlated model described in the paper."
        ),
    )
    # Parameters for IID models
    ap.add_argument(
        "--p",
        type=float,
        default=0.03,
        help="Depolarizing probability p (for --model depol).",
    )
    ap.add_argument(
        "--pX",
        type=float,
        default=None,
        help="P(X) for --model asym.",
    )
    ap.add_argument(
        "--pY",
        type=float,
        default=None,
        help="P(Y) for --model asym.",
    )
    ap.add_argument(
        "--pZ",
        type=float,
        default=None,
        help="P(Z) for --model asym.",
    )
    # Parameters for the Markov-correlated model
    ap.add_argument(
        "--pZ-base",
        type=float,
        default=9.0e-3,
        help=(
            "Baseline phase-error probability pZ for the Markov model (low-noise state)."
        ),
    )
    ap.add_argument(
        "--pX-fraction",
        type=float,
        default=0.3,
        help=(
            "Fraction relating amplitude-error probability to phase-error probability. "
            "pX = fraction * pZ in each state."
        ),
    )
    ap.add_argument(
        "--high-factor",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to pZ-base (and consequently pX) when in the high-noise state."
        ),
    )
    ap.add_argument(
        "--rho",
        type=float,
        default=0.6,
        help=(
            "State persistence probability for the first-order Markov chain. "
            "At each position, with probability rho we remain in the current noise state; "
            "otherwise we switch between low and high."
        ),
    )

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


def simulate_markov_failures(
    n: int,
    d: int,
    pZ_base: float,
    pX_fraction: float,
    high_factor: float,
    rho: float,
    trials: int,
    batch_size: int,
    seed: int,
) -> Tuple[int, int]:
    """
    Simulate logical failure for a CSS code under a two-state Markov-correlated
    Pauli error model as described in the paper.

    Each qubit position belongs to either a low-noise or high-noise state. The
    state evolves as a first-order Markov chain: with probability `rho` the
    next position remains in the same state; with probability (1-rho) it
    switches to the other state. The initial state is assumed low noise.

    In the low-noise state, the phase error probability is `pZ_base` and the
    amplitude error probability is `pX_base = pX_fraction * pZ_base`. In the
    high-noise state, both probabilities are multiplied by `high_factor`.

    Amplitude and phase errors are drawn independently for each position
    conditioned on the state. A Y error occurs whenever both an X and a Z
    error occur on the same qubit. Y errors count towards both X-type and
    Z-type error tallies.

    A trial is considered to fail under the t-bound model if either the
    number of X-type errors or the number of Z-type errors exceeds
    t = floor((d-1)/2).

    Returns: (failures, total_trials)
    """
    t = t_from_distance(d)
    rng = np.random.default_rng(seed)

    failures = 0
    done = 0
    while done < trials:
        m = int(min(batch_size, trials - done))
        # Generate Markov states for all m trials and n positions.
        # 0 denotes low noise, 1 denotes high noise.
        states = np.zeros((m, n), dtype=np.int8)
        # Draw random numbers for transitions: shape (m, n-1)
        transitions = rng.random((m, n - 1))
        for pos in range(1, n):
            # Switch when random >= rho, else stay in same state. Equivalent to
            # using a condition transitions[:, pos-1] >= rho.
            prev = states[:, pos - 1]
            switch = (transitions[:, pos - 1] >= rho).astype(np.int8)
            # XOR with switch toggles state when switch==1.
            states[:, pos] = prev ^ switch
        # Compute per-position probabilities for Z and X errors.
        # pZ_base and pX_base for low state; multiplied by high_factor for high state.
        pZ = pZ_base * (1.0 + states * (high_factor - 1.0))
        pX = pZ * pX_fraction
        # Sample Z and X occurrences. Each is a Bernoulli matrix (m,n).
        z_rand = rng.random((m, n))
        x_rand = rng.random((m, n))
        z_occurs = z_rand < pZ
        x_occurs = x_rand < pX
        # Count X-type errors (X or Y) and Z-type errors (Z or Y)
        x_counts = np.count_nonzero(x_occurs, axis=1)
        z_counts = np.count_nonzero(z_occurs, axis=1)
        # Y errors occur when both x_occurs and z_occurs, they contribute to both counts
        # x_counts and z_counts already include Y via the independent draws since
        # x_occurs and z_occurs overlap. No extra handling needed.
        failures += np.count_nonzero((x_counts > t) | (z_counts > t))
        done += m
    return failures, trials

def main() -> None:
    args = parse_args()
    # Compute attenuation report (independent of noise model)
    A = attenuation_linear(args.alpha_db_per_km, args.length_km)

    # Run the appropriate noise model
    if args.model == "markov":
        failures, total = simulate_markov_failures(
            n=args.n,
            d=args.d,
            pZ_base=args.pZ_base,
            pX_fraction=args.pX_fraction,
            high_factor=args.high_factor,
            rho=args.rho,
            trials=args.trials,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        # Estimate empirical logical error probability and Wilson interval
        p_hat = failures / float(total)
        lo, hi = wilson_interval(failures, total)
        result = {
            "n": args.n,
            "d": args.d,
            "t": t_from_distance(args.d),
            "model": args.model,
            "markov_params": {
                "pZ_base": args.pZ_base,
                "pX_fraction": args.pX_fraction,
                "high_factor": args.high_factor,
                "rho": args.rho,
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
            print(
                f"Model=markov  pZ_base={args.pZ_base:.3e}, pX_fraction={args.pX_fraction}, "
                f"high_factor={args.high_factor}, rho={args.rho}"
            )
            print(f"Trials={total:,}  Failures={failures:,}")
            print(f"p_logical ≈ {p_hat:.6e}  (Wilson 95%: [{lo:.6e}, {hi:.6e}])")
            print(
                f"Attenuation: alpha={args.alpha_db_per_km} dB/km, L={args.length_km} km, "
                f"A_linear={A:.6e}"
            )
        return

    # IID models (depol or asym)
    probs = probs_from_args(args)  # [PI, PX, PY, PZ]
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
        print(f"p_logical ≈ {p_hat:.6e}  (Wilson 95%: [{lo:.6e}, {hi:.6e}])")
        print(
            f"Attenuation: alpha={args.alpha_db_per_km} dB/km, L={args.length_km} km, "
            f"A_linear={A:.6e}"
        )

if __name__ == "__main__":
    main()
