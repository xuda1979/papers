#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AG-QEC minimal simulator and CLI.

Features
--------
* Repetition-code phenomenological simulation with bit-flip, phase-flip, and
  depolarizing noise; optional measurement error.
* Two decoders:
    - "majority": classical majority vote on qubit readouts
    - "threshold": AI-style data-calibrated threshold on syndrome count S
* Deterministic seeds, timestamped artifact folders, JSON/CSV outputs.

Usage
-----
python simulation.py --code repetition --distance 5 --noise bitflip \
    --p-x 0.05 --p-meas 0.02 --n-samples 100000 --decoder threshold --seed 123
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ---------------------------
# Config and utilities
# ---------------------------

@dataclass
class Config:
    code: str = "repetition"
    distance: int = 5
    noise: str = "bitflip"           # bitflip | phaseflip | depolarizing
    p_x: float = 0.0
    p_y: float = 0.0
    p_z: float = 0.0
    p_meas: float = 0.0
    n_rounds: int = 1
    n_samples: int = 10000
    decoder: str = "majority"        # majority | threshold
    calib_frac: float = 0.1          # only used for threshold
    seed: int = 123
    save_dir: str = ""               # auto-created if empty
    save_syndromes: bool = False

def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    np.random.seed(seed)

def validate_config(cfg: Config) -> None:
    assert cfg.code == "repetition", "This reference implementation supports --code repetition"
    assert cfg.distance >= 3 and cfg.distance % 2 == 1, "--distance must be odd and >= 3"
    assert cfg.noise in {"bitflip", "phaseflip", "depolarizing"}
    for p, name in [(cfg.p_x, "p-x"), (cfg.p_y, "p-y"), (cfg.p_z, "p-z"), (cfg.p_meas, "p-meas")]:
        assert 0.0 <= p <= 1.0, f"--{name} must be in [0,1]"
    assert 0.0 <= cfg.calib_frac < 1.0, "--calib-frac must be in [0,1)"
    assert cfg.n_samples > 0
    assert cfg.n_rounds >= 1
    assert cfg.decoder in {"majority", "threshold"}

# ---------------------------
# Repetition code primitives
# ---------------------------

def encode_repetition(logical_bit: int, d: int) -> np.ndarray:
    """Return encoded physical bitstring of length d (0->all zeros, 1->all ones)."""
    return np.full(shape=(d,), fill_value=logical_bit, dtype=np.int8)

def apply_pauli_noise(bits: np.ndarray, noise: str, p_x: float, p_y: float, p_z: float) -> np.ndarray:
    """
    Phenomenological model on classical readouts:
    - bitflip: flips with prob p_x
    - phaseflip: no effect on classical readout for a Z-basis measurement (kept for interface compatibility)
    - depolarizing: flips with prob p where X/Y/Z equi-probable -> effective classical flip ~ 2/3 p,
      but we model as independent X,Y,Z draws and flip if an odd number of X/Y occurs.
    """
    d = bits.shape[0]
    out = bits.copy()
    if noise == "bitflip":
        mask = (np.random.rand(d) < p_x).astype(np.int8)
        out ^= mask
        return out
    elif noise == "phaseflip":
        # Phase (Z) errors don't flip Z-basis measurement outcomes of a classical repetition code.
        # We keep this branch for completeness / future extension.
        return out
    elif noise == "depolarizing":
        # Draw independent X,Y,Z; for classical bit values, X and Y flip the outcome, Z does not.
        x = np.random.rand(d) < p_x
        y = np.random.rand(d) < p_y
        # z = np.random.rand(d) < p_z  # unused for classical readout
        flips = (x | y).astype(np.int8)
        out ^= flips
        return out
    else:
        raise ValueError(f"unknown noise: {noise}")

def apply_measurement_error(meas: np.ndarray, p_meas: float) -> np.ndarray:
    """Flip each measurement outcome with prob p_meas."""
    d = meas.shape[0]
    flips = (np.random.rand(d) < p_meas).astype(np.int8)
    return meas ^ flips

def syndrome_from_measurements(meas: np.ndarray) -> np.ndarray:
    """Edge-parity syndrome: s_i = meas_i XOR meas_{i+1}."""
    return meas[:-1] ^ meas[1:]

# ---------------------------
# Decoders
# ---------------------------

def decode_majority(meas: np.ndarray) -> int:
    """Return estimated logical bit via majority vote."""
    ones = int(meas.sum())
    zeros = meas.size - ones
    return 1 if ones > zeros else 0

def calibrate_threshold(syndromes: np.ndarray, labels: np.ndarray) -> int:
    """
    Fit a threshold t* on the statistic S = sum(syndrome) to minimize 0/1 error on labels in {0,1}.
    Returns an integer threshold in [0, d-1].
    """
    S = syndromes.sum(axis=1)  # shape (n,)
    d_minus_1 = syndromes.shape[1]
    # brute-force over possible thresholds on S
    best_t, best_err = 0, 1.0
    for t in range(d_minus_1 + 1):
        preds = (S >= t).astype(np.int8)
        err = float((preds != labels).mean())
        if err < best_err:
            best_err, best_t = err, t
    return int(best_t)

def decode_threshold(syndrome: np.ndarray, t_star: int) -> int:
    """Predict logical=1 if sum(syndrome) >= t_star, else 0."""
    return int(syndrome.sum() >= t_star)

# ---------------------------
# Simulation core
# ---------------------------

def simulate_shot(cfg: Config, logical_bit: int) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Simulate a single encoded shot.
    Returns: (true_logical, measured_majority, measurements, syndrome)
    """
    d = cfg.distance
    phys = encode_repetition(logical_bit, d)
    # (Optional) multiple rounds: here we use last round's measurement as readout;
    # extend to temporal majority or 3D matching as needed.
    meas = phys.copy()
    for _ in range(cfg.n_rounds):
        meas = apply_pauli_noise(meas, cfg.noise, cfg.p_x, cfg.p_y, cfg.p_z)
        meas = apply_measurement_error(meas, cfg.p_meas)
    syn = syndrome_from_measurements(meas)
    est_majority = decode_majority(meas)
    return logical_bit, est_majority, meas, syn

def run(cfg: Config) -> Dict[str, object]:
    validate_config(cfg)
    set_seed(cfg.seed)

    # Prepare save folder
    save_root = Path(cfg.save_dir) if cfg.save_dir else Path("results") / _now_tag()
    ensure_dir(save_root)

    # Generate a label-balanced stream for robust calibration
    n_calib = int(cfg.calib_frac * cfg.n_samples) if cfg.decoder == "threshold" else 0
    n_eval = cfg.n_samples

    # Calibration split
    t_star = None
    if cfg.decoder == "threshold":
        calib_L = np.random.randint(0, 2, size=(n_calib,), dtype=np.int8)
        calib_syn = []
        for L in calib_L:
            _, _, _, syn = simulate_shot(cfg, int(L))
            calib_syn.append(syn)
        calib_syn = np.stack(calib_syn, axis=0) if len(calib_syn) else np.zeros((0, cfg.distance-1), dtype=np.int8)
        t_star = calibrate_threshold(calib_syn, calib_L) if len(calib_syn) else 0

    # Evaluation split
    eval_L = np.random.randint(0, 2, size=(n_eval,), dtype=np.int8)
    logical_err_majority = 0
    logical_err_threshold = 0
    all_rows = []
    for L in eval_L:
        true_L, est_maj, meas, syn = simulate_shot(cfg, int(L))
        if est_maj != true_L:
            logical_err_majority += 1
        if cfg.decoder == "threshold":
            est_thr = decode_threshold(syn, t_star)
            if est_thr != true_L:
                logical_err_threshold += 1
        # optional artifact
        if cfg.save_syndromes:
            all_rows.append({
                "true_L": int(true_L),
                "est_majority": int(est_maj),
                "syn_sum": int(syn.sum()),
                "syn": "".join(map(str, syn.tolist())),
            })

    # Metrics
    maj_rate = logical_err_majority / float(n_eval)
    out = {
        "config": asdict(cfg),
        "metrics": {
            "n_eval": n_eval,
            "logical_error_majority": maj_rate,
        },
    }
    if cfg.decoder == "threshold":
        thr_rate = logical_err_threshold / float(n_eval)
        out["metrics"].update({
            "t_star": int(t_star),
            "logical_error_threshold": thr_rate,
        })

    # Save artifacts
    (save_root / "artifacts").mkdir(parents=True, exist_ok=True)
    with open(save_root / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(save_root / "metrics.json", "w") as f:
        json.dump(out["metrics"], f, indent=2)
    if cfg.save_syndromes and all_rows:
        # lightweight CSV; avoid pandas dependency
        csv_path = save_root / "artifacts" / "syndromes.csv"
        with open(csv_path, "w") as f:
            f.write("true_L,est_majority,syn_sum,syndrome_bits\n")
            for r in all_rows:
                f.write(f"{r['true_L']},{r['est_majority']},{r['syn_sum']},{r['syn']}\n")

    return out

# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AG-QEC minimal simulator (repetition code).")
    p.add_argument("--code", type=str, default="repetition", help="only 'repetition' supported in this reference")
    p.add_argument("--distance", type=int, default=5, help="odd >= 3")
    p.add_argument("--noise", type=str, default="bitflip",
                   choices=["bitflip", "phaseflip", "depolarizing"])
    p.add_argument("--p-x", dest="p_x", type=float, default=0.05)
    p.add_argument("--p-y", dest="p_y", type=float, default=0.0)
    p.add_argument("--p-z", dest="p_z", type=float, default=0.0)
    p.add_argument("--p-meas", dest="p_meas", type=float, default=0.0)
    p.add_argument("--n-rounds", dest="n_rounds", type=int, default=1)
    p.add_argument("--n-samples", dest="n_samples", type=int, default=10000)
    p.add_argument("--decoder", type=str, default="majority", choices=["majority", "threshold"])
    p.add_argument("--calib-frac", dest="calib_frac", type=float, default=0.1,
                   help="fraction of n-samples used to calibrate threshold decoder")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-dir", dest="save_dir", type=str, default="")
    p.add_argument("--save-syndromes", dest="save_syndromes", action="store_true")
    return p

def main() -> None:
    args = build_argparser().parse_args()
    cfg = Config(**vars(args))
    out = run(cfg)
    # Pretty-print a short summary
    print(json.dumps(out["metrics"], indent=2))

if __name__ == "__main__":
    main()
