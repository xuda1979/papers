#!/usr/bin/env python3
"""
AG-QEC simulator: CLI, configs, seeds, resource accounting, and plots.
Supports: depolarizing (bit-flip proxy), biased dephasing (phase-flip proxy),
and leakage augmentation. Produces logical error vs p curves, pseudo-thresholds,
CSV/JSON logs, and a figure saved to results/figs/l_vs_p.png.
"""
from __future__ import annotations
import argparse, json, math, os, time, pathlib, random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import yaml
import matplotlib.pyplot as plt

# -------- utilities --------
def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = int(time.time() * 1e6) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return seed

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# -------- toy repetition-code core (distance d odd) --------
@dataclass
class RepetitionCode:
    d: int  # odd
    ancillas: int = 1
    def resources(self, rounds: int = 1) -> Dict[str, int]:
        # nearest-neighbor chain; example schedule (illustrative)
        oneq = 4 * self.d * rounds
        twoq = 4 * ((self.d - 1) // 2) * rounds
        swaps = 0
        depth2q = 4 * ((self.d - 1) // 2) * rounds
        return {
            "qubits_data": self.d,
            "qubits_ancilla": self.ancillas,
            "oneq_gates": oneq,
            "twoq_gates": twoq,
            "swaps": swaps,
            "twoq_depth": depth2q,
        }

def sample_bitflip(n: int, p: float) -> np.ndarray:
    return (np.random.rand(n) < p)

def sample_phaseflip(n: int, p: float) -> np.ndarray:
    # phase flips do not flip measurement in computational basis; proxy for dephasing bias
    return (np.random.rand(n) < p)

def simulate_trial_repetition(
    d: int,
    p: float,
    sigma: float,
    decoder: str = "analog",
    leakage_prob: float = 0.0,
    noise_model: str = "bit_flip",
) -> bool:
    """
    One logical 0 trial under chosen noise; returns True if logical error occurs.
    - bit_flip: flips measurement outcome with prob p.
    - phase_flip: leaves computational-basis outcome unchanged (proxy), but we treat
      it as degrading analog readout quality.
    Leakage: with probability p_l, a qubit yields near-zero-SNR readout.
    """
    assert d % 2 == 1, "distance must be odd."
    # ideal outcomes for logical 0 in repetition: +1 on all data
    m = np.ones(d, dtype=int)
    leaked = (np.random.rand(d) < leakage_prob)
    if noise_model == "bit_flip":
        flips = sample_bitflip(d, p)
        m = np.where(flips, -m, m)
        # analog readout: r = m + N(0, sigma^2), leakage -> N(0, sigma^2)
        r = m.astype(float) + np.random.normal(0.0, sigma, size=d)
        r[leaked] = np.random.normal(0.0, sigma, size=leaked.sum())
    elif noise_model == "phase_flip":
        # phase flip doesn't change m, but analog quality decreases with prob p
        phase = sample_phaseflip(d, p)
        noise_scale = sigma * (1.0 + 2.0 * phase.astype(float))
        r = m.astype(float) + np.random.normal(0.0, noise_scale, size=d)
        r[leaked] = np.random.normal(0.0, sigma, size=leaked.sum())
    else:
        raise ValueError(f"unknown noise_model: {noise_model}")

    if decoder == "analog":
        score = (r / (sigma * sigma)).sum()
        decision = +1 if score >= 0.0 else -1
    elif decoder == "digital":
        # per-qubit threshold then majority
        y = np.where(r >= 0.0, +1, -1)
        decision = +1 if y.sum() >= 0 else -1
    else:
        raise ValueError(f"unknown decoder: {decoder}")

    return bool(decision == -1)  # logical error if we decode to -1

def logical_error_rate(
    d: int,
    p: float,
    shots: int,
    sigma: float,
    decoder: str,
    leakage_prob: float,
    noise_model: str,
) -> float:
    errs = 0
    for _ in range(shots):
        errs += simulate_trial_repetition(
            d, p, sigma, decoder=decoder, leakage_prob=leakage_prob, noise_model=noise_model
        )
    return errs / float(shots)

def scan_curve(
    d_list: List[int],
    p_min: float,
    p_max: float,
    points: int,
    shots: int,
    sigma: float,
    decoder: str,
    leakage_prob: float,
    noise_model: str,
) -> Dict:
    grid = np.logspace(np.log10(p_min), np.log10(p_max), points)
    out = {"p": grid.tolist(), "curves": {}, "meta": {}}
    for d in d_list:
        vals = []
        for p in grid:
            vals.append(
                logical_error_rate(d, float(p), shots, sigma, decoder, leakage_prob, noise_model)
            )
        out["curves"][str(d)] = vals
    out["meta"] = {
        "decoder": decoder,
        "sigma": sigma,
        "shots": shots,
        "leakage_prob": leakage_prob,
        "noise_model": noise_model,
        "d_list": d_list,
    }
    return out

def estimate_pseudo_threshold(p: np.ndarray, L: np.ndarray) -> float | None:
    # return the smallest p where L(p) <= p (grid-based)
    mask = L <= p
    if not mask.any():
        return None
    idx = np.argmax(mask)  # first True
    return float(p[idx])

def save_results(outdir: str, results: Dict, resources: Dict[str, Dict]):
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "figs"))
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(outdir, "resources.json"), "w") as f:
        json.dump(resources, f, indent=2)
    # simple CSV for convenience
    p = np.array(results["p"])
    header = "p," + ",".join([f"d{d}" for d in results["meta"]["d_list"]])
    rows = []
    for i in range(len(p)):
        row = [f"{p[i]:.8e}"]
        for d in results["meta"]["d_list"]:
            row.append(f"{results['curves'][str(d)][i]:.8e}")
        rows.append(",".join(row))
    with open(os.path.join(outdir, "results.csv"), "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")

def plot_curves(outdir: str, results: Dict):
    p = np.array(results["p"])
    plt.figure()
    for d in results["meta"]["d_list"]:
        y = np.array(results["curves"][str(d)])
        plt.loglog(p, y, label=f"d={d}")
    plt.loglog(p, p, linestyle="--", label="L(p)=p")
    plt.xlabel("physical error p")
    plt.ylabel("logical error L(p)")
    plt.legend()
    figpath = os.path.join(outdir, "figs", "l_vs_p.png")
    plt.savefig(figpath, bbox_inches="tight", dpi=160)
    plt.close()

def parse_args():
    ap = argparse.ArgumentParser(description="AG-QEC simulator")
    ap.add_argument("--config", type=str, default="configs/depolarizing.yaml")
    ap.add_argument("--outdir", type=str, default="results/exp1")
    ap.add_argument("--seed", type=int, default=None)
    # overrides
    ap.add_argument("--decoder", type=str, choices=["analog", "digital"], default=None)
    ap.add_argument("--noise_model", type=str, choices=["bit_flip", "phase_flip"], default=None)
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--pmin", type=float, default=None)
    ap.add_argument("--pmax", type=float, default=None)
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--leakage", type=float, default=None)
    return ap.parse_args()

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    args = parse_args()
    seed = set_seed(args.seed)
    cfg = load_config(args.config)
    # apply overrides
    decoder = args.decoder or cfg["decoder"]
    noise_model = args.noise_model or cfg["noise_model"]
    shots = args.shots or cfg["shots"]
    pmin = args.pmin or cfg["p_range"]["min"]
    pmax = args.pmax or cfg["p_range"]["max"]
    points = args.points or cfg["p_range"]["points"]
    sigma = args.sigma or cfg["readout"]["sigma"]
    leakage = args.leakage if args.leakage is not None else cfg.get("leakage", 0.0)
    d_list = cfg["code"]["distances"]
    rounds = cfg.get("rounds", 1)

    # resources
    resources = {}
    for d in d_list:
        r = RepetitionCode(d).resources(rounds=rounds)
        resources[str(d)] = r

    # curves
    results = scan_curve(
        d_list=d_list,
        p_min=pmin, p_max=pmax, points=points,
        shots=shots, sigma=sigma, decoder=decoder,
        leakage_prob=leakage, noise_model=noise_model,
    )
    # pseudo-thresholds
    p = np.array(results["p"])
    thresholds = {}
    for d in d_list:
        L = np.array(results["curves"][str(d)])
        th = estimate_pseudo_threshold(p, L)
        thresholds[str(d)] = th
    results["meta"]["pseudo_thresholds"] = thresholds
    results["meta"]["seed"] = seed

    save_results(args.outdir, results, resources)
    plot_curves(args.outdir, results)
    print(json.dumps({"outdir": args.outdir, "seed": seed, "thresholds": thresholds}, indent=2))

if __name__ == "__main__":
    main()
