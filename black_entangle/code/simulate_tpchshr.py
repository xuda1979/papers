#!/usr/bin/env python3
"""
simulate_tpchshr.py
Toy simulator for the GR-corrected CHSH functional S_GR as a function of
impact parameter b and spin a (dimensionless Kerr parameter).

This is a pedagogical stub: it does NOT integrate Kerr geodesics. It applies
simple, controllable "drag" and "transport" ansätze to show how S_GR varies,
so readers can sanity-check the inequalities and reproduce figures quickly.
"""
from __future__ import annotations
import argparse
import math
import os
from typing import Tuple, List

def redshift_factor(b: float, a: float) -> float:
    """
    Crude monotone model for gravitational redshift along a lensed path.
    b: impact parameter (in units of GM/c^2)
    a: dimensionless spin (0..1)
    Returns z >= 0; observed frequency ~ nu_emitted / (1+z).
    """
    # Larger spin and smaller b -> stronger effective redshift in this toy model.
    return max(0.0, 0.15 * (1.0 + 0.8*a) * (1.0 / max(1.0, b-1.5)))

def transport_angle_shift(b: float, a: float) -> float:
    """
    Very rough model for polarization rotation (in radians) from parallel transport
    plus frame dragging along a null geodesic near a Kerr BH.
    """
    # Increase with spin; decay with b; bounded small-angle regime.
    return 0.4 * a / max(1.2, b)  # ~O(0.05-0.2) radians in typical ranges

def sgr_corrected(b: float, a: float) -> float:
    """
    Construct a GR-corrected CHSH functional S_GR by "warping" the effective
    measurement settings (angles) via transport_angle_shift, and attenuating
    correlations via a redshift-induced visibility factor.
    """
    # Baseline optimal CHSH settings in flat space (radians):
    a1, a2 = 0.0, math.pi/4.0
    b1, b2 = math.pi/8.0, 3.0*math.pi/8.0

    # GR "warps": rotate effective analyzers; reduce visibility by z-dependent factor.
    dth = transport_angle_shift(b, a)
    z = redshift_factor(b, a)
    visibility = 1.0 / (1.0 + z)  # simple attenuation with redshift

    a1p, a2p = a1 + dth, a2 - dth
    b1p, b2p = b1 - 0.5*dth, b2 + 0.5*dth

    def E(th1: float, th2: float) -> float:
        # E ≈ visibility * cos(2(θ1-θ2)) for polarization-entangled photons
        return visibility * math.cos(2.0*(th1 - th2))

    S = abs(E(a1p,b1p) + E(a1p,b2p) + E(a2p,b1p) - E(a2p,b2p))
    return S

def scan(bmin: float, bmax: float, samples: int, a: float) -> List[Tuple[float, float]]:
    xs = []
    for i in range(samples):
        t = i/(samples-1) if samples>1 else 0.0
        b = (1.0 - t)*bmin + t*bmax
        xs.append((b, sgr_corrected(b, a)))
    return xs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spin", type=float, default=0.6, help="Dimensionless spin a in [0,1]")
    ap.add_argument("--bmin", type=float, default=3.0, help="Min impact parameter (GM/c^2 units)")
    ap.add_argument("--bmax", type=float, default=8.0, help="Max impact parameter (GM/c^2 units)")
    ap.add_argument("--samples", type=int, default=40, help="Number of sample points")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"sgr_scan_a{args.spin:.2f}.tsv")
    data = scan(args.bmin, args.bmax, args.samples, args.spin)

    print("# impact_parameter\tS_GR")
    for b, s in data:
        print(f"{b:.6f}\t{s:.6f}")
    with open(out, "w") as f:
        f.write("# impact_parameter\tS_GR\n")
        for b, s in data:
            f.write(f"{b:.9f}\t{s:.9f}\n")
    print(f"\n[Saved] {out}")
    # Simple exceedance count (did we beat classical 2?)
    n_exceed = sum(1 for _, s in data if s>2.0)
    print(f"[Summary] Points with S_GR > 2: {n_exceed}/{len(data)}")

if __name__ == "__main__":
    main()
