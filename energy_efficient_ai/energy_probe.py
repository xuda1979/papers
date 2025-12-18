"""energy_probe.py

Device energy/power measurement helper.

On Ascend, *real* power measurement depends on your cluster tooling.
This module provides a thin abstraction:
- Local CPU: estimates energy via FLOPs proxy and wall time.
- Remote Ascend: optionally shells out to an external power tool if available,
  and otherwise records wall time + throughput.

The idea is that you can plug in your site's standard voltage/power logging
(e.g., npu-smi / ascend-smi / dcmi / vendor exporter).

"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class EnergySample:
    wall_time_sec: float
    avg_power_w: Optional[float]
    energy_j: Optional[float]


def _try_run(cmd: str) -> Optional[str]:
    try:
        p = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return p.stdout.strip()
    except Exception:
        return None


def measure_with_external_power(
    fn: Callable[[], None],
    *,
    power_cmd_before: Optional[str] = None,
    power_cmd_after: Optional[str] = None,
) -> EnergySample:
    """Measure wall time and optionally power (W) using external commands."""

    p0 = _try_run(power_cmd_before) if power_cmd_before else None
    t0 = time.time()
    fn()
    wall = time.time() - t0
    p1 = _try_run(power_cmd_after) if power_cmd_after else None

    # If the caller provides commands that output a number in watts,
    # we can compute energy.
    def parse_w(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            return float(s.split()[0])
        except Exception:
            return None

    w0 = parse_w(p0)
    w1 = parse_w(p1)
    avg_w = None
    energy = None
    if w0 is not None and w1 is not None:
        avg_w = 0.5 * (w0 + w1)
        energy = avg_w * wall

    return EnergySample(wall_time_sec=wall, avg_power_w=avg_w, energy_j=energy)


def flops_proxy_energy(wall_time_sec: float, flops: float, assumed_efficiency: float = 1e12) -> EnergySample:
    """Proxy: convert FLOPs into an energy (J) using a rough efficiency.

    assumed_efficiency: FLOPs per Joule (defaults to 1e12 FLOP/J = 1 TOP/J).
    Tune this later using measured Ascend power.
    """

    energy = float(flops) / float(assumed_efficiency)
    avg_power = energy / max(1e-9, wall_time_sec)
    return EnergySample(wall_time_sec=wall_time_sec, avg_power_w=avg_power, energy_j=energy)
