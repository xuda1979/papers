"""
Time-varying quantum channel models (simulation only).
We synthesize depolarization, dephasing, erasure (loss), and measurement error,
with burst processes (two-state Markov) and slow drift.
"""
from __future__ import annotations
import numpy as np

class MarkovBurst:
    def __init__(self, p_on=0.02, p_off=0.1, strength=3.0, drift=0.0, rng=None):
        self.p_on = p_on
        self.p_off = p_off
        self.strength = strength
        self.drift = drift
        self.state = 0
        self.rng = np.random.default_rng(rng)

    def step(self, base):
        # toggle state
        if self.state == 0 and self.rng.random() < self.p_on:
            self.state = 1
        elif self.state == 1 and self.rng.random() < self.p_off:
            self.state = 0
        # slow drift
        base = max(0.0, base + self.drift * (self.rng.standard_normal()))
        if self.state == 1:
            return min(1.0, base * self.strength)
        return min(1.0, base)

class QuantumTimeVaryingChannel:
    """
    Generates per-slice noise params for (depolar, dephase, erasure, meas_err).
    """
    def __init__(self, depolar=0.005, dephase=0.01, erasure=0.02, meas_err=0.001,
                 burst=None, rng=None):
        self.base = np.array([depolar, dephase, erasure, meas_err], dtype=float)
        self.bursts = [MarkovBurst(**burst) if burst else None for _ in range(4)]
        self.rng = np.random.default_rng(rng)

    def step(self):
        params = []
        for i, b in enumerate(self.bursts):
            basei = self.base[i]
            if b is None:
                params.append(basei)
            else:
                params.append(b.step(basei))
        return np.array(params, dtype=float)
