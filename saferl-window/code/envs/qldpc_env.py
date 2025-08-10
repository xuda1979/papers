"""
Gym-like environment wrapper for sliding-window decoding with SafeRL control.
State: features from recent syndromes, residual stats, noise fingerprint
Action: (W, F, iters, damping, schedule) chosen from discrete sets
Reward: throughput - penalty(failure, latency/compute)
Safety: hard budgets via action masking; baseline fallback hook left to controller
"""
from __future__ import annotations
import numpy as np
from .channels import QuantumTimeVaryingChannel
from .codes import sc_qldpc_placeholder, tiny_css_example
from ..decoders.bp_decoder import css_bp_round

class SlidingWindowEnv:
    def __init__(self, code_len=6000, channel_cfg=None,
                 action_sets=None, latency_budget_ms=5.0, compute_budget=1e9, rng=None):
        self.rng = np.random.default_rng(rng)
        self.latency_budget = latency_budget_ms
        self.compute_budget = compute_budget
        ch = channel_cfg or {}
        self.channel = QuantumTimeVaryingChannel(**{k:v for k,v in ch.items() if k not in ["burst","seed"]},
                                                 burst=ch.get("burst", None), rng=ch.get("seed", None))
        self.Hx, self.Hz = sc_qldpc_placeholder(n=code_len, seed=0)
        self.L = 50  # toy number of "slices"
        self.t = 0
        # action sets
        self.A = action_sets or {
            "W": [3,5,7,9],
            "F": [1,2],
            "iters": [8,12,16,20,24],
            "damp": [0.2,0.4,0.6,0.8],
            "schedule": ["layered","flooding"]
        }
        self.prev_failure = 0

    def reset(self):
        self.t = 0
        self.prev_failure = 0
        obs = self._observe()
        return obs

    def step(self, a):
        # unpack action dict
        W,F,iters,damp,schedule = a["W"], a["F"], a["iters"], a["damp"], a["schedule"]
        # simulate W slices
        latency_ms = 0.0
        compute = 0.0
        failures = 0
        for _ in range(W):
            depol, deph, era, meas = self.channel.step()
            # toy syndromes: random Bernoulli proportional to params
            n_x = self.Hz.shape[0]
            n_z = self.Hx.shape[0]
            syn_x = (self.rng.random(n_x) < min(0.5, depol+era+meas)).astype(int)
            syn_z = (self.rng.random(n_z) < min(0.5, deph+era+meas)).astype(int)
            ex, ez = css_bp_round(self.Hx, self.Hz, syn_x, syn_z, iters=iters, damping=damp, schedule=schedule, rng=self.rng)
            # simplistic success criterion: residual syndrome weight
            res = syn_x.sum() + syn_z.sum()
            fail = int(res > 0 and self.rng.random() < 0.3)  # toy failure model
            failures += fail
            latency_ms += 0.05 * iters  # toy latency model
            compute += 1e6 * iters      # toy compute estimate
        self.t += F
        done = self.t >= self.L
        # reward: throughput - penalties
        throughput = (W - failures) / max(1,F)
        reward = throughput - 5.0*failures - 0.1*(latency_ms/self.latency_budget)
        obs = self._observe()
        info = {"latency_ms": latency_ms, "compute": compute, "failures": failures}
        self.prev_failure = failures
        return obs, reward, done, info

    def _observe(self):
        # minimal observation vector
        return np.array([self.t/self.L, float(self.prev_failure)], dtype=float)

    def feasible(self, a):
        # simple feasibility check against budgets (approximate)
        iters = a["iters"]
        W = a["W"]
        est_latency = 0.05 * iters * W
        est_compute = 1e6 * iters * W
        return (est_latency <= self.latency_budget) and (est_compute <= self.compute_budget)
