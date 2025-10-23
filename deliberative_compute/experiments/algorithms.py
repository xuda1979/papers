import math
import random
from typing import List, Tuple

from simulators import ThreadState, geometric_improvement, rollout_value

Rnd = random.Random


def igd_step(
    threads: List[ThreadState],
    rnd: Rnd,
    lam: float = 0.0,
    gamma: float = 0.95,
    c: float = 1.0,
) -> float:
    """One IGD step: compute a crude index tau_i from expected discounted gains."""
    indices = []
    for i, th in enumerate(threads):
        r = geometric_improvement(rnd)
        tau = r / max(1e-9, c)
        indices.append((tau, i, r))
    tau, idx, r = max(indices, key=lambda t: t[0])
    if tau <= lam:
        return 0.0  # stop signal
    threads[idx].value = min(1.0, threads[idx].value + r)
    threads[idx].depth += 1
    return c


def ucb_igd_step(
    threads: List[ThreadState],
    rnd: Rnd,
    t: int,
    counts: List[int],
    means: List[float],
    lam: float = 0.0,
    c: float = 1.0,
) -> float:
    """UCB surrogate for IGD index."""
    ucbs = []
    for i, _ in enumerate(threads):
        m = means[i]
        n = max(1, counts[i])
        bonus = math.sqrt(2 * math.log(max(2, t)) / n)
        ucbs.append((m + bonus, i))
    u, idx = max(ucbs, key=lambda z: z[0])
    if u <= lam:
        return 0.0
    r = geometric_improvement(rnd)
    counts[idx] += 1
    means[idx] = (means[idx] * (counts[idx] - 1) + r) / counts[idx]
    threads[idx].value = min(1.0, threads[idx].value + r)
    threads[idx].depth += 1
    return c


def rs_mctt_budget(
    rnd: Rnd,
    depth: int = 4,
    eta_schedule=lambda d: -0.5,
    c: float = 1.0,
) -> Tuple[float, float]:
    """Toy RS-MCTT: build a shallow tree with risk-averse selection."""
    B = 0.0

    def node_value(d, base):
        nonlocal B
        if d == depth:
            v = rollout_value(rnd, d, base)
            return v
        eta = eta_schedule(d)
        vals = []
        for _ in range(2):
            child1 = rollout_value(rnd, d, base + 0.05)
            child2 = rollout_value(rnd, d, base + 0.02)
            vals.append(child1)
            vals.append(child2)
        if eta == 0.0:
            score = sum(vals) / len(vals)
        else:
            score = (1.0 / eta) * math.log(
                sum(math.exp(eta * v) for v in vals) / len(vals)
            )
        B += c
        return node_value(d + 1, score)

    v = node_value(0, 0.2)
    return v, B


def csc_vote(rnd: Rnd, k: int = 8, eps: float = 0.2, epsp: float = 0.05) -> float:
    """Counterfactual Self-Consistency: filter k chains with rejection rates eps, eps'."""
    delta = 0.2
    kept = 0
    correct = 0
    for _ in range(k):
        is_correct = rnd.random() < 0.5 + delta
        reject = rnd.random() < (eps if not is_correct else epsp)
        if not reject:
            kept += 1
            correct += int(is_correct)
    if kept == 0:
        return 0.0
    return correct / kept
