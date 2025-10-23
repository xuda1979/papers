import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

Rnd = random.Random


@dataclass
class ThreadState:
    depth: int = 0
    value: float = 0.0  # best-so-far utility


def geometric_improvement(rnd: Rnd, q: float = 0.8, scale: float = 0.1) -> float:
    """Stochastically decreasing expected increments."""
    return max(0.0, rnd.random() * scale) * (q ** rnd.random())


def make_threads(n: int, rnd: Rnd) -> List[ThreadState]:
    return [ThreadState() for _ in range(n)]


def rollout_value(rnd: Rnd, depth: int, base: float) -> float:
    """Toy rollout for RS-MCTT: bounded [0,1], slightly worse near leaves."""
    noise = rnd.random() * 0.1
    return max(0.0, min(1.0, base + noise - 0.02 * depth))


def verifier_likelihood_ratios(rnd: Rnd, m: int = 3, advantage: float = 0.6) -> List[float]:
    """Return m likelihood ratios >1 when solution is correct, <1 otherwise."""
    lrs = []
    for _ in range(m):
        if rnd.random() < advantage:
            lrs.append(1.5 + rnd.random() * 0.5)  # supportive check
        else:
            lrs.append(0.6 + rnd.random() * 0.3)  # skeptical check
    return lrs
