import random
from typing import List, Tuple


def trapezoid_area(points: List[Tuple[float, float]]) -> float:
    pts = sorted(points)
    area = 0.0
    for (b1, u1), (b2, u2) in zip(pts[:-1], pts[1:]):
        area += 0.5 * (b2 - b1) * (u1 + u2)
    return area


def bootstrap_ci(
    values: List[float], rnd: random.Random, iters: int = 500, alpha: float = 0.05
):
    n = len(values)
    if n == 0:
        raise ValueError("Cannot bootstrap empty sequence")
    boots = []
    for _ in range(iters):
        samp = [values[rnd.randrange(n)] for _ in range(n)]
        boots.append(sum(samp) / len(samp))
    boots.sort()
    lo = boots[int(alpha / 2 * iters)]
    hi = boots[int((1 - alpha / 2) * iters) - 1]
    return lo, hi
