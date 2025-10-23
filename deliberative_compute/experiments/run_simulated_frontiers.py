import argparse
import random
from typing import List, Tuple

from simulators import make_threads
from algorithms import igd_step, ucb_igd_step, rs_mctt_budget, csc_vote
from metrics import trapezoid_area, bootstrap_ci


def eval_igd(rnd: random.Random, budgets: List[int], lam: float = 0.0):
    vals = []
    for B in budgets:
        threads = make_threads(6, rnd)
        spent = 0.0
        t = 1
        while spent < B:
            c = igd_step(threads, rnd, lam=lam)
            if c == 0.0:
                break
            spent += c
            t += 1
        vals.append((B, max(th.value for th in threads)))
    return vals


def eval_ucb_igd(rnd: random.Random, budgets: List[int], lam: float = 0.0):
    vals = []
    for B in budgets:
        threads = make_threads(6, rnd)
        counts = [0] * 6
        means = [0.0] * 6
        spent = 0.0
        t = 1
        while spent < B:
            c = ucb_igd_step(threads, rnd, t, counts, means, lam=lam)
            if c == 0.0:
                break
            spent += c
            t += 1
        vals.append((B, max(th.value for th in threads)))
    return vals


def eval_rs_mctt(rnd: random.Random, budgets: List[int]):
    vals = []
    for B in budgets:
        u = 0.0
        spent = 0.0
        while spent < B:
            v, b = rs_mctt_budget(rnd)
            u = max(u, v)
            spent += b
        vals.append((B, u))
    return vals


def eval_csc(rnd: random.Random, budgets: List[int]):
    vals = []
    for B in budgets:
        k = max(2, B // 8)
        u = csc_vote(rnd, k=k)
        vals.append((B, u))
    return vals


def summarize(name: str, pts: List[Tuple[float, float]], rnd: random.Random) -> None:
    area = trapezoid_area(pts)
    vals = [u for _, u in pts]
    lo, hi = bootstrap_ci(vals, rnd)
    print(f"{name:8s} FA={area:.3f}  meanU={sum(vals)/len(vals):.3f}  CI[{lo:.3f},{hi:.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128])
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    igd = eval_igd(rnd, args.budgets)
    ucb = eval_ucb_igd(rnd, args.budgets)
    mctt = eval_rs_mctt(rnd, args.budgets)
    csc = eval_csc(rnd, args.budgets)

    for name, pts in [
        ("IGD", igd),
        ("UCB-IGD", ucb),
        ("RS-MCTT", mctt),
        ("CSC", csc),
    ]:
        summarize(name, pts, rnd)

if __name__ == "__main__":
    main()
