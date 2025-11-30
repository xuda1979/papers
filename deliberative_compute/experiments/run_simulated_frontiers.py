import argparse
import random
import json
import os
from typing import List, Tuple, Dict

from simulators import GameOf24Task, BitstringTask, SearchState, DeliberationTask
from algorithms import igd_step, ucb_igd_step, rs_mctt_budget, csc_vote, adr_loop
from metrics import trapezoid_area

def make_threads(n: int, rnd: random.Random, task: DeliberationTask) -> List[SearchState]:
    return [task.initial_state(rnd) for _ in range(n)]

def eval_igd(rnd: random.Random, budgets: List[int], task: DeliberationTask, lam: float = 0.0):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(20): # Reduced eps for speed
            threads = make_threads(6, rnd, task)
            spent = 0.0
            while spent < B:
                c = igd_step(threads, rnd, task, lam=lam)
                if c == 0.0: break
                spent += c
            ep_vals.append(max(th.value for th in threads))
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_ucb_igd(rnd: random.Random, budgets: List[int], task: DeliberationTask, lam: float = 0.0):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(20):
            threads = make_threads(6, rnd, task)
            counts = [0] * 6
            means = [0.0] * 6
            spent = 0.0
            t = 1
            while spent < B:
                c = ucb_igd_step(threads, rnd, t, counts, means, task, lam=lam)
                if c == 0.0: break
                spent += c
                t += 1
            ep_vals.append(max(th.value for th in threads))
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_rs_mctt(rnd: random.Random, budgets: List[int], task: DeliberationTask):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(20):
            v, b = rs_mctt_budget(rnd, task, B)
            ep_vals.append(v)
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_csc(rnd: random.Random, budgets: List[int], task: DeliberationTask):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(20):
            k = max(1, int(B / 5)) # Approx cost per chain ~ 5
            u, cost = csc_vote(rnd, task, k=k)
            ep_vals.append(u)
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_adr(rnd: random.Random, budgets: List[int], task: DeliberationTask):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(20):
            u, cost = adr_loop(rnd, task, B)
            ep_vals.append(u)
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals


def summarize(name: str, pts: List[Tuple[float, float]], rnd: random.Random) -> Dict:
    area = trapezoid_area(pts)
    vals = [u for _, u in pts]
    print(f"{name:8s} FA={area:.3f}  meanU={sum(vals)/len(vals):.3f}")
    return {
        "name": name,
        "frontier_area": area,
        "mean_utility": sum(vals)/len(vals),
        "points": pts
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 80, 160, 240])
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    results = {}

    tasks = [GameOf24Task(), BitstringTask(length=12)]

    for task in tasks:
        print(f"\nRunning {task.name} Task...")

        igd = eval_igd(rnd, args.budgets, task)
        ucb = eval_ucb_igd(rnd, args.budgets, task)
        mctt = eval_rs_mctt(rnd, args.budgets, task)
        csc = eval_csc(rnd, args.budgets, task)
        adr = eval_adr(rnd, args.budgets, task)

        task_results = []
        for name, pts in [
            ("IGD", igd),
            ("UCB-IGD", ucb),
            ("RS-MCTT", mctt),
            ("CSC", csc),
            ("ADR", adr)
        ]:
            res = summarize(name, pts, rnd)
            task_results.append(res)

        results[task.name] = task_results

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "simulated_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
