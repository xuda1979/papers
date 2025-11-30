import argparse
import random
import json
import os
from typing import List, Tuple, Dict

from simulators import make_threads, SMR_PROFILE, SCG_PROFILE, TaskProfile
from algorithms import igd_step, ucb_igd_step, rs_mctt_budget, csc_vote, adr_loop
from metrics import trapezoid_area, bootstrap_ci

def eval_igd(rnd: random.Random, budgets: List[int], profile: TaskProfile, lam: float = 0.0):
    vals = []
    for B in budgets:
        # Run multiple episodes to get expected utility
        ep_vals = []
        for _ in range(50):
            threads = make_threads(6, rnd)
            spent = 0.0
            while spent < B:
                c = igd_step(threads, rnd, profile, lam=lam)
                if c == 0.0:
                    break
                spent += c
            ep_vals.append(max(th.value for th in threads))
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_ucb_igd(rnd: random.Random, budgets: List[int], profile: TaskProfile, lam: float = 0.0):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(50):
            threads = make_threads(6, rnd)
            counts = [0] * 6
            means = [0.0] * 6
            spent = 0.0
            t = 1
            while spent < B:
                c = ucb_igd_step(threads, rnd, t, counts, means, profile, lam=lam)
                if c == 0.0: # Should rarely happen with UCB unless explicit
                    if spent > B/2: break # Force stop if UCB just spins
                spent += c
                t += 1
            ep_vals.append(max(th.value for th in threads))
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_rs_mctt(rnd: random.Random, budgets: List[int], profile: TaskProfile):
    vals = []
    for B in budgets:
        ep_vals = []
        # RS-MCTT builds one tree per budget;
        # But 'budgets' list implies we want to see performance AT that budget.
        # So we run the algorithm until it hits B.
        for _ in range(50):
            u = 0.0
            spent = 0.0
            # Repeatedly run trees or expand one tree?
            # Paper says "Budgeted". Let's assume we run MCTT iterations.
            while spent < B:
                v, b = rs_mctt_budget(rnd, profile)
                u = max(u, v)
                spent += b
                if b == 0: break # Safety
            ep_vals.append(u)
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_csc(rnd: random.Random, budgets: List[int], profile: TaskProfile):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(50):
            # Determine k based on budget
            k = max(1, int(B / profile.cost_per_step))
            u, cost = csc_vote(rnd, profile, k=k)
            ep_vals.append(u)
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals

def eval_adr(rnd: random.Random, budgets: List[int], profile: TaskProfile):
    vals = []
    for B in budgets:
        ep_vals = []
        for _ in range(50):
            u, cost = adr_loop(rnd, profile, B)
            ep_vals.append(u)
        vals.append((B, sum(ep_vals)/len(ep_vals)))
    return vals


def summarize(name: str, pts: List[Tuple[float, float]], rnd: random.Random) -> Dict:
    area = trapezoid_area(pts)
    vals = [u for _, u in pts]
    # lo, hi = bootstrap_ci(vals, rnd) # Not meaningful for means of means without raw data
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
    # Normalized budgets (1x, 2x, 4x, 8x, 12x baseline)
    # Assuming baseline cost is around 10-15 units (e.g. one chain)
    # Let's use raw units
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 40, 80, 120])
    args = ap.parse_args()

    rnd = random.Random(args.seed)

    results = {}

    for profile in [SMR_PROFILE, SCG_PROFILE]:
        print(f"\nRunning {profile.name} Task...")

        igd = eval_igd(rnd, args.budgets, profile)
        ucb = eval_ucb_igd(rnd, args.budgets, profile)
        mctt = eval_rs_mctt(rnd, args.budgets, profile)
        csc = eval_csc(rnd, args.budgets, profile)
        adr = eval_adr(rnd, args.budgets, profile)

        # Baselines
        # Greedy CoT is basically 1 thread, 1 step (or budget 10)
        # Self-Consistency is CSC without counterfactuals (eps=0, eps'=0) -> equivalent to simple majority
        # We can simulate SC by running CSC with params that don't filter much or filter randomly

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

        results[profile.name] = task_results

    # Save to JSON relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "simulated_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
