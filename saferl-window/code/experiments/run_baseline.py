"""
Run a fixed-parameter sliding-window baseline.
Logs minimal stats to a CSV for plotting.
"""
from __future__ import annotations
import yaml, csv
from pathlib import Path
from ..envs.qldpc_env import SlidingWindowEnv

def run(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    ch = cfg.get("channel", {})
    env = SlidingWindowEnv(code_len=cfg.get("code", {}).get("N",6000),
                           channel_cfg=ch,
                           latency_budget_ms=cfg.get("eval", {}).get("latency_budget_ms", 5.0),
                           compute_budget=cfg.get("eval", {}).get("compute_budget", 1e9))
    W = cfg["decoder"]["window"]["W"]
    F = cfg["decoder"]["window"]["F"]
    iters = cfg["decoder"]["iters"]
    damp = cfg["decoder"]["damping"]
    schedule = cfg["decoder"]["schedule"]
    frames = int(cfg["eval"]["frames"])
    outdir = Path(cfg.get("logdir","./data/logs/baseline"))
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "baseline_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","latency_ms","compute","failures"])
        for ep in range(frames):
            env.reset()
            done = False
            latency = compute = fails = 0.0
            while not done:
                a = {"W":W, "F":F, "iters":iters, "damp":damp, "schedule":schedule}
                _, _, done, info = env.step(a)
                latency += info["latency_ms"]
                compute += info["compute"]
                fails += info["failures"]
            w.writerow([ep, latency, compute, int(fails)])
    print(f"Wrote {csv_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = p.parse_args()
    run(args.config)
