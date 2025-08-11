"""
Safe RL training harness using stable-baselines3 (PPO) with a simple Lagrangian penalty
and action masking/fallback hooks. This is a skeleton; adjust for your environment.

NOTE: This is a research skeleton. The GymWrapper and training loop are functional,
but they rely on a toy environment (`SlidingWindowEnv`) that uses placeholder logic
for observations, rewards, and safety checks. The core PPO algorithm is used, but
the Lagrangian penalties and sophisticated safety mechanisms described in the
accompanying paper are not implemented here and must be added by the user.
"""
from __future__ import annotations
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
except Exception as e:
    PPO = None
    gym = None
    spaces = None

from ..envs.qldpc_env import SlidingWindowEnv

class GymWrapper(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, env: SlidingWindowEnv, action_sets: Dict[str, list], safety_cfg: Dict[str, Any]):
        super().__init__()
        self.env = env
        self.keys = ["W","F","iters","damping","schedule"]
        self.action_sets = action_sets
        self.safety_cfg = safety_cfg or {}
        # Discrete multi-action via MultiDiscrete
        self.action_space = spaces.MultiDiscrete([len(self.action_sets[k]) for k in self.keys])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        obs = self.env.reset().astype(np.float32)
        return obs, {}

    def step(self, action):
        a = {k: self.action_sets[k][int(idx)] for k, idx in zip(self.keys, action)}
        # safety mask
        if self.safety_cfg.get("mask_actions", True) and not self.env.feasible(a):
            # project to nearest feasible by shrinking iters or W
            for key in ["iters","W"]:
                for val in sorted(self.action_sets[key]):
                    a[key] = val
                    if self.env.feasible(a): break
        obs, reward, done, info = self.env.step(a)
        return obs.astype(np.float32), float(reward), done, False, info

def train_saferl(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    ch = cfg.get("channel", {})
    env = SlidingWindowEnv(code_len=cfg.get("code", {}).get("N",6000),
                           channel_cfg=ch,
                           latency_budget_ms=cfg.get("eval", {}).get("latency_budget_ms", 5.0),
                           compute_budget=cfg.get("eval", {}).get("compute_budget", 1e9))

    action_sets = cfg.get("controller_actions", {
        "W":[3,5,7,9],
        "F":[1,2],
        "iters":[8,12,16,20,24],
        "damping":[0.2,0.4,0.6,0.8],
        "schedule":["layered","flooding"]
    })
    gym_env = GymWrapper(env, action_sets, cfg.get("rl", {}).get("safety", {}))
    if PPO is None:
        print("stable-baselines3/gymnasium not available in this environment. Please install dependencies.")
        return
    model = PPO("MlpPolicy", gym_env, verbose=1, tensorboard_log=None,
                learning_rate=cfg["rl"]["lr"],
                n_steps=cfg["rl"]["n_steps"],
                batch_size=cfg["rl"]["batch_size"],
                ent_coef=cfg["rl"]["ent_coef"],
                gamma=cfg["rl"]["gamma"],
                clip_range=cfg["rl"]["clip_range"])
    steps = int(cfg["rl"]["steps"])
    model.learn(total_timesteps=steps)
    outdir = Path(cfg.get("logdir","./data/logs/saferl"))
    outdir.mkdir(parents=True, exist_ok=True)
    model.save(outdir / "policy.zip")
    print(f"Saved policy to {outdir / 'policy.zip'}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/saferl.yaml")
    args = p.parse_args()
    train_saferl(args.config)
