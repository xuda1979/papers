"""
Train SafeRL-Window controller using stable-baselines3 PPO (skeleton).
"""
from __future__ import annotations
from policy.safe_rl import train_saferl
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/saferl.yaml")
    args = p.parse_args()
    train_saferl(args.config)

