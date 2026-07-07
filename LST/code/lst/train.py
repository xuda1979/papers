"""Training and evaluation utilities for LST experiments."""
from __future__ import annotations
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 1
    steps_per_epoch: int = 200
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_steps: int = 50
    log_every: int = 50
    device: str = "mps"
    seed: int = 0


def get_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    # cosine to 0.1*lr
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@contextmanager
def timing() -> float:
    t0 = time.time()
    yield lambda: time.time() - t0


def train_model(model: nn.Module, train_iter: Callable[[], tuple],
                eval_iter: Optional[Callable[[], tuple]], cfg: TrainConfig,
                label: str = "model") -> dict:
    """Train a model and return a history dict with train/eval metrics.

    train_iter() returns (input, target) tensors on CPU; this function moves
    them to the configured device.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    optim = make_optimizer(model, cfg)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, cfg.warmup_steps, total_steps)
    )
    history = {
        "label": label,
        "steps": [],
        "train_loss": [],
        "eval_loss": [],
        "eval_acc": [],
        "step_time": [],
    }
    step = 0
    for epoch in range(cfg.epochs):
        for _ in range(cfg.steps_per_epoch):
            t0 = time.time()
            model.train()
            x, y = train_iter()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            # Flatten and compute cross-entropy, ignoring -100
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), ignore_index=-100)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            dt = time.time() - t0
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                history["steps"].append(step)
                history["train_loss"].append(loss.item())
                history["step_time"].append(dt)
                print(f"[{label}] step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | {dt*1000:.0f}ms")
            if eval_iter is not None and step % cfg.eval_every == 0:
                eloss, eacc = evaluate_model(model, eval_iter, cfg.eval_steps, device)
                history["eval_loss"].append((step, eloss))
                history["eval_acc"].append((step, eacc))
                print(f"[{label}] EVAL step {step:4d} | loss {eloss:.4f} | acc {eacc:.4f}")
    # Final eval
    if eval_iter is not None:
        eloss, eacc = evaluate_model(model, eval_iter, cfg.eval_steps, device)
        history["eval_loss"].append((step, eloss))
        history["eval_acc"].append((step, eacc))
        print(f"[{label}] FINAL EVAL | loss {eloss:.4f} | acc {eacc:.4f}")
    return history


@torch.no_grad()
def evaluate_model(model: nn.Module, eval_iter: Callable[[], tuple],
                   n_steps: int, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for _ in range(n_steps):
        x, y = eval_iter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100,
                               reduction="sum")
        total_loss += loss.item()
        # Accuracy on non-ignored positions
        mask = y != -100
        preds = logits.argmax(-1)
        total_correct += (preds[mask] == y[mask]).sum().item()
        total_tokens += mask.sum().item()
    avg_loss = total_loss / max(1, total_tokens)
    acc = total_correct / max(1, total_tokens)
    return avg_loss, acc
