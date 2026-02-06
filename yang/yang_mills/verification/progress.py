"""progress.py

Tiny progress-reporting utilities for long-running verification loops.

Design goals:
- Zero required dependencies.
- When `tqdm` is installed and enabled, show a progress bar.
- Otherwise, print periodic, low-noise progress lines.

The verification codebase often runs on Windows and in minimal Python
installations, so this module must remain pure-Python.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def progress_enabled() -> bool:
    """Whether fine-grained progress prints are enabled."""
    return _env_flag("YM_PROGRESS", default=True)


def tqdm_enabled() -> bool:
    """Whether tqdm progress bars are allowed (requires tqdm installed)."""
    return _env_flag("YM_TQDM", default=False)


@dataclass
class ProgressState:
    label: str
    total_volume: float
    print_every: int = 500
    time_every_seconds: float = 2.0

    processed: int = 0
    verified_volume: float = 0.0
    max_stack: int = 0
    start_time: float = 0.0
    last_print_time: float = 0.0


class ProgressReporter:
    """Periodic console progress reporter."""

    def __init__(self, state: ProgressState):
        self.s = state
        self.s.start_time = time.perf_counter()
        self.s.last_print_time = self.s.start_time

    def update(self, *, processed_inc: int = 0, verified_volume_inc: float = 0.0, stack_size: int = 0) -> None:
        self.s.processed += processed_inc
        self.s.verified_volume += verified_volume_inc
        if stack_size > self.s.max_stack:
            self.s.max_stack = stack_size

        if not progress_enabled():
            return

        now = time.perf_counter()
        if (self.s.processed % self.s.print_every) != 0 and (now - self.s.last_print_time) < self.s.time_every_seconds:
            return

        self.s.last_print_time = now
        frac = 0.0
        if self.s.total_volume > 0:
            frac = max(0.0, min(1.0, self.s.verified_volume / self.s.total_volume))

        elapsed = now - self.s.start_time
        # Heuristic ETA: assume verified_volume grows roughly linearly.
        eta: Optional[float] = None
        if frac > 1e-9:
            eta = elapsed * (1.0 - frac) / frac

        eta_str = "?" if eta is None else f"{eta:,.1f}s"
        print(
            f"  [PROGRESS] {self.s.label}: nodes={self.s.processed:,} "
            f"verified={100.0*frac:6.2f}% stack={stack_size:,} max_stack={self.s.max_stack:,} "
            f"elapsed={elapsed:,.1f}s eta~{eta_str}"
        )


def maybe_make_tqdm(total: Optional[int], desc: str):
    """Best-effort tqdm factory; returns None if disabled or not installed."""
    if not tqdm_enabled():
        return None
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc)
    except Exception:
        return None
