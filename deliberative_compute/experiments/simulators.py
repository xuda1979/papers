import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

Rnd = random.Random


@dataclass
class ThreadState:
    depth: int = 0
    value: float = 0.0  # best-so-far utility
    history: List[float] = None  # Track history for more complex state

    def __post_init__(self):
        if self.history is None:
            self.history = []

@dataclass
class TaskProfile:
    name: str
    improvement_rate: float
    improvement_scale: float
    noise_level: float
    branching_factor: int
    cost_per_step: float
    verifier_noise: float # 0.0 means perfect, 0.5 means random

# Define profiles for the two tasks mentioned in the paper
SMR_PROFILE = TaskProfile(
    name="SMR",
    improvement_rate=0.3, # Harder to find improvements
    improvement_scale=0.3, # But improvements are significant
    noise_level=0.1,
    branching_factor=4,
    cost_per_step=1.0,
    verifier_noise=0.2
)

SCG_PROFILE = TaskProfile(
    name="SCG",
    improvement_rate=0.7, # Easier to make incremental progress
    improvement_scale=0.1, # But steps are smaller (passing one more test)
    noise_level=0.05,
    branching_factor=2,
    cost_per_step=1.5, # Code generation is more expensive
    verifier_noise=0.05 # Unit tests are reliable
)


def geometric_improvement(rnd: Rnd, profile: TaskProfile) -> float:
    """Stochastically decreasing expected increments based on profile."""
    if rnd.random() > profile.improvement_rate:
        return 0.0

    # Base improvement
    imp = rnd.random() * profile.improvement_scale
    # Decaying returns based on current value would be handled by the caller or state,
    # but here we simulate 'difficulty' by just returning the raw potential improvement.
    return imp


def make_threads(n: int, rnd: Rnd) -> List[ThreadState]:
    return [ThreadState() for _ in range(n)]


def rollout_value(rnd: Rnd, depth: int, base: float, profile: TaskProfile) -> float:
    """Rollout value simulation."""
    noise = (rnd.random() - 0.5) * 2 * profile.noise_level
    # Diminishing returns with depth
    penalty = 0.01 * depth
    return max(0.0, min(1.0, base + noise - penalty))


def verifier_likelihood_ratios(rnd: Rnd, m: int, is_correct: bool, profile: TaskProfile) -> List[float]:
    """Return m likelihood ratios."""
    lrs = []
    # If verifier is noisy, it flips the signal with prob verifier_noise
    effective_correct = is_correct
    if rnd.random() < profile.verifier_noise:
        effective_correct = not is_correct

    for _ in range(m):
        # Separation between correct/incorrect distributions
        # Correct: N(1.5, 0.5), Incorrect: N(0.5, 0.5) roughly
        if effective_correct:
            val = 1.2 + rnd.random() * 0.6
        else:
            val = 0.4 + rnd.random() * 0.6
        lrs.append(val)
    return lrs
