import math
import random
from typing import List, Tuple

from simulators import ThreadState, geometric_improvement, rollout_value, TaskProfile

Rnd = random.Random


def igd_step(
    threads: List[ThreadState],
    rnd: Rnd,
    profile: TaskProfile,
    lam: float = 0.0,
    gamma: float = 0.95,
) -> float:
    """One IGD step: compute a crude index tau_i from expected discounted gains."""
    indices = []
    c = profile.cost_per_step
    for i, th in enumerate(threads):
        # Estimate potential improvement (oracle-ish for simulation)
        # In a real system, this comes from a learned value model
        # Here we simulate the 'estimated' improvement using the profile parameters + noise
        est_imp = profile.improvement_rate * profile.improvement_scale * (0.5 + rnd.random()) * (gamma ** th.depth)

        tau = est_imp / max(1e-9, c)
        indices.append((tau, i))

    tau, idx = max(indices, key=lambda t: t[0])

    if tau <= lam:
        return 0.0  # stop signal

    # Execute step
    real_imp = geometric_improvement(rnd, profile)
    # Diminishing returns: harder to improve if already high
    scaling = (1.0 - threads[idx].value)
    threads[idx].value = min(1.0, threads[idx].value + real_imp * scaling)
    threads[idx].depth += 1

    return c


def ucb_igd_step(
    threads: List[ThreadState],
    rnd: Rnd,
    t: int,
    counts: List[int],
    means: List[float],
    profile: TaskProfile,
    lam: float = 0.0,
) -> float:
    """UCB surrogate for IGD index."""
    ucbs = []
    c = profile.cost_per_step
    for i, _ in enumerate(threads):
        m = means[i]
        n = max(1, counts[i])
        bonus = math.sqrt(2 * math.log(max(2, t)) / n)
        # Heuristic value estimate
        val = m + bonus
        ucbs.append((val, i))

    u, idx = max(ucbs, key=lambda z: z[0])

    # Simple stopping rule for UCB (can be improved)
    if u * (0.95 ** threads[idx].depth) / c <= lam:
         pass # Don't stop immediately in UCB usually, but for consistency:
         # return 0.0

    # Execute
    r = geometric_improvement(rnd, profile)
    scaling = (1.0 - threads[idx].value)
    actual_gain = r * scaling

    counts[idx] += 1
    # Update mean reward for this arm
    means[idx] = (means[idx] * (counts[idx] - 1) + actual_gain) / counts[idx]

    threads[idx].value = min(1.0, threads[idx].value + actual_gain)
    threads[idx].depth += 1

    return c


def rs_mctt_budget(
    rnd: Rnd,
    profile: TaskProfile,
    depth: int = 4,
    eta_schedule=lambda d: -1.0, # Risk averse
) -> Tuple[float, float]:
    """Toy RS-MCTT: build a shallow tree with risk-averse selection."""
    B = 0.0
    c = profile.cost_per_step

    def node_value(d, base):
        nonlocal B
        if d == depth:
            v = rollout_value(rnd, d, base, profile)
            return v

        eta = eta_schedule(d)
        vals = []
        # Branching factor
        for _ in range(profile.branching_factor):
            # Simulate child outcomes
            imp = geometric_improvement(rnd, profile) * (1.0 - base)
            child_base = base + imp
            vals.append(node_value(d + 1, child_base))

        if eta == 0.0:
            score = sum(vals) / len(vals)
        else:
            # Softmin/Softmax
            try:
                score = (1.0 / eta) * math.log(
                    sum(math.exp(eta * v) for v in vals) / len(vals)
                )
            except ValueError:
                score = min(vals) if eta < 0 else max(vals)

        B += c
        return score

    # Start with some base value
    v = node_value(0, 0.2)
    return v, B


def csc_vote(rnd: Rnd, profile: TaskProfile, k: int = 8) -> Tuple[float, float]:
    """Counterfactual Self-Consistency."""
    # Probability of a chain being "correct"
    # SMR is harder than SCG usually for baseline models
    p_correct = 0.4 if profile.name == "SMR" else 0.5

    kept = 0
    correct_count = 0

    # Counterfactual check parameters
    # Rejection rate for incorrect chains (high is good)
    p_reject_incorrect = 0.7 if profile.verifier_noise < 0.1 else 0.5
    # Rejection rate for correct chains (low is good)
    p_reject_correct = 0.1

    cost = k * profile.cost_per_step # Parallel generation cost

    for _ in range(k):
        is_correct = rnd.random() < p_correct

        # Apply filter
        if is_correct:
            rejected = rnd.random() < p_reject_correct
        else:
            rejected = rnd.random() < p_reject_incorrect

        if not rejected:
            kept += 1
            if is_correct:
                correct_count += 1

    if kept == 0:
        return 0.0, cost

    # Majority vote accuracy approximation
    # If majority of kept are correct, we say the answer is correct (utility 1.0)
    # else utility 0.0
    # Or smooth utility: ratio
    final_acc = correct_count / kept
    # Boost it for majority voting effect
    if final_acc > 0.5:
        return 1.0, cost
    else:
        return 0.0, cost

def adr_loop(rnd: Rnd, profile: TaskProfile, budget: float) -> Tuple[float, float]:
    """Abduction-Deduction-Refutation loop simulation."""
    spent = 0.0
    best_val = 0.0
    c = profile.cost_per_step

    while spent + c <= budget:
        # Abduct: Propose hypothesis
        spent += c * 0.5
        hyp_quality = rnd.random()

        # Deduce & Refute: Test it
        spent += c * 0.5

        # Chance to find error
        if rnd.random() < (1.0 - hyp_quality) * 0.8: # Flawed hypothesis refuted
            continue # Try again

        # If not refuted, we accept it as candidate
        # Value depends on quality
        best_val = max(best_val, hyp_quality)

        # Stopping condition: if quality is high enough
        if best_val > 0.9:
            break

    return best_val, spent
