import math
import random
from typing import List, Tuple, Dict, Any

from simulators import DeliberationTask, SearchState, Rnd

# --- Algorithms ---

# Noise parameters for realistic LLM estimation simulation
# These control how "oracle-like" the value estimates are
DEFAULT_ESTIMATION_NOISE = 0.25  # 25% noise in value estimates (increased from 20%)
DEFAULT_SELECTION_ERROR_PROB = 0.20  # 20% chance of selecting wrong action (increased from 15%)


def add_estimation_noise(true_value: float, rnd: Rnd, noise_level: float = DEFAULT_ESTIMATION_NOISE) -> float:
    """
    Add realistic noise to value estimates to simulate imperfect LLM scoring.
    Models the gap between oracle evaluation and LLM-based heuristics.
    """
    if noise_level <= 0:
        return true_value
    # Multiplicative noise (proportional to value) + additive noise
    noise = rnd.gauss(0, noise_level * max(0.1, abs(true_value)))
    return max(0.0, true_value + noise)


def igd_step(
    threads: List[SearchState],
    rnd: Rnd,
    task: DeliberationTask,
    lam: float = 0.0,
    gamma: float = 0.95,
    estimation_noise: float = DEFAULT_ESTIMATION_NOISE,
    selection_error_prob: float = DEFAULT_SELECTION_ERROR_PROB,
) -> float:
    """
    Index-Guided Deliberation step.
    Selects thread with highest estimated future gain / cost.
    
    Args:
        threads: List of active reasoning threads
        rnd: Random number generator for reproducibility
        task: The deliberation task
        lam: Stopping threshold (shadow price of compute)
        gamma: Discount factor for depth
        estimation_noise: Noise level for value estimates (0 = oracle, 0.2 = realistic)
        selection_error_prob: Probability of selecting suboptimal action
    """
    indices = []

    # 1. Compute Indices
    for i, state in enumerate(threads):
        if state.is_terminal:
            indices.append((-1.0, i, None)) # Don't expand
            continue

        actions = task.get_actions(state)
        if not actions:
            indices.append((-1.0, i, None))
            continue

        # Estimate gain for each action
        # In practice, LLM generates actions and scores them with inherent noise
        # We simulate this with explicit estimation noise

        best_a = None
        best_idx_val = -1.0
        all_candidates = []  # Track all for potential selection error

        # Sample a few candidate actions (LLM proposals)
        candidates = rnd.sample(actions, min(len(actions), 3))

        for a in candidates:
            # Simulate "Thought" - predict value after action
            next_s, cost = task.transition(state, a, rnd)

            # Get true value, then add noise to simulate imperfect LLM estimation
            true_new_val = task.evaluate(next_s, rnd)
            est_new_val = add_estimation_noise(true_new_val, rnd, estimation_noise)
            
            # Estimated Gain = (NewValue - OldValue) * gamma^depth
            # Also add noise to current state estimate for consistency
            est_curr_val = add_estimation_noise(state.value, rnd, estimation_noise * 0.5)
            gain = max(0.0, est_new_val - est_curr_val)

            # Gittins-like Index ~ Gain / Cost
            idx_val = (gain * (gamma ** state.depth)) / max(1e-9, cost)
            
            all_candidates.append((idx_val, a))

            if idx_val > best_idx_val:
                best_idx_val = idx_val
                best_a = a

        # Simulate selection error: sometimes pick suboptimal action
        if all_candidates and rnd.random() < selection_error_prob:
            # Pick a random action instead of the best one
            _, best_a = rnd.choice(all_candidates)
            # Recalculate index for the chosen action
            best_idx_val = next(idx for idx, a in all_candidates if a == best_a)

        indices.append((best_idx_val, i, best_a))

    # 2. Select Best Thread
    best_tau, best_idx, best_action = max(indices, key=lambda x: x[0])

    # 3. Stop if value too low
    if best_tau <= lam:
        return 0.0

    # 4. Execute
    state = threads[best_idx]
    next_s, cost = task.transition(state, best_action, rnd)
    threads[best_idx] = next_s # Update thread

    return cost


def ucb_igd_step(
    threads: List[SearchState],
    rnd: Rnd,
    t: int,
    counts: List[int],
    means: List[float],
    task: DeliberationTask,
    lam: float = 0.0,
) -> float:
    """
    UCB-IGD: Treat threads as arms.
    """
    ucbs = []
    candidate_actions = {}

    for i, state in enumerate(threads):
        if state.is_terminal:
            ucbs.append((-1e9, i))
            continue

        # UCB Score
        n = max(1, counts[i])
        bonus = 1.0 * math.sqrt(2 * math.log(max(2, t)) / n)
        val = means[i] + bonus
        ucbs.append((val, i))

        # Pre-select action (random for now)
        acts = task.get_actions(state)
        if acts:
            candidate_actions[i] = rnd.choice(acts)
        else:
            candidate_actions[i] = None

    score, idx = max(ucbs, key=lambda x: x[0])
    action = candidate_actions.get(idx)

    if action is None: return 0.0

    # Execute
    state = threads[idx]
    next_s, cost = task.transition(state, action, rnd)

    # Feedback: Improvement
    gain = max(0.0, next_s.value - state.value)

    # Update Stats
    counts[idx] += 1
    means[idx] = (means[idx] * (counts[idx]-1) + gain) / counts[idx]

    threads[idx] = next_s

    return cost


def rs_mctt_budget(
    rnd: Rnd,
    task: DeliberationTask,
    budget: float,
    eta: float = -5.0 # Risk averse
) -> Tuple[float, float]:
    """
    Risk-Sensitive MCTT (Budgeted).
    """
    root = task.initial_state(rnd)
    spent = 0.0

    # Simple MCTS-like rollout or Tree Search
    # With limited budget, we expand the tree.

    # Let's maintain a tree structure.
    # Nodes: (State, children, visits, value_est)
    tree = {0: {'state': root, 'children': [], 'n': 1, 'q': 0.0}}
    next_node_id = 1
    max_tree_size = 500  # Limit tree size to prevent infinite growth
    max_iterations = int(budget * 2)  # Limit iterations
    iteration = 0

    while spent < budget and next_node_id < max_tree_size and iteration < max_iterations:
        iteration += 1
        # Selection
        node_id = 0
        path = [0]

        # Select leaf
        depth = 0
        max_depth = 20
        while tree[node_id]['children'] and depth < max_depth:
            depth += 1
            # Risk-sensitive selection using UCB
            best_c = -1
            best_score = -1e9

            for child_id in tree[node_id]['children']:
                child = tree[child_id]
                if child['n'] == 0:
                    best_c = child_id
                    break

                # UCB with parent visit count safety check
                parent_n = max(1, tree[node_id]['n'])
                child_n = max(1, child['n'])
                u = child['q'] + 1.0 * math.sqrt(math.log(parent_n) / child_n)
                if u > best_score:
                    best_score = u
                    best_c = child_id

            if best_c == -1:
                break
            node_id = best_c
            path.append(node_id)

        # Expansion
        curr_state = tree[node_id]['state']
        if not curr_state.is_terminal and next_node_id < max_tree_size:
            actions = task.get_actions(curr_state)
            if actions:
                # Sample k actions
                k = min(len(actions), 3)
                sampled_acts = rnd.sample(actions, k)

                for a in sampled_acts:
                    if spent > budget or next_node_id >= max_tree_size:
                        break
                    next_s, c = task.transition(curr_state, a, rnd)
                    spent += c

                    tree[next_node_id] = {'state': next_s, 'children': [], 'n': 0, 'q': 0.0}
                    tree[node_id]['children'].append(next_node_id)
                    next_node_id += 1

                if tree[node_id]['children']:
                    # Pick one child to simulate
                    node_id = rnd.choice(tree[node_id]['children'])
                    path.append(node_id)

        # Simulation / Evaluation
        leaf_state = tree[node_id]['state']
        val = task.evaluate(leaf_state, rnd)

        # Backprop with risk-sensitive update
        for nid in reversed(path):
            tree[nid]['n'] += 1
            # Standard avg update
            tree[nid]['q'] += (val - tree[nid]['q']) / tree[nid]['n']

    # Return best value found in tree
    best_v = 0.0
    for nid in tree:
        if tree[nid]['state'].value > best_v:
            best_v = tree[nid]['state'].value

    return best_v, spent


def csc_vote(rnd: Rnd, task: DeliberationTask, k: int = 8) -> Tuple[float, float]:
    """Counterfactual Self-Consistency."""
    # Run k independent chains (greedy rollouts)
    chains = []
    cost = 0.0

    for _ in range(k):
        curr = task.initial_state(rnd)
        # Run chain for some steps (e.g. 5)
        for _ in range(5):
            if curr.is_terminal: break
            actions = task.get_actions(curr)
            if not actions: break

            # Greedy step
            # Sample a few, pick best eval
            best_a = None
            best_v = -1.0
            candidates = rnd.sample(actions, min(len(actions), 2))

            step_cost = 0
            for a in candidates:
                ns, c = task.transition(curr, a, rnd)
                v = task.evaluate(ns, rnd)
                step_cost += c # We pay for all evaluations
                if v > best_v:
                    best_v = v
                    best_a = a

            cost += step_cost
            if best_a:
                curr, c = task.transition(curr, best_a, rnd)
                # cost += c # Already paid in eval? No, transition cost separate?
                # Let's say eval included transition cost.

        chains.append(curr)

    # CSC Logic: Verify Consistency
    # In Game24: "Consistency" is hard to define without multiple equivalent expressions.
    # But we can verify correctness.

    # "Message Passing": Discard chains that are obviously wrong (Counterfactual check)
    # Check: does the expression evaluate to 24?
    # This is "verification", which CSC uses.

    valid_chains = []
    for c_state in chains:
        # Check if consistent (simulated cheap check)
        # In CSC paper, we use relations. Here we use `verify` as the check
        if task.verify(c_state, rnd):
            valid_chains.append(c_state)

    if not valid_chains:
        return 0.0, cost

    # Majority vote? Or max value?
    # Return 1.0 if any valid found (since valid means solved in 24)
    return 1.0, cost

def adr_loop(rnd: Rnd, task: DeliberationTask, budget: float) -> Tuple[float, float]:
    """Abduction-Deduction-Refutation."""
    # A simplified version:
    # 1. Propose solution (Abduction)
    # 2. Verify/Refute (Deduction/Refutation)
    # 3. If refuted, refine/retry.

    spent = 0.0
    best_val = 0.0

    while spent < budget:
        # Abduct (Propose)
        state = task.initial_state(rnd)
        # Fast rollout to get a proposal
        for _ in range(5):
            if state.is_terminal: break
            acts = task.get_actions(state)
            if not acts: break
            a = rnd.choice(acts)
            state, c = task.transition(state, a, rnd)
            spent += c

        if spent > budget: break

        # Deduce/Refute (Verify)
        # Check validity
        is_correct = task.verify(state, rnd)
        spent += 1.0 # Verification cost

        if is_correct:
            return 1.0, spent

        if state.value > best_val:
            best_val = state.value

    return best_val, spent


def best_of_n(
    rnd: Rnd, 
    task: DeliberationTask, 
    n: int, 
    steps_per_sample: int = 5,
    use_verifier: bool = True
) -> Tuple[float, float]:
    """
    Best-of-N baseline: Sample N independent solutions and return the best.
    
    This is the most common test-time compute baseline. It samples N independent
    reasoning chains and selects the one with the highest heuristic score
    (or the first verified correct solution if use_verifier=True).
    
    Args:
        rnd: Random number generator
        task: The deliberation task
        n: Number of samples to generate
        steps_per_sample: Maximum reasoning steps per sample
        use_verifier: If True, return first verified solution; else return highest-scored
        
    Returns:
        (best_value, total_cost): Best solution quality and compute spent
    """
    samples = []
    total_cost = 0.0
    
    for _ in range(n):
        # Generate one complete reasoning chain
        state = task.initial_state(rnd)
        sample_cost = 0.0
        
        for _ in range(steps_per_sample):
            if state.is_terminal:
                break
            
            actions = task.get_actions(state)
            if not actions:
                break
            
            # Greedy selection: pick action that looks best (with noise)
            best_action = None
            best_score = -float('inf')
            
            # Sample subset of actions (simulates LLM proposal distribution)
            candidates = rnd.sample(actions, min(len(actions), 3))
            
            for a in candidates:
                next_s, c = task.transition(state, a, rnd)
                # Add estimation noise for realistic simulation
                score = add_estimation_noise(task.evaluate(next_s, rnd), rnd, DEFAULT_ESTIMATION_NOISE)
                sample_cost += c * 0.1  # Partial cost for evaluation
                
                if score > best_score:
                    best_score = score
                    best_action = a
            
            if best_action is None:
                break
                
            state, c = task.transition(state, best_action, rnd)
            sample_cost += c
        
        total_cost += sample_cost
        samples.append(state)
        
        # Early exit if verified correct (optional optimization)
        if use_verifier and task.verify(state, rnd):
            total_cost += 1.0  # Verification cost
            return 1.0, total_cost
    
    # If using verifier, check all samples
    if use_verifier:
        for state in samples:
            total_cost += 1.0  # Verification cost per sample
            if task.verify(state, rnd):
                return 1.0, total_cost
    
    # Return best heuristic value if no verified solution found
    if samples:
        best_val = max(s.value for s in samples)
        return best_val, total_cost
    
    return 0.0, total_cost


def best_of_n_budget(
    rnd: Rnd, 
    task: DeliberationTask, 
    budget: float,
    steps_per_sample: int = 5,
) -> Tuple[float, float]:
    """
    Best-of-N with budget constraint: Sample as many solutions as budget allows.
    
    Args:
        rnd: Random number generator
        task: The deliberation task
        budget: Maximum compute budget
        steps_per_sample: Maximum reasoning steps per sample
        
    Returns:
        (best_value, total_cost): Best solution quality and compute spent
    """
    samples = []
    spent = 0.0
    
    while spent < budget:
        # Generate one complete reasoning chain
        state = task.initial_state(rnd)
        sample_cost = 0.0
        
        for _ in range(steps_per_sample):
            if state.is_terminal:
                break
            
            actions = task.get_actions(state)
            if not actions:
                break
            
            # Random action selection (simulates temperature sampling)
            a = rnd.choice(actions)
            state, c = task.transition(state, a, rnd)
            sample_cost += c
            
            if spent + sample_cost > budget:
                break
        
        spent += sample_cost
        
        if spent > budget:
            break
            
        samples.append(state)
        
        # Verification cost
        spent += 1.0
        if task.verify(state, rnd):
            return 1.0, spent
    
    # Return best heuristic value if no verified solution found
    if samples:
        best_val = max(s.value for s in samples)
        return best_val, spent
    
    return 0.0, spent
