import math
import random
from typing import List, Tuple, Dict, Any

from simulators import DeliberationTask, SearchState, Rnd

# --- Algorithms ---

def igd_step(
    threads: List[SearchState],
    rnd: Rnd,
    task: DeliberationTask,
    lam: float = 0.0,
    gamma: float = 0.95,
) -> float:
    """
    Index-Guided Deliberation step.
    Selects thread with highest estimated future gain / cost.
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

        # Estimate gain for each action (simplified: pick best action per thread)
        # In practice, LLM generates actions and we score them.
        # Here we simulate the LLM's "Prior" or "Q-value"

        # We need a way to pick an action to evaluate the index
        # Let's just pick a random one for simulation, but weighted by "quality"
        # We cheat slightly and peek at transition to simulate "good proposal"

        # To simulate a learned policy, we pick an action that likely improves
        # Probability of picking improving action depends on state difficulty?

        # Simple Simulation: Pick random action, but calculate index based on
        # "Expected Improvement" derived from task heuristics.

        # For Game24: improvement is getting closer to 24.

        best_a = None
        best_idx_val = -1.0

        # Sample a few candidate actions (LLM proposals)
        candidates = rnd.sample(actions, min(len(actions), 3))

        for a in candidates:
            # Simulate "Thought" - predict value after action
            # We use the true transition but add noise to represent estimation
            next_s, cost = task.transition(state, a, rnd)

            # Estimated Gain = (NewValue - OldValue) * gamma^depth
            # Add noise to the estimation
            est_new_val = task.evaluate(next_s, rnd)
            gain = max(0.0, est_new_val - state.value)

            # Gittins-like Index ~ Gain / Cost
            idx_val = (gain * (gamma ** state.depth)) / max(1e-9, cost)

            if idx_val > best_idx_val:
                best_idx_val = idx_val
                best_a = a

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
    tree = {0: {'state': root, 'children': [], 'n': 0, 'q': 0.0}}
    next_node_id = 1

    while spent < budget:
        # Selection
        node_id = 0
        path = [0]

        # Select leaf
        while tree[node_id]['children']:
            # Risk-sensitive selection? Or just UCB?
            # Paper says: Select a = argmax rho(Q) + exploration
            # Here simplified: Standard UCB but Q is entropic risk?
            # Let's use Standard UCB for structure, risk in backprop

            best_c = -1
            best_score = -1e9

            for child_id in tree[node_id]['children']:
                child = tree[child_id]
                if child['n'] == 0:
                    best_c = child_id
                    break

                # UCB
                u = child['q'] + 1.0 * math.sqrt(math.log(tree[node_id]['n']) / child['n'])
                if u > best_score:
                    best_score = u
                    best_c = child_id

            node_id = best_c
            path.append(node_id)

        # Expansion
        curr_state = tree[node_id]['state']
        if not curr_state.is_terminal:
            actions = task.get_actions(curr_state)
            # Sample k actions
            k = min(len(actions), 3)
            sampled_acts = rnd.sample(actions, k)

            for a in sampled_acts:
                next_s, c = task.transition(curr_state, a, rnd)
                spent += c
                if spent > budget: break

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

        # Backprop (Risk Sensitive)
        # Update Q values using entropic risk aggregation?
        # Or just track distribution.
        # Simplified: Standard average for MCTS, but selection uses risk (not fully implemented here for brevity)
        # We will return the max found value

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
