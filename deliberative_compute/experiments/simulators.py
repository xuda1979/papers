import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
from abc import ABC, abstractmethod

Rnd = random.Random

@dataclass
class SearchState:
    data: Any
    value: float = 0.0
    depth: int = 0
    history: List[str] = field(default_factory=list)
    is_terminal: bool = False

class DeliberationTask(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def initial_state(self, rnd: Rnd) -> SearchState:
        pass

    @abstractmethod
    def get_actions(self, state: SearchState) -> List[Any]:
        """Return list of possible actions."""
        pass

    @abstractmethod
    def transition(self, state: SearchState, action: Any, rnd: Rnd) -> Tuple[SearchState, float]:
        """Apply action, return (next_state, cost)."""
        pass

    @abstractmethod
    def evaluate(self, state: SearchState, rnd: Rnd) -> float:
        """Heuristic evaluation (noisy)."""
        pass

    @abstractmethod
    def verify(self, state: SearchState, rnd: Rnd) -> bool:
        """Ground truth verification (or high-fidelity check)."""
        pass

# --- 1. Game of 24 Task ---

@dataclass
class Game24State_Data:
    nums: List[float]
    ops_history: List[str]

class GameOf24Task(DeliberationTask):
    def __init__(self):
        self._name = "GameOf24"
        self.ops = ['+', '-', '*', '/']
        # Hardcoded set of solvable 24 puzzles for stability
        self.puzzles = [
            [4, 4, 10, 10], # (10*10 - 4) / 4 = 24
            [1, 5, 5, 5],   # 5 * (5 - 1/5) = 24
            [3, 3, 8, 8],   # 8 / (3 - 8/3) = 24
            [5, 5, 5, 1],
            [1, 3, 4, 6],   # 6 / (1 - 3/4) = 24
            [8, 8, 3, 3],
            [2, 4, 10, 10],
            [1, 2, 8, 8],
            # Harder puzzles (require more creative solutions)
            [1, 2, 7, 7],   # 7 * (2 + 7 - 1) / ? - harder
            [2, 3, 5, 12],  # Multiple paths but not obvious
            [1, 4, 5, 6],   # (6 - 1) * 4 + 5 - 1 = 24
            [3, 6, 6, 11],  # Requires careful ordering
            [2, 5, 5, 10],  # (5 - 2) * 10 - 5 - 1 = 24
            [1, 1, 5, 5],   # Hard: 5 * 5 - 1 * 1 = 24
        ]
        # Track solve difficulty - some puzzles are harder
        self.puzzle_difficulty = {
            tuple([4, 4, 10, 10]): 0.3,
            tuple([1, 5, 5, 5]): 0.5,
            tuple([3, 3, 8, 8]): 0.6,
            tuple([1, 2, 7, 7]): 0.8,
            tuple([2, 3, 5, 12]): 0.7,
        }

    @property
    def name(self) -> str:
        return self._name

    def initial_state(self, rnd: Rnd) -> SearchState:
        nums = list(rnd.choice(self.puzzles))
        return SearchState(
            data=Game24State_Data(nums=nums, ops_history=[]),
            value=self._heuristic(nums),
            depth=0
        )

    def get_actions(self, state: SearchState) -> List[Any]:
        if state.is_terminal:
            return []

        nums = state.data.nums
        actions = []
        n = len(nums)
        if n < 2: return []

        # Generate all pairs and ops
        # To limit branching factor for simulation, we can sample
        # But 4 numbers -> 4*3/2 = 6 pairs * 4 ops = 24 actions. Manageable.
        for i in range(n):
            for j in range(n):
                if i == j: continue
                for op in self.ops:
                    if op in ['+', '*'] and i > j: continue # Commutativity
                    if op == '/' and abs(nums[j]) < 1e-4: continue # Div by zero
                    actions.append((i, j, op))
        return actions

    def transition(self, state: SearchState, action: Any, rnd: Rnd) -> Tuple[SearchState, float]:
        i, j, op = action
        nums = state.data.nums
        a, b = nums[i], nums[j]

        res = 0.0
        if op == '+': res = a + b
        elif op == '-': res = a - b
        elif op == '*': res = a * b
        elif op == '/': res = a / b

        new_nums = [x for k, x in enumerate(nums) if k != i and k != j] + [res]
        new_hist = state.data.ops_history + [f"{a} {op} {b} = {res}"]

        is_sol = False
        if len(new_nums) == 1 and abs(new_nums[0] - 24.0) < 1e-4:
            is_sol = True

        new_state = SearchState(
            data=Game24State_Data(nums=new_nums, ops_history=new_hist),
            value=1.0 if is_sol else self._heuristic(new_nums),
            depth=state.depth + 1,
            is_terminal=is_sol or len(new_nums) == 1
        )

        # Cost: arithmetic is cheap, but let's say 1.0 unit
        return new_state, 1.0

    def _heuristic(self, nums: List[float]) -> float:
        # Distance to 24.
        # Simple heuristic: how close is the closest number or pair sum to 24?
        if not nums: return 0.0
        best_dist = min(abs(x - 24.0) for x in nums)
        # Check simple 1-step reachability (pairs)
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                a, b = nums[i], nums[j]
                for v in [a+b, a-b, b-a, a*b, a/b if abs(b)>1e-4 else 1e9]:
                    best_dist = min(best_dist, abs(v - 24.0))

        return 1.0 / (1.0 + 0.1 * best_dist)

    def evaluate(self, state: SearchState, rnd: Rnd) -> float:
        # Simulated LLM self-eval: add noise to heuristic
        # More noise to simulate imperfect LLM scoring
        base = state.value
        noise = rnd.gauss(0, 0.15)  # Increased noise
        return max(0.0, min(1.0, base + noise))

    def verify(self, state: SearchState, rnd: Rnd) -> bool:
        # Check if 24
        if not state.data.nums: return False
        for x in state.data.nums:
            if abs(x - 24.0) < 1e-4:
                return True
        return False

# --- 2. Bitstring Search (Code Gen Proxy) ---

@dataclass
class BitstringData:
    bits: List[int]
    target: List[int]

class BitstringTask(DeliberationTask):
    """
    Task: Find a hidden bitstring.
    Analogy: Writing code to pass N unit tests.
    """
    def __init__(self, length=12):
        self._name = "BitSearch"
        self.length = length

    @property
    def name(self) -> str:
        return self._name

    def initial_state(self, rnd: Rnd) -> SearchState:
        # Each episode/thread has its own target
        target = [rnd.randint(0, 1) for _ in range(self.length)]
        start_bits = [rnd.randint(0, 1) for _ in range(self.length)]

        # Calculate initial score
        score = self._score(start_bits, target)

        return SearchState(
            data=BitstringData(bits=start_bits, target=target),
            value=score,
            depth=0
        )

    def _score(self, bits: List[int], target: List[int]) -> float:
        match = sum(1 for a, b in zip(bits, target) if a == b)
        return match / self.length

    def get_actions(self, state: SearchState) -> List[Any]:
        if state.is_terminal: return []
        # Actions: Flip 1 bit, Flip 2 bits
        actions = []
        indices = list(range(self.length))
        # 1-bit flips
        for i in indices:
            actions.append(('flip', i))
        # 2-bit flips (a few)
        for _ in range(5):
             i, j = random.sample(indices, 2)
             actions.append(('flip2', i, j))
        return actions

    def transition(self, state: SearchState, action: Any, rnd: Rnd) -> Tuple[SearchState, float]:
        bits = list(state.data.bits)
        op = action[0]
        if op == 'flip':
            bits[action[1]] = 1 - bits[action[1]]
        elif op == 'flip2':
            bits[action[1]] = 1 - bits[action[1]]
            bits[action[2]] = 1 - bits[action[2]]

        # Use target from state
        val = self._score(bits, state.data.target)
        is_solved = (val == 1.0)

        new_state = SearchState(
            data=BitstringData(bits=bits, target=state.data.target),
            value=val,
            depth=state.depth + 1,
            is_terminal=is_solved
        )
        return new_state, 1.0 # Cost 1

    def evaluate(self, state: SearchState, rnd: Rnd) -> float:
        # Self-eval: Noisy estimate
        true_val = state.value
        noise = (rnd.random() - 0.5) * 0.1
        return max(0.0, min(1.0, true_val + noise))

    def verify(self, state: SearchState, rnd: Rnd) -> bool:
        return state.value == 1.0
