import numpy as np


class RuntimeGate:
    """A simple runtime-enforced gate based on a consumable budget.

    This class explicitly implements the concept of a runtime-enforced
    constraint as described in the paper. The model's internal logic can
    query this gate to see if an action (like structural growth) is
    permitted. The gate's state is controlled externally.
    """
    def __init__(self, budget: int | None):
        self._budget = budget

    def is_open(self) -> bool:
        """Returns True if the gate allows the action."""
        if self._budget is None:
            return True
        return self._budget > 0

    def consume(self):
        """Consumes one unit from the budget, effectively closing the gate
        by one step."""
        if self._budget is not None:
            self._budget -= 1


class GrowingRBFNetPlastic:
    """A Growing RBF Network with plastic synapses and advanced features.

    This model implements several mechanisms for online, continual learning in
    non-stationary environments:
    1.  **Structural Growth:** Spawns new RBF prototypes for novel data.
    2.  **Structural Pruning:** Removes least-used prototypes.
    3.  **Structural Merging:** Fuses nearby prototypes to compact the model.
    4.  **Dual Timescale Plasticity:** Learns slow base weights (W) and fast
        Hebbian weights (H) for rapid adaptation.
    5.  **Context-Key Gating:** A gating mechanism (`_gate`) uses learned keys
        (K) to select a sparse subset of active prototypes for each input.
    6.  **Runtime-Enforced Budgeting:** Growth is controlled by an external
        `RuntimeGate`, demonstrating the core concept of the paper.

    Parameters
    ----------
    d : int
        Input dimension.
    k : int
        Number of output classes.
    sigma : float
        Width of the RBF kernels.
    lr_w, lr_c : float
        Learning rates for slow weights and prototype centers.
    wd : float
        Weight decay for the slow weights.
    min_phi_to_cover : float
        Activation threshold below which a new prototype may be spawned.
    usage_decay : float
        Decay rate for the usage counter of each prototype.
    max_prototypes : int
        Hard limit on the number of prototypes.
    growth_gate : RuntimeGate, optional
        External gate to control growth. If None, growth is unlimited.
    hebb_eta, hebb_decay : float
        Learning rate and decay for the fast Hebbian weights.
    gate_beta, gate_top : float, int
        Parameters for the context-key gating mechanism.
    key_lr : float
        Learning rate for the gating keys.
    r_merge : float
        Distance threshold for merging prototypes.
    merge_every : int
        Frequency (in steps) at which the merge check is performed.
    name : str
        Identifier for the model.
    """
    def __init__(self, d, k, sigma=0.65, lr_w=0.25, lr_c=0.06, wd=1e-5,
                 min_phi_to_cover=0.25, usage_decay=0.995, max_prototypes=90,
                 growth_gate: RuntimeGate | None = None,
                 hebb_eta=0.5, hebb_decay=0.95, gate_beta=1.0, gate_top=16, key_lr=0.02,
                 r_merge=0.4, merge_every=500, name="GRBFN+Plastic"):
        # Store hyperparameters
        self.d = d; self.k = k; self.sigma = sigma
        self.lr_w = lr_w; self.lr_c = lr_c; self.wd = wd
        self.min_phi_to_cover = min_phi_to_cover
        self.usage_decay = usage_decay; self.max_prototypes = max_prototypes; self.name = name
        self.growth_gate = growth_gate if growth_gate is not None else RuntimeGate(None)
        self.hebb_eta = hebb_eta; self.hebb_decay = hebb_decay
        self.gate_beta = gate_beta; self.gate_top = gate_top; self.key_lr = key_lr
        self.r_merge = r_merge; self.merge_every = merge_every
        self.t = 0  # Internal timestep

        # Initialize model parameters
        rng = np.random.RandomState(0)
        self.centers = rng.randn(k, d) * 0.01  # Prototypes
        self.W = np.zeros((k, k))              # Slow weights
        for c in range(k): self.W[c, c] = 1.0
        self.H = np.zeros((k, k))              # Fast (Hebbian) weights
        self.K = rng.randn(k, d) * 0.01        # Gating keys
        self.usage = np.ones(k)                # Usage counter for pruning
        self.M = k                             # Number of prototypes

    def _phi(self, x):
        """Calculates RBF activations for a given input."""
        diffs = self.centers - x
        dist2 = np.sum(diffs * diffs, axis=1)
        phi = np.exp(-dist2 / (2.0 * (self.sigma ** 2)))
        return phi, dist2

    def _gate(self, x, phi):
        """Applies context-gating to select a sparse subset of prototypes."""
        g = 1.0 / (1.0 + np.exp(-self.gate_beta * (self.K @ x)))  # (M,)
        s = phi * g
        if self.M <= self.gate_top:
            # If fewer prototypes than the gate size, use all of them
            mask = np.ones(self.M, dtype=bool)
        else:
            # Select the top-k most active gated units
            idx = np.argpartition(s, -self.gate_top)[-self.gate_top:]
            mask = np.zeros(self.M, dtype=bool); mask[idx] = True
        return s, mask

    def predict(self, x):
        """Predicts the class label for a given input."""
        phi, _ = self._phi(x)
        s, mask = self._gate(x, phi)
        # Prediction uses both slow and fast weights
        scores = (phi * mask) @ (self.W + self.H)
        zmax = np.max(scores); p = np.exp(scores - zmax); p /= np.sum(p)
        return int(np.argmax(p))

    def _spawn(self, x, y):
        """Creates a new prototype, subject to the runtime gate and capacity."""
        if not self.growth_gate.is_open():
            return  # Growth is blocked by the external gate

        if self.M >= self.max_prototypes:
            self._prune()  # Attempt to prune a unit to make space
            if self.M >= self.max_prototypes: return

        # Add the new prototype
        self.centers = np.vstack([self.centers, x.reshape(1,-1)])
        new_w = np.zeros((1, self.k)); new_h = np.zeros((1, self.k))
        new_w[0, y] = 1.0 # Initialize weight to predict the target class
        self.W = np.vstack([self.W, new_w])
        self.H = np.vstack([self.H, new_h])
        self.K = np.vstack([self.K, x.reshape(1,-1)]) # Initialize key
        self.usage = np.concatenate([self.usage, np.array([1.0])])
        self.M += 1
        self.growth_gate.consume() # Inform the gate that a unit of budget was used

    def _prune(self):
        """Removes the least-used prototype."""
        if self.M <= self.k: return # Don't prune below the initial number
        idx = int(np.argmin(self.usage))
        self.centers = np.delete(self.centers, idx, axis=0)
        self.W = np.delete(self.W, idx, axis=0)
        self.H = np.delete(self.H, idx, axis=0)
        self.K = np.delete(self.K, idx, axis=0)
        self.usage = np.delete(self.usage, idx, axis=0)
        self.M -= 1

    def _maybe_merge(self):
        """Periodically checks for and merges nearby prototypes."""
        if self.t % self.merge_every != 0 or self.M <= self.k:
            return

        keep = np.ones(self.M, dtype=bool)
        for i in range(self.M):
            if not keep[i]: continue
            for j in range(i + 1, self.M):
                if not keep[j]: continue
                # If two prototypes are too close, merge them
                if np.sum((self.centers[i] - self.centers[j]) ** 2) < self.r_merge ** 2:
                    # Merge by averaging centers/keys and summing weights
                    self.centers[i] = 0.5 * (self.centers[i] + self.centers[j])
                    self.W[i] += self.W[j]
                    self.H[i] += self.H[j]
                    self.K[i] = 0.5 * (self.K[i] + self.K[j])
                    keep[j] = False # Mark the second prototype for deletion

        if np.any(~keep):
            self.centers = self.centers[keep]
            self.W = self.W[keep]
            self.H = self.H[keep]
            self.K = self.K[keep]
            self.usage = self.usage[keep]
            self.M = self.centers.shape[0]

    def update(self, x, y):
        """Performs a full online update step."""
        self.t += 1
        self.usage *= self.usage_decay

        phi, dist2 = self._phi(x)
        s, mask = self._gate(x, phi)

        # If input is not well-covered by any prototype, spawn a new one
        if np.max(s) < self.min_phi_to_cover:
            self._spawn(x, y)
            # Re-compute activations after spawning
            phi, dist2 = self._phi(x)
            s, mask = self._gate(x, phi)

        # --- Standard supervised update using cross-entropy loss ---
        scores = (phi * mask) @ (self.W + self.H)
        zmax = np.max(scores); exp_scores = np.exp(scores - zmax); p = exp_scores / np.sum(exp_scores)
        y_one = np.zeros(self.k); y_one[y] = 1.0
        ds = p - y_one  # Gradient of loss w.r.t. scores

        # Update slow weights
        gW = np.outer(phi * mask, ds) + self.wd * self.W
        self.W -= self.lr_w * gW

        # Update prototype centers
        dphi = ((self.W + self.H) @ ds) * mask
        dcenters = (phi[:, None] * (x - self.centers) / (self.sigma ** 2)) * dphi[:, None]
        self.centers -= self.lr_c * dcenters

        # --- Plasticity updates ---
        # Update fast (Hebbian) weights
        self.H *= self.hebb_decay
        self.H += self.hebb_eta * np.outer(phi * mask, (y_one - p))

        # Update gating keys to be more responsive to this input
        corr = -np.sum(ds) # A measure of correctness
        self.K += self.key_lr * ((phi * mask)[:, None] * (x - self.K)) * corr

        # --- Structural maintenance ---
        self.usage += s * 0.05
        if self.M > self.max_prototypes: self._prune()
        self._maybe_merge()
