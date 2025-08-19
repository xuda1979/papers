"""Growing radial-basis-function network with online structural adaptation."""

from typing import Tuple

import numpy as np


class GrowingRBFNet:
    """Self-assembling RBF classifier with prototype growth and pruning.

    The model starts with one prototype per class and can spawn new
    prototypes when an input is poorly covered. Least-used prototypes are
    pruned to keep the model compact.

    Parameters
    ----------
    d : int
        Input dimension.
    k : int
        Number of classes (and initial prototypes).
    sigma : float, optional
        Width of the radial basis function.
    lr_w, lr_c : float, optional
        Learning rates for class weights and prototype centers.
    wd : float, optional
        Weight decay for class weights.
    min_phi_to_cover : float, optional
        Minimum activation required to avoid spawning a new prototype.
    usage_decay : float, optional
        Exponential decay applied to prototype usage counts.
    max_prototypes : int, optional
        Hard limit on the number of prototypes maintained.
    growth_budget : int, optional
        Maximum number of new prototypes allowed to spawn after
        initialization. ``None`` means unlimited growth.
    seed : int, optional
        Random seed for reproducibility.
    name : str, optional
        Identifier used in logging and plots.
    """

    def __init__(
        self,
        d: int,
        k: int,
        sigma: float = 0.65,
        lr_w: float = 0.25,
        lr_c: float = 0.06,
        wd: float = 1e-5,
        min_phi_to_cover: float = 0.25,
        usage_decay: float = 0.995,
        max_prototypes: int = 90,
        growth_budget: int | None = None,
        seed: int = 0,
        name: str = "GRBFN",
    ) -> None:
        self.d = d
        self.k = k
        self.sigma = sigma
        self.lr_w = lr_w
        self.lr_c = lr_c
        self.wd = wd
        self.min_phi_to_cover = min_phi_to_cover
        self.usage_decay = usage_decay
        self.max_prototypes = max_prototypes
        self.growth_budget = growth_budget
        self.name = name
        rng = np.random.RandomState(seed)
        self.centers = rng.randn(k, d) * 0.01
        self.W = np.zeros((k, k))
        for c in range(k):
            self.W[c, c] = 1.0
        self.usage = np.ones(k)
        self.M = k

    def _phi(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return basis activations and squared distances for ``x``."""
        diffs = self.centers - x
        dist2 = np.sum(diffs * diffs, axis=1)
        phi = np.exp(-dist2 / (2.0 * (self.sigma ** 2)))
        return phi, dist2

    def predict(self, x: np.ndarray) -> int:
        """Predict the class label for input ``x``."""
        phi, _ = self._phi(x)
        scores = phi @ self.W
        zmax = np.max(scores)
        p = np.exp(scores - zmax)
        p /= np.sum(p)
        return int(np.argmax(p))

    def _spawn(self, x: np.ndarray, y: int) -> None:
        """Create a new prototype centered at ``x`` for class ``y``."""
        if self.growth_budget is not None and self.growth_budget <= 0:
            return
        if self.M >= self.max_prototypes:
            self._prune()
            if self.M >= self.max_prototypes:
                return
        self.centers = np.vstack([self.centers, x.reshape(1, -1)])
        new_w = np.zeros((1, self.k))
        new_w[0, y] = 1.0
        self.W = np.vstack([self.W, new_w])
        self.usage = np.concatenate([self.usage, np.array([1.0])])
        self.M += 1
        if self.growth_budget is not None:
            self.growth_budget -= 1

    def _prune(self) -> None:
        """Remove the least-used prototype when capacity is exceeded."""
        if self.M <= self.k:
            return
        idx = int(np.argmin(self.usage))
        self.centers = np.delete(self.centers, idx, axis=0)
        self.W = np.delete(self.W, idx, axis=0)
        self.usage = np.delete(self.usage, idx, axis=0)
        self.M -= 1

    def update(self, x: np.ndarray, y: int) -> None:
        """Online update given input ``x`` and target label ``y``."""
        self.usage *= self.usage_decay
        phi, dist2 = self._phi(x)
        scores = phi @ self.W
        zmax = np.max(scores)
        exp_scores = np.exp(scores - zmax)
        p = exp_scores / np.sum(exp_scores)
        self.usage += phi * 0.05

        if np.max(phi) < self.min_phi_to_cover:
            self._spawn(x, y)
            phi, dist2 = self._phi(x)
            scores = phi @ self.W
            zmax = np.max(scores)
            exp_scores = np.exp(scores - zmax)
            p = exp_scores / np.sum(exp_scores)

        y_one = np.zeros(self.k)
        y_one[y] = 1.0
        ds = p - y_one
        gW = np.outer(phi, ds) + self.wd * self.W
        dphi = self.W @ ds
        dcenters = (phi[:, None] * (x - self.centers) / (self.sigma ** 2)) * dphi[:, None]
        self.W -= self.lr_w * gW
        self.centers -= self.lr_c * dcenters
        if self.M > self.max_prototypes:
            self._prune()
