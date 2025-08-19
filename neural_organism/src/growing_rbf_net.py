
import numpy as np

class GrowingRBFNet:
    # Baseline self-assembling network with radial basis activations and class weights.
    def __init__(self, d, k, sigma=0.65, lr_w=0.25, lr_c=0.06, wd=1e-5,
                 min_phi_to_cover=0.25, usage_decay=0.995, max_prototypes=90, name="GRBFN"):
        self.d = d; self.k = k; self.sigma = sigma
        self.lr_w = lr_w; self.lr_c = lr_c; self.wd = wd
        self.min_phi_to_cover = min_phi_to_cover
        self.usage_decay = usage_decay; self.max_prototypes = max_prototypes; self.name = name
        rng = np.random.RandomState(0)
        self.centers = rng.randn(k, d) * 0.01
        self.W = np.zeros((k, k))
        for c in range(k): self.W[c, c] = 1.0
        self.usage = np.ones(k); self.M = k

    def _phi(self, x):
        diffs = self.centers - x
        dist2 = np.sum(diffs * diffs, axis=1)
        phi = np.exp(-dist2 / (2.0 * (self.sigma ** 2)))
        return phi, dist2

    def predict(self, x):
        phi, _ = self._phi(x)
        scores = phi @ self.W
        zmax = np.max(scores); p = np.exp(scores - zmax); p /= np.sum(p)
        return int(np.argmax(p))

    def _spawn(self, x, y):
        if self.M >= self.max_prototypes:
            self._prune()
            if self.M >= self.max_prototypes: return
        self.centers = np.vstack([self.centers, x.reshape(1, -1)])
        new_w = np.zeros((1, self.k)); new_w[0, y] = 1.0
        self.W = np.vstack([self.W, new_w])
        self.usage = np.concatenate([self.usage, np.array([1.0])])
        self.M += 1

    def _prune(self):
        if self.M <= self.k: return
        idx = int(np.argmin(self.usage))
        self.centers = np.delete(self.centers, idx, axis=0)
        self.W = np.delete(self.W, idx, axis=0)
        self.usage = np.delete(self.usage, idx, axis=0)
        self.M -= 1

    def update(self, x, y):
        self.usage *= self.usage_decay
        phi, dist2 = self._phi(x)
        scores = phi @ self.W
        zmax = np.max(scores); exp_scores = np.exp(scores - zmax); p = exp_scores / np.sum(exp_scores)
        self.usage += phi * 0.05

        if np.max(phi) < self.min_phi_to_cover:
            self._spawn(x, y)
            phi, dist2 = self._phi(x)
            scores = phi @ self.W
            zmax = np.max(scores); exp_scores = np.exp(scores - zmax); p = exp_scores / np.sum(exp_scores)

        y_one = np.zeros(self.k); y_one[y] = 1.0
        ds = p - y_one
        gW = np.outer(phi, ds) + self.wd * self.W
        dphi = self.W @ ds
        dcenters = (phi[:, None] * (x - self.centers) / (self.sigma ** 2)) * dphi[:, None]
        self.W -= self.lr_w * gW
        self.centers -= self.lr_c * dcenters
        if self.M > self.max_prototypes: self._prune()
