
import numpy as np

class GrowingRBFNetPlastic:
    # GRBFN with dual timescales (slow weights + Hebbian fast weights), context-key gating,
    # and simple structural merges of nearby prototypes.
    def __init__(self, d, k, sigma=0.65, lr_w=0.25, lr_c=0.06, wd=1e-5,
                 min_phi_to_cover=0.25, usage_decay=0.995, max_prototypes=90,
                 hebb_eta=0.5, hebb_decay=0.95, gate_beta=1.0, gate_top=16, key_lr=0.02,
                 r_merge=0.4, merge_every=500, name="GRBFN+Plastic"):
        self.d = d; self.k = k; self.sigma = sigma
        self.lr_w = lr_w; self.lr_c = lr_c; self.wd = wd
        self.min_phi_to_cover = min_phi_to_cover
        self.usage_decay = usage_decay; self.max_prototypes = max_prototypes; self.name = name
        self.hebb_eta = hebb_eta; self.hebb_decay = hebb_decay
        self.gate_beta = gate_beta; self.gate_top = gate_top; self.key_lr = key_lr
        self.r_merge = r_merge; self.merge_every = merge_every
        self.t = 0

        rng = np.random.RandomState(0)
        self.centers = rng.randn(k, d) * 0.01
        self.W = np.zeros((k, k))
        for c in range(k): self.W[c, c] = 1.0
        self.H = np.zeros((k, k))         # fast weights
        self.K = rng.randn(k, d) * 0.01   # keys for gating
        self.usage = np.ones(k); self.M = k

    def _phi(self, x):
        diffs = self.centers - x
        dist2 = np.sum(diffs * diffs, axis=1)
        phi = np.exp(-dist2 / (2.0 * (self.sigma ** 2)))
        return phi, dist2

    def _gate(self, x, phi):
        g = 1.0 / (1.0 + np.exp(-self.gate_beta * (self.K @ x)))  # (M,)
        s = phi * g
        if self.M <= self.gate_top:
            mask = np.ones(self.M, dtype=bool)
        else:
            idx = np.argpartition(s, -self.gate_top)[-self.gate_top:]
            mask = np.zeros(self.M, dtype=bool); mask[idx] = True
        return s, mask

    def predict(self, x):
        phi, _ = self._phi(x)
        s, mask = self._gate(x, phi)
        scores = (phi * mask) @ (self.W + self.H)
        zmax = np.max(scores); p = np.exp(scores - zmax); p /= np.sum(p)
        return int(np.argmax(p))

    def _spawn(self, x, y):
        if self.M >= self.max_prototypes:
            self._prune()
            if self.M >= self.max_prototypes: return
        self.centers = np.vstack([self.centers, x.reshape(1,-1)])
        new_w = np.zeros((1, self.k)); new_h = np.zeros((1, self.k))
        new_w[0, y] = 1.0
        self.W = np.vstack([self.W, new_w])
        self.H = np.vstack([self.H, new_h])
        self.K = np.vstack([self.K, x.reshape(1,-1)])
        self.usage = np.concatenate([self.usage, np.array([1.0])])
        self.M += 1

    def _prune(self):
        if self.M <= self.k: return
        idx = int(np.argmin(self.usage))
        self.centers = np.delete(self.centers, idx, axis=0)
        self.W = np.delete(self.W, idx, axis=0)
        self.H = np.delete(self.H, idx, axis=0)
        self.K = np.delete(self.K, idx, axis=0)
        self.usage = np.delete(self.usage, idx, axis=0)
        self.M -= 1

    def _maybe_merge(self):
        if self.t % self.merge_every != 0: return
        if self.M <= self.k: return
        keep = np.ones(self.M, dtype=bool)
        for i in range(self.M):
            if not keep[i]: continue
            for j in range(i+1, self.M):
                if not keep[j]: continue
                if np.sum((self.centers[i]-self.centers[j])**2) < self.r_merge**2:
                    self.centers[i] = 0.5*(self.centers[i]+self.centers[j])
                    self.W[i] += self.W[j]
                    self.H[i] += self.H[j]
                    self.K[i] = 0.5*(self.K[i]+self.K[j])
                    keep[j] = False
        if np.any(~keep):
            self.centers = self.centers[keep]
            self.W = self.W[keep]
            self.H = self.H[keep]
            self.K = self.K[keep]
            self.usage = self.usage[keep]
            self.M = self.centers.shape[0]

    def update(self, x, y):
        self.t += 1
        self.usage *= self.usage_decay

        phi, dist2 = self._phi(x)
        s, mask = self._gate(x, phi)

        if np.max(s) < self.min_phi_to_cover:
            self._spawn(x, y)
            phi, dist2 = self._phi(x)
            s, mask = self._gate(x, phi)

        scores = (phi * mask) @ (self.W + self.H)
        zmax = np.max(scores); exp_scores = np.exp(scores - zmax); p = exp_scores / np.sum(exp_scores)
        y_one = np.zeros(self.k); y_one[y] = 1.0
        ds = p - y_one

        gW = np.outer(phi * mask, ds) + self.wd * self.W
        self.W -= self.lr_w * gW

        dphi = ((self.W + self.H) @ ds) * mask
        dcenters = (phi[:, None] * (x - self.centers) / (self.sigma ** 2)) * dphi[:, None]
        self.centers -= self.lr_c * dcenters

        # Hebbian fast weights and gating-key adaptation
        self.H *= self.hebb_decay
        self.H += self.hebb_eta * np.outer(phi * mask, (y_one - p))
        corr = -np.sum(ds)
        self.K += self.key_lr * ((phi * mask)[:, None] * (x - self.K)) * corr

        self.usage += s * 0.05
        if self.M > self.max_prototypes: self._prune()
        self._maybe_merge()
