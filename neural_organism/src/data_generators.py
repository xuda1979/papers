
import numpy as np

class MixtureDriftGaussians:
    # 2-D, K-class classification. Each class distribution is a mixture of n_components Gaussians.
    # Every 'steps_per_phase' samples, component means are resampled (abrupt drift).
    def __init__(self, d=2, k=3, n_components=2, sigma=0.4, range_lim=3.0,
                 steps_per_phase=2000, num_phases=6, seed=0):
        self.d = d; self.k = k; self.n_components = n_components
        self.sigma = sigma; self.range_lim = range_lim
        self.steps_per_phase = steps_per_phase; self.num_phases = num_phases
        self.total_steps = steps_per_phase * num_phases
        self.cur_phase = 0
        self.rng = np.random.RandomState(seed)
        self.centers = self._sample_new_centers()

    def _sample_new_centers(self):
        return self.rng.uniform(-self.range_lim, self.range_lim,
                                size=(self.k, self.n_components, self.d))

    def reset_phase(self, phase_idx):
        self.cur_phase = phase_idx
        self.centers = self._sample_new_centers()

    def sample(self, t):
        phase_idx = t // self.steps_per_phase
        if phase_idx != self.cur_phase:
            self.reset_phase(phase_idx)
        y = self.rng.randint(0, self.k)
        comp = self.rng.randint(0, self.n_components)
        mean = self.centers[y, comp]
        x = mean + self.rng.normal(0, self.sigma, size=(self.d,))
        return x.astype(np.float64), int(y)


class ContextualBanditDrift:
    # Non-stationary contextual bandit in R^2 (context x). A arms.
    # Reward for arm a is Bernoulli with probability determined by mixture-of-Gaussians
    # hotspots unique to each arm. Hotspots resample at each drift boundary.
    def __init__(self, d=2, A=3, n_components=2, sigma=0.5, range_lim=3.0,
                 steps_per_phase=2000, num_phases=6, seed=0):
        self.d = d; self.A = A; self.n_components = n_components
        self.sigma = sigma; self.range_lim = range_lim
        self.steps_per_phase = steps_per_phase; self.num_phases = num_phases
        self.total_steps = steps_per_phase * num_phases
        self.cur_phase = 0
        self.rng = np.random.RandomState(seed)
        self.centers = self._new_centers()  # shape (A, n_components, d)
        self.scales  = self._new_scales()   # shape (A, n_components)

    def _new_centers(self):
        return self.rng.uniform(-self.range_lim, self.range_lim,
                                size=(self.A, self.n_components, self.d))

    def _new_scales(self):
        # scale in (0.6, 1.4), controls hotspot strength per arm/component
        return 0.6 + 0.8 * self.rng.rand(self.A, self.n_components)

    def reset_phase(self, phase_idx):
        self.cur_phase = phase_idx
        self.centers = self._new_centers()
        self.scales  = self._new_scales()

    def _reward_prob(self, x, a):
        # probability is a softplus-like function of distances to component centers for arm a,
        # then squashed to (0,1) with logistic.
        diffs = self.centers[a] - x  # (n_components, d)
        dist2 = np.sum(diffs * diffs, axis=1)  # (n_components,)
        logits = -dist2 / (2.0 * (self.sigma ** 2)) * self.scales[a]
        val = np.log1p(np.sum(np.exp(logits)))  # softplus over components
        p = 1.0 / (1.0 + np.exp(-val))
        return float(p)

    def sample(self, t):
        phase_idx = t // self.steps_per_phase
        if phase_idx != self.cur_phase:
            self.reset_phase(phase_idx)
        x = self.rng.uniform(-self.range_lim, self.range_lim, size=(self.d,))
        return x.astype(np.float64)

    def pull(self, x, a):
        p = self._reward_prob(x, a)
        r = 1 if self.rng.rand() < p else 0
        return r, p
