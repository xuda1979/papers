
import numpy as np
from dataclasses import dataclass, field

def softmax(z):
    zmax = np.max(z)
    e = np.exp(z - zmax)
    return e / np.sum(e)

@dataclass
class LogisticSGD:
    d: int; k: int; lr: float = 0.05; wd: float = 1e-4
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    name: str = "LogReg"
    def __post_init__(self):
        rng = np.random.RandomState(0)
        self.W = rng.randn(self.d, self.k) * 0.01
        self.b = np.zeros(self.k)
    def predict(self, x):
        scores = x @ self.W + self.b
        p = softmax(scores)
        return int(np.argmax(p))
    def update(self, x, y):
        scores = x @ self.W + self.b
        p = softmax(scores)
        y_one = np.zeros_like(p); y_one[y] = 1.0
        grad_scores = p - y_one
        gW = np.outer(x, grad_scores) + self.wd * self.W
        gb = grad_scores
        self.W -= self.lr * gW
        self.b -= self.lr * gb


@dataclass
class MLPOnline:
    d: int; k: int; h: int = 32; lr: float = 0.01; wd: float = 1e-5
    name: str = "MLP(32)"
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    def __post_init__(self):
        rng = np.random.RandomState(0)
        limit1 = np.sqrt(6 / (self.d + self.h))
        limit2 = np.sqrt(6 / (self.h + self.k))
        self.W1 = rng.uniform(-limit1, limit1, size=(self.d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = rng.uniform(-limit2, limit2, size=(self.h, self.k))
        self.b2 = np.zeros(self.k)
    def _forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.tanh(z1)
        scores = h @ self.W2 + self.b2
        p = softmax(scores)
        return h, p
    def predict(self, x):
        _, p = self._forward(x)
        return int(np.argmax(p))
    def update(self, x, y):
        h, p = self._forward(x)
        y_one = np.zeros_like(p); y_one[y] = 1.0
        dscores = p - y_one
        gW2 = np.outer(h, dscores) + self.wd * self.W2
        gb2 = dscores
        dh = self.W2 @ dscores
        dz1 = (1.0 - h * h) * dh
        gW1 = np.outer(x, dz1) + self.wd * self.W1
        gb1 = dz1
        self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
        self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1


class LinUCB:
    # Standard LinUCB for contextual bandits.
    def __init__(self, d, A, alpha=0.8, lam=1.0):
        self.d = d + 1  # bias
        self.A = A
        self.alpha = alpha
        self.lam = lam
        self.As = [lam * np.eye(self.d) for _ in range(A)]
        self.bs = [np.zeros(self.d) for _ in range(A)]
        self.name = f"LinUCB(alpha={alpha})"

    def _feat(self, x):
        return np.concatenate([x, [1.0]])

    def select(self, x):
        phi = self._feat(x)
        best_a, best_u = 0, -1e9
        for a in range(self.A):
            A_inv = np.linalg.inv(self.As[a])
            theta = A_inv @ self.bs[a]
            mu = theta @ phi
            ucb = mu + self.alpha * np.sqrt(phi @ A_inv @ phi)
            if ucb > best_u:
                best_u, best_a = ucb, a
        return best_a

    def update(self, x, a, r):
        phi = self._feat(x)
        self.As[a] += np.outer(phi, phi)
        self.bs[a] += r * phi
