import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from collections import deque
import random
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

class HCDStream:
    """
    Generates a stream of data with Hierarchical Concept Drift.
    Includes a 'Noise Phase' to test stability.
    """
    def __init__(self, n_steps, drift_step, noise_interval=(2500, 3500), noise_prob=0.3, dim=20, n_classes=4):
        self.n_steps = n_steps
        self.drift_step = drift_step
        self.dim = dim
        self.n_classes = n_classes
        self.current_step = 0
        self.noise_interval = noise_interval
        self.noise_prob = noise_prob

        # Define prototypes for Phase 1 (active dims 0-4)
        # Using larger separation to make it learnable
        self.protos_p1 = np.random.randn(n_classes, 5) * 3.0

        # Define prototypes for Phase 2 (active dims 5-9)
        self.protos_p2 = np.random.randn(n_classes, 5) * 3.0

    def next_batch(self, batch_size=1):
        if self.current_step >= self.n_steps:
            return None, None

        is_phase2 = self.current_step >= self.drift_step
        is_noise_phase = (self.noise_interval[0] <= self.current_step < self.noise_interval[1])

        X = np.zeros((batch_size, self.dim))
        y = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            # Check for noise injection
            if is_noise_phase and np.random.random() < self.noise_prob:
                # Generate random noise vector
                X[i, :] = np.random.randn(self.dim)
                # Assign a dummy label -1 to indicate noise (evaluator should handle this)
                # But to keep accuracy metric simple, we can assign a random label
                # and expect the model to fail (accuracy drops), but structure should remain stable.
                y[i] = np.random.randint(0, self.n_classes)
            else:
                label = np.random.randint(0, self.n_classes)
                y[i] = label
                if not is_phase2:
                    # Phase 1: Signal in 0-5
                    signal = self.protos_p1[label] + np.random.randn(5) * 0.5
                    X[i, 0:5] = signal
                    X[i, 5:] = np.random.randn(self.dim - 5) * 0.1 # Noise
                else:
                    # Phase 2: Signal in 5-10
                    signal = self.protos_p2[label] + np.random.randn(5) * 0.5
                    X[i, 5:10] = signal
                    X[i, :5] = np.random.randn(5) * 0.1 # Noise
                    X[i, 10:] = np.random.randn(self.dim - 10) * 0.1 # Noise

        self.current_step += batch_size
        return X, y

class GRUR_Node:
    def __init__(self, input_dim, initial_weight, label_dist=None):
        self.W_BU = initial_weight.copy() # Bottom-Up Weights (Prototype)
        # Normalize
        norm = np.linalg.norm(self.W_BU)
        if norm > 0:
            self.W_BU = self.W_BU / norm

        if label_dist is None:
            self.label_counts = {}
        else:
            self.label_counts = label_dist

        self.lr = 0.1 # Faster learning

    def activation(self, x):
        # Cosine similarity
        x_norm = np.linalg.norm(x)
        if x_norm == 0: return 0
        x_n = x / x_norm
        return np.dot(self.W_BU, x_n)

    def learn(self, x, label):
        # Hebbian-like update (move prototype towards input)
        # Standard instar learning rule
        x_norm = np.linalg.norm(x)
        if x_norm > 0:
            x_n = x / x_norm
            self.W_BU = self.W_BU + self.lr * (x_n - self.W_BU)

            # Renormalize
            norm = np.linalg.norm(self.W_BU)
            if norm > 0:
                self.W_BU = self.W_BU / norm

        # Update label statistics
        self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def predict_label(self):
        if not self.label_counts:
            return np.random.randint(0, 4)
        return max(self.label_counts, key=self.label_counts.get)

class ARH:
    def __init__(self, input_dim, vigilance=0.85, dissonance_threshold=3.0):
        self.input_dim = input_dim
        self.vigilance = vigilance
        self.dissonance_threshold = dissonance_threshold

        self.nodes = [] # List of GRUR_Node

        # Dissonance tracking
        self.dissonance_accumulator = 0.0
        self.gamma = 0.05 # Decay
        self.beta = 0.5   # Accumulation - Increased this to trigger growth faster

        self.buffer_X = []
        self.buffer_y = []

        # Metrics
        self.dissonance_history = []
        self.nodes_history = []

    def predict(self, x):
        if not self.nodes:
            return np.random.randint(0, 4)

        activations = [node.activation(x) for node in self.nodes]
        winner_idx = np.argmax(activations)

        return self.nodes[winner_idx].predict_label()

    def train_step(self, x, y):
        self.nodes_history.append(len(self.nodes))

        # 1. If empty, initialize
        if not self.nodes:
            self.nodes.append(GRUR_Node(self.input_dim, x, {y: 1}))
            self.dissonance_accumulator = 0
            self.dissonance_history.append(0)
            return

        # 2. Recognition
        activations = [node.activation(x) for node in self.nodes]
        winner_idx = np.argmax(activations)
        match_score = activations[winner_idx]

        # 3. Resonance Check
        if match_score >= self.vigilance:
            # Resonance!
            self.nodes[winner_idx].learn(x, y)

            # Decay dissonance
            self.dissonance_accumulator = (1 - self.gamma) * self.dissonance_accumulator
        else:
            # Dissonance!
            # Accumulate dissonance
            q = 1.0 # Mismatch
            self.dissonance_accumulator = (1 - self.gamma) * self.dissonance_accumulator + self.beta * q

            # Buffer the unknown pattern
            self.buffer_X.append(x)
            self.buffer_y.append(y)

        # 4. Structural Adaptation Check
        if self.dissonance_accumulator >= self.dissonance_threshold:
            self._consolidate_dissonance()

        self.dissonance_history.append(self.dissonance_accumulator)

    def _consolidate_dissonance(self):
        # Spawn new nodes from buffer
        if len(self.buffer_X) < 1:
            return

        X_buf = np.array(self.buffer_X)
        y_buf = np.array(self.buffer_y)

        # If we have enough points, cluster them.
        k = min(len(X_buf), 4) # Attempt to find up to 4 new clusters

        if k > 0:
            try:
                kmeans = KMeans(n_clusters=k, n_init=10)
                kmeans.fit(X_buf)

                for i in range(k):
                    centroid = kmeans.cluster_centers_[i]

                    # Determine label
                    cluster_indices = np.where(kmeans.labels_ == i)[0]
                    cluster_labels = y_buf[cluster_indices]
                    if len(cluster_labels) > 0:
                        counts = np.bincount(cluster_labels, minlength=10)
                        maj_label = np.argmax(counts)
                        label_dist = {maj_label: len(cluster_indices)}

                        self.nodes.append(GRUR_Node(self.input_dim, centroid, label_dist))
            except Exception as e:
                # Fallback if KMeans fails (e.g. singular matrix)
                pass

        # Reset buffer and dissonance
        self.buffer_X = []
        self.buffer_y = []
        self.dissonance_accumulator = 0.0

class StaticBaseline:
    def __init__(self, input_dim, n_nodes=4):
        self.nodes = []
        # Initialize nodes randomly, but normalized
        for _ in range(n_nodes):
            w = np.random.randn(input_dim)
            w = w / np.linalg.norm(w)
            self.nodes.append(GRUR_Node(input_dim, w))

    def predict(self, x):
        activations = [node.activation(x) for node in self.nodes]
        winner_idx = np.argmax(activations)
        return self.nodes[winner_idx].predict_label()

    def train_step(self, x, y):
        activations = [node.activation(x) for node in self.nodes]
        winner_idx = np.argmax(activations)

        # Always learn, but slowly
        self.nodes[winner_idx].lr = 0.01
        self.nodes[winner_idx].learn(x, y)

def run_simulation():
    total_steps = 10000
    drift_step = 5000
    noise_start = 2500
    noise_end = 3500

    stream = HCDStream(total_steps, drift_step, noise_interval=(noise_start, noise_end), noise_prob=0.3)

    # Tuned parameters: lower threshold, higher beta to ensure growth
    # Threshold increased to 5.0 to ensure stability against 30% noise
    arh = ARH(input_dim=20, vigilance=0.85, dissonance_threshold=5.0)
    baseline = StaticBaseline(input_dim=20, n_nodes=4)

    arh_acc = []
    base_acc = []
    arh_dissonance = []

    window_size = 200
    arh_correct_window = deque(maxlen=window_size)
    base_correct_window = deque(maxlen=window_size)

    # Initialize windows with chance accuracy (0.25)
    for _ in range(window_size):
        arh_correct_window.append(1 if np.random.random() < 0.25 else 0)
        base_correct_window.append(1 if np.random.random() < 0.25 else 0)

    print("Starting Simulation...")
    for t in range(total_steps):
        X_batch, y_batch = stream.next_batch(1)
        x, y = X_batch[0], y_batch[0]

        # Predict
        p_arh = arh.predict(x)
        p_base = baseline.predict(x)

        # Record Accuracy
        # If it was noise injection (which we don't know explicitly here without peeking),
        # accuracy will drop. That's fine.
        arh_correct = 1 if p_arh == y else 0
        base_correct = 1 if p_base == y else 0

        arh_correct_window.append(arh_correct)
        base_correct_window.append(base_correct)

        arh_acc.append(np.mean(arh_correct_window) * 100)
        base_acc.append(np.mean(base_correct_window) * 100)

        # Train
        arh.train_step(x, y)
        baseline.train_step(x, y)

        arh_dissonance.append(arh.dissonance_accumulator)

        if t % 1000 == 0:
            print(f"Step {t}: ARH Acc={arh_acc[-1]:.1f}%, Nodes={len(arh.nodes)} | Base Acc={base_acc[-1]:.1f}%")

    # Save Results
    results_df = pd.DataFrame({
        'step': np.arange(total_steps),
        'ARH_Accuracy': arh_acc,
        'Baseline_Accuracy': base_acc,
        'ARH_Dissonance': arh_dissonance,
        'ARH_Nodes': arh.nodes_history
    })

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results_df.to_csv(os.path.join(out_dir, 'simulation_results.csv'), index=False)
    print("Simulation Complete. Data saved.")

if __name__ == "__main__":
    run_simulation()
