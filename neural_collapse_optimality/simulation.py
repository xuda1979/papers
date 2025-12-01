import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration for professional plots
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "figure.figsize": (10, 6),
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_synthetic_data(n_samples, n_classes, dim):
    """Generates synthetic data indices."""
    samples_per_class = n_samples // n_classes
    y = np.repeat(np.arange(n_classes), samples_per_class)
    # One-hot encoding
    y_onehot = np.zeros((n_classes, n_samples))
    y_onehot[y, np.arange(n_samples)] = 1
    return y, y_onehot

def softmax(Z):
    """Compute softmax values for each set of scores in x."""
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_Z / e_Z.sum(axis=0, keepdims=True)

def compute_loss(W, H, y_onehot, lambda_decay):
    """Computes CE loss with weight decay."""
    Z = W @ H
    probs = softmax(Z)
    N = H.shape[1]

    # Cross Entropy
    ce_loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N

    # L2 Regularization
    reg_loss = (lambda_decay / 2) * (np.linalg.norm(W)**2 + np.linalg.norm(H)**2)

    total_loss = ce_loss + reg_loss
    return total_loss, probs

def train_ufm(n_classes=10, dim=64, samples_per_class=100, epochs=2000, lr=0.1, lambda_decay=1e-4):
    """Trains the Unconstrained Feature Model (UFM)."""
    n_samples = n_classes * samples_per_class
    y, y_onehot = generate_synthetic_data(n_samples, n_classes, dim)

    # Initialize W and H randomly
    W = np.random.randn(n_classes, dim) * 0.01
    H = np.random.randn(dim, n_samples) * 0.01

    metrics = {
        'loss': [],
        'nc1_within_var': [],
        'nc2_etf_error': [],
        'nc3_alignment': []
    }

    for epoch in range(epochs):
        # Forward
        Z = W @ H
        probs = softmax(Z)

        # Backward (Gradient of CE w.r.t Z is (P - Y)/N)
        dZ = (probs - y_onehot) / n_samples

        dW = dZ @ H.T + lambda_decay * W
        dH = W.T @ dZ + lambda_decay * H

        # Update
        W -= lr * dW
        H -= lr * dH

        # Compute Metrics
        if epoch % 10 == 0:
            loss, _ = compute_loss(W, H, y_onehot, lambda_decay)

            # NC1: Within-class variation
            # Sigma_W = (1/K) sum_k Sigma_k
            within_cov = 0
            class_means = np.zeros((dim, n_classes))

            for k in range(n_classes):
                indices = np.where(y == k)[0]
                H_k = H[:, indices]
                mu_k = np.mean(H_k, axis=1)
                class_means[:, k] = mu_k
                centered = H_k - mu_k[:, None]
                within_cov += np.trace(centered @ centered.T) / len(indices)

            nc1 = within_cov / n_classes

            # NC2: Equiangular Tight Frame (ETF) structure of class means
            # Normalize class means
            M = class_means
            M_centered = M - np.mean(M, axis=1, keepdims=True)
            # For ETF, Gram matrix G should be proportional to (K/(K-1)) * (I - 1/K 11^T)
            # Actually, standard NC metric is || M_norm^T M_norm - (K/(K-1)) (I - 1/K 11^T) ||_F
            # But let's just check if Gram matrix is close to centered identity structure
            G = M_centered.T @ M_centered
            G_norm = G / (np.linalg.norm(G, 'fro') + 1e-9) # Normalize for shape comparison

            # Ideal ETF Gram Matrix structure (centered)
            target_G = np.eye(n_classes) - np.ones((n_classes, n_classes)) / n_classes
            target_G = target_G / np.linalg.norm(target_G, 'fro')

            nc2 = np.linalg.norm(G_norm - target_G, 'fro')

            # NC3: Alignment between W and M
            # Duality: W proportional to M
            W_centered = W - np.mean(W, axis=0, keepdims=True)
            product = W @ class_means # Should be proportional to I
            # Or simpler: measure angle between W_k and mu_k

            # Let's use the metric: || W/||W|| - M^T/||M|| ||_F
            W_norm = W / (np.linalg.norm(W, 'fro') + 1e-9)
            M_norm = class_means.T / (np.linalg.norm(class_means, 'fro') + 1e-9)
            nc3 = np.linalg.norm(W_norm - M_norm, 'fro') # This assumes they match index-wise (which they do by training)

            metrics['loss'].append(loss)
            metrics['nc1_within_var'].append(nc1)
            metrics['nc2_etf_error'].append(nc2)
            metrics['nc3_alignment'].append(nc3)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, NC1 = {nc1:.4e}, NC2 = {nc2:.4e}, NC3 = {nc3:.4e}")

    return metrics, W, H

def plot_metrics(metrics):
    epochs = np.arange(0, len(metrics['loss'])) * 10

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Loss
    axes[0].plot(epochs, metrics['loss'], color='#1f77b4', linewidth=3)
    axes[0].set_title('Training Loss (Cross Entropy)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')

    # NC1
    axes[1].plot(epochs, metrics['nc1_within_var'], color='#ff7f0e', linewidth=3)
    axes[1].set_title('NC1: Within-Class Variation')
    axes[1].set_xlabel('Epochs')
    axes[1].set_yscale('log')

    # NC2
    axes[2].plot(epochs, metrics['nc2_etf_error'], color='#2ca02c', linewidth=3)
    axes[2].set_title('NC2: ETF Structure Error')
    axes[2].set_xlabel('Epochs')
    axes[2].set_yscale('log')

    # NC3
    axes[3].plot(epochs, metrics['nc3_alignment'], color='#d62728', linewidth=3)
    axes[3].set_title('NC3: W-H Alignment Error')
    axes[3].set_xlabel('Epochs')
    axes[3].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nc_metrics.png'), dpi=300)
    plt.close()

def plot_geometry(W, H, y):
    """Visualizes the geometry of features and weights using PCA."""
    from sklearn.decomposition import PCA

    # Combine W and H for joint PCA
    # W: (K, d)
    # H: (d, N) -> (N, d)

    pca = PCA(n_components=2)
    H_T = H.T

    # Fit on features
    H_pca = pca.fit_transform(H_T)
    W_pca = pca.transform(W)

    plt.figure(figsize=(10, 10))

    # Plot features
    scatter = plt.scatter(H_pca[:, 0], H_pca[:, 1], c=y, cmap='tab10', alpha=0.3, s=10, label='Features')

    # Plot weights (as arrows or large points)
    # We use the same color map
    colors = plt.cm.tab10(np.arange(W.shape[0]))

    for k in range(W.shape[0]):
        plt.scatter(W_pca[k, 0], W_pca[k, 1], color=colors[k], s=200, marker='X', edgecolors='black', linewidth=2, label='Class Weight' if k == 0 else "")
        # Draw line from origin
        plt.arrow(0, 0, W_pca[k, 0], W_pca[k, 1], color=colors[k], alpha=0.5, width=0.002)

    plt.title('2D PCA Visualization of Neural Collapse Geometry')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_geometry.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Starting Neural Collapse Simulation...")
    metrics, W, H = train_ufm(n_classes=5, dim=128, samples_per_class=50, epochs=3000, lr=0.1)

    print("Generating Plots...")
    plot_metrics(metrics)

    # Generate data again just for labels for plotting
    y, _ = generate_synthetic_data(5 * 50, 5, 128)
    plot_geometry(W, H, y)

    print(f"Simulation Complete. Results saved to {OUTPUT_DIR}")
