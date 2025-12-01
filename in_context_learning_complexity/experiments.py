
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication-quality plots
# sns.set(style="whitegrid", context="paper")
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['text.usetex'] = True # Disabled due to missing latex
# plt.rcParams['figure.dpi'] = 300

# Alternative high-quality settings without LaTeX
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def generate_task(dim, n_examples):
    # Target weight vector
    w_star = np.random.randn(dim)
    w_star = w_star / np.linalg.norm(w_star)

    # Context data
    X = np.random.randn(n_examples, dim)
    noise = 0.01 * np.random.randn(n_examples)
    y = X @ w_star + noise

    return w_star, X, y

def linear_attention_ic(w_star, X_context, y_context, x_query, n_layers=5, step_size=0.1):
    """
    Simulates ICL via Gradient Descent implemented by Linear Attention layers.
    Ref: Von Oswald et al. (2023).

    Each layer performs: w_{l+1} = w_l + step_size * X^T (y - X w_l)
    This is effectively GD on the least squares objective.
    """
    dim = len(w_star)
    w_current = np.zeros(dim) # Initialize with zero knowledge

    errors = []

    # Theoretical fixed point (Ridge Regression)
    # w_ridge = (X^T X + lambda I)^-1 X^T y
    # But here we simulate the layer-wise dynamics

    for layer in range(n_layers):
        # Calculate prediction on context
        y_pred_context = X_context @ w_current
        residuals = y_context - y_pred_context

        # Gradient update (Attention mechanism)
        # In a transformer, this is W_V * Attention * W_O
        # For linear regression, it simplifies to adding correlated features
        grad = X_context.T @ residuals / len(y_context)

        w_current = w_current + step_size * grad

        # Test error on query
        y_query_true = x_query @ w_star
        y_query_pred = x_query @ w_current
        error = (y_query_pred - y_query_true)**2
        errors.append(error)

    return errors

def run_sample_complexity_experiment():
    dim = 10
    n_values = np.logspace(1, 2.5, 15).astype(int) # 10 to 300
    n_trials = 50
    final_errors = []

    print("Running Sample Complexity Experiment...")
    for n in n_values:
        trial_errors = []
        for _ in range(n_trials):
            w_star, X, y = generate_task(dim, n)
            x_query = np.random.randn(dim)

            # Run for enough layers to converge
            errors = linear_attention_ic(w_star, X, y, x_query, n_layers=10, step_size=0.5)
            trial_errors.append(errors[-1])

        final_errors.append(np.mean(trial_errors))

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.loglog(n_values, final_errors, 'o-', linewidth=2, label='Empirical MSE')

    # Theoretical reference 1/n
    ref_x = n_values
    ref_y = final_errors[0] * (n_values[0] / n_values) # Match start point
    plt.loglog(ref_x, ref_y, '--', color='gray', label=r'Theory $O(1/n)$')

    plt.xlabel('Number of Context Examples ($n$)')
    plt.ylabel('Mean Squared Error')
    plt.title('Sample Complexity of In-Context Learning')
    plt.legend()
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tex/fig')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'sample_complexity.pdf')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def run_contraction_experiment():
    dim = 20
    n_examples = 50
    n_layers = 20
    n_trials = 20

    avg_errors = np.zeros(n_layers)

    print("Running Contraction Dynamics Experiment...")
    for _ in range(n_trials):
        w_star, X, y = generate_task(dim, n_examples)
        x_query = np.random.randn(dim)

        errors = linear_attention_ic(w_star, X, y, x_query, n_layers=n_layers, step_size=0.2)
        avg_errors += np.array(errors)

    avg_errors /= n_trials

    # Plotting
    plt.figure(figsize=(6, 4))
    layers = np.arange(1, n_layers + 1)
    plt.plot(layers, avg_errors, 'o-', color='crimson', linewidth=2)
    plt.yscale('log')

    plt.xlabel('Transformer Layer Depth')
    plt.ylabel('MSE (Log Scale)')
    plt.title('Contraction Dynamics of Attention Layers')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tex/fig')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'contraction_dynamics.pdf')
    plt.savefig(save_path)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    np.random.seed(42)
    run_sample_complexity_experiment()
    run_contraction_experiment()
