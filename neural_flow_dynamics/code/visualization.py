"""
Visualization Utilities for Neural Flow Dynamics
==================================================

This module provides visualization tools for:
- Training progress plots
- Quantum state distributions
- Flow trajectories
- Benchmark comparisons

Author: Neural Flow Dynamics Team
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json


# ============================================
# Style Configuration
# ============================================

# Use a clean, publication-ready style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#dc2626',    # Red
    'tertiary': '#16a34a',     # Green
    'quaternary': '#9333ea',   # Purple
    'accent': '#f59e0b',       # Orange
    'gray': '#6b7280'
}

# Figure settings
FIGURE_DPI = 150
FIGURE_SIZE = (10, 6)


def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': FIGURE_SIZE,
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ============================================
# Training Progress Visualization
# ============================================

def plot_training_loss(
    losses: List[float],
    output_path: Optional[str] = None,
    title: str = "Training Loss"
):
    """
    Plot training loss curve.
    
    Args:
        losses: List of loss values per epoch
        output_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, color=COLORS['primary'], linewidth=2, label='Training Loss')
    
    # Add smoothed curve
    if len(losses) > 10:
        window = min(10, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(epochs[window-1:], smoothed, color=COLORS['secondary'], 
                linewidth=2, linestyle='--', alpha=0.8, label='Smoothed')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, List[Dict]],
    metric_name: str = 'fidelity',
    output_path: Optional[str] = None
):
    """
    Compare metrics across different methods.
    
    Args:
        metrics_dict: Dict mapping method names to their metrics
        metric_name: Which metric to plot
        output_path: Path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_list = list(COLORS.values())
    
    for i, (method, metrics) in enumerate(metrics_dict.items()):
        steps = [m['step'] for m in metrics]
        values = [m[metric_name] for m in metrics]
        
        ax.plot(steps, values, color=color_list[i % len(color_list)],
                linewidth=2, marker='o', markersize=4, label=method)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison')
    ax.legend()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


# ============================================
# Quantum State Visualization
# ============================================

def plot_probability_distribution(
    psi: torch.Tensor,
    n_qubits: int,
    top_k: int = 20,
    output_path: Optional[str] = None,
    title: str = "Probability Distribution"
):
    """
    Plot probability distribution over computational basis states.
    
    Args:
        psi: Quantum state tensor (complex or real representation)
        n_qubits: Number of qubits
        top_k: Show only top-k probabilities
        output_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()
    
    # Convert to probabilities
    if psi.shape[-1] == 2:  # Real representation
        probs = psi[..., 0]**2 + psi[..., 1]**2
    else:
        probs = torch.abs(psi)**2
    
    probs = probs.flatten().numpy()
    
    # Get top-k states
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    
    # Create labels
    labels = [f"|{format(idx, f'0{n_qubits}b')}⟩" for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(top_k), top_probs, color=COLORS['primary'], alpha=0.8)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Basis State')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    
    # Add value labels on bars
    for bar, prob in zip(bars, top_probs):
        if prob > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def plot_phase_distribution(
    psi: torch.Tensor,
    n_qubits: int,
    output_path: Optional[str] = None
):
    """
    Plot phase distribution of quantum state amplitudes.
    
    Args:
        psi: Complex quantum state tensor
        n_qubits: Number of qubits
        output_path: Path to save figure
    """
    setup_plot_style()
    
    # Convert to complex
    if psi.shape[-1] == 2:
        psi_complex = psi[..., 0] + 1j * psi[..., 1]
    else:
        psi_complex = psi
    
    psi_complex = psi_complex.flatten()
    
    # Get phases and magnitudes
    magnitudes = torch.abs(psi_complex).numpy()
    phases = torch.angle(psi_complex).numpy()
    
    # Only show significant amplitudes
    mask = magnitudes > 1e-4
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    scatter = ax.scatter(phases[mask], magnitudes[mask], 
                         c=magnitudes[mask], cmap='viridis',
                         alpha=0.6, s=50)
    
    ax.set_title('Amplitude Phase Distribution', pad=20)
    plt.colorbar(scatter, ax=ax, label='Magnitude')
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def plot_wigner_function(
    psi: torch.Tensor,
    x_range: Tuple[float, float] = (-5, 5),
    p_range: Tuple[float, float] = (-5, 5),
    resolution: int = 100,
    output_path: Optional[str] = None
):
    """
    Plot Wigner quasi-probability distribution (for single-mode states).
    
    Args:
        psi: Quantum state in Fock basis
        x_range: Range for position axis
        p_range: Range for momentum axis
        resolution: Grid resolution
        output_path: Path to save figure
    """
    setup_plot_style()
    
    # Create grid
    x = np.linspace(*x_range, resolution)
    p = np.linspace(*p_range, resolution)
    X, P = np.meshgrid(x, p)
    
    # Convert state to numpy
    if isinstance(psi, torch.Tensor):
        if psi.shape[-1] == 2:
            psi = (psi[..., 0] + 1j * psi[..., 1]).numpy()
        else:
            psi = psi.numpy()
    
    n_max = len(psi)
    
    # Compute Wigner function
    # W(x, p) = (1/π) ∫ <x-y|ρ|x+y> exp(2ipy) dy
    # For pure states: W(x, p) = (1/π) ψ*(x-y) ψ(x+y) exp(2ipy) dy
    
    # Simplified calculation using Fock basis
    W = np.zeros_like(X)
    
    for n in range(n_max):
        for m in range(n_max):
            # Wigner function matrix element
            # Using harmonic oscillator eigenfunctions
            W_nm = wigner_matrix_element(n, m, X, P)
            W += np.real(np.conj(psi[n]) * psi[m] * W_nm)
    
    W /= np.pi
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at zero
    vmax = np.max(np.abs(W))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.pcolormesh(X, P, W, cmap='RdBu_r', norm=norm, shading='auto')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Momentum p')
    ax.set_title('Wigner Function')
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='W(x, p)')
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def wigner_matrix_element(n: int, m: int, X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute Wigner function matrix element for harmonic oscillator states."""
    from scipy.special import factorial, hermite
    
    alpha = (X + 1j * P) / np.sqrt(2)
    
    # Simplified approximation for small n, m
    if n == m:
        # Diagonal element
        return np.exp(-np.abs(alpha)**2) * (-1)**n * eval_laguerre(n, 2*np.abs(alpha)**2)
    else:
        return np.zeros_like(X)


def eval_laguerre(n: int, x: np.ndarray) -> np.ndarray:
    """Evaluate Laguerre polynomial L_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 1 - x
    else:
        L_prev, L_curr = np.ones_like(x), 1 - x
        for k in range(2, n + 1):
            L_next = ((2*k - 1 - x) * L_curr - (k - 1) * L_prev) / k
            L_prev, L_curr = L_curr, L_next
        return L_curr


# ============================================
# Flow Trajectory Visualization
# ============================================

def plot_flow_trajectory_2d(
    trajectory: torch.Tensor,
    output_path: Optional[str] = None,
    title: str = "Flow Trajectory"
):
    """
    Visualize 2D projection of flow trajectory.
    
    Args:
        trajectory: Tensor of shape (time_steps, batch, state_dim, 2)
        output_path: Path to save figure
        title: Plot title
    """
    setup_plot_style()
    
    # Use first two dimensions for visualization
    traj = trajectory.numpy()
    
    if len(traj.shape) == 4:
        # (time, batch, dim, 2) -> (time, batch, 2)
        traj = traj[:, :, :2, 0]  # Take real part of first 2 dims
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories for first few samples
    n_samples = min(10, traj.shape[1])
    
    for i in range(n_samples):
        x = traj[:, i, 0]
        y = traj[:, i, 1]
        
        # Color by time
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        t = np.linspace(0, 1, len(x))
        ax.scatter(x, y, c=t, cmap='viridis', s=20, alpha=0.7)
        
        # Mark start and end
        ax.scatter(x[0], y[0], color=COLORS['secondary'], s=100, 
                  marker='o', zorder=5, label='Start' if i == 0 else '')
        ax.scatter(x[-1], y[-1], color=COLORS['tertiary'], s=100,
                  marker='*', zorder=5, label='End' if i == 0 else '')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def create_flow_animation(
    trajectory: torch.Tensor,
    output_path: str,
    fps: int = 30
):
    """
    Create animation of flow evolution.
    
    Args:
        trajectory: Tensor of shape (time_steps, batch, state_dim, 2)
        output_path: Path to save animation (mp4)
        fps: Frames per second
    """
    setup_plot_style()
    
    traj = trajectory.numpy()
    n_steps = traj.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def init():
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return []
    
    def animate(frame):
        ax.clear()
        
        # Current state
        state = traj[frame, :, :2, 0]  # (batch, 2)
        
        ax.scatter(state[:, 0], state[:, 1], c=COLORS['primary'], 
                  s=50, alpha=0.6)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(f'Flow Evolution (t = {frame/n_steps:.2f})')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=n_steps, interval=1000/fps, blit=True)
    
    anim.save(output_path, writer='ffmpeg', fps=fps)
    plt.close()


# ============================================
# Benchmark Visualization
# ============================================

def plot_fidelity_comparison(
    results: Dict[str, Dict[str, float]],
    system_sizes: List[int],
    output_path: Optional[str] = None
):
    """
    Compare fidelity across methods and system sizes.
    
    Args:
        results: Dict[method_name][system_size] = fidelity
        system_sizes: List of system sizes (qubits)
        output_path: Path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_list = list(COLORS.values())
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (method, fidelities) in enumerate(results.items()):
        sizes = list(fidelities.keys())
        values = list(fidelities.values())
        
        ax.plot(sizes, values, color=color_list[i % len(color_list)],
                marker=markers[i % len(markers)], markersize=8,
                linewidth=2, label=method)
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Fidelity')
    ax.set_title('Fidelity vs System Size')
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def plot_scaling_analysis(
    methods: List[str],
    system_sizes: List[int],
    times: Dict[str, List[float]],
    memories: Dict[str, List[float]],
    output_path: Optional[str] = None
):
    """
    Plot computational scaling analysis.
    
    Args:
        methods: List of method names
        system_sizes: List of system sizes
        times: Dict[method] = list of wall-clock times
        memories: Dict[method] = list of memory usages (GB)
        output_path: Path to save figure
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    color_list = list(COLORS.values())
    
    # Time plot
    for i, method in enumerate(methods):
        ax1.semilogy(system_sizes, times[method], 
                    color=color_list[i % len(color_list)],
                    marker='o', markersize=8, linewidth=2, label=method)
    
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Wall-clock Time (s)')
    ax1.set_title('Computational Time Scaling')
    ax1.legend()
    
    # Memory plot
    for i, method in enumerate(methods):
        ax2.semilogy(system_sizes, memories[method],
                    color=color_list[i % len(color_list)],
                    marker='s', markersize=8, linewidth=2, label=method)
    
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Memory Usage (GB)')
    ax2.set_title('Memory Scaling')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


def plot_dynamics_comparison(
    times: np.ndarray,
    observables: Dict[str, np.ndarray],
    observable_name: str = "Observable",
    exact: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
):
    """
    Compare dynamics predictions across methods.
    
    Args:
        times: Array of time points
        observables: Dict[method_name] = array of observable values
        observable_name: Name of observable for labeling
        exact: Exact solution (if available)
        output_path: Path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_list = list(COLORS.values())
    
    # Plot exact solution first
    if exact is not None:
        ax.plot(times, exact, color='black', linewidth=3, 
               label='Exact', zorder=10)
    
    # Plot each method
    for i, (method, values) in enumerate(observables.items()):
        ax.plot(times, values, color=color_list[i % len(color_list)],
                linewidth=2, linestyle='--', alpha=0.8, label=method)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(observable_name)
    ax.set_title(f'{observable_name} Dynamics')
    ax.legend()
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


# ============================================
# Entanglement Visualization
# ============================================

def plot_entanglement_growth(
    times: np.ndarray,
    entropies: Dict[str, np.ndarray],
    output_path: Optional[str] = None
):
    """
    Plot entanglement entropy growth over time.
    
    Args:
        times: Array of time points
        entropies: Dict[method_name] = array of entropy values
        output_path: Path to save figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_list = list(COLORS.values())
    
    for i, (method, values) in enumerate(entropies.items()):
        ax.plot(times, values, color=color_list[i % len(color_list)],
                linewidth=2, marker='o', markersize=4, label=method)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('Entanglement Growth During Dynamics')
    ax.legend()
    
    # Add theoretical bounds if applicable
    ax.axhline(y=np.log(2), color='gray', linestyle=':', 
               label='max (1 ebit)', alpha=0.5)
    
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return fig


# ============================================
# Load and Plot from Metrics File
# ============================================

def load_and_plot_training(metrics_path: str, output_dir: str):
    """
    Load training metrics and generate all plots.
    
    Args:
        metrics_path: Path to metrics.json file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Training loss
    if 'train_losses' in metrics:
        plot_training_loss(
            metrics['train_losses'],
            output_path=output_dir / 'training_loss.png'
        )
    
    # Evaluation metrics
    if 'eval_metrics' in metrics:
        # Fidelity over training
        steps = [m['step'] for m in metrics['eval_metrics']]
        fidelities = [m.get('fidelity', 0) for m in metrics['eval_metrics']]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, fidelities, color=COLORS['primary'], 
               linewidth=2, marker='o')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity During Training')
        plt.savefig(output_dir / 'fidelity_training.png')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    
    # Demo: Generate sample plots
    setup_plot_style()
    
    # Create demo data
    np.random.seed(42)
    
    # Training loss demo
    losses = np.exp(-np.linspace(0, 3, 100)) + 0.1 * np.random.randn(100) * np.exp(-np.linspace(0, 3, 100))
    losses = np.maximum(losses, 0.01)
    
    fig = plot_training_loss(losses, title="Demo Training Loss")
    plt.show()
    
    print("Demo visualization complete!")
