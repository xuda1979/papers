"""
可视化工具
=========
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple
from mpl_toolkits.mplot3d import Axes3D


def plot_training_curves(history: Dict[str, List], 
                        title: str = "Training Progress",
                        save_path: Optional[str] = None):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史字典
        title: 图标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 奖励曲线
    if 'rewards' in history:
        ax = axes[0, 0]
        rewards = history['rewards']
        ax.plot(rewards, alpha=0.3, label='Episode reward')
        # 平滑曲线
        window = min(50, len(rewards) // 10 + 1)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 损失曲线
    if 'losses' in history:
        ax = axes[0, 1]
        losses = history['losses']
        ax.plot(losses, alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
    
    # Episode长度
    if 'lengths' in history:
        ax = axes[1, 0]
        lengths = history['lengths']
        ax.plot(lengths, alpha=0.3)
        window = min(50, len(lengths) // 10 + 1)
        if len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(lengths)), smoothed)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
    
    # 评估奖励
    if 'eval_rewards' in history and len(history['eval_rewards']) > 0:
        ax = axes[1, 1]
        eval_rewards = history['eval_rewards']
        ax.plot(eval_rewards, 'o-')
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Evaluation Performance')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_fidelity_progress(fidelities: List[float],
                          threshold: float = 0.99,
                          title: str = "Fidelity Progress",
                          save_path: Optional[str] = None):
    """
    绘制保真度进展
    
    Args:
        fidelities: 保真度列表
        threshold: 目标阈值
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(fidelities, 'b-', linewidth=2, label='Fidelity')
    ax.axhline(y=threshold, color='r', linestyle='--', 
               label=f'Threshold ({threshold})')
    
    ax.fill_between(range(len(fidelities)), fidelities, alpha=0.3)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_bloch_sphere(states: List[np.ndarray],
                     labels: Optional[List[str]] = None,
                     title: str = "Bloch Sphere",
                     save_path: Optional[str] = None):
    """
    绘制布洛赫球上的量子态
    
    Args:
        states: 量子态列表
        labels: 标签列表
        title: 图标题
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制球面
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='cyan')
    
    # 绘制轴
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
    
    # 标记轴
    ax.text(1.1, 0, 0, 'X', fontsize=12)
    ax.text(0, 1.1, 0, 'Y', fontsize=12)
    ax.text(0, 0, 1.1, '|0⟩', fontsize=12)
    ax.text(0, 0, -1.1, '|1⟩', fontsize=12)
    
    # 绘制态
    colors = plt.cm.viridis(np.linspace(0, 1, len(states)))
    
    for i, state in enumerate(states):
        # 转换为布洛赫坐标
        if len(state) == 2:
            alpha, beta = state
            bx = 2 * np.real(np.conj(alpha) * beta)
            by = 2 * np.imag(np.conj(alpha) * beta)
            bz = np.abs(alpha)**2 - np.abs(beta)**2
            
            label = labels[i] if labels else f'State {i}'
            ax.scatter([bx], [by], [bz], color=colors[i], s=100, label=label)
            ax.plot([0, bx], [0, by], [0, bz], color=colors[i], alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if len(states) <= 10:
        ax.legend()
    
    ax.set_box_aspect([1, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_quantum_circuit(circuit_history: List[Dict],
                        n_qubits: int,
                        title: str = "Quantum Circuit",
                        save_path: Optional[str] = None):
    """
    简单的量子电路可视化
    
    Args:
        circuit_history: 电路历史
        n_qubits: 量子比特数
        title: 图标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(max(12, len(circuit_history) * 0.8), n_qubits * 1.5))
    
    # 绘制量子比特线
    for i in range(n_qubits):
        y = n_qubits - 1 - i
        ax.axhline(y=y, color='black', linewidth=1)
        ax.text(-0.5, y, f'q{i}', ha='right', va='center', fontsize=12)
    
    # 绘制门
    for step, gate_info in enumerate(circuit_history):
        gate = gate_info.get('gate', '?')
        qubits = gate_info.get('qubits', [0])
        params = gate_info.get('params', None)
        
        if isinstance(qubits, int):
            qubits = [qubits]
        
        x = step + 0.5
        
        if len(qubits) == 1:
            # 单量子比特门
            y = n_qubits - 1 - qubits[0]
            
            # 门框
            rect = plt.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                                fill=True, facecolor='lightblue',
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # 门名称
            label = gate[:3].upper()
            ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
            
        else:
            # 双量子比特门
            y1 = n_qubits - 1 - qubits[0]
            y2 = n_qubits - 1 - qubits[1]
            
            # 连接线
            ax.plot([x, x], [y1, y2], 'k-', linewidth=2)
            
            # 控制点
            ax.plot(x, y1, 'ko', markersize=10)
            
            # 目标门
            if gate.upper() in ['CNOT', 'CX']:
                circle = plt.Circle((x, y2), 0.2, fill=False, linewidth=2)
                ax.add_patch(circle)
                ax.plot([x - 0.2, x + 0.2], [y2, y2], 'k-', linewidth=2)
                ax.plot([x, x], [y2 - 0.2, y2 + 0.2], 'k-', linewidth=2)
            else:
                rect = plt.Rectangle((x - 0.3, y2 - 0.3), 0.6, 0.6,
                                    fill=True, facecolor='lightgreen',
                                    edgecolor='black', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y2, gate[:2], ha='center', va='center', fontsize=8)
    
    ax.set_xlim(-1, len(circuit_history) + 0.5)
    ax.set_ylim(-0.5, n_qubits - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_pulse_sequence(pulse_sequence: np.ndarray,
                       time_steps: Optional[np.ndarray] = None,
                       labels: Optional[List[str]] = None,
                       title: str = "Control Pulse Sequence",
                       save_path: Optional[str] = None):
    """
    绘制控制脉冲序列
    
    Args:
        pulse_sequence: 脉冲序列 (n_steps, n_controls)
        time_steps: 时间步
        labels: 控制标签
        title: 图标题
        save_path: 保存路径
    """
    if pulse_sequence.ndim == 1:
        pulse_sequence = pulse_sequence.reshape(-1, 1)
    
    n_steps, n_controls = pulse_sequence.shape
    
    if time_steps is None:
        time_steps = np.arange(n_steps)
    
    if labels is None:
        labels = [f'Control {i}' for i in range(n_controls)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.arange(n_controls))
    
    for i in range(n_controls):
        ax.step(time_steps, pulse_sequence[:, i], where='mid',
               label=labels[i], color=colors[i], linewidth=2)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_unitary_heatmap(U: np.ndarray,
                        title: str = "Unitary Matrix",
                        save_path: Optional[str] = None):
    """
    绘制酉矩阵热图
    
    Args:
        U: 酉矩阵
        title: 图标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 实部
    im1 = axes[0].imshow(U.real, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Real Part')
    plt.colorbar(im1, ax=axes[0])
    
    # 虚部
    im2 = axes[1].imshow(U.imag, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('Imaginary Part')
    plt.colorbar(im2, ax=axes[1])
    
    for ax in axes:
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("=== 可视化工具测试 ===\n")
    
    # 测试训练曲线
    print("1. 训练曲线测试")
    history = {
        'rewards': np.cumsum(np.random.randn(100)) + 50,
        'losses': np.exp(-np.arange(100) / 30) + 0.1 * np.random.randn(100),
        'lengths': np.random.randint(10, 50, 100),
        'eval_rewards': [20, 30, 40, 45, 50]
    }
    # plot_training_curves(history, "Test Training Curves")
    print("训练曲线绘制函数已准备好")
    
    # 测试保真度进展
    print("\n2. 保真度进展测试")
    fidelities = 1 - np.exp(-np.arange(50) / 10) * 0.5
    # plot_fidelity_progress(fidelities, threshold=0.99)
    print("保真度曲线绘制函数已准备好")
    
    # 测试布洛赫球
    print("\n3. 布洛赫球测试")
    states = [
        np.array([1, 0]),  # |0⟩
        np.array([0, 1]),  # |1⟩
        np.array([1, 1]) / np.sqrt(2),  # |+⟩
        np.array([1, 1j]) / np.sqrt(2),  # |+i⟩
    ]
    # plot_bloch_sphere(states, ['|0⟩', '|1⟩', '|+⟩', '|+i⟩'])
    print("布洛赫球绘制函数已准备好")
    
    # 测试脉冲序列
    print("\n4. 脉冲序列测试")
    pulses = np.random.randn(20, 2)
    # plot_pulse_sequence(pulses, labels=['X control', 'Y control'])
    print("脉冲序列绘制函数已准备好")
