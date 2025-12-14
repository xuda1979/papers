"""
评估指标
=======
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ControlMetrics:
    """控制性能指标"""
    fidelity: float
    circuit_depth: int
    success: bool
    total_time: float
    n_gates: int
    
    def to_dict(self) -> Dict:
        return {
            'fidelity': self.fidelity,
            'circuit_depth': self.circuit_depth,
            'success': self.success,
            'total_time': self.total_time,
            'n_gates': self.n_gates
        }


@dataclass
class TrainingMetrics:
    """训练性能指标"""
    episode: int
    reward: float
    loss: float
    fidelity: float
    episode_length: int
    
    def to_dict(self) -> Dict:
        return {
            'episode': self.episode,
            'reward': self.reward,
            'loss': self.loss,
            'fidelity': self.fidelity,
            'episode_length': self.episode_length
        }


def compute_gate_complexity(circuit_history: List[Dict]) -> Dict:
    """
    计算电路复杂度指标
    
    Args:
        circuit_history: 电路操作历史
        
    Returns:
        复杂度指标字典
    """
    n_single = 0
    n_two = 0
    gate_counts = {}
    
    for op in circuit_history:
        gate = op.get('gate', 'unknown')
        qubits = op.get('qubits', [0])
        
        if isinstance(qubits, int):
            n_single += 1
        elif len(qubits) == 1:
            n_single += 1
        else:
            n_two += 1
        
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    
    return {
        'total_gates': len(circuit_history),
        'single_qubit_gates': n_single,
        'two_qubit_gates': n_two,
        'circuit_depth': len(circuit_history),  # 简化估计
        'gate_counts': gate_counts
    }


def compute_control_efficiency(fidelity: float, 
                              n_operations: int,
                              max_operations: int = 100) -> float:
    """
    计算控制效率
    
    Args:
        fidelity: 达到的保真度
        n_operations: 使用的操作数
        max_operations: 最大允许操作数
        
    Returns:
        效率分数 (0-1)
    """
    # 高保真度、少操作 = 高效率
    fidelity_score = fidelity
    efficiency_score = 1 - n_operations / max_operations
    
    return 0.7 * fidelity_score + 0.3 * efficiency_score


def compute_robustness(fidelities_with_noise: List[float],
                      fidelity_without_noise: float) -> float:
    """
    计算鲁棒性指标
    
    Args:
        fidelities_with_noise: 有噪声时的保真度列表
        fidelity_without_noise: 无噪声时的保真度
        
    Returns:
        鲁棒性分数 (0-1)
    """
    if fidelity_without_noise == 0:
        return 0.0
    
    mean_noisy = np.mean(fidelities_with_noise)
    std_noisy = np.std(fidelities_with_noise)
    
    # 鲁棒性 = 保真度保持率 * 稳定性
    retention = mean_noisy / fidelity_without_noise
    stability = 1 / (1 + std_noisy * 10)  # 标准差越小越稳定
    
    return np.clip(retention * stability, 0, 1)


def compute_learning_rate_metrics(rewards: List[float]) -> Dict:
    """
    计算学习速率相关指标
    
    Args:
        rewards: 奖励历史
        
    Returns:
        学习指标字典
    """
    if len(rewards) < 2:
        return {
            'convergence_episode': None,
            'learning_rate': 0,
            'final_performance': rewards[-1] if rewards else 0,
            'max_performance': max(rewards) if rewards else 0
        }
    
    rewards = np.array(rewards)
    
    # 找收敛点（奖励变化小于阈值）
    window = min(20, len(rewards) // 5)
    if window > 0:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        if len(smoothed) > 1:
            changes = np.abs(np.diff(smoothed))
            convergence_idx = np.where(changes < 0.01 * np.max(np.abs(smoothed)))[0]
            convergence_episode = convergence_idx[0] + window if len(convergence_idx) > 0 else None
        else:
            convergence_episode = None
    else:
        convergence_episode = None
    
    # 学习速率（早期梯度）
    early_rewards = rewards[:min(50, len(rewards)//2)]
    if len(early_rewards) > 1:
        learning_rate = (early_rewards[-1] - early_rewards[0]) / len(early_rewards)
    else:
        learning_rate = 0
    
    return {
        'convergence_episode': convergence_episode,
        'learning_rate': learning_rate,
        'final_performance': rewards[-1],
        'max_performance': np.max(rewards),
        'mean_performance': np.mean(rewards),
        'std_performance': np.std(rewards)
    }


def compute_success_rate(results: List[Dict], threshold: float = 0.99) -> Dict:
    """
    计算成功率统计
    
    Args:
        results: 评估结果列表
        threshold: 成功阈值
        
    Returns:
        成功率统计
    """
    successes = [r.get('fidelity', 0) >= threshold for r in results]
    fidelities = [r.get('fidelity', 0) for r in results]
    
    return {
        'success_rate': np.mean(successes),
        'n_success': sum(successes),
        'n_total': len(results),
        'mean_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'min_fidelity': np.min(fidelities),
        'max_fidelity': np.max(fidelities)
    }


def compare_methods(results: Dict[str, List[Dict]]) -> Dict:
    """
    比较不同方法的性能
    
    Args:
        results: {方法名: 结果列表} 字典
        
    Returns:
        比较结果
    """
    comparison = {}
    
    for method, method_results in results.items():
        fidelities = [r.get('fidelity', 0) for r in method_results]
        depths = [r.get('circuit_depth', 0) for r in method_results]
        successes = [r.get('success', False) for r in method_results]
        
        comparison[method] = {
            'mean_fidelity': np.mean(fidelities),
            'std_fidelity': np.std(fidelities),
            'mean_depth': np.mean(depths),
            'success_rate': np.mean(successes),
            'n_trials': len(method_results)
        }
    
    # 排名
    methods = list(comparison.keys())
    
    # 按保真度排名
    fidelity_rank = sorted(methods, 
                          key=lambda m: comparison[m]['mean_fidelity'],
                          reverse=True)
    
    # 按效率排名（高保真度、低深度）
    efficiency_scores = {
        m: comparison[m]['mean_fidelity'] / (comparison[m]['mean_depth'] + 1)
        for m in methods
    }
    efficiency_rank = sorted(methods, key=lambda m: efficiency_scores[m], reverse=True)
    
    return {
        'details': comparison,
        'fidelity_ranking': fidelity_rank,
        'efficiency_ranking': efficiency_rank,
        'efficiency_scores': efficiency_scores
    }


class MetricsLogger:
    """指标记录器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def log(self, step: int, **kwargs):
        """记录指标"""
        entry = {'step': step, **kwargs}
        self.history.append(entry)
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_metric(self, key: str) -> List:
        """获取特定指标"""
        return self.metrics.get(key, [])
    
    def get_latest(self, key: str, default=None):
        """获取最新值"""
        values = self.metrics.get(key, [])
        return values[-1] if values else default
    
    def get_mean(self, key: str, window: Optional[int] = None) -> float:
        """获取均值"""
        values = self.metrics.get(key, [])
        if not values:
            return 0.0
        if window:
            values = values[-window:]
        return np.mean(values)
    
    def summary(self) -> Dict:
        """获取摘要"""
        return {
            key: {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'last': values[-1]
            }
            for key, values in self.metrics.items()
            if values
        }
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'metrics': self.metrics,
            'history': self.history
        }


if __name__ == "__main__":
    print("=== 评估指标测试 ===\n")
    
    # 测试控制指标
    print("1. 控制指标")
    metrics = ControlMetrics(
        fidelity=0.995,
        circuit_depth=10,
        success=True,
        total_time=1.5,
        n_gates=15
    )
    print(f"控制指标: {metrics.to_dict()}")
    
    # 测试电路复杂度
    print("\n2. 电路复杂度")
    circuit = [
        {'gate': 'H', 'qubits': 0},
        {'gate': 'CNOT', 'qubits': [0, 1]},
        {'gate': 'RX', 'qubits': 0, 'params': [0.5]},
    ]
    complexity = compute_gate_complexity(circuit)
    print(f"复杂度: {complexity}")
    
    # 测试学习率指标
    print("\n3. 学习率指标")
    rewards = list(np.cumsum(np.random.randn(100)) + 50)
    learning_metrics = compute_learning_rate_metrics(rewards)
    print(f"学习指标: {learning_metrics}")
    
    # 测试成功率
    print("\n4. 成功率统计")
    results = [
        {'fidelity': 0.995, 'success': True},
        {'fidelity': 0.98, 'success': False},
        {'fidelity': 0.999, 'success': True},
    ]
    success_stats = compute_success_rate(results)
    print(f"成功率: {success_stats}")
    
    # 测试指标记录器
    print("\n5. 指标记录器")
    logger = MetricsLogger()
    for i in range(10):
        logger.log(i, reward=np.random.randn() + i * 0.1, fidelity=0.9 + i * 0.01)
    
    print(f"摘要: {logger.summary()}")
