"""
脉冲优化环境
===========

强化学习环境：优化量子控制脉冲序列
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List, Union
from scipy.linalg import expm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_backend.backend_manager import BackendManager
from quantum_backend.simulator import QuantumGates


class PulseOptimizationEnv(gym.Env):
    """
    脉冲优化环境
    
    目标：优化控制脉冲参数来实现目标量子演化
    
    这个环境模拟了量子系统的哈密顿量演化，
    智能体需要学习最优的脉冲参数来实现目标操作。
    
    哈密顿量: H(t) = H_0 + Σ_k u_k(t) H_k
    其中 H_0 是漂移哈密顿量，H_k 是控制哈密顿量，u_k(t) 是控制脉冲
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self,
                 n_qubits: int = 1,
                 n_time_steps: int = 10,
                 total_time: float = 1.0,
                 target_unitary: Optional[np.ndarray] = None,
                 drift_strength: float = 0.1,
                 max_control_amplitude: float = 1.0,
                 fidelity_threshold: float = 0.99,
                 render_mode: Optional[str] = None):
        """
        初始化脉冲优化环境
        
        Args:
            n_qubits: 量子比特数量
            n_time_steps: 时间步数
            total_time: 总演化时间
            target_unitary: 目标酉矩阵
            drift_strength: 漂移哈密顿量强度
            max_control_amplitude: 最大控制振幅
            fidelity_threshold: 成功阈值
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_time_steps = n_time_steps
        self.total_time = total_time
        self.dt = total_time / n_time_steps
        self.drift_strength = drift_strength
        self.max_control_amplitude = max_control_amplitude
        self.fidelity_threshold = fidelity_threshold
        self.render_mode = render_mode
        
        # 系统维度
        self.dim = 2 ** n_qubits
        
        # 设置哈密顿量
        self._setup_hamiltonians()
        
        # 设置目标
        if target_unitary is None:
            # 默认目标：X门
            self.target_unitary = QuantumGates.X if n_qubits == 1 else QuantumGates.CNOT
        else:
            self.target_unitary = target_unitary
        
        # 控制参数数量：每个时间步有n_controls个控制振幅
        self.n_controls = len(self.control_hamiltonians)
        
        # 观测空间：当前酉矩阵与目标的差异 + 当前时间步
        obs_dim = 2 * self.dim * self.dim + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 动作空间：所有控制振幅（连续）
        self.action_space = spaces.Box(
            low=-max_control_amplitude,
            high=max_control_amplitude,
            shape=(self.n_controls,),
            dtype=np.float32
        )
        
        # 状态变量
        self.current_step = 0
        self.current_unitary = np.eye(self.dim, dtype=np.complex128)
        self.pulse_history = []
        self.fidelity_history = []
    
    def _setup_hamiltonians(self):
        """设置系统哈密顿量"""
        # 漂移哈密顿量 (系统固有演化)
        if self.n_qubits == 1:
            # 单量子比特：Z方向漂移
            self.drift_hamiltonian = self.drift_strength * QuantumGates.Z
            
            # 控制哈密顿量：X和Y方向控制
            self.control_hamiltonians = [
                QuantumGates.X,
                QuantumGates.Y,
            ]
        else:
            # 多量子比特
            # 漂移：局部Z场 + ZZ耦合
            H_drift = np.zeros((self.dim, self.dim), dtype=np.complex128)
            for i in range(self.n_qubits):
                H_i = self._single_qubit_operator(QuantumGates.Z, i)
                H_drift += self.drift_strength * H_i
            
            # 添加耦合
            if self.n_qubits >= 2:
                for i in range(self.n_qubits - 1):
                    ZZ = self._two_qubit_operator(
                        np.kron(QuantumGates.Z, QuantumGates.Z),
                        i, i+1
                    )
                    H_drift += 0.5 * self.drift_strength * ZZ
            
            self.drift_hamiltonian = H_drift
            
            # 控制：每个量子比特的X和Y控制
            self.control_hamiltonians = []
            for i in range(self.n_qubits):
                H_x = self._single_qubit_operator(QuantumGates.X, i)
                H_y = self._single_qubit_operator(QuantumGates.Y, i)
                self.control_hamiltonians.extend([H_x, H_y])
    
    def _single_qubit_operator(self, op: np.ndarray, qubit: int) -> np.ndarray:
        """扩展单量子比特算子到完整空间"""
        result = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.n_qubits):
            if i == qubit:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    def _two_qubit_operator(self, op: np.ndarray, 
                           q1: int, q2: int) -> np.ndarray:
        """扩展双量子比特算子到完整空间"""
        if q1 > q2:
            q1, q2 = q2, q1
        
        result = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.n_qubits):
            if i == q1:
                result = np.kron(result, op[:2, :2])
            elif i == q2:
                result = np.kron(result, op[2:, 2:])
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置状态
        self.current_step = 0
        self.current_unitary = np.eye(self.dim, dtype=np.complex128)
        self.pulse_history = []
        self.fidelity_history = []
        
        # 可选：设置新的目标
        if options and 'target_unitary' in options:
            self.target_unitary = options['target_unitary']
        
        obs = self._get_observation()
        info = {
            'fidelity': self._compute_fidelity()
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步演化"""
        # 构建当前时间步的哈密顿量
        H = self.drift_hamiltonian.copy()
        for k, u_k in enumerate(action):
            H += u_k * self.control_hamiltonians[k]
        
        # 演化
        U_step = expm(-1j * H * self.dt)
        self.current_unitary = U_step @ self.current_unitary
        
        # 记录
        self.pulse_history.append(action.copy())
        self.current_step += 1
        
        # 计算保真度
        fidelity = self._compute_fidelity()
        self.fidelity_history.append(fidelity)
        
        # 计算奖励
        reward = self._compute_reward(fidelity)
        
        # 检查终止条件
        terminated = fidelity >= self.fidelity_threshold
        truncated = self.current_step >= self.n_time_steps
        
        # 获取观测
        obs = self._get_observation()
        
        info = {
            'fidelity': fidelity,
            'step': self.current_step,
            'success': terminated and fidelity >= self.fidelity_threshold,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        # 当前酉矩阵与目标的差异
        diff = self.target_unitary - self.current_unitary
        
        # 展平并分离实部虚部
        obs = np.concatenate([
            diff.real.flatten(),
            diff.imag.flatten(),
            [self.current_step / self.n_time_steps]  # 归一化时间
        ]).astype(np.float32)
        
        return obs
    
    def _compute_fidelity(self) -> float:
        """计算门保真度"""
        d = self.dim
        trace = np.abs(np.trace(self.target_unitary.conj().T @ self.current_unitary))
        fidelity = (trace ** 2 + d) / (d * (d + 1))
        return fidelity
    
    def _compute_reward(self, fidelity: float) -> float:
        """计算奖励"""
        # 保真度提升
        if len(self.fidelity_history) > 1:
            improvement = fidelity - self.fidelity_history[-2]
        else:
            improvement = fidelity - 0.5
        
        # 成功奖励
        success_bonus = 10.0 if fidelity >= self.fidelity_threshold else 0.0
        
        # 最终步的奖励调整
        if self.current_step >= self.n_time_steps:
            final_bonus = fidelity * 5.0  # 鼓励高保真度
        else:
            final_bonus = 0.0
        
        reward = improvement * 2.0 + success_bonus + final_bonus
        
        return reward
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            print(f"\n=== 脉冲优化环境 ===")
            print(f"时间步: {self.current_step}/{self.n_time_steps}")
            print(f"当前保真度: {self._compute_fidelity():.6f}")
            
            if self.pulse_history:
                print(f"最后脉冲振幅: {self.pulse_history[-1]}")
    
    def get_pulse_sequence(self) -> np.ndarray:
        """获取完整的脉冲序列"""
        return np.array(self.pulse_history)
    
    def get_optimization_result(self) -> Dict:
        """获取优化结果"""
        return {
            'target_unitary': self.target_unitary.copy(),
            'achieved_fidelity': self._compute_fidelity(),
            'pulse_sequence': self.get_pulse_sequence(),
            'n_time_steps': self.n_time_steps,
            'total_time': self.total_time,
            'success': self._compute_fidelity() >= self.fidelity_threshold,
            'final_unitary': self.current_unitary.copy()
        }


class RobustPulseOptimizationEnv(PulseOptimizationEnv):
    """
    鲁棒脉冲优化环境
    
    在有噪声的情况下优化控制脉冲
    """
    
    def __init__(self, 
                 noise_strength: float = 0.01,
                 **kwargs):
        """
        Args:
            noise_strength: 噪声强度
        """
        super().__init__(**kwargs)
        self.noise_strength = noise_strength
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """带噪声的演化"""
        # 添加控制噪声
        noisy_action = action + self.noise_strength * np.random.randn(*action.shape)
        noisy_action = np.clip(noisy_action, 
                              -self.max_control_amplitude,
                              self.max_control_amplitude)
        
        # 哈密顿量噪声
        H = self.drift_hamiltonian.copy()
        
        # 添加漂移噪声
        drift_noise = self.noise_strength * np.random.randn() * QuantumGates.Z
        if self.n_qubits == 1:
            H += drift_noise
        
        for k, u_k in enumerate(noisy_action):
            H += u_k * self.control_hamiltonians[k]
        
        # 演化
        U_step = expm(-1j * H * self.dt)
        self.current_unitary = U_step @ self.current_unitary
        
        # 记录（原始动作）
        self.pulse_history.append(action.copy())
        self.current_step += 1
        
        # 计算保真度
        fidelity = self._compute_fidelity()
        self.fidelity_history.append(fidelity)
        
        # 计算奖励
        reward = self._compute_reward(fidelity)
        
        # 检查终止条件
        terminated = fidelity >= self.fidelity_threshold
        truncated = self.current_step >= self.n_time_steps
        
        obs = self._get_observation()
        
        info = {
            'fidelity': fidelity,
            'step': self.current_step,
            'success': terminated and fidelity >= self.fidelity_threshold,
            'noise_strength': self.noise_strength
        }
        
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    print("=== 脉冲优化环境测试 ===\n")
    
    # 测试单量子比特脉冲优化
    print("1. 单量子比特脉冲优化 (目标: X门)")
    env = PulseOptimizationEnv(
        n_qubits=1,
        n_time_steps=20,
        total_time=1.0,
        target_unitary=QuantumGates.X,
        fidelity_threshold=0.95
    )
    
    obs, info = env.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    # 随机策略测试
    total_reward = 0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    
    print(f"随机策略最终保真度: {info['fidelity']:.6f}")
    print(f"总奖励: {total_reward:.4f}")
    
    # 测试带噪声的脉冲优化
    print("\n2. 鲁棒脉冲优化 (带噪声)")
    env_robust = RobustPulseOptimizationEnv(
        n_qubits=1,
        n_time_steps=20,
        noise_strength=0.05,
        fidelity_threshold=0.90
    )
    
    obs, info = env_robust.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    for _ in range(20):
        action = env_robust.action_space.sample()
        obs, reward, terminated, truncated, info = env_robust.step(action)
    
    print(f"带噪声最终保真度: {info['fidelity']:.6f}")
    
    # 测试H门脉冲优化
    print("\n3. H门脉冲优化")
    env_h = PulseOptimizationEnv(
        n_qubits=1,
        n_time_steps=30,
        total_time=2.0,
        target_unitary=QuantumGates.H,
        fidelity_threshold=0.95
    )
    
    obs, info = env_h.reset()
    
    # 使用简单的贪心策略
    best_fidelity = info['fidelity']
    for step in range(30):
        best_action = None
        best_reward = -np.inf
        
        # 尝试几个随机动作，选择最好的
        for _ in range(10):
            action = env_h.action_space.sample()
            # 需要保存状态进行回滚
            saved_unitary = env_h.current_unitary.copy()
            saved_step = env_h.current_step
            
            obs, reward, terminated, truncated, info = env_h.step(action)
            
            if reward > best_reward:
                best_reward = reward
                best_action = action.copy()
                best_fidelity = info['fidelity']
            
            # 回滚
            env_h.current_unitary = saved_unitary
            env_h.current_step = saved_step
            env_h.pulse_history = env_h.pulse_history[:-1] if env_h.pulse_history else []
        
        # 应用最佳动作
        obs, reward, terminated, truncated, info = env_h.step(best_action)
        
        if terminated:
            print(f"成功! 步数: {env_h.current_step}, 保真度: {info['fidelity']:.6f}")
            break
    
    if not terminated:
        print(f"最终保真度: {info['fidelity']:.6f}")
    
    # 获取优化结果
    result = env_h.get_optimization_result()
    print(f"\n脉冲序列形状: {result['pulse_sequence'].shape}")
