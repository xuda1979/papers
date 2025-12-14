"""
基础量子控制环境
===============

实现Gymnasium兼容的量子控制强化学习环境基类
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
from abc import abstractmethod

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_backend.backend_manager import BackendManager, BackendType


class QuantumControlEnv(gym.Env):
    """
    量子控制环境基类
    
    将量子控制问题建模为强化学习环境，支持：
    - 离散动作空间（选择预定义的门）
    - 连续动作空间（门参数优化）
    - 混合动作空间
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 n_qubits: int = 1,
                 max_steps: int = 50,
                 backend_type: str = "simulator",
                 action_type: str = "discrete",
                 render_mode: Optional[str] = None,
                 **backend_kwargs):
        """
        初始化环境
        
        Args:
            n_qubits: 量子比特数量
            max_steps: 最大步数
            backend_type: 后端类型
            action_type: 动作类型 ("discrete", "continuous", "hybrid")
            render_mode: 渲染模式
            **backend_kwargs: 后端参数
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.action_type = action_type
        self.render_mode = render_mode
        
        # 初始化后端
        self.backend = BackendManager(n_qubits)
        self.backend.set_backend(backend_type, **backend_kwargs)
        
        # 状态维度：量子态的实部和虚部
        self.state_dim = 2 * (2 ** n_qubits)
        
        # 定义观测空间
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # 定义动作空间
        self._setup_action_space()
        
        # 环境状态
        self.current_step = 0
        self.done = False
        self.truncated = False
        self.info = {}
        
        # 用于记录
        self.episode_reward = 0.0
        self.action_history = []
    
    @abstractmethod
    def _setup_action_space(self):
        """设置动作空间（子类实现）"""
        pass
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """计算奖励（子类实现）"""
        pass
    
    @abstractmethod
    def _is_done(self) -> bool:
        """检查是否终止（子类实现）"""
        pass
    
    @abstractmethod
    def _apply_action(self, action):
        """应用动作（子类实现）"""
        pass
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Returns:
            observation: 初始观测
            info: 附加信息
        """
        super().reset(seed=seed)
        
        # 重置后端
        self.backend.reset()
        
        # 重置状态
        self.current_step = 0
        self.done = False
        self.truncated = False
        self.episode_reward = 0.0
        self.action_history = []
        self.info = {}
        
        # 获取初始观测
        observation = self._get_observation()
        
        return observation, self.info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 动作
            
        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 附加信息
        """
        # 应用动作
        self._apply_action(action)
        self.action_history.append(action)
        self.current_step += 1
        
        # 计算奖励
        reward = self._compute_reward()
        self.episode_reward += reward
        
        # 检查终止条件
        terminated = self._is_done()
        truncated = self.current_step >= self.max_steps
        
        # 获取观测
        observation = self._get_observation()
        
        # 更新信息
        self.info.update({
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'fidelity': self._get_fidelity() if hasattr(self, '_get_fidelity') else None,
        })
        
        return observation, reward, terminated, truncated, self.info
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        if self.backend.supports_statevector():
            sv = self.backend.get_statevector()
            # 将复数状态向量转换为实数观测
            obs = np.concatenate([sv.real, sv.imag]).astype(np.float32)
        else:
            # 对于不支持状态向量的后端，使用测量结果构建观测
            counts = self.backend.measure(shots=100)
            dim = 2 ** self.n_qubits
            obs = np.zeros(self.state_dim, dtype=np.float32)
            for bitstring, count in counts.items():
                idx = int(bitstring, 2)
                prob = count / 100
                obs[idx] = np.sqrt(prob)
        
        return obs
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            print(f"\n步数: {self.current_step}/{self.max_steps}")
            print(f"累计奖励: {self.episode_reward:.4f}")
            if self.backend.supports_statevector():
                sv = self.backend.get_statevector()
                print(f"状态向量: {sv}")
            print(f"动作历史: {self.action_history[-5:]}")  # 最后5个动作
    
    def close(self):
        """关闭环境"""
        pass
    
    def switch_backend(self, backend_type: str, **kwargs):
        """
        切换后端
        
        Args:
            backend_type: 新的后端类型
            **kwargs: 后端参数
        """
        self.backend.set_backend(backend_type, **kwargs)
    
    def get_circuit_depth(self) -> int:
        """获取当前电路深度"""
        return self.current_step


class DiscreteGateEnv(QuantumControlEnv):
    """
    离散门选择环境
    
    动作空间是预定义的量子门集合
    """
    
    # 预定义的门集合
    GATE_SET = ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 
                'RX_PI4', 'RX_PI2', 'RX_PI', 
                'RY_PI4', 'RY_PI2', 'RY_PI',
                'RZ_PI4', 'RZ_PI2', 'RZ_PI']
    
    def __init__(self, n_qubits: int = 1, **kwargs):
        self.gate_to_params = {
            'I': ('I', None),
            'X': ('X', None),
            'Y': ('Y', None),
            'Z': ('Z', None),
            'H': ('H', None),
            'S': ('S', None),
            'T': ('T', None),
            'RX_PI4': ('RX', [np.pi/4]),
            'RX_PI2': ('RX', [np.pi/2]),
            'RX_PI': ('RX', [np.pi]),
            'RY_PI4': ('RY', [np.pi/4]),
            'RY_PI2': ('RY', [np.pi/2]),
            'RY_PI': ('RY', [np.pi]),
            'RZ_PI4': ('RZ', [np.pi/4]),
            'RZ_PI2': ('RZ', [np.pi/2]),
            'RZ_PI': ('RZ', [np.pi]),
        }
        super().__init__(n_qubits=n_qubits, action_type="discrete", **kwargs)
    
    def _setup_action_space(self):
        """设置离散动作空间"""
        # 每个量子比特可以选择一个门
        n_gates = len(self.GATE_SET)
        self.action_space = spaces.Discrete(n_gates * self.n_qubits)
    
    def _apply_action(self, action: int):
        """应用离散动作"""
        # 解析动作：哪个门，哪个量子比特
        gate_idx = action % len(self.GATE_SET)
        qubit = action // len(self.GATE_SET)
        
        gate_name = self.GATE_SET[gate_idx]
        gate, params = self.gate_to_params[gate_name]
        
        self.backend.apply_gate(gate, qubit, params)


class ContinuousGateEnv(QuantumControlEnv):
    """
    连续参数环境
    
    动作空间是连续的门参数
    """
    
    def __init__(self, n_qubits: int = 1, n_params_per_step: int = 3, **kwargs):
        """
        Args:
            n_qubits: 量子比特数量
            n_params_per_step: 每步的参数数量（默认3个对应U3门）
        """
        self.n_params_per_step = n_params_per_step
        super().__init__(n_qubits=n_qubits, action_type="continuous", **kwargs)
    
    def _setup_action_space(self):
        """设置连续动作空间"""
        # 每个量子比特有n_params_per_step个参数
        action_dim = self.n_qubits * self.n_params_per_step
        self.action_space = spaces.Box(
            low=-np.pi, high=np.pi,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def _apply_action(self, action: np.ndarray):
        """应用连续动作"""
        for qubit in range(self.n_qubits):
            start_idx = qubit * self.n_params_per_step
            params = action[start_idx:start_idx + self.n_params_per_step]
            
            if self.n_params_per_step == 3:
                # U3门
                self.backend.apply_gate('U3', qubit, params.tolist())
            elif self.n_params_per_step == 1:
                # 只有一个旋转
                self.backend.apply_gate('RY', qubit, params.tolist())
            else:
                # 分解为多个旋转
                for i, param in enumerate(params):
                    gate = ['RX', 'RY', 'RZ'][i % 3]
                    self.backend.apply_gate(gate, qubit, [float(param)])


if __name__ == "__main__":
    print("=== 量子控制环境测试 ===\n")
    
    # 这里只是基类测试，具体功能在子类中实现
    print("基类已定义，请使用具体的环境实现（如GateSynthesisEnv）")
