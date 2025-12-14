"""
量子态制备环境
=============

强化学习环境：学习制备目标量子态
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List, Union

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_backend.backend_manager import BackendManager
from quantum_backend.simulator import QuantumGates


class StatePreparationEnv(gym.Env):
    """
    量子态制备环境
    
    目标：从|0⟩态出发，通过一系列门操作制备目标量子态
    
    观测空间：当前量子态
    动作空间：门选择或连续参数
    奖励：基于保真度的奖励
    """
    
    # 常用目标态
    COMMON_STATES = {
        'plus': np.array([1, 1]) / np.sqrt(2),  # |+⟩
        'minus': np.array([1, -1]) / np.sqrt(2),  # |-⟩
        'plus_i': np.array([1, 1j]) / np.sqrt(2),  # |+i⟩
        'minus_i': np.array([1, -1j]) / np.sqrt(2),  # |-i⟩
        'one': np.array([0, 1]),  # |1⟩
        'zero': np.array([1, 0]),  # |0⟩
    }
    
    # 双量子比特常用态
    TWO_QUBIT_STATES = {
        'bell_00': np.array([1, 0, 0, 1]) / np.sqrt(2),  # (|00⟩ + |11⟩)/√2
        'bell_01': np.array([1, 0, 0, -1]) / np.sqrt(2),  # (|00⟩ - |11⟩)/√2
        'bell_10': np.array([0, 1, 1, 0]) / np.sqrt(2),  # (|01⟩ + |10⟩)/√2
        'bell_11': np.array([0, 1, -1, 0]) / np.sqrt(2),  # (|01⟩ - |10⟩)/√2
        'ghz_2': np.array([1, 0, 0, 1]) / np.sqrt(2),  # GHZ态（2量子比特）
    }
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self,
                 target_state: Union[str, np.ndarray] = 'plus',
                 n_qubits: int = 1,
                 max_steps: int = 20,
                 action_type: str = "discrete",
                 fidelity_threshold: float = 0.99,
                 step_penalty: float = 0.01,
                 backend_type: str = "simulator",
                 render_mode: Optional[str] = None,
                 **backend_kwargs):
        """
        初始化态制备环境
        
        Args:
            target_state: 目标态（名称或向量）
            n_qubits: 量子比特数量
            max_steps: 最大步数
            action_type: "discrete" 或 "continuous"
            fidelity_threshold: 成功阈值
            step_penalty: 步数惩罚
            backend_type: 后端类型
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.max_steps = max_steps
        self.action_type = action_type
        self.fidelity_threshold = fidelity_threshold
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        
        # 设置目标态
        self._set_target_state(target_state)
        
        # 初始化后端
        self.backend = BackendManager(n_qubits)
        self.backend.set_backend(backend_type, **backend_kwargs)
        
        # 状态维度
        self.dim = 2 ** n_qubits
        
        # 观测空间：量子态的实部和虚部
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * self.dim,),
            dtype=np.float32
        )
        
        # 动作空间
        self._setup_action_space()
        
        # 状态变量
        self.current_step = 0
        self.action_history = []
        self.fidelity_history = []
    
    def _set_target_state(self, target_state: Union[str, np.ndarray]):
        """设置目标态"""
        if isinstance(target_state, str):
            all_states = {**self.COMMON_STATES, **self.TWO_QUBIT_STATES}
            if target_state in all_states:
                self.target_state = all_states[target_state].astype(np.complex128)
                self.target_name = target_state
            else:
                raise ValueError(f"未知的目标态: {target_state}")
        else:
            self.target_state = target_state.astype(np.complex128)
            self.target_name = "custom"
        
        # 归一化
        self.target_state = self.target_state / np.linalg.norm(self.target_state)
    
    def _setup_action_space(self):
        """设置动作空间"""
        if self.action_type == "discrete":
            # 离散动作：选择预定义门
            # 单量子比特门 * n_qubits + 双量子比特门（如果n_qubits > 1）
            n_single_gates = 10  # I, X, Y, Z, H, S, T, Rx, Ry, Rz
            n_actions = n_single_gates * self.n_qubits
            
            if self.n_qubits > 1:
                # 添加CNOT门（所有可能的控制-目标对）
                n_actions += self.n_qubits * (self.n_qubits - 1)
            
            self.action_space = spaces.Discrete(n_actions)
            
            # 门映射
            self._build_gate_map()
        else:
            # 连续动作：每个量子比特3个U3参数
            action_dim = 3 * self.n_qubits
            self.action_space = spaces.Box(
                low=-np.pi, high=np.pi,
                shape=(action_dim,),
                dtype=np.float32
            )
    
    def _build_gate_map(self):
        """构建动作到门的映射"""
        self.gate_map = {}
        idx = 0
        
        # 单量子比特门
        single_gates = [
            ('I', None), ('X', None), ('Y', None), ('Z', None),
            ('H', None), ('S', None), ('T', None),
            ('RX', np.pi/4), ('RY', np.pi/4), ('RZ', np.pi/4)
        ]
        
        for qubit in range(self.n_qubits):
            for gate_name, param in single_gates:
                self.gate_map[idx] = ('single', gate_name, qubit, param)
                idx += 1
        
        # 双量子比特门
        if self.n_qubits > 1:
            for ctrl in range(self.n_qubits):
                for tgt in range(self.n_qubits):
                    if ctrl != tgt:
                        self.gate_map[idx] = ('two', 'CNOT', [ctrl, tgt], None)
                        idx += 1
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置后端
        self.backend.reset()
        
        # 重置状态
        self.current_step = 0
        self.action_history = []
        self.fidelity_history = []
        
        # 可选：设置新的目标态
        if options and 'target_state' in options:
            self._set_target_state(options['target_state'])
        
        obs = self._get_observation()
        info = {
            'target_state': self.target_name,
            'fidelity': self._compute_fidelity()
        }
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 应用动作
        self._apply_action(action)
        self.action_history.append(action)
        self.current_step += 1
        
        # 计算保真度
        fidelity = self._compute_fidelity()
        self.fidelity_history.append(fidelity)
        
        # 计算奖励
        reward = self._compute_reward(fidelity)
        
        # 检查终止条件
        terminated = fidelity >= self.fidelity_threshold
        truncated = self.current_step >= self.max_steps
        
        # 获取观测
        obs = self._get_observation()
        
        # 信息
        info = {
            'fidelity': fidelity,
            'step': self.current_step,
            'success': terminated,
            'circuit_depth': self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """应用动作"""
        if self.action_type == "discrete":
            gate_info = self.gate_map[action]
            gate_type, gate_name, qubit, param = gate_info
            
            if gate_type == 'single':
                if param is not None:
                    self.backend.apply_gate(gate_name, qubit, [param])
                else:
                    self.backend.apply_gate(gate_name, qubit)
            else:
                # 双量子比特门
                self.backend.apply_gate(gate_name, qubit)
        else:
            # 连续动作
            for qubit in range(self.n_qubits):
                params = action[qubit*3:(qubit+1)*3]
                self.backend.apply_gate('U3', qubit, params.tolist())
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        sv = self.backend.get_statevector()
        obs = np.concatenate([sv.real, sv.imag]).astype(np.float32)
        return obs
    
    def _compute_fidelity(self) -> float:
        """计算态保真度"""
        current_state = self.backend.get_statevector()
        fidelity = np.abs(np.vdot(self.target_state, current_state)) ** 2
        return fidelity
    
    def _compute_reward(self, fidelity: float) -> float:
        """计算奖励"""
        # 基础奖励：保真度提升
        if len(self.fidelity_history) > 1:
            fidelity_improvement = fidelity - self.fidelity_history[-2]
        else:
            # 初始态与目标态的保真度
            initial_fidelity = np.abs(self.target_state[0]) ** 2
            fidelity_improvement = fidelity - initial_fidelity
        
        # 步数惩罚
        step_cost = -self.step_penalty
        
        # 成功奖励
        success_bonus = 10.0 if fidelity >= self.fidelity_threshold else 0.0
        
        # 总奖励
        reward = fidelity_improvement * 5.0 + step_cost + success_bonus
        
        return reward
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            print(f"\n=== 态制备环境 ===")
            print(f"目标态: {self.target_name}")
            print(f"步数: {self.current_step}/{self.max_steps}")
            print(f"当前保真度: {self._compute_fidelity():.6f}")
            
            if self.backend.supports_statevector():
                print(f"当前态: {self.backend.get_statevector()}")
                print(f"目标态: {self.target_state}")
    
    def get_preparation_result(self) -> Dict:
        """获取制备结果"""
        return {
            'target_state': self.target_name,
            'achieved_fidelity': self._compute_fidelity(),
            'circuit_depth': self.current_step,
            'action_sequence': self.action_history.copy(),
            'success': self._compute_fidelity() >= self.fidelity_threshold,
            'prepared_state': self.backend.get_statevector().copy()
        }


class GHZPreparationEnv(StatePreparationEnv):
    """
    GHZ态制备环境
    
    专门用于制备多量子比特GHZ态
    """
    
    def __init__(self, n_qubits: int = 3, **kwargs):
        """
        初始化GHZ制备环境
        
        Args:
            n_qubits: 量子比特数量
        """
        # 构建GHZ态
        ghz_state = np.zeros(2 ** n_qubits, dtype=np.complex128)
        ghz_state[0] = 1 / np.sqrt(2)  # |00...0⟩
        ghz_state[-1] = 1 / np.sqrt(2)  # |11...1⟩
        
        super().__init__(
            target_state=ghz_state,
            n_qubits=n_qubits,
            **kwargs
        )
        self.target_name = f"GHZ_{n_qubits}"


class WStatePreparationEnv(StatePreparationEnv):
    """
    W态制备环境
    
    专门用于制备多量子比特W态
    """
    
    def __init__(self, n_qubits: int = 3, **kwargs):
        """
        初始化W态制备环境
        
        Args:
            n_qubits: 量子比特数量
        """
        # 构建W态: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
        w_state = np.zeros(2 ** n_qubits, dtype=np.complex128)
        for i in range(n_qubits):
            idx = 1 << (n_qubits - 1 - i)
            w_state[idx] = 1 / np.sqrt(n_qubits)
        
        super().__init__(
            target_state=w_state,
            n_qubits=n_qubits,
            **kwargs
        )
        self.target_name = f"W_{n_qubits}"


class RandomStatePreparationEnv(StatePreparationEnv):
    """
    随机态制备环境
    
    每次重置时生成新的随机目标态
    """
    
    def __init__(self, n_qubits: int = 1, **kwargs):
        # 先用零态初始化
        super().__init__(
            target_state='zero' if n_qubits == 1 else 'bell_00',
            n_qubits=n_qubits,
            **kwargs
        )
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置并生成新的随机目标态"""
        if seed is not None:
            np.random.seed(seed)
        
        # 生成随机态
        random_state = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
        random_state = random_state / np.linalg.norm(random_state)
        
        self._set_target_state(random_state)
        self.target_name = "random"
        
        return super().reset(seed=seed, options=options)


if __name__ == "__main__":
    print("=== 量子态制备环境测试 ===\n")
    
    # 测试单量子比特态制备
    print("1. 单量子比特态制备 (目标: |+⟩)")
    env = StatePreparationEnv(target_state='plus', n_qubits=1, max_steps=10)
    
    obs, info = env.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    # 应用H门应该直接达到目标
    # H门在离散动作空间中的索引是4
    obs, reward, terminated, truncated, info = env.step(4)
    print(f"应用H门后保真度: {info['fidelity']:.6f}")
    print(f"成功: {info['success']}")
    
    # 测试Bell态制备
    print("\n2. Bell态制备")
    env_bell = StatePreparationEnv(target_state='bell_00', n_qubits=2, max_steps=20)
    
    obs, info = env_bell.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    # 手动执行Bell态制备电路
    # H on qubit 0, then CNOT(0,1)
    env_bell.backend.apply_gate('H', 0)
    env_bell.backend.apply_gate('CNOT', [0, 1])
    fidelity = env_bell._compute_fidelity()
    print(f"H+CNOT后保真度: {fidelity:.6f}")
    
    # 测试GHZ态制备
    print("\n3. GHZ态制备 (3量子比特)")
    env_ghz = GHZPreparationEnv(n_qubits=3, max_steps=30)
    
    obs, info = env_ghz.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    # GHZ制备电路: H on qubit 0, CNOT(0,1), CNOT(0,2)
    env_ghz.backend.apply_gate('H', 0)
    env_ghz.backend.apply_gate('CNOT', [0, 1])
    env_ghz.backend.apply_gate('CNOT', [0, 2])
    fidelity = env_ghz._compute_fidelity()
    print(f"手动GHZ电路保真度: {fidelity:.6f}")
    
    # 测试随机态制备
    print("\n4. 随机态制备")
    env_random = RandomStatePreparationEnv(n_qubits=1, max_steps=20)
    
    for i in range(3):
        obs, info = env_random.reset(seed=i)
        print(f"  随机态 {i+1}: 初始保真度 = {info['fidelity']:.6f}")
