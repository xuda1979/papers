"""
量子门合成环境
=============

强化学习环境：学习合成目标量子门
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


class GateSynthesisEnv(gym.Env):
    """
    量子门合成环境
    
    目标：学习一系列基础门操作来合成目标量子门
    
    观测空间：当前酉矩阵与目标门的差异
    动作空间：选择基础门或连续参数
    奖励：基于保真度的奖励
    """
    
    # 基础门集合
    BASIC_GATES = {
        0: ('I', None),
        1: ('X', None),
        2: ('Y', None),
        3: ('Z', None),
        4: ('H', None),
        5: ('S', None),
        6: ('T', None),
        7: ('RX', 'param'),
        8: ('RY', 'param'),
        9: ('RZ', 'param'),
    }
    
    # 预定义目标门
    TARGET_GATES = {
        'X': QuantumGates.X,
        'Y': QuantumGates.Y,
        'Z': QuantumGates.Z,
        'H': QuantumGates.H,
        'S': QuantumGates.S,
        'T': QuantumGates.T,
        'SQRT_X': np.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2,
        'SQRT_Y': np.array([[1+1j, -1-1j], [1+1j, 1+1j]]) / 2,
    }
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self,
                 target_gate: Union[str, np.ndarray] = 'H',
                 n_qubits: int = 1,
                 max_steps: int = 20,
                 action_type: str = "discrete",
                 fidelity_threshold: float = 0.99,
                 step_penalty: float = 0.01,
                 backend_type: str = "simulator",
                 render_mode: Optional[str] = None,
                 **backend_kwargs):
        """
        初始化门合成环境
        
        Args:
            target_gate: 目标门（名称或矩阵）
            n_qubits: 量子比特数量（门合成通常为1）
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
        
        # 设置目标门
        if isinstance(target_gate, str):
            if target_gate in self.TARGET_GATES:
                self.target_gate = self.TARGET_GATES[target_gate]
                self.target_name = target_gate
            else:
                raise ValueError(f"未知的目标门: {target_gate}")
        else:
            self.target_gate = target_gate
            self.target_name = "custom"
        
        # 初始化后端
        self.backend = BackendManager(n_qubits)
        self.backend.set_backend(backend_type, **backend_kwargs)
        
        # 当前酉矩阵
        self.dim = 2 ** n_qubits
        self.current_unitary = np.eye(self.dim, dtype=np.complex128)
        
        # 观测空间：酉矩阵的实部和虚部展平
        obs_dim = 2 * self.dim * self.dim
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 动作空间
        if action_type == "discrete":
            # 离散：选择基础门
            self.action_space = spaces.Discrete(len(self.BASIC_GATES))
        else:
            # 连续：3个U3参数
            self.action_space = spaces.Box(
                low=-np.pi, high=np.pi,
                shape=(3,),
                dtype=np.float32
            )
        
        # 状态变量
        self.current_step = 0
        self.action_history = []
        self.fidelity_history = []
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置后端
        self.backend.reset()
        
        # 重置酉矩阵
        self.current_unitary = np.eye(self.dim, dtype=np.complex128)
        
        # 重置状态
        self.current_step = 0
        self.action_history = []
        self.fidelity_history = []
        
        # 可选：设置新的目标门
        if options and 'target_gate' in options:
            target = options['target_gate']
            if isinstance(target, str):
                self.target_gate = self.TARGET_GATES[target]
                self.target_name = target
            else:
                self.target_gate = target
                self.target_name = "custom"
        
        obs = self._get_observation()
        info = {
            'target_gate': self.target_name,
            'fidelity': self._compute_fidelity()
        }
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 应用动作
        gate_matrix = self._action_to_gate(action)
        self.current_unitary = gate_matrix @ self.current_unitary
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
    
    def _action_to_gate(self, action) -> np.ndarray:
        """将动作转换为门矩阵"""
        if self.action_type == "discrete":
            gate_name, param_type = self.BASIC_GATES[action]
            
            if param_type is None:
                # 无参数门
                return getattr(QuantumGates, gate_name)
            else:
                # 参数化门，使用随机参数（或可以扩展为子动作）
                # 这里简化为固定角度
                angles = [np.pi/4, np.pi/2, np.pi]
                theta = angles[self.current_step % 3]
                
                if gate_name == 'RX':
                    return QuantumGates.Rx(theta)
                elif gate_name == 'RY':
                    return QuantumGates.Ry(theta)
                elif gate_name == 'RZ':
                    return QuantumGates.Rz(theta)
        else:
            # 连续动作：U3门
            theta, phi, lam = action
            return QuantumGates.U3(theta, phi, lam)
    
    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        # 计算当前酉矩阵与目标的差异矩阵
        diff = self.target_gate - self.current_unitary
        
        # 展平并分离实部虚部
        obs = np.concatenate([
            diff.real.flatten(),
            diff.imag.flatten()
        ]).astype(np.float32)
        
        # 归一化到[-1, 1]
        max_val = np.max(np.abs(obs)) + 1e-8
        obs = obs / max_val
        
        return obs
    
    def _compute_fidelity(self) -> float:
        """计算门保真度"""
        # 平均门保真度
        d = self.dim
        trace = np.abs(np.trace(self.target_gate.conj().T @ self.current_unitary))
        fidelity = (trace ** 2 + d) / (d * (d + 1))
        return fidelity
    
    def _compute_reward(self, fidelity: float) -> float:
        """计算奖励"""
        # 基础奖励：保真度提升
        if len(self.fidelity_history) > 1:
            fidelity_improvement = fidelity - self.fidelity_history[-2]
        else:
            fidelity_improvement = fidelity - 0.5  # 初始保真度约0.5
        
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
            print(f"\n=== 门合成环境 ===")
            print(f"目标门: {self.target_name}")
            print(f"步数: {self.current_step}/{self.max_steps}")
            print(f"当前保真度: {self._compute_fidelity():.6f}")
            print(f"目标保真度: {self.fidelity_threshold}")
            print(f"动作历史: {self.action_history[-5:]}")
            
            print(f"\n当前酉矩阵:")
            print(self.current_unitary)
    
    def get_synthesis_result(self) -> Dict:
        """获取合成结果"""
        return {
            'target_gate': self.target_name,
            'achieved_fidelity': self._compute_fidelity(),
            'circuit_depth': self.current_step,
            'action_sequence': self.action_history.copy(),
            'success': self._compute_fidelity() >= self.fidelity_threshold,
            'synthesized_unitary': self.current_unitary.copy()
        }


class MultiQubitGateSynthesisEnv(GateSynthesisEnv):
    """
    多量子比特门合成环境
    
    支持2量子比特门如CNOT、CZ等的合成
    """
    
    # 双量子比特基础门
    TWO_QUBIT_GATES = {
        10: ('CNOT', [0, 1]),
        11: ('CNOT', [1, 0]),
        12: ('CZ', [0, 1]),
        13: ('SWAP', [0, 1]),
    }
    
    def __init__(self, 
                 target_gate: Union[str, np.ndarray] = 'CNOT',
                 max_steps: int = 30,
                 **kwargs):
        """
        初始化多量子比特门合成环境
        """
        # 添加多量子比特目标门
        self.TARGET_GATES.update({
            'CNOT': QuantumGates.CNOT,
            'CZ': QuantumGates.CZ,
            'SWAP': QuantumGates.SWAP,
        })
        
        super().__init__(
            target_gate=target_gate,
            n_qubits=2,
            max_steps=max_steps,
            **kwargs
        )
        
        # 扩展动作空间
        if self.action_type == "discrete":
            # 单量子比特门 * 2 + 双量子比特门
            n_single = len(self.BASIC_GATES)
            n_two = len(self.TWO_QUBIT_GATES)
            self.action_space = spaces.Discrete(n_single * 2 + n_two)
    
    def _action_to_gate(self, action) -> np.ndarray:
        """将动作转换为门矩阵"""
        if self.action_type == "discrete":
            n_single = len(self.BASIC_GATES)
            
            if action < n_single * 2:
                # 单量子比特门
                qubit = action // n_single
                gate_idx = action % n_single
                gate_name, param_type = self.BASIC_GATES[gate_idx]
                
                if param_type is None:
                    single_gate = getattr(QuantumGates, gate_name)
                else:
                    theta = np.pi / 4
                    if gate_name == 'RX':
                        single_gate = QuantumGates.Rx(theta)
                    elif gate_name == 'RY':
                        single_gate = QuantumGates.Ry(theta)
                    else:
                        single_gate = QuantumGates.Rz(theta)
                
                # 扩展到2量子比特空间
                if qubit == 0:
                    return np.kron(single_gate, np.eye(2))
                else:
                    return np.kron(np.eye(2), single_gate)
            else:
                # 双量子比特门
                two_qubit_idx = action - n_single * 2 + 10
                gate_name, qubits = self.TWO_QUBIT_GATES[two_qubit_idx]
                return getattr(QuantumGates, gate_name)
        else:
            # 连续动作空间
            return super()._action_to_gate(action)


if __name__ == "__main__":
    print("=== 量子门合成环境测试 ===\n")
    
    # 测试单量子比特门合成
    print("1. 单量子比特门合成 (目标: H门)")
    env = GateSynthesisEnv(target_gate='H', max_steps=20, action_type='discrete')
    
    obs, info = env.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    # 随机策略测试
    total_reward = 0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"成功! 保真度: {info['fidelity']:.6f}, 步数: {info['step']}")
            break
    
    if not terminated:
        print(f"未成功. 最终保真度: {info['fidelity']:.6f}")
    
    print(f"总奖励: {total_reward:.4f}")
    
    # 测试连续动作空间
    print("\n2. 连续动作空间测试")
    env_cont = GateSynthesisEnv(target_gate='X', max_steps=10, action_type='continuous')
    obs, info = env_cont.reset()
    
    # 尝试直接的X门参数
    action = np.array([np.pi, 0, np.pi])  # U3参数使得U3(π,0,π) ≈ X
    obs, reward, terminated, truncated, info = env_cont.step(action)
    print(f"单步后保真度: {info['fidelity']:.6f}")
    
    # 测试多量子比特门合成
    print("\n3. 多量子比特门合成 (目标: CNOT)")
    env_multi = MultiQubitGateSynthesisEnv(target_gate='CNOT', max_steps=30)
    obs, info = env_multi.reset()
    print(f"初始保真度: {info['fidelity']:.6f}")
    
    for _ in range(30):
        action = env_multi.action_space.sample()
        obs, reward, terminated, truncated, info = env_multi.step(action)
        
        if terminated:
            print(f"成功! 保真度: {info['fidelity']:.6f}, 步数: {info['step']}")
            break
    
    if not terminated:
        print(f"随机策略未成功. 最终保真度: {info['fidelity']:.6f}")
