"""
量子模拟器模块
============

实现基于NumPy的量子计算模拟器，支持：
- 任意量子比特数量（受内存限制）
- 各种量子门操作
- 噪声模型
- 测量操作
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class QuantumState:
    """量子态表示类"""
    
    def __init__(self, n_qubits: int):
        """
        初始化量子态
        
        Args:
            n_qubits: 量子比特数量
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        # 初始化为|0...0⟩态
        self.state_vector = np.zeros(self.dim, dtype=np.complex128)
        self.state_vector[0] = 1.0
    
    def reset(self):
        """重置为|0...0⟩态"""
        self.state_vector = np.zeros(self.dim, dtype=np.complex128)
        self.state_vector[0] = 1.0
    
    def set_state(self, state_vector: np.ndarray):
        """设置量子态"""
        if len(state_vector) != self.dim:
            raise ValueError(f"状态向量维度必须为 {self.dim}")
        # 归一化
        norm = np.linalg.norm(state_vector)
        self.state_vector = state_vector / norm
    
    def get_density_matrix(self) -> np.ndarray:
        """获取密度矩阵"""
        return np.outer(self.state_vector, np.conj(self.state_vector))
    
    def get_probabilities(self) -> np.ndarray:
        """获取测量概率分布"""
        return np.abs(self.state_vector) ** 2
    
    def fidelity(self, target_state: np.ndarray) -> float:
        """计算与目标态的保真度"""
        target_norm = target_state / np.linalg.norm(target_state)
        return np.abs(np.vdot(target_norm, self.state_vector)) ** 2
    
    def copy(self) -> 'QuantumState':
        """复制量子态"""
        new_state = QuantumState(self.n_qubits)
        new_state.state_vector = self.state_vector.copy()
        return new_state


class QuantumGates:
    """量子门库"""
    
    # 单量子比特门
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    
    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """绕X轴旋转门"""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
    
    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """绕Y轴旋转门"""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
    
    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """绕Z轴旋转门"""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=np.complex128)
    
    @staticmethod
    def U3(theta: float, phi: float, lam: float) -> np.ndarray:
        """通用单量子比特门"""
        return np.array([
            [np.cos(theta/2), -np.exp(1j * lam) * np.sin(theta/2)],
            [np.exp(1j * phi) * np.sin(theta/2), 
             np.exp(1j * (phi + lam)) * np.cos(theta/2)]
        ], dtype=np.complex128)
    
    # 双量子比特门
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)
    
    CZ = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex128)
    
    SWAP = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)
    
    @staticmethod
    def CRx(theta: float) -> np.ndarray:
        """受控Rx门"""
        rx = QuantumGates.Rx(theta)
        return np.block([
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), rx]
        ])
    
    @staticmethod
    def CRy(theta: float) -> np.ndarray:
        """受控Ry门"""
        ry = QuantumGates.Ry(theta)
        return np.block([
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), ry]
        ])
    
    @staticmethod
    def CRz(theta: float) -> np.ndarray:
        """受控Rz门"""
        rz = QuantumGates.Rz(theta)
        return np.block([
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), rz]
        ])


class NoiseModel:
    """噪声模型"""
    
    def __init__(self, 
                 depolarizing_rate: float = 0.0,
                 amplitude_damping_rate: float = 0.0,
                 phase_damping_rate: float = 0.0,
                 readout_error: float = 0.0):
        """
        初始化噪声模型
        
        Args:
            depolarizing_rate: 去极化噪声率
            amplitude_damping_rate: 振幅阻尼率
            phase_damping_rate: 相位阻尼率
            readout_error: 读出错误率
        """
        self.depolarizing_rate = depolarizing_rate
        self.amplitude_damping_rate = amplitude_damping_rate
        self.phase_damping_rate = phase_damping_rate
        self.readout_error = readout_error
    
    def apply_depolarizing(self, density_matrix: np.ndarray, 
                          n_qubits: int) -> np.ndarray:
        """应用去极化噪声"""
        if self.depolarizing_rate == 0:
            return density_matrix
        
        p = self.depolarizing_rate
        dim = 2 ** n_qubits
        identity = np.eye(dim) / dim
        return (1 - p) * density_matrix + p * identity
    
    def apply_amplitude_damping(self, density_matrix: np.ndarray,
                                qubit: int, n_qubits: int) -> np.ndarray:
        """应用振幅阻尼噪声"""
        if self.amplitude_damping_rate == 0:
            return density_matrix
        
        gamma = self.amplitude_damping_rate
        # Kraus算子
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        
        # 扩展到完整希尔伯特空间
        K0_full = self._expand_operator(K0, qubit, n_qubits)
        K1_full = self._expand_operator(K1, qubit, n_qubits)
        
        return K0_full @ density_matrix @ K0_full.conj().T + \
               K1_full @ density_matrix @ K1_full.conj().T
    
    def apply_readout_error(self, probabilities: np.ndarray) -> np.ndarray:
        """应用读出错误"""
        if self.readout_error == 0:
            return probabilities
        
        p = self.readout_error
        n = len(probabilities)
        # 简化的读出错误模型
        noisy_probs = (1 - p) * probabilities + p / n
        return noisy_probs / np.sum(noisy_probs)
    
    def _expand_operator(self, op: np.ndarray, qubit: int, 
                        n_qubits: int) -> np.ndarray:
        """将单量子比特算子扩展到完整空间"""
        result = np.array([[1.0]])
        for i in range(n_qubits):
            if i == qubit:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2))
        return result


class QuantumSimulator:
    """量子模拟器"""
    
    def __init__(self, n_qubits: int, noise_model: Optional[NoiseModel] = None):
        """
        初始化量子模拟器
        
        Args:
            n_qubits: 量子比特数量
            noise_model: 噪声模型（可选）
        """
        self.n_qubits = n_qubits
        self.state = QuantumState(n_qubits)
        self.noise_model = noise_model
        self.gates = QuantumGates()
        self.circuit_history = []
        
    def reset(self):
        """重置模拟器状态"""
        self.state.reset()
        self.circuit_history = []
    
    def apply_gate(self, gate: Union[str, np.ndarray], 
                   qubits: Union[int, List[int]], 
                   params: Optional[List[float]] = None):
        """
        应用量子门
        
        Args:
            gate: 门名称或门矩阵
            qubits: 作用的量子比特
            params: 门参数（对于参数化门）
        """
        if isinstance(qubits, int):
            qubits = [qubits]
        
        # 获取门矩阵
        if isinstance(gate, str):
            gate_matrix = self._get_gate_matrix(gate, params)
        else:
            gate_matrix = gate
        
        # 构建完整的算子
        full_operator = self._build_full_operator(gate_matrix, qubits)
        
        # 应用门
        self.state.state_vector = full_operator @ self.state.state_vector
        
        # 记录电路历史
        self.circuit_history.append({
            'gate': gate if isinstance(gate, str) else 'custom',
            'qubits': qubits,
            'params': params
        })
        
        # 应用噪声（如果有）
        if self.noise_model and self.noise_model.depolarizing_rate > 0:
            dm = self.state.get_density_matrix()
            dm = self.noise_model.apply_depolarizing(dm, self.n_qubits)
            # 从密度矩阵恢复纯态（近似）
            eigenvalues, eigenvectors = np.linalg.eigh(dm)
            max_idx = np.argmax(eigenvalues)
            self.state.state_vector = eigenvectors[:, max_idx]
    
    def _get_gate_matrix(self, gate_name: str, 
                        params: Optional[List[float]] = None) -> np.ndarray:
        """获取门矩阵"""
        gate_name = gate_name.upper()
        
        # 无参数门
        simple_gates = {
            'I': self.gates.I,
            'X': self.gates.X,
            'Y': self.gates.Y,
            'Z': self.gates.Z,
            'H': self.gates.H,
            'S': self.gates.S,
            'T': self.gates.T,
            'CNOT': self.gates.CNOT,
            'CX': self.gates.CNOT,
            'CZ': self.gates.CZ,
            'SWAP': self.gates.SWAP,
        }
        
        if gate_name in simple_gates:
            return simple_gates[gate_name]
        
        # 参数化门
        if params is None:
            raise ValueError(f"参数化门 {gate_name} 需要参数")
        
        param_gates = {
            'RX': lambda p: self.gates.Rx(p[0]),
            'RY': lambda p: self.gates.Ry(p[0]),
            'RZ': lambda p: self.gates.Rz(p[0]),
            'U3': lambda p: self.gates.U3(p[0], p[1], p[2]),
            'CRX': lambda p: self.gates.CRx(p[0]),
            'CRY': lambda p: self.gates.CRy(p[0]),
            'CRZ': lambda p: self.gates.CRz(p[0]),
        }
        
        if gate_name in param_gates:
            return param_gates[gate_name](params)
        
        raise ValueError(f"未知的门: {gate_name}")
    
    def _build_full_operator(self, gate_matrix: np.ndarray, 
                            qubits: List[int]) -> np.ndarray:
        """构建完整的算子矩阵"""
        n_gate_qubits = int(np.log2(gate_matrix.shape[0]))
        
        if n_gate_qubits == 1:
            return self._single_qubit_operator(gate_matrix, qubits[0])
        elif n_gate_qubits == 2:
            return self._two_qubit_operator(gate_matrix, qubits[0], qubits[1])
        else:
            raise ValueError("目前只支持1-2量子比特门")
    
    def _single_qubit_operator(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """构建单量子比特门的完整算子"""
        result = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.n_qubits):
            if i == qubit:
                result = np.kron(result, gate)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    def _two_qubit_operator(self, gate: np.ndarray, 
                           qubit1: int, qubit2: int) -> np.ndarray:
        """构建双量子比特门的完整算子"""
        # 确保qubit1 < qubit2
        if qubit1 > qubit2:
            qubit1, qubit2 = qubit2, qubit1
            # 需要交换门的qubit顺序
            swap = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
            gate = swap @ gate @ swap
        
        # 构建算子
        dim = 2 ** self.n_qubits
        result = np.zeros((dim, dim), dtype=np.complex128)
        
        for i in range(dim):
            for j in range(dim):
                # 提取qubit1和qubit2的位
                i1 = (i >> (self.n_qubits - 1 - qubit1)) & 1
                i2 = (i >> (self.n_qubits - 1 - qubit2)) & 1
                j1 = (j >> (self.n_qubits - 1 - qubit1)) & 1
                j2 = (j >> (self.n_qubits - 1 - qubit2)) & 1
                
                # 其他位必须相同
                other_i = i & ~((1 << (self.n_qubits - 1 - qubit1)) | 
                               (1 << (self.n_qubits - 1 - qubit2)))
                other_j = j & ~((1 << (self.n_qubits - 1 - qubit1)) | 
                               (1 << (self.n_qubits - 1 - qubit2)))
                
                if other_i == other_j:
                    gate_i = i1 * 2 + i2
                    gate_j = j1 * 2 + j2
                    result[i, j] = gate[gate_i, gate_j]
        
        return result
    
    def measure(self, shots: int = 1024) -> dict:
        """
        测量量子态
        
        Args:
            shots: 测量次数
            
        Returns:
            测量结果统计
        """
        probabilities = self.state.get_probabilities()
        
        # 应用读出错误
        if self.noise_model:
            probabilities = self.noise_model.apply_readout_error(probabilities)
        
        # 采样
        outcomes = np.random.choice(self.state.dim, size=shots, p=probabilities)
        
        # 统计结果
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def get_statevector(self) -> np.ndarray:
        """获取当前状态向量"""
        return self.state.state_vector.copy()
    
    def get_unitary(self) -> np.ndarray:
        """获取整个电路的酉矩阵"""
        dim = 2 ** self.n_qubits
        unitary = np.eye(dim, dtype=np.complex128)
        
        for gate_info in self.circuit_history:
            gate_name = gate_info['gate']
            qubits = gate_info['qubits']
            params = gate_info['params']
            
            if gate_name == 'custom':
                continue
            
            gate_matrix = self._get_gate_matrix(gate_name, params)
            full_operator = self._build_full_operator(gate_matrix, qubits)
            unitary = full_operator @ unitary
        
        return unitary
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """
        计算可观测量的期望值
        
        Args:
            observable: 可观测量矩阵
            
        Returns:
            期望值
        """
        dm = self.state.get_density_matrix()
        return np.real(np.trace(observable @ dm))
    
    def fidelity(self, target_state: np.ndarray) -> float:
        """计算与目标态的保真度"""
        return self.state.fidelity(target_state)


def create_simulator(n_qubits: int, 
                    with_noise: bool = False,
                    noise_params: Optional[dict] = None) -> QuantumSimulator:
    """
    创建量子模拟器的工厂函数
    
    Args:
        n_qubits: 量子比特数量
        with_noise: 是否添加噪声
        noise_params: 噪声参数
        
    Returns:
        QuantumSimulator实例
    """
    noise_model = None
    if with_noise:
        default_params = {
            'depolarizing_rate': 0.001,
            'amplitude_damping_rate': 0.001,
            'phase_damping_rate': 0.001,
            'readout_error': 0.01
        }
        if noise_params:
            default_params.update(noise_params)
        noise_model = NoiseModel(**default_params)
    
    return QuantumSimulator(n_qubits, noise_model)


if __name__ == "__main__":
    # 测试示例
    print("=== 量子模拟器测试 ===\n")
    
    # 创建2量子比特模拟器
    sim = create_simulator(2)
    
    # 创建Bell态
    sim.apply_gate('H', 0)
    sim.apply_gate('CNOT', [0, 1])
    
    print("Bell态:")
    print(f"状态向量: {sim.get_statevector()}")
    print(f"测量结果: {sim.measure(1000)}")
    
    # 测试保真度
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    print(f"与理想Bell态的保真度: {sim.fidelity(bell_state):.6f}")
    
    # 测试带噪声的模拟
    print("\n=== 带噪声的模拟 ===")
    noisy_sim = create_simulator(2, with_noise=True, 
                                 noise_params={'depolarizing_rate': 0.05})
    noisy_sim.apply_gate('H', 0)
    noisy_sim.apply_gate('CNOT', [0, 1])
    print(f"带噪声的保真度: {noisy_sim.fidelity(bell_state):.6f}")
    print(f"带噪声的测量结果: {noisy_sim.measure(1000)}")
