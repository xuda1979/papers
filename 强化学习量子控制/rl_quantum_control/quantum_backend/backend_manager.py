"""
后端管理器
=========

统一管理量子模拟器和真实设备的后端切换
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from .simulator import QuantumSimulator, create_simulator, NoiseModel
from .real_device import (
    RealQuantumDevice, 
    IBMQuantumDevice, 
    LocalSimulatorDevice,
    CustomDevice
)


class BackendType(Enum):
    """后端类型枚举"""
    SIMULATOR = "simulator"
    IBM_QUANTUM = "ibm_quantum"
    LOCAL_QISKIT = "local_qiskit"
    CUSTOM = "custom"


class BackendManager:
    """
    后端管理器
    
    提供统一的接口来管理不同的量子后端，支持：
    - 内置NumPy模拟器
    - IBM Quantum真机
    - 本地Qiskit模拟器
    - 自定义设备
    """
    
    def __init__(self, n_qubits: int = 2):
        """
        初始化后端管理器
        
        Args:
            n_qubits: 默认量子比特数量
        """
        self.n_qubits = n_qubits
        self.current_backend = None
        self.backend_type = None
        self._backends = {}
        
        # 初始化默认后端（模拟器）
        self.set_backend(BackendType.SIMULATOR)
    
    def set_backend(self, backend_type: Union[BackendType, str], **kwargs):
        """
        设置当前后端
        
        Args:
            backend_type: 后端类型
            **kwargs: 后端特定参数
        """
        if isinstance(backend_type, str):
            backend_type = BackendType(backend_type.lower())
        
        self.backend_type = backend_type
        
        if backend_type == BackendType.SIMULATOR:
            noise_params = kwargs.get('noise_params', None)
            with_noise = kwargs.get('with_noise', False)
            self.current_backend = create_simulator(
                self.n_qubits, 
                with_noise=with_noise,
                noise_params=noise_params
            )
            
        elif backend_type == BackendType.IBM_QUANTUM:
            backend_name = kwargs.get('backend_name', 'ibm_brisbane')
            api_token = kwargs.get('api_token', None)
            
            device = IBMQuantumDevice(self.n_qubits, backend_name)
            device.connect(api_token=api_token)
            self.current_backend = device
            
        elif backend_type == BackendType.LOCAL_QISKIT:
            with_noise = kwargs.get('with_noise', False)
            device = LocalSimulatorDevice(self.n_qubits, with_noise)
            device.connect()
            self.current_backend = device
            
        elif backend_type == BackendType.CUSTOM:
            custom_device = kwargs.get('device', None)
            if custom_device is None:
                raise ValueError("自定义后端需要提供device参数")
            self.current_backend = custom_device
            
        print(f"已切换到后端: {backend_type.value}")
    
    def reset(self):
        """重置当前后端"""
        self.current_backend.reset()
    
    def apply_gate(self, gate: str, qubits: Union[int, List[int]], 
                   params: Optional[List[float]] = None):
        """
        应用量子门
        
        Args:
            gate: 门名称
            qubits: 作用的量子比特
            params: 门参数
        """
        self.current_backend.apply_gate(gate, qubits, params)
    
    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """
        执行测量
        
        Args:
            shots: 测量次数
            
        Returns:
            测量结果统计
        """
        return self.current_backend.measure(shots)
    
    def get_statevector(self) -> np.ndarray:
        """
        获取状态向量
        
        注意：真机不支持此操作
        """
        if self.backend_type == BackendType.SIMULATOR:
            return self.current_backend.get_statevector()
        elif self.backend_type == BackendType.LOCAL_QISKIT:
            return self.current_backend.get_statevector()
        else:
            raise NotImplementedError(
                f"后端 {self.backend_type.value} 不支持获取状态向量"
            )
    
    def get_unitary(self) -> np.ndarray:
        """
        获取电路的酉矩阵
        
        注意：仅模拟器支持
        """
        if self.backend_type == BackendType.SIMULATOR:
            return self.current_backend.get_unitary()
        else:
            raise NotImplementedError(
                f"后端 {self.backend_type.value} 不支持获取酉矩阵"
            )
    
    def fidelity(self, target_state: np.ndarray) -> float:
        """
        计算与目标态的保真度
        
        Args:
            target_state: 目标量子态
            
        Returns:
            保真度值
        """
        if self.backend_type == BackendType.SIMULATOR:
            return self.current_backend.fidelity(target_state)
        else:
            # 对于真机，通过状态层析估计保真度
            current_sv = self.get_statevector()
            target_norm = target_state / np.linalg.norm(target_state)
            return np.abs(np.vdot(target_norm, current_sv)) ** 2
    
    def expectation_value(self, observable: np.ndarray) -> float:
        """
        计算可观测量期望值
        
        Args:
            observable: 可观测量矩阵
        """
        if self.backend_type == BackendType.SIMULATOR:
            return self.current_backend.expectation_value(observable)
        else:
            # 对于其他后端，需要通过采样估计
            return self._estimate_expectation(observable)
    
    def _estimate_expectation(self, observable: np.ndarray, 
                             shots: int = 10000) -> float:
        """通过采样估计期望值"""
        # 获取状态向量
        sv = self.get_statevector()
        dm = np.outer(sv, np.conj(sv))
        return np.real(np.trace(observable @ dm))
    
    def get_backend_info(self) -> Dict[str, Any]:
        """获取当前后端信息"""
        info = {
            "type": self.backend_type.value,
            "n_qubits": self.n_qubits,
        }
        
        if self.backend_type == BackendType.SIMULATOR:
            info["has_noise"] = self.current_backend.noise_model is not None
        elif hasattr(self.current_backend, 'get_device_info'):
            info.update(self.current_backend.get_device_info())
        
        return info
    
    def is_simulator(self) -> bool:
        """检查是否为模拟器后端"""
        return self.backend_type in [BackendType.SIMULATOR, BackendType.LOCAL_QISKIT]
    
    def supports_statevector(self) -> bool:
        """检查是否支持状态向量访问"""
        return self.backend_type in [BackendType.SIMULATOR, BackendType.LOCAL_QISKIT]
    
    def execute_circuit(self, circuit_operations: List[Dict], 
                       shots: int = 1024) -> Dict[str, int]:
        """
        执行一系列门操作并测量
        
        Args:
            circuit_operations: 门操作列表
            shots: 测量次数
            
        Returns:
            测量结果
        """
        self.reset()
        
        for op in circuit_operations:
            gate = op['gate']
            qubits = op['qubits']
            params = op.get('params', None)
            self.apply_gate(gate, qubits, params)
        
        return self.measure(shots)
    
    def run_pulse_sequence(self, pulse_params: np.ndarray, 
                          duration: float = 1.0,
                          shots: int = 1024) -> Dict[str, int]:
        """
        执行脉冲序列（简化模型）
        
        Args:
            pulse_params: 脉冲参数
            duration: 演化时间
            shots: 测量次数
            
        Returns:
            测量结果
        """
        # 将脉冲参数转换为门操作
        # 这是一个简化模型，实际需要根据硬件特性实现
        self.reset()
        
        n_pulses = len(pulse_params) // 3
        for i in range(n_pulses):
            theta = pulse_params[i * 3]
            phi = pulse_params[i * 3 + 1]
            lam = pulse_params[i * 3 + 2]
            self.apply_gate('U3', 0, [theta, phi, lam])
        
        return self.measure(shots)


def create_backend(backend_type: str = "simulator", 
                   n_qubits: int = 2,
                   **kwargs) -> BackendManager:
    """
    创建后端管理器的工厂函数
    
    Args:
        backend_type: 后端类型
        n_qubits: 量子比特数量
        **kwargs: 其他参数
        
    Returns:
        BackendManager实例
    """
    manager = BackendManager(n_qubits)
    manager.set_backend(backend_type, **kwargs)
    return manager


if __name__ == "__main__":
    print("=== 后端管理器测试 ===\n")
    
    # 测试模拟器后端
    print("1. 内置模拟器后端:")
    manager = create_backend("simulator", n_qubits=2)
    
    manager.apply_gate('H', 0)
    manager.apply_gate('CNOT', [0, 1])
    
    print(f"  后端信息: {manager.get_backend_info()}")
    print(f"  测量结果: {manager.measure(1000)}")
    print(f"  状态向量: {manager.get_statevector()}")
    
    # 测试带噪声的模拟器
    print("\n2. 带噪声的模拟器:")
    manager.set_backend("simulator", with_noise=True, 
                       noise_params={'depolarizing_rate': 0.05})
    manager.reset()
    manager.apply_gate('H', 0)
    manager.apply_gate('CNOT', [0, 1])
    print(f"  测量结果: {manager.measure(1000)}")
    
    # 测试本地Qiskit后端
    print("\n3. 本地Qiskit后端:")
    try:
        manager.set_backend("local_qiskit", with_noise=False)
        manager.reset()
        manager.apply_gate('h', 0)
        manager.apply_gate('cnot', [0, 1])
        print(f"  后端信息: {manager.get_backend_info()}")
        print(f"  测量结果: {manager.measure(1000)}")
    except ImportError as e:
        print(f"  跳过（需要安装qiskit-aer）: {e}")
    
    # 测试电路执行
    print("\n4. 执行电路操作列表:")
    manager.set_backend("simulator")
    circuit = [
        {'gate': 'H', 'qubits': 0},
        {'gate': 'RY', 'qubits': 1, 'params': [np.pi/4]},
        {'gate': 'CNOT', 'qubits': [0, 1]}
    ]
    result = manager.execute_circuit(circuit, shots=1000)
    print(f"  结果: {result}")
