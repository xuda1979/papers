"""
真实量子设备接口
===============

提供与真实量子计算机的接口，支持：
- IBM Quantum
- 自定义硬件接口
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
import warnings


class RealQuantumDevice(ABC):
    """真实量子设备的抽象基类"""
    
    def __init__(self, n_qubits: int, name: str = "generic_device"):
        """
        初始化真实量子设备
        
        Args:
            n_qubits: 量子比特数量
            name: 设备名称
        """
        self.n_qubits = n_qubits
        self.name = name
        self.is_connected = False
        self.circuit_history = []
    
    @abstractmethod
    def connect(self, **kwargs):
        """连接到设备"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开设备连接"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置设备状态"""
        pass
    
    @abstractmethod
    def apply_gate(self, gate: str, qubits: Union[int, List[int]], 
                   params: Optional[List[float]] = None):
        """应用量子门"""
        pass
    
    @abstractmethod
    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """测量"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        pass
    
    def get_statevector(self) -> np.ndarray:
        """
        获取状态向量（真机通常不支持）
        """
        raise NotImplementedError("真实量子设备不支持直接获取状态向量")
    
    def get_unitary(self) -> np.ndarray:
        """
        获取酉矩阵（真机通常不支持）
        """
        raise NotImplementedError("真实量子设备不支持直接获取酉矩阵")


class IBMQuantumDevice(RealQuantumDevice):
    """IBM Quantum设备接口"""
    
    def __init__(self, n_qubits: int = 5, backend_name: str = "ibm_brisbane"):
        """
        初始化IBM Quantum设备
        
        Args:
            n_qubits: 使用的量子比特数量
            backend_name: IBM后端名称
        """
        super().__init__(n_qubits, backend_name)
        self.backend_name = backend_name
        self.api_token = None
        self.service = None
        self.backend = None
        self.circuit = None
        
    def connect(self, api_token: str = None, channel: str = "ibm_quantum"):
        """
        连接到IBM Quantum
        
        Args:
            api_token: IBM Quantum API令牌
            channel: 服务通道
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            if api_token:
                self.api_token = api_token
                # 保存账户
                QiskitRuntimeService.save_account(
                    channel=channel,
                    token=api_token,
                    overwrite=True
                )
            
            # 连接服务
            self.service = QiskitRuntimeService(channel=channel)
            self.backend = self.service.backend(self.backend_name)
            self.is_connected = True
            
            # 初始化电路
            self._init_circuit()
            
            print(f"成功连接到 {self.backend_name}")
            
        except ImportError:
            raise ImportError("请安装 qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
        except Exception as e:
            self.is_connected = False
            raise ConnectionError(f"连接IBM Quantum失败: {e}")
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        self.service = None
        self.backend = None
        self.circuit = None
        print(f"已断开与 {self.backend_name} 的连接")
    
    def reset(self):
        """重置电路"""
        self._init_circuit()
        self.circuit_history = []
    
    def _init_circuit(self):
        """初始化量子电路"""
        try:
            from qiskit import QuantumCircuit
            self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        except ImportError:
            raise ImportError("请安装 qiskit: pip install qiskit")
    
    def apply_gate(self, gate: str, qubits: Union[int, List[int]], 
                   params: Optional[List[float]] = None):
        """
        应用量子门
        
        Args:
            gate: 门名称
            qubits: 作用的量子比特
            params: 门参数
        """
        if not self.is_connected:
            raise RuntimeError("设备未连接")
        
        if isinstance(qubits, int):
            qubits = [qubits]
        
        gate = gate.lower()
        
        # 映射门操作
        if gate == 'h':
            self.circuit.h(qubits[0])
        elif gate == 'x':
            self.circuit.x(qubits[0])
        elif gate == 'y':
            self.circuit.y(qubits[0])
        elif gate == 'z':
            self.circuit.z(qubits[0])
        elif gate == 's':
            self.circuit.s(qubits[0])
        elif gate == 't':
            self.circuit.t(qubits[0])
        elif gate == 'rx':
            self.circuit.rx(params[0], qubits[0])
        elif gate == 'ry':
            self.circuit.ry(params[0], qubits[0])
        elif gate == 'rz':
            self.circuit.rz(params[0], qubits[0])
        elif gate == 'u3':
            self.circuit.u(params[0], params[1], params[2], qubits[0])
        elif gate in ['cnot', 'cx']:
            self.circuit.cx(qubits[0], qubits[1])
        elif gate == 'cz':
            self.circuit.cz(qubits[0], qubits[1])
        elif gate == 'swap':
            self.circuit.swap(qubits[0], qubits[1])
        elif gate == 'crx':
            self.circuit.crx(params[0], qubits[0], qubits[1])
        elif gate == 'cry':
            self.circuit.cry(params[0], qubits[0], qubits[1])
        elif gate == 'crz':
            self.circuit.crz(params[0], qubits[0], qubits[1])
        else:
            raise ValueError(f"不支持的门: {gate}")
        
        self.circuit_history.append({
            'gate': gate,
            'qubits': qubits,
            'params': params
        })
    
    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """
        执行测量
        
        Args:
            shots: 测量次数
            
        Returns:
            测量结果统计
        """
        if not self.is_connected:
            raise RuntimeError("设备未连接")
        
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            
            # 添加测量
            circuit = self.circuit.copy()
            circuit.measure_all()
            
            # 转译电路
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            transpiled = pm.run(circuit)
            
            # 执行
            sampler = Sampler(self.backend)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            
            # 获取计数
            counts = result[0].data.meas.get_counts()
            
            return counts
            
        except Exception as e:
            raise RuntimeError(f"测量执行失败: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        if not self.is_connected:
            return {"status": "disconnected", "name": self.backend_name}
        
        try:
            return {
                "name": self.backend_name,
                "status": "connected",
                "n_qubits": self.backend.num_qubits,
                "basis_gates": list(self.backend.basis_gates) if hasattr(self.backend, 'basis_gates') else [],
                "online": self.backend.status().operational if hasattr(self.backend, 'status') else True
            }
        except Exception as e:
            return {
                "name": self.backend_name,
                "status": "connected",
                "error": str(e)
            }
    
    def run_circuit(self, circuit, shots: int = 1024) -> Dict[str, int]:
        """
        直接运行Qiskit电路
        
        Args:
            circuit: Qiskit QuantumCircuit
            shots: 测量次数
            
        Returns:
            测量结果
        """
        if not self.is_connected:
            raise RuntimeError("设备未连接")
        
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
            transpiled = pm.run(circuit)
            
            sampler = Sampler(self.backend)
            job = sampler.run([transpiled], shots=shots)
            result = job.result()
            
            return result[0].data.meas.get_counts()
            
        except Exception as e:
            raise RuntimeError(f"电路执行失败: {e}")


class LocalSimulatorDevice(RealQuantumDevice):
    """
    本地Qiskit模拟器设备
    
    用于测试，模拟真机接口但使用本地模拟器
    """
    
    def __init__(self, n_qubits: int = 5, with_noise: bool = False):
        """
        初始化本地模拟器
        
        Args:
            n_qubits: 量子比特数量
            with_noise: 是否添加噪声
        """
        super().__init__(n_qubits, "local_simulator")
        self.with_noise = with_noise
        self.circuit = None
        self.backend = None
    
    def connect(self, **kwargs):
        """连接（初始化模拟器）"""
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            
            if self.with_noise:
                # 创建噪声模型
                noise_model = NoiseModel()
                error_1q = depolarizing_error(0.001, 1)
                error_2q = depolarizing_error(0.01, 2)
                noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
                noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
                self.backend = AerSimulator(noise_model=noise_model)
            else:
                self.backend = AerSimulator()
            
            self._init_circuit()
            self.is_connected = True
            print(f"本地模拟器已初始化 (噪声: {self.with_noise})")
            
        except ImportError:
            raise ImportError("请安装 qiskit-aer: pip install qiskit-aer")
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        self.backend = None
        self.circuit = None
    
    def reset(self):
        """重置电路"""
        self._init_circuit()
        self.circuit_history = []
    
    def _init_circuit(self):
        """初始化电路"""
        from qiskit import QuantumCircuit
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
    
    def apply_gate(self, gate: str, qubits: Union[int, List[int]], 
                   params: Optional[List[float]] = None):
        """应用量子门"""
        if isinstance(qubits, int):
            qubits = [qubits]
        
        gate = gate.lower()
        
        if gate == 'h':
            self.circuit.h(qubits[0])
        elif gate == 'x':
            self.circuit.x(qubits[0])
        elif gate == 'y':
            self.circuit.y(qubits[0])
        elif gate == 'z':
            self.circuit.z(qubits[0])
        elif gate == 's':
            self.circuit.s(qubits[0])
        elif gate == 't':
            self.circuit.t(qubits[0])
        elif gate == 'rx':
            self.circuit.rx(params[0], qubits[0])
        elif gate == 'ry':
            self.circuit.ry(params[0], qubits[0])
        elif gate == 'rz':
            self.circuit.rz(params[0], qubits[0])
        elif gate == 'u3':
            self.circuit.u(params[0], params[1], params[2], qubits[0])
        elif gate in ['cnot', 'cx']:
            self.circuit.cx(qubits[0], qubits[1])
        elif gate == 'cz':
            self.circuit.cz(qubits[0], qubits[1])
        elif gate == 'swap':
            self.circuit.swap(qubits[0], qubits[1])
        else:
            raise ValueError(f"不支持的门: {gate}")
        
        self.circuit_history.append({
            'gate': gate,
            'qubits': qubits,
            'params': params
        })
    
    def measure(self, shots: int = 1024) -> Dict[str, int]:
        """执行测量"""
        if not self.is_connected:
            raise RuntimeError("设备未连接")
        
        circuit = self.circuit.copy()
        circuit.measure_all()
        
        job = self.backend.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        return counts
    
    def get_statevector(self) -> np.ndarray:
        """获取状态向量"""
        from qiskit_aer import AerSimulator
        from qiskit import QuantumCircuit
        
        # 使用状态向量模拟器
        sv_backend = AerSimulator(method='statevector')
        
        circuit = self.circuit.copy()
        circuit.save_statevector()
        
        job = sv_backend.run(circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        return np.array(statevector)
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        return {
            "name": "local_simulator",
            "status": "connected" if self.is_connected else "disconnected",
            "n_qubits": self.n_qubits,
            "with_noise": self.with_noise,
            "type": "simulator"
        }


class CustomDevice(RealQuantumDevice):
    """
    自定义设备接口
    
    允许用户实现自己的量子硬件接口
    """
    
    def __init__(self, n_qubits: int, 
                 connect_fn=None,
                 disconnect_fn=None,
                 apply_gate_fn=None,
                 measure_fn=None):
        """
        初始化自定义设备
        
        Args:
            n_qubits: 量子比特数量
            connect_fn: 连接函数
            disconnect_fn: 断开函数
            apply_gate_fn: 应用门函数
            measure_fn: 测量函数
        """
        super().__init__(n_qubits, "custom_device")
        self._connect_fn = connect_fn
        self._disconnect_fn = disconnect_fn
        self._apply_gate_fn = apply_gate_fn
        self._measure_fn = measure_fn
    
    def connect(self, **kwargs):
        if self._connect_fn:
            self._connect_fn(**kwargs)
        self.is_connected = True
    
    def disconnect(self):
        if self._disconnect_fn:
            self._disconnect_fn()
        self.is_connected = False
    
    def reset(self):
        self.circuit_history = []
    
    def apply_gate(self, gate: str, qubits: Union[int, List[int]], 
                   params: Optional[List[float]] = None):
        if self._apply_gate_fn:
            self._apply_gate_fn(gate, qubits, params)
        self.circuit_history.append({
            'gate': gate,
            'qubits': qubits,
            'params': params
        })
    
    def measure(self, shots: int = 1024) -> Dict[str, int]:
        if self._measure_fn:
            return self._measure_fn(self.circuit_history, shots)
        raise NotImplementedError("未提供测量函数")
    
    def get_device_info(self) -> Dict[str, Any]:
        return {
            "name": "custom_device",
            "status": "connected" if self.is_connected else "disconnected",
            "n_qubits": self.n_qubits,
            "type": "custom"
        }


if __name__ == "__main__":
    # 测试本地模拟器设备
    print("=== 本地模拟器设备测试 ===\n")
    
    device = LocalSimulatorDevice(n_qubits=2)
    device.connect()
    
    # 创建Bell态
    device.apply_gate('h', 0)
    device.apply_gate('cnot', [0, 1])
    
    print(f"设备信息: {device.get_device_info()}")
    print(f"测量结果: {device.measure(1000)}")
    print(f"状态向量: {device.get_statevector()}")
    
    device.disconnect()
    
    print("\n=== 带噪声的模拟器测试 ===\n")
    
    noisy_device = LocalSimulatorDevice(n_qubits=2, with_noise=True)
    noisy_device.connect()
    
    noisy_device.apply_gate('h', 0)
    noisy_device.apply_gate('cnot', [0, 1])
    
    print(f"带噪声测量结果: {noisy_device.measure(1000)}")
    
    noisy_device.disconnect()
