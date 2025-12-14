"""
真机连接示例
==========

演示如何连接IBM量子计算机并运行强化学习控制
"""

import numpy as np
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_quantum_control.main import QuantumControlSystem, create_system
from rl_quantum_control.quantum_backend.backend_manager import BackendManager, BackendType
from rl_quantum_control.quantum_backend.simulator import QuantumSimulator
from rl_quantum_control.quantum_backend.real_device import IBMQuantumDevice


def example_backend_switching():
    """示例：模拟器和真机之间切换"""
    print("=" * 60)
    print("示例1：后端切换演示")
    print("=" * 60)
    
    # 创建后端管理器
    manager = BackendManager(n_qubits=2)
    
    # 默认使用模拟器
    print(f"当前后端: {manager.backend_type.value}")
    print(f"量子比特数: {manager.n_qubits}")
    
    # 执行量子操作
    result = manager.execute_circuit([
        ('H', 0),
        ('CNOT', 0, 1)
    ])
    print(f"模拟器测量结果分布: {result['measurement_results']}")
    
    # 切换到带噪声的模拟器
    print("\n切换到带噪声模拟器...")
    manager.set_backend(
        BackendType.SIMULATOR,
        with_noise=True,
        noise_params={
            'depolarizing_rate': 0.01,
            'readout_error': 0.02
        }
    )
    
    result_noisy = manager.execute_circuit([
        ('H', 0),
        ('CNOT', 0, 1)
    ])
    print(f"带噪声测量结果: {result_noisy['measurement_results']}")
    
    # 获取后端信息
    info = manager.get_backend_info()
    print(f"\n后端信息: {info}")
    
    return manager


def example_prepare_for_real_device():
    """示例：准备真机连接"""
    print("\n" + "=" * 60)
    print("示例2：准备真机连接")
    print("=" * 60)
    
    print("""
要连接IBM量子计算机，您需要：
1. 在 https://quantum.ibm.com 注册账号
2. 获取API Token
3. 选择可用的量子后端

支持的后端包括：
- ibm_brisbane (127量子比特)
- ibm_osaka (127量子比特)  
- ibm_kyoto (127量子比特)
- ibmq_qasm_simulator (模拟器)
""")
    
    # 模拟真机连接流程
    print("\n模拟连接流程...")
    
    system = QuantumControlSystem(
        backend='simulator',
        n_qubits=2,
        with_noise=True
    )
    
    # 在模拟器上训练
    system.setup_task(
        task='gate_synthesis',
        target='H',
        action_type='discrete',
        max_steps=10
    )
    system.setup_agent('dqn')
    
    print("在模拟器上训练策略...")
    history = system.train(n_episodes=50, verbose=True)
    
    # 展示如何切换到真机
    print("\n" + "-" * 40)
    print("如何切换到真机：")
    print("-" * 40)
    print("""
# 方法1：使用系统方法
system.connect_real_device(
    api_token='your_ibm_quantum_api_token',
    backend_name='ibm_brisbane'
)

# 方法2：直接创建真机后端
from rl_quantum_control.quantum_backend.real_device import IBMQuantumDevice
device = IBMQuantumDevice(
    api_token='your_token',
    backend_name='ibm_brisbane',
    n_qubits=2
)

# 方法3：通过BackendManager
manager = BackendManager(n_qubits=2)
manager.set_backend(
    BackendType.IBM_QUANTUM,
    api_token='your_token',
    backend_name='ibm_brisbane'
)
""")
    
    return system


def example_hybrid_training():
    """示例：混合训练（模拟器+真机）"""
    print("\n" + "=" * 60)
    print("示例3：混合训练策略")
    print("=" * 60)
    
    print("""
混合训练策略：
1. 在模拟器上进行大量预训练
2. 在真机上进行微调（验证）
3. 周期性返回模拟器继续训练
""")
    
    # 创建系统
    system = QuantumControlSystem(
        backend='simulator',
        n_qubits=1,
        with_noise=False
    )
    
    # 阶段1：理想环境预训练
    print("\n阶段1：理想环境预训练")
    print("-" * 40)
    
    system.setup_task(
        task='state_preparation',
        target='plus',
        action_type='discrete',
        max_steps=10
    )
    system.setup_agent('ppo')
    
    history1 = system.train(n_episodes=50, verbose=True)
    eval1 = system.evaluate(n_episodes=10)
    print(f"理想环境保真度: {eval1['mean_fidelity']:.4f}")
    
    # 阶段2：噪声环境适应
    print("\n阶段2：噪声环境适应")
    print("-" * 40)
    
    system.use_simulator(
        with_noise=True,
        noise_params={
            'depolarizing_rate': 0.005,
            'readout_error': 0.01
        }
    )
    
    # 继续训练（在噪声环境）
    history2 = system.train(n_episodes=50, verbose=True)
    eval2 = system.evaluate(n_episodes=10)
    print(f"噪声环境保真度: {eval2['mean_fidelity']:.4f}")
    
    # 阶段3：准备真机测试
    print("\n阶段3：真机测试（模拟）")
    print("-" * 40)
    
    # 使用更高噪声模拟真机
    system.use_simulator(
        with_noise=True,
        noise_params={
            'depolarizing_rate': 0.01,
            'amplitude_damping_rate': 0.005,
            'readout_error': 0.02
        }
    )
    
    eval3 = system.evaluate(n_episodes=10)
    print(f"高噪声环境保真度: {eval3['mean_fidelity']:.4f}")
    
    # 比较
    print("\n训练效果比较:")
    print(f"  理想 -> 噪声: {eval1['mean_fidelity']:.4f} -> {eval2['mean_fidelity']:.4f}")
    print(f"  噪声 -> 高噪声: {eval2['mean_fidelity']:.4f} -> {eval3['mean_fidelity']:.4f}")
    
    return system, history1, history2


def example_noise_calibration():
    """示例：噪声校准"""
    print("\n" + "=" * 60)
    print("示例4：噪声模型校准")
    print("=" * 60)
    
    print("""
噪声校准流程：
1. 在真机上运行基准测试
2. 从测量结果估计噪声参数
3. 在模拟器中使用校准后的噪声模型
""")
    
    # 模拟噪声校准过程
    def run_benchmark_circuits(backend):
        """运行基准电路"""
        # T1测量（振幅衰减）
        # T2测量（相位衰减）
        # 门错误率
        pass
    
    def estimate_noise_params(results):
        """从结果估计噪声参数"""
        # 这里使用模拟值
        return {
            'depolarizing_rate': 0.008,
            'amplitude_damping_rate': 0.003,
            'phase_damping_rate': 0.002,
            'readout_error': 0.015
        }
    
    # 模拟校准
    print("\n运行噪声校准...")
    estimated_params = estimate_noise_params(None)
    print(f"估计的噪声参数: {estimated_params}")
    
    # 使用校准后的参数
    system = QuantumControlSystem(
        backend='simulator',
        n_qubits=1,
        with_noise=True,
        noise_params=estimated_params
    )
    
    system.setup_task('state_preparation', target='plus')
    system.setup_agent('sac')
    
    # 训练
    print("\n使用校准噪声模型训练...")
    history = system.train(n_episodes=80, verbose=True)
    
    eval_result = system.evaluate(n_episodes=10)
    print(f"\n校准后训练保真度: {eval_result['mean_fidelity']:.4f}")
    
    return system, estimated_params


def example_real_device_api():
    """示例：真机API使用演示"""
    print("\n" + "=" * 60)
    print("示例5：真机API使用")
    print("=" * 60)
    
    print("""
IBMQuantumDevice类提供以下功能：

1. 连接管理:
   device = IBMQuantumDevice(api_token, backend_name, n_qubits)
   device.connect()
   device.disconnect()

2. 电路执行:
   result = device.execute_circuit(operations, shots=1024)
   # operations: [('H', 0), ('CNOT', 0, 1), ...]

3. 作业管理:
   job_id = device.submit_job(circuit)
   status = device.get_job_status(job_id)
   result = device.get_job_result(job_id)

4. 设备信息:
   info = device.get_backend_info()
   # 返回: 量子比特数、连接性、错误率等

5. 校准数据:
   calibration = device.get_calibration_data()
   # 返回: T1, T2, 门保真度等
""")
    
    # 演示代码（不实际连接）
    demo_code = '''
# 实际使用示例
from rl_quantum_control.quantum_backend.real_device import IBMQuantumDevice

# 创建设备连接
device = IBMQuantumDevice(
    api_token='YOUR_IBM_QUANTUM_API_TOKEN',
    backend_name='ibm_brisbane',
    n_qubits=5
)

# 连接
device.connect()

# 执行简单电路
result = device.execute_circuit([
    ('H', 0),
    ('CNOT', 0, 1)
], shots=1024)

print("测量结果:", result['measurement_results'])
print("执行时间:", result['execution_time'])

# 获取设备信息
info = device.get_backend_info()
print("设备信息:", info)

# 断开连接
device.disconnect()
'''
    
    print("\n示例代码:")
    print(demo_code)
    
    return demo_code


def example_production_workflow():
    """示例：生产环境工作流"""
    print("\n" + "=" * 60)
    print("示例6：生产环境工作流")
    print("=" * 60)
    
    workflow = """
生产环境量子控制工作流：

┌─────────────────────────────────────────────────────────────┐
│                      开发阶段                               │
├─────────────────────────────────────────────────────────────┤
│  1. 理想模拟器训练                                          │
│     └─ 验证算法正确性                                       │
│  2. 噪声模拟器训练                                          │
│     └─ 提高鲁棒性                                           │
│  3. 超参数调优                                              │
│     └─ 网格搜索/贝叶斯优化                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      验证阶段                               │
├─────────────────────────────────────────────────────────────┤
│  4. 真机噪声校准                                            │
│     └─ 获取当前设备噪声参数                                 │
│  5. 使用校准参数微调                                        │
│     └─ 适应特定设备特性                                     │
│  6. 小规模真机测试                                          │
│     └─ 验证迁移效果                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      部署阶段                               │
├─────────────────────────────────────────────────────────────┤
│  7. 完整真机运行                                            │
│     └─ 收集性能数据                                         │
│  8. 持续监控                                                │
│     └─ 检测设备漂移                                         │
│  9. 在线适应                                                │
│     └─ 根据反馈调整策略                                     │
└─────────────────────────────────────────────────────────────┘
"""
    
    print(workflow)
    
    # 演示工作流
    print("\n执行简化工作流演示...")
    
    # 步骤1-2
    print("\n步骤1-2: 模拟器训练")
    system = create_system(
        task='gate_synthesis',
        target='H',
        agent='dqn',
        n_qubits=1,
        max_steps=10
    )
    
    system.train(n_episodes=30, verbose=True)
    
    # 步骤3
    print("\n步骤3: 评估")
    eval_result = system.evaluate(n_episodes=5)
    print(f"模拟器保真度: {eval_result['mean_fidelity']:.4f}")
    
    # 步骤4-5
    print("\n步骤4-5: 噪声适应")
    system.use_simulator(with_noise=True)
    system.train(n_episodes=30, verbose=True)
    
    # 步骤6
    print("\n步骤6: 最终评估")
    final_eval = system.evaluate(n_episodes=10)
    print(f"最终保真度: {final_eval['mean_fidelity']:.4f}")
    print(f"成功率: {final_eval['success_rate']:.2%}")
    
    return system


def check_qiskit_installation():
    """检查Qiskit安装状态"""
    print("\n" + "=" * 60)
    print("检查依赖安装")
    print("=" * 60)
    
    dependencies = [
        ('qiskit', 'Qiskit'),
        ('qiskit_aer', 'Qiskit Aer'),
        ('qiskit_ibm_runtime', 'Qiskit IBM Runtime'),
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('gymnasium', 'Gymnasium'),
    ]
    
    print("\n依赖状态:")
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name}: 已安装")
        except ImportError:
            print(f"  ✗ {name}: 未安装")
    
    print("\n安装命令:")
    print("  pip install qiskit qiskit-aer qiskit-ibm-runtime")


if __name__ == "__main__":
    print("强化学习量子控制 - 真机连接示例\n")
    
    try:
        # 检查依赖
        check_qiskit_installation()
        
        # 示例1：后端切换
        manager = example_backend_switching()
        
        # 示例2：准备真机
        system2 = example_prepare_for_real_device()
        
        # 示例3：混合训练
        system3, h1, h2 = example_hybrid_training()
        
        # 示例4：噪声校准
        system4, params = example_noise_calibration()
        
        # 示例5：API使用
        code = example_real_device_api()
        
        # 示例6：生产工作流
        system6 = example_production_workflow()
        
        print("\n" + "=" * 60)
        print("所有真机示例完成!")
        print("=" * 60)
        print("""
下一步：
1. 注册IBM Quantum账号
2. 获取API Token
3. 修改示例中的token参数
4. 运行真机测试
""")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
