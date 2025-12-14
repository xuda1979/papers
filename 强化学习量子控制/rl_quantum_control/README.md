# 强化学习量子控制系统 (RL-QCS)

## 概述

本系统是一个基于强化学习的量子控制软件，集成了：
- **强化学习模块**：实现多种RL算法（DQN、PPO、SAC等）用于量子控制优化
- **量子模拟器**：基于NumPy/Qiskit的量子计算模拟
- **真机接口**：支持无缝切换到IBM Quantum等真实量子计算机

## 系统架构

```
rl_quantum_control/
├── rl_quantum_control/       # 主模块
│   ├── __init__.py           # 模块初始化
│   ├── main.py               # 主程序入口
│   ├── cli.py                # 命令行接口
│   │
│   ├── quantum_backend/      # 量子后端（模拟器+真机接口）
│   │   ├── __init__.py
│   │   ├── simulator.py      # 量子模拟器
│   │   ├── real_device.py    # IBM Quantum真机接口
│   │   └── backend_manager.py # 后端管理器
│   │
│   ├── rl_agents/            # 强化学习代理
│   │   ├── __init__.py
│   │   ├── base_agent.py     # 基础代理类
│   │   ├── dqn_agent.py      # DQN算法
│   │   ├── ppo_agent.py      # PPO算法
│   │   └── sac_agent.py      # SAC算法
│   │
│   ├── environments/         # 量子控制环境
│   │   ├── __init__.py
│   │   ├── base_env.py       # 基础环境
│   │   ├── gate_synthesis.py # 量子门合成环境
│   │   ├── state_preparation.py # 量子态制备环境
│   │   └── pulse_optimization.py # 脉冲优化环境
│   │
│   ├── utils/                # 工具函数
│   │   ├── __init__.py
│   │   ├── quantum_utils.py  # 量子计算工具
│   │   ├── visualization.py  # 可视化工具
│   │   └── metrics.py        # 评估指标
│   │
│   ├── examples/             # 示例代码
│   │   ├── __init__.py
│   │   ├── gate_synthesis_example.py
│   │   ├── state_preparation_example.py
│   │   ├── pulse_optimization_example.py
│   │   └── real_device_example.py
│   │
│   └── tests/                # 测试代码
│       ├── __init__.py
│       └── test_system.py
│
├── README.md
└── requirements.txt
```

## 安装

```bash
# 克隆或下载项目后
cd rl_quantum_control

# 安装依赖
pip install -r requirements.txt

# （可选）安装Qiskit以连接IBM Quantum
pip install qiskit qiskit-aer qiskit-ibm-runtime
```

## 快速开始

### 方式一：使用Python API

```python
from rl_quantum_control.main import QuantumControlSystem, create_system

# 方法1：快速创建系统
system = create_system(
    task='gate_synthesis',       # 任务类型
    target='H',                  # 目标：Hadamard门
    agent='dqn',                 # RL算法
    n_qubits=1,                  # 量子比特数
    max_steps=20                 # 最大步数
)

# 训练
history = system.train(n_episodes=500, verbose=True)

# 评估
result = system.evaluate(n_episodes=100)
print(f"成功率: {result['success_rate']:.2%}")
print(f"平均保真度: {result['mean_fidelity']:.4f}")

# 方法2：详细配置
system = QuantumControlSystem(
    backend='simulator',
    n_qubits=2,
    with_noise=True,
    noise_params={'depolarizing_rate': 0.01}
)

system.setup_task(
    task='state_preparation',
    target='bell_00',
    action_type='discrete',
    max_steps=15,
    fidelity_threshold=0.99
)

system.setup_agent('ppo', config={
    'lr': 3e-4,
    'clip_epsilon': 0.2
})

system.train(n_episodes=1000)

# 切换到真机
system.connect_real_device(
    api_token='your_ibm_quantum_token',
    backend_name='ibm_brisbane'
)

# 在真机上运行
real_result = system.run_single()
```

### 方式二：使用命令行

```bash
# 训练门合成
python -m rl_quantum_control.cli train --task gate_synthesis --target H --agent dqn --episodes 500

# 训练态制备（带噪声）
python -m rl_quantum_control.cli train --task state_preparation --target bell_00 --agent ppo --noise

# 脉冲优化
python -m rl_quantum_control.cli train --task pulse_optimization --target X --agent sac

# 评估模型
python -m rl_quantum_control.cli evaluate --model model.pt --task gate_synthesis --target H

# 列出可用选项
python -m rl_quantum_control.cli list --type gates
python -m rl_quantum_control.cli list --type states
python -m rl_quantum_control.cli list --type agents

# 生成配置文件
python -m rl_quantum_control.cli config --template advanced --output config.json
```

## 主要功能

### 1. 量子后端

#### 量子模拟器
- 支持1-20量子比特模拟
- 完整的量子门集合（I, X, Y, Z, H, S, T, RX, RY, RZ, CNOT, CZ, SWAP等）
- 多种噪声模型：
  - 去极化噪声 (Depolarizing)
  - 振幅衰减 (Amplitude Damping)
  - 相位衰减 (Phase Damping)
  - 读出错误 (Readout Error)

```python
from rl_quantum_control.quantum_backend.simulator import QuantumSimulator

sim = QuantumSimulator(n_qubits=2, with_noise=True)
result = sim.execute_circuit([
    ('H', 0),
    ('CNOT', 0, 1)
], shots=1024)
```

#### IBM Quantum真机接口
```python
from rl_quantum_control.quantum_backend.real_device import IBMQuantumDevice

device = IBMQuantumDevice(
    api_token='your_token',
    backend_name='ibm_brisbane',
    n_qubits=5
)
device.connect()
result = device.execute_circuit([('H', 0), ('CNOT', 0, 1)])
```

#### 后端管理器
```python
from rl_quantum_control.quantum_backend.backend_manager import BackendManager, BackendType

manager = BackendManager(n_qubits=2)
manager.set_backend(BackendType.SIMULATOR, with_noise=True)
# 或
manager.set_backend(BackendType.IBM_QUANTUM, api_token='token')
```

### 2. 强化学习算法

| 算法 | 动作空间 | 适用场景 |
|------|---------|---------|
| DQN | 离散 | 门合成、态制备（离散门选择）|
| PPO | 离散/连续 | 通用，门合成、态制备、脉冲优化 |
| SAC | 连续 | 脉冲优化（连续控制参数）|

```python
from rl_quantum_control.rl_agents.dqn_agent import DQNAgent
from rl_quantum_control.rl_agents.ppo_agent import PPOAgent
from rl_quantum_control.rl_agents.sac_agent import SACAgent

# DQN
agent = DQNAgent(state_dim=10, action_dim=6, config={'lr': 1e-3})

# PPO (连续)
agent = PPOAgent(state_dim=10, action_dim=3, continuous=True)

# SAC
agent = SACAgent(state_dim=10, action_dim=3)
```

### 3. 控制任务

#### 量子门合成
学习将目标量子门分解为基本门序列。

```python
env = GateSynthesisEnv(
    target_gate='H',
    n_qubits=1,
    max_steps=20,
    gate_fidelity_threshold=0.99
)
```

支持的目标门：I, X, Y, Z, H, S, T, RX, RY, RZ, CNOT, CZ, SWAP

#### 量子态制备
学习从|0⟩态制备目标量子态。

```python
env = StatePreparationEnv(
    target_state='plus',  # |+⟩态
    n_qubits=1,
    fidelity_threshold=0.99
)
```

支持的目标态：
- 单比特：zero, one, plus, minus, plus_i, minus_i
- 双比特：bell_00, bell_01, bell_10, bell_11
- 多比特：ghz, w

#### 脉冲优化
学习优化控制脉冲波形实现量子门。

```python
env = PulseOptimizationEnv(
    target_gate='X',
    n_time_steps=20,
    max_amplitude=1.0
)
```

### 4. 工具和可视化

```python
from rl_quantum_control.utils.visualization import Visualizer
from rl_quantum_control.utils.metrics import MetricsCalculator

# 可视化
viz = Visualizer()
viz.plot_training_curves(history)
viz.plot_bloch_sphere(state_vector)
viz.plot_circuit(operations)

# 指标计算
metrics = MetricsCalculator()
fidelity = metrics.calculate_fidelity(state1, state2)
```

### 5. 课程学习 (Curriculum Learning)

支持通过逐步增加任务难度来加速训练收敛。

```python
# 设置课程学习策略
system.setup_curriculum(
    strategy_type='fidelity',  # 或 'steps', 'noise'
    start_threshold=0.8,
    end_threshold=0.99,
    update_freq=100
)
```

## 示例

运行示例代码：

```bash
# 门合成示例
python -m rl_quantum_control.examples.gate_synthesis_example

# 态制备示例
python -m rl_quantum_control.examples.state_preparation_example

# 脉冲优化示例
python -m rl_quantum_control.examples.pulse_optimization_example

# 真机连接示例
python -m rl_quantum_control.examples.real_device_example
```

## 测试

```bash
python -m rl_quantum_control.tests.test_system
```

## 配置选项

### 环境配置
| 参数 | 说明 | 默认值 |
|-----|------|-------|
| n_qubits | 量子比特数 | 1 |
| max_steps | 最大步数 | 20 |
| action_type | 动作类型 | 'discrete' |
| fidelity_threshold | 保真度阈值 | 0.99 |

### 代理配置
| 参数 | 说明 | 默认值 |
|-----|------|-------|
| lr | 学习率 | 1e-3 |
| gamma | 折扣因子 | 0.99 |
| buffer_size | 经验回放大小 | 10000 |

### 噪声配置
| 参数 | 说明 | 默认值 |
|-----|------|-------|
| depolarizing_rate | 去极化率 | 0.01 |
| amplitude_damping_rate | 振幅衰减率 | 0.005 |
| phase_damping_rate | 相位衰减率 | 0.002 |
| readout_error | 读出错误率 | 0.01 |

## 参考文献

1. Reinforcement Learning for Quantum Control (量子控制中的强化学习)
2. Deep Reinforcement Learning for Quantum Gate Synthesis
3. Model-Free Quantum Control with Reinforcement Learning

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
