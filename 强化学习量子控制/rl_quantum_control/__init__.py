"""
强化学习量子控制系统 (RL-QCS)
===========================

一个集成强化学习与量子控制的软件系统，支持模拟器和真机两种模式。
"""

__version__ = "1.0.0"
__author__ = "RL-QCS Team"

from .quantum_backend.backend_manager import BackendManager
from .quantum_backend.simulator import QuantumSimulator
from .quantum_backend.real_device import RealQuantumDevice
from .environments.base_env import QuantumControlEnv
from .environments.gate_synthesis import GateSynthesisEnv
from .environments.state_preparation import StatePreparationEnv
from .rl_agents.dqn_agent import DQNAgent
from .rl_agents.ppo_agent import PPOAgent
from .main import QuantumControlSystem

__all__ = [
    'QuantumControlSystem',
    'BackendManager',
    'QuantumSimulator',
    'RealQuantumDevice',
    'QuantumControlEnv',
    'GateSynthesisEnv',
    'StatePreparationEnv',
    'DQNAgent',
    'PPOAgent',
]
