"""
强化学习量子控制系统 - 主程序
============================

提供统一的接口来进行量子控制任务的强化学习训练和评估
"""

import numpy as np
from typing import Dict, Optional, Union, List
import os
import json
import time

from .quantum_backend.backend_manager import BackendManager, BackendType
from .quantum_backend.simulator import QuantumGates
from .environments.gate_synthesis import GateSynthesisEnv, MultiQubitGateSynthesisEnv
from .environments.state_preparation import StatePreparationEnv, GHZPreparationEnv
from .environments.pulse_optimization import PulseOptimizationEnv
from .rl_agents.dqn_agent import DQNAgent, train_dqn
from .rl_agents.ppo_agent import PPOAgent, train_ppo
from .rl_agents.sac_agent import SACAgent, train_sac
from .utils.metrics import MetricsLogger, compute_success_rate
from .utils.curriculum import CurriculumStrategy, FidelityCurriculum, MaxStepsCurriculum, NoiseCurriculum


class QuantumControlSystem:
    """
    强化学习量子控制系统
    
    集成量子后端管理、强化学习训练和评估功能
    """
    
    # 支持的任务类型
    TASKS = {
        'gate_synthesis': GateSynthesisEnv,
        'multi_qubit_gate': MultiQubitGateSynthesisEnv,
        'state_preparation': StatePreparationEnv,
        'ghz_preparation': GHZPreparationEnv,
        'pulse_optimization': PulseOptimizationEnv,
    }
    
    # 支持的代理类型
    AGENTS = {
        'dqn': DQNAgent,
        'ppo': PPOAgent,
        'sac': SACAgent,
    }
    
    def __init__(self,
                 backend: str = 'simulator',
                 n_qubits: int = 1,
                 **backend_kwargs):
        """
        初始化量子控制系统
        
        Args:
            backend: 后端类型 ('simulator', 'ibm_quantum', 'local_qiskit')
            n_qubits: 量子比特数量
            **backend_kwargs: 后端参数
        """
        self.n_qubits = n_qubits
        self.backend_type = backend
        self.backend_kwargs = backend_kwargs
        
        # 初始化后端管理器
        self.backend_manager = BackendManager(n_qubits)
        self.backend_manager.set_backend(backend, **backend_kwargs)
        
        # 当前环境和代理
        self.env = None
        self.agent = None
        self.curriculum = None
        
        # 训练历史
        self.training_history = {}
        self.metrics_logger = MetricsLogger()
        
        print(f"量子控制系统初始化完成")
        print(f"  后端: {backend}")
        print(f"  量子比特数: {n_qubits}")
    
    def setup_task(self,
                   task: str = 'gate_synthesis',
                   target: Union[str, np.ndarray] = 'H',
                   action_type: str = 'discrete',
                   max_steps: int = 50,
                   **task_kwargs) -> 'QuantumControlSystem':
        """
        设置控制任务
        
        Args:
            task: 任务类型
            target: 目标（门名称、态名称或矩阵/向量）
            action_type: 动作类型 ('discrete' 或 'continuous')
            max_steps: 最大步数
            **task_kwargs: 任务特定参数
            
        Returns:
            self (支持链式调用)
        """
        if task not in self.TASKS:
            raise ValueError(f"未知任务类型: {task}. 支持: {list(self.TASKS.keys())}")
        
        EnvClass = self.TASKS[task]
        
        # 准备环境参数
        env_kwargs = {
            'n_qubits': self.n_qubits,
            'max_steps': max_steps,
            'backend_type': self.backend_type,
            **self.backend_kwargs,
            **task_kwargs
        }
        
        # 设置目标
        if task in ['gate_synthesis', 'multi_qubit_gate']:
            env_kwargs['target_gate'] = target
            env_kwargs['action_type'] = action_type
        elif task in ['state_preparation', 'ghz_preparation']:
            if task != 'ghz_preparation':
                env_kwargs['target_state'] = target
            env_kwargs['action_type'] = action_type
        elif task == 'pulse_optimization':
            if isinstance(target, str):
                env_kwargs['target_unitary'] = getattr(QuantumGates, target, QuantumGates.X)
            else:
                env_kwargs['target_unitary'] = target
        
        self.env = EnvClass(**env_kwargs)
        self.current_task = task
        self.current_target = target
        
        print(f"\n任务设置完成")
        print(f"  任务类型: {task}")
        print(f"  目标: {target}")
        print(f"  动作类型: {action_type}")
        print(f"  最大步数: {max_steps}")
        
        return self
    
    def setup_agent(self,
                    agent_type: str = 'dqn',
                    config: Optional[Dict] = None) -> 'QuantumControlSystem':
        """
        设置强化学习代理
        
        Args:
            agent_type: 代理类型 ('dqn', 'ppo', 'sac')
            config: 代理配置
            
        Returns:
            self
        """
        if self.env is None:
            raise RuntimeError("请先调用 setup_task() 设置任务")
        
        if agent_type not in self.AGENTS:
            raise ValueError(f"未知代理类型: {agent_type}. 支持: {list(self.AGENTS.keys())}")
        
        AgentClass = self.AGENTS[agent_type]
        
        # 获取环境空间信息
        state_dim = self.env.observation_space.shape[0]
        
        if hasattr(self.env.action_space, 'n'):
            # 离散动作空间
            action_dim = self.env.action_space.n
            continuous = False
        else:
            # 连续动作空间
            action_dim = self.env.action_space.shape[0]
            continuous = True
        
        # 创建代理
        default_config = config or {}
        
        if agent_type == 'ppo':
            self.agent = AgentClass(state_dim, action_dim, 
                                   continuous=continuous,
                                   config=default_config)
        else:
            self.agent = AgentClass(state_dim, action_dim, 
                                   config=default_config)
        
        self.current_agent_type = agent_type
        
        print(f"\n代理设置完成")
        print(f"  代理类型: {agent_type}")
        print(f"  状态维度: {state_dim}")
        print(f"  动作维度: {action_dim}")
        print(f"  动作类型: {'连续' if continuous else '离散'}")
        
        return self
    
    def setup_curriculum(self, strategy_type: str = 'fidelity', **kwargs) -> 'QuantumControlSystem':
        """
        设置课程学习策略
        
        Args:
            strategy_type: 策略类型 ('fidelity', 'steps', 'noise')
            **kwargs: 策略参数
        """
        if self.env is None:
            raise RuntimeError("请先调用 setup_task() 设置任务")
            
        if strategy_type == 'fidelity':
            self.curriculum = FidelityCurriculum(self.env, **kwargs)
        elif strategy_type == 'steps':
            self.curriculum = MaxStepsCurriculum(self.env, **kwargs)
        elif strategy_type == 'noise':
            self.curriculum = NoiseCurriculum(self.env, self.backend_manager, **kwargs)
        else:
            raise ValueError(f"未知课程策略: {strategy_type}")
            
        print(f"\n课程学习策略已设置: {strategy_type}")
        return self

    def train(self,
              n_episodes: int = 1000,
              eval_freq: int = 100,
              save_freq: int = 500,
              save_path: Optional[str] = None,
              verbose: bool = True) -> Dict:
        """
        训练代理
        
        Args:
            n_episodes: 训练episode数
            eval_freq: 评估频率
            save_freq: 保存频率
            save_path: 模型保存路径
            verbose: 是否打印日志
            
        Returns:
            训练历史
        """
        if self.env is None or self.agent is None:
            raise RuntimeError("请先设置任务和代理")
        
        print(f"\n开始训练...")
        print(f"  总episode数: {n_episodes}")
        print(f"  评估频率: 每{eval_freq}个episode")
        
        start_time = time.time()
        
        history = {
            'rewards': [],
            'lengths': [],
            'losses': [],
            'fidelities': [],
            'eval_rewards': [],
            'eval_fidelities': []
        }
        
        best_eval_reward = float('-inf')
        
        for episode in range(n_episodes):
            # 训练一个episode
            metrics = self.agent.train_episode(self.env)
            
            history['rewards'].append(metrics['episode_reward'])
            history['lengths'].append(metrics['episode_length'])
            history['losses'].append(metrics.get('loss', 0))
            
            # 获取保真度（如果可用）
            if hasattr(self.env, '_compute_fidelity'):
                fidelity = self.env._compute_fidelity()
            else:
                fidelity = metrics.get('fidelity', 0)
            history['fidelities'].append(fidelity)
            
            # 记录指标
            self.metrics_logger.log(
                episode,
                reward=metrics['episode_reward'],
                length=metrics['episode_length'],
                loss=metrics.get('loss', 0),
                fidelity=fidelity
            )
            
            # 打印进度
            if verbose and (episode + 1) % 10 == 0:
                mean_reward = np.mean(history['rewards'][-10:])
                mean_fidelity = np.mean(history['fidelities'][-10:])
                print(f"Episode {episode + 1}: "
                      f"reward={mean_reward:.2f}, "
                      f"fidelity={mean_fidelity:.4f}")
            
            # 评估
            if (episode + 1) % eval_freq == 0:
                eval_stats = self.evaluate(n_episodes=10, verbose=False)
                history['eval_rewards'].append(eval_stats['mean_reward'])
                history['eval_fidelities'].append(eval_stats.get('mean_fidelity', 0))
                
                if verbose:
                    print(f"  [评估] reward={eval_stats['mean_reward']:.2f}, "
                          f"success_rate={eval_stats['success_rate']:.2%}")
                
                # 课程学习更新
                if self.curriculum:
                    self.curriculum.update(eval_stats)

                # 保存最佳模型
                if eval_stats['mean_reward'] > best_eval_reward and save_path:
                    best_eval_reward = eval_stats['mean_reward']
                    self.save(os.path.join(save_path, 'best_model.pt'))
            
            # 定期保存
            if save_freq and (episode + 1) % save_freq == 0 and save_path:
                self.save(os.path.join(save_path, f'checkpoint_{episode+1}.pt'))
        
        training_time = time.time() - start_time
        
        history['training_time'] = training_time
        history['n_episodes'] = n_episodes
        
        self.training_history = history
        
        print(f"\n训练完成!")
        print(f"  总时间: {training_time:.1f}秒")
        print(f"  平均奖励: {np.mean(history['rewards'][-100:]):.2f}")
        print(f"  最终保真度: {history['fidelities'][-1]:.4f}")
        
        return history
    
    def evaluate(self,
                 n_episodes: int = 10,
                 deterministic: bool = True,
                 verbose: bool = True) -> Dict:
        """
        评估代理性能
        
        Args:
            n_episodes: 评估episode数
            deterministic: 是否使用确定性策略
            verbose: 是否打印日志
            
        Returns:
            评估统计
        """
        if self.env is None or self.agent is None:
            raise RuntimeError("请先设置任务和代理")
        
        results = []
        
        for ep in range(n_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.select_action(state, deterministic=deterministic)
                state, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # 获取最终结果
            if hasattr(self.env, 'get_synthesis_result'):
                result = self.env.get_synthesis_result()
            elif hasattr(self.env, 'get_preparation_result'):
                result = self.env.get_preparation_result()
            elif hasattr(self.env, 'get_optimization_result'):
                result = self.env.get_optimization_result()
            else:
                result = {'fidelity': info.get('fidelity', 0)}
            
            result['episode_reward'] = episode_reward
            result['episode_length'] = episode_length
            results.append(result)
        
        # 计算统计
        rewards = [r['episode_reward'] for r in results]
        fidelities = [r.get('achieved_fidelity', r.get('fidelity', 0)) for r in results]
        successes = [r.get('success', False) for r in results]
        
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_fidelity': np.mean(fidelities),
            'std_fidelity': np.std(fidelities),
            'success_rate': np.mean(successes),
            'n_episodes': n_episodes,
            'results': results
        }
        
        if verbose:
            print(f"\n评估结果 ({n_episodes} episodes):")
            print(f"  平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"  平均保真度: {stats['mean_fidelity']:.4f} ± {stats['std_fidelity']:.4f}")
            print(f"  成功率: {stats['success_rate']:.2%}")
        
        return stats
    
    def run_single(self, deterministic: bool = True, 
                   render: bool = False) -> Dict:
        """
        运行单次控制任务
        
        Args:
            deterministic: 是否使用确定性策略
            render: 是否渲染
            
        Returns:
            运行结果
        """
        if self.env is None or self.agent is None:
            raise RuntimeError("请先设置任务和代理")
        
        state, info = self.env.reset()
        total_reward = 0
        
        while True:
            if render:
                self.env.render()
            
            action = self.agent.select_action(state, deterministic=deterministic)
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        if render:
            self.env.render()
        
        # 获取结果
        if hasattr(self.env, 'get_synthesis_result'):
            result = self.env.get_synthesis_result()
        elif hasattr(self.env, 'get_preparation_result'):
            result = self.env.get_preparation_result()
        elif hasattr(self.env, 'get_optimization_result'):
            result = self.env.get_optimization_result()
        else:
            result = info
        
        result['total_reward'] = total_reward
        
        return result
    
    def switch_backend(self, backend: str, **kwargs):
        """
        切换量子后端
        
        Args:
            backend: 新的后端类型
            **kwargs: 后端参数
        """
        self.backend_type = backend
        self.backend_kwargs.update(kwargs)
        self.backend_manager.set_backend(backend, **kwargs)
        
        # 如果环境存在，也需要更新
        if self.env is not None:
            self.env.switch_backend(backend, **kwargs)
        
        print(f"已切换到后端: {backend}")
    
    def connect_real_device(self, api_token: str = None,
                           backend_name: str = 'ibm_brisbane'):
        """
        连接到真实量子设备
        
        Args:
            api_token: API令牌
            backend_name: 后端名称
        """
        self.switch_backend('ibm_quantum',
                           api_token=api_token,
                           backend_name=backend_name)
        print(f"已连接到真实设备: {backend_name}")
    
    def use_simulator(self, with_noise: bool = False,
                     noise_params: Optional[Dict] = None):
        """
        切换到模拟器
        
        Args:
            with_noise: 是否添加噪声
            noise_params: 噪声参数
        """
        self.switch_backend('simulator',
                           with_noise=with_noise,
                           noise_params=noise_params)
        print(f"已切换到模拟器 (噪声: {with_noise})")
    
    def save(self, path: str):
        """
        保存模型和状态
        
        Args:
            path: 保存路径
        """
        if self.agent is None:
            raise RuntimeError("没有代理可保存")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
        
        # 保存配置
        config_path = path.replace('.pt', '_config.json')
        config = {
            'task': self.current_task,
            'target': str(self.current_target),
            'agent_type': self.current_agent_type,
            'n_qubits': self.n_qubits,
            'backend': self.backend_type
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if self.agent is None:
            raise RuntimeError("请先设置代理")
        
        self.agent.load(path)
        print(f"模型已加载: {path}")
    
    def get_backend_info(self) -> Dict:
        """获取当前后端信息"""
        return self.backend_manager.get_backend_info()
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        return self.metrics_logger.summary()


def create_system(task: str = 'gate_synthesis',
                  target: str = 'H',
                  agent: str = 'dqn',
                  n_qubits: int = 1,
                  use_npu: bool = False,
                  **kwargs) -> QuantumControlSystem:
    """
    快速创建并配置量子控制系统
    
    Args:
        task: 任务类型
        target: 目标
        agent: 代理类型
        n_qubits: 量子比特数
        use_npu: 是否使用NPU
        **kwargs: 其他参数
        
    Returns:
        配置好的QuantumControlSystem
    """
    system = QuantumControlSystem(n_qubits=n_qubits)
    system.setup_task(task, target, **kwargs)
    
    agent_config = {}
    if use_npu:
        agent_config['device'] = 'npu'
        
    system.setup_agent(agent, config=agent_config)
    return system


if __name__ == "__main__":
    print("=" * 50)
    print("强化学习量子控制系统")
    print("=" * 50)
    
    # 示例1：量子门合成
    print("\n示例1：H门合成")
    system = create_system(
        task='gate_synthesis',
        target='H',
        agent='dqn',
        n_qubits=1,
        max_steps=20,
        action_type='discrete'
    )
    
    # 快速训练测试
    history = system.train(n_episodes=50, eval_freq=25, verbose=True)
    
    # 评估
    system.evaluate(n_episodes=5)
    
    # 示例2：态制备
    print("\n示例2：|+⟩态制备")
    system2 = create_system(
        task='state_preparation',
        target='plus',
        agent='ppo',
        n_qubits=1,
        max_steps=10,
        action_type='discrete'
    )
    
    history2 = system2.train(n_episodes=30, eval_freq=15, verbose=True)
    
    print("\n系统测试完成!")
