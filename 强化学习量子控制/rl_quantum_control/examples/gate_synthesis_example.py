"""
量子门合成示例
=============

演示如何使用强化学习进行量子门合成
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_quantum_control.main import QuantumControlSystem, create_system
from rl_quantum_control.quantum_backend.simulator import QuantumGates


def example_h_gate_synthesis():
    """示例：使用DQN合成H门"""
    print("=" * 60)
    print("示例1：使用DQN合成Hadamard门")
    print("=" * 60)
    
    # 创建系统
    system = QuantumControlSystem(backend='simulator', n_qubits=1)
    
    # 设置门合成任务
    system.setup_task(
        task='gate_synthesis',
        target='H',
        action_type='discrete',
        max_steps=20,
        fidelity_threshold=0.99
    )
    
    # 设置DQN代理
    system.setup_agent('dqn', config={
        'lr': 1e-3,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'double_dqn': True,
        'dueling': True
    })
    
    # 训练
    print("\n开始训练...")
    history = system.train(
        n_episodes=200,
        eval_freq=50,
        verbose=True
    )
    
    # 评估
    print("\n最终评估:")
    eval_result = system.evaluate(n_episodes=10)
    
    # 运行单次并显示结果
    print("\n单次运行结果:")
    result = system.run_single(deterministic=True)
    print(f"  保真度: {result['achieved_fidelity']:.6f}")
    print(f"  电路深度: {result['circuit_depth']}")
    print(f"  成功: {result['success']}")
    
    return system, history


def example_x_gate_continuous():
    """示例：使用PPO和连续动作空间合成X门"""
    print("\n" + "=" * 60)
    print("示例2：使用PPO（连续动作）合成X门")
    print("=" * 60)
    
    system = create_system(
        task='gate_synthesis',
        target='X',
        agent='ppo',
        n_qubits=1,
        max_steps=10,
        action_type='continuous',
        fidelity_threshold=0.99
    )
    
    # 训练
    history = system.train(
        n_episodes=100,
        eval_freq=25,
        verbose=True
    )
    
    # 评估
    eval_result = system.evaluate(n_episodes=5)
    
    return system, history


def example_cnot_synthesis():
    """示例：双量子比特CNOT门合成"""
    print("\n" + "=" * 60)
    print("示例3：使用DQN合成CNOT门")
    print("=" * 60)
    
    from rl_quantum_control.environments.gate_synthesis import MultiQubitGateSynthesisEnv
    from rl_quantum_control.rl_agents.dqn_agent import DQNAgent
    
    # 创建环境
    env = MultiQubitGateSynthesisEnv(
        target_gate='CNOT',
        max_steps=30,
        action_type='discrete',
        fidelity_threshold=0.95
    )
    
    # 创建代理
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            'lr': 1e-3,
            'epsilon_decay': 0.99
        }
    )
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 训练
    print("\n开始训练...")
    best_fidelity = 0
    
    for episode in range(100):
        metrics = agent.train_episode(env)
        
        if (episode + 1) % 20 == 0:
            # 评估
            eval_stats = agent.evaluate(env, n_episodes=5)
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"eval_fidelity={eval_stats.get('mean_reward', 0):.2f}")
            
            if eval_stats.get('success_rate', 0) > 0:
                print(f"  成功率: {eval_stats['success_rate']:.2%}")
    
    return env, agent


def example_custom_gate():
    """示例：合成自定义量子门"""
    print("\n" + "=" * 60)
    print("示例4：合成自定义量子门 (√X)")
    print("=" * 60)
    
    # 定义√X门
    sqrt_x = np.array([
        [1+1j, 1-1j],
        [1-1j, 1+1j]
    ], dtype=np.complex128) / 2
    
    print("目标门 (√X):")
    print(sqrt_x)
    
    # 创建系统
    system = create_system(
        task='gate_synthesis',
        target=sqrt_x,
        agent='sac',
        n_qubits=1,
        max_steps=15,
        action_type='continuous',
        fidelity_threshold=0.98
    )
    
    # 训练
    history = system.train(
        n_episodes=150,
        eval_freq=30,
        verbose=True
    )
    
    # 显示结果
    result = system.run_single()
    print(f"\n合成结果:")
    print(f"  保真度: {result['achieved_fidelity']:.6f}")
    print(f"  合成的酉矩阵:")
    print(result['synthesized_unitary'])
    
    return system, history


def example_noisy_synthesis():
    """示例：在有噪声环境下进行门合成"""
    print("\n" + "=" * 60)
    print("示例5：有噪声环境下的门合成")
    print("=" * 60)
    
    # 创建带噪声的系统
    system = QuantumControlSystem(
        backend='simulator',
        n_qubits=1,
        with_noise=True,
        noise_params={
            'depolarizing_rate': 0.01,
            'readout_error': 0.02
        }
    )
    
    system.setup_task(
        task='gate_synthesis',
        target='H',
        action_type='discrete',
        max_steps=25,
        fidelity_threshold=0.95  # 降低阈值
    )
    
    system.setup_agent('dqn', config={
        'lr': 5e-4,
        'epsilon_decay': 0.99
    })
    
    # 训练
    history = system.train(
        n_episodes=150,
        eval_freq=30,
        verbose=True
    )
    
    # 在无噪声环境下评估
    print("\n切换到无噪声环境评估...")
    system.use_simulator(with_noise=False)
    eval_result = system.evaluate(n_episodes=10)
    
    return system, history


def compare_algorithms():
    """比较不同RL算法的性能"""
    print("\n" + "=" * 60)
    print("算法比较：DQN vs PPO vs SAC")
    print("=" * 60)
    
    results = {}
    
    for agent_type in ['dqn', 'ppo', 'sac']:
        print(f"\n训练 {agent_type.upper()}...")
        
        if agent_type in ['ppo', 'sac']:
            action_type = 'continuous'
        else:
            action_type = 'discrete'
        
        system = create_system(
            task='gate_synthesis',
            target='H',
            agent=agent_type,
            n_qubits=1,
            max_steps=20,
            action_type=action_type,
            fidelity_threshold=0.99
        )
        
        history = system.train(n_episodes=100, eval_freq=50, verbose=False)
        eval_result = system.evaluate(n_episodes=10, verbose=False)
        
        results[agent_type] = {
            'final_reward': np.mean(history['rewards'][-20:]),
            'success_rate': eval_result['success_rate'],
            'mean_fidelity': eval_result['mean_fidelity']
        }
        
        print(f"  {agent_type.upper()}: fidelity={eval_result['mean_fidelity']:.4f}, "
              f"success_rate={eval_result['success_rate']:.2%}")
    
    print("\n比较结果:")
    for agent, res in results.items():
        print(f"  {agent.upper()}: fidelity={res['mean_fidelity']:.4f}, "
              f"success={res['success_rate']:.2%}")
    
    return results


if __name__ == "__main__":
    print("强化学习量子门合成示例\n")
    
    # 运行示例
    try:
        # 示例1：基础H门合成
        system1, history1 = example_h_gate_synthesis()
        
        # 示例2：连续动作X门合成
        system2, history2 = example_x_gate_continuous()
        
        # 示例3：CNOT门合成
        env3, agent3 = example_cnot_synthesis()
        
        # 示例4：自定义门
        system4, history4 = example_custom_gate()
        
        # 示例5：有噪声环境
        system5, history5 = example_noisy_synthesis()
        
        # 算法比较
        comparison = compare_algorithms()
        
        print("\n所有示例运行完成!")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
