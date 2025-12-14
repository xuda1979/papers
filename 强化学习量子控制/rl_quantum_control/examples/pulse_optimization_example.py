"""
脉冲优化示例
==========

演示如何使用强化学习进行量子控制脉冲优化
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_quantum_control.main import QuantumControlSystem, create_system
from rl_quantum_control.environments.pulse_optimization import (
    PulseOptimizationEnv,
    SingleQubitPulseEnv,
    TwoQubitPulseEnv,
    RobustPulseOptimizationEnv
)
from rl_quantum_control.rl_agents.sac_agent import SACAgent
from rl_quantum_control.rl_agents.ppo_agent import PPOAgent


def example_x_gate_pulse():
    """示例：优化X门的控制脉冲"""
    print("=" * 60)
    print("示例1：X门脉冲优化")
    print("=" * 60)
    
    # 使用连续动作空间的脉冲优化环境
    env = SingleQubitPulseEnv(
        target_gate='X',
        n_time_steps=10,
        max_amplitude=1.0,
        gate_time=np.pi
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim} (控制脉冲分量)")
    print(f"目标: 实现X门 (π旋转)")
    
    # 使用SAC算法（适合连续控制）
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            'lr_actor': 1e-3,
            'lr_critic': 1e-3,
            'gamma': 0.99,
            'alpha': 0.2
        }
    )
    
    # 训练
    print("\n开始脉冲优化训练...")
    best_fidelity = 0
    training_history = []
    
    for episode in range(200):
        metrics = agent.train_episode(env)
        
        # 评估当前脉冲
        state, _ = env.reset()
        pulse_sequence = []
        for _ in range(env.n_time_steps):
            action = agent.select_action(state, deterministic=True)
            pulse_sequence.append(action.copy())
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        fidelity = info.get('fidelity', 0)
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_pulse = np.array(pulse_sequence)
        
        training_history.append({
            'episode': episode,
            'reward': metrics['episode_reward'],
            'fidelity': fidelity
        })
        
        if (episode + 1) % 40 == 0:
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"fidelity={fidelity:.6f}, "
                  f"best={best_fidelity:.6f}")
    
    print(f"\n最佳门保真度: {best_fidelity:.6f}")
    
    return env, agent, best_pulse if 'best_pulse' in dir() else None


def example_hadamard_pulse():
    """示例：优化Hadamard门的控制脉冲"""
    print("\n" + "=" * 60)
    print("示例2：Hadamard门脉冲优化")
    print("=" * 60)
    
    env = SingleQubitPulseEnv(
        target_gate='H',
        n_time_steps=15,
        max_amplitude=1.0,
        gate_time=np.pi / 2
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"目标: 实现Hadamard门")
    
    # 使用PPO（连续版本）
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=True,
        config={
            'lr': 3e-4,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01
        }
    )
    
    print("\n开始训练...")
    fidelities = []
    
    for episode in range(250):
        metrics = agent.train_episode(env)
        
        if (episode + 1) % 50 == 0:
            # 评估
            eval_fidelities = []
            for _ in range(5):
                state, _ = env.reset()
                for _ in range(env.n_time_steps):
                    action = agent.select_action(state, deterministic=True)
                    state, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                eval_fidelities.append(info.get('fidelity', 0))
            
            mean_fidelity = np.mean(eval_fidelities)
            fidelities.append(mean_fidelity)
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"mean_fidelity={mean_fidelity:.6f}")
    
    return env, agent, fidelities


def example_cnot_pulse():
    """示例：优化CNOT门的控制脉冲"""
    print("\n" + "=" * 60)
    print("示例3：CNOT门脉冲优化（两量子比特）")
    print("=" * 60)
    
    env = TwoQubitPulseEnv(
        target_gate='CNOT',
        n_time_steps=20,
        max_amplitude=1.0
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim} (包含耦合控制)")
    print("目标: 实现CNOT门（两比特纠缠门）")
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            'lr_actor': 3e-4,
            'lr_critic': 3e-4,
            'buffer_size': 100000
        }
    )
    
    print("\n开始训练（CNOT较复杂，需要更多训练）...")
    
    for episode in range(300):
        metrics = agent.train_episode(env)
        
        if (episode + 1) % 60 == 0:
            # 评估
            state, _ = env.reset()
            for _ in range(env.n_time_steps):
                action = agent.select_action(state, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"fidelity={info.get('fidelity', 0):.6f}")
    
    return env, agent


def example_robust_pulse():
    """示例：鲁棒脉冲优化（对噪声不敏感）"""
    print("\n" + "=" * 60)
    print("示例4：鲁棒脉冲优化")
    print("=" * 60)
    
    # 创建考虑系统不确定性的环境
    env = RobustPulseOptimizationEnv(
        target_gate='X',
        n_time_steps=15,
        max_amplitude=1.0,
        parameter_uncertainty=0.05,  # 5% 参数不确定性
        n_samples=5  # 每次评估采样5个参数实例
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"目标: 找到对参数波动鲁棒的X门脉冲")
    print(f"参数不确定性: ±5%")
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={'lr_actor': 3e-4}
    )
    
    print("\n开始训练...")
    
    for episode in range(200):
        metrics = agent.train_episode(env)
        
        if (episode + 1) % 40 == 0:
            # 测试鲁棒性：在多个参数变化下评估
            test_fidelities = []
            for _ in range(10):
                state, _ = env.reset()
                for _ in range(env.n_time_steps):
                    action = agent.select_action(state, deterministic=True)
                    state, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                test_fidelities.append(info.get('fidelity', 0))
            
            mean_fid = np.mean(test_fidelities)
            std_fid = np.std(test_fidelities)
            print(f"Episode {episode + 1}: "
                  f"fidelity={mean_fid:.4f}±{std_fid:.4f}")
    
    return env, agent


def example_time_optimal():
    """示例：时间最优脉冲"""
    print("\n" + "=" * 60)
    print("示例5：时间最优脉冲设计")
    print("=" * 60)
    
    # 比较不同时间步数的性能
    results = []
    
    for n_steps in [5, 10, 15, 20]:
        print(f"\n测试 {n_steps} 个时间步...")
        
        env = SingleQubitPulseEnv(
            target_gate='H',
            n_time_steps=n_steps,
            max_amplitude=1.5  # 允许更高幅度
        )
        
        agent = SACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
        
        # 快速训练
        for episode in range(100):
            agent.train_episode(env)
        
        # 评估
        fidelities = []
        for _ in range(10):
            state, _ = env.reset()
            for _ in range(n_steps):
                action = agent.select_action(state, deterministic=True)
                state, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            fidelities.append(info.get('fidelity', 0))
        
        mean_fidelity = np.mean(fidelities)
        results.append({
            'n_steps': n_steps,
            'fidelity': mean_fidelity,
            'time': n_steps * env.dt  # 假设dt是单位时间
        })
        
        print(f"  {n_steps}步: 保真度={mean_fidelity:.4f}")
    
    print("\n时间-保真度权衡:")
    for r in results:
        print(f"  {r['n_steps']}步 ({r['time']:.3f}时间单位): "
              f"保真度={r['fidelity']:.4f}")
    
    return results


def visualize_pulse(env, agent, filename='pulse_visualization.png'):
    """可视化优化后的脉冲"""
    print("\n" + "=" * 60)
    print("脉冲可视化")
    print("=" * 60)
    
    state, _ = env.reset()
    pulses = []
    
    for t in range(env.n_time_steps):
        action = agent.select_action(state, deterministic=True)
        pulses.append(action.copy())
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    
    pulses = np.array(pulses)
    time_axis = np.arange(len(pulses)) * getattr(env, 'dt', 1.0)
    
    fig, axes = plt.subplots(pulses.shape[1], 1, figsize=(10, 3 * pulses.shape[1]))
    if pulses.shape[1] == 1:
        axes = [axes]
    
    labels = ['X控制', 'Y控制', 'Z控制', '耦合控制']
    
    for i, ax in enumerate(axes):
        ax.plot(time_axis, pulses[:, i], 'b-', linewidth=2)
        ax.fill_between(time_axis, 0, pulses[:, i], alpha=0.3)
        ax.set_xlabel('时间')
        ax.set_ylabel(f'{labels[i]}幅度' if i < len(labels) else f'控制{i+1}')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('优化后的控制脉冲')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"脉冲图保存到: {filename}")
    
    return pulses


def example_with_main_system():
    """使用主系统进行脉冲优化"""
    print("\n" + "=" * 60)
    print("示例6：使用主系统进行脉冲优化")
    print("=" * 60)
    
    system = create_system(
        task='pulse_optimization',
        target='X',
        agent='sac',
        n_qubits=1,
        max_steps=15,
        action_type='continuous'
    )
    
    # 训练
    print("\n训练中...")
    history = system.train(n_episodes=150, eval_freq=30, verbose=True)
    
    # 评估
    print("\n评估结果:")
    eval_result = system.evaluate(n_episodes=10, verbose=True)
    
    return system, history


if __name__ == "__main__":
    print("强化学习量子控制脉冲优化示例\n")
    
    try:
        # 示例1：X门脉冲
        env1, agent1, best_pulse = example_x_gate_pulse()
        
        # 示例2：Hadamard门脉冲
        env2, agent2, fidelities = example_hadamard_pulse()
        
        # 示例3：CNOT门脉冲
        env3, agent3 = example_cnot_pulse()
        
        # 示例4：鲁棒脉冲
        env4, agent4 = example_robust_pulse()
        
        # 示例5：时间最优
        time_results = example_time_optimal()
        
        # 示例6：使用主系统
        system, history = example_with_main_system()
        
        # 可视化最佳脉冲
        if agent1 is not None:
            visualize_pulse(env1, agent1, 'x_gate_pulse.png')
        
        print("\n所有脉冲优化示例完成!")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
