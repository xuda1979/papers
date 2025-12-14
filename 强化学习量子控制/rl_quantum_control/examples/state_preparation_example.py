"""
量子态制备示例
=============

演示如何使用强化学习进行量子态制备
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_quantum_control.main import QuantumControlSystem, create_system
from rl_quantum_control.environments.state_preparation import (
    StatePreparationEnv,
    GHZPreparationEnv,
    WStatePreparationEnv,
    RandomStatePreparationEnv
)
from rl_quantum_control.rl_agents.ppo_agent import PPOAgent


def example_plus_state():
    """示例：制备|+⟩态"""
    print("=" * 60)
    print("示例1：制备|+⟩态")
    print("=" * 60)
    
    system = create_system(
        task='state_preparation',
        target='plus',
        agent='dqn',
        n_qubits=1,
        max_steps=10,
        action_type='discrete'
    )
    
    # 训练
    history = system.train(n_episodes=100, eval_freq=25, verbose=True)
    
    # 评估
    print("\n评估结果:")
    result = system.run_single()
    print(f"  目标态: |+⟩ = (|0⟩ + |1⟩)/√2")
    print(f"  制备态: {result['prepared_state']}")
    print(f"  保真度: {result['achieved_fidelity']:.6f}")
    
    return system, history


def example_bell_state():
    """示例：制备Bell态"""
    print("\n" + "=" * 60)
    print("示例2：制备Bell态 (|00⟩ + |11⟩)/√2")
    print("=" * 60)
    
    system = create_system(
        task='state_preparation',
        target='bell_00',
        agent='ppo',
        n_qubits=2,
        max_steps=15,
        action_type='discrete'
    )
    
    # 训练
    history = system.train(n_episodes=150, eval_freq=30, verbose=True)
    
    # 评估
    result = system.run_single()
    print(f"\n制备结果:")
    print(f"  目标态: (|00⟩ + |11⟩)/√2")
    print(f"  保真度: {result['achieved_fidelity']:.6f}")
    print(f"  电路深度: {result['circuit_depth']}")
    
    return system, history


def example_ghz_state():
    """示例：制备GHZ态"""
    print("\n" + "=" * 60)
    print("示例3：制备3量子比特GHZ态")
    print("=" * 60)
    
    # 使用专门的GHZ环境
    env = GHZPreparationEnv(
        n_qubits=3,
        max_steps=20,
        action_type='discrete',
        fidelity_threshold=0.95
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    print(f"目标: GHZ态 = (|000⟩ + |111⟩)/√2")
    
    # 创建PPO代理
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        config={'lr': 3e-4}
    )
    
    # 训练
    print("\n开始训练...")
    for episode in range(150):
        metrics = agent.train_episode(env)
        
        if (episode + 1) % 30 == 0:
            eval_stats = agent.evaluate(env, n_episodes=5)
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"success_rate={eval_stats['success_rate']:.2%}")
    
    # 最终评估
    print("\n最终评估:")
    final_eval = agent.evaluate(env, n_episodes=10)
    print(f"  成功率: {final_eval['success_rate']:.2%}")
    print(f"  平均奖励: {final_eval['mean_reward']:.2f}")
    
    return env, agent


def example_w_state():
    """示例：制备W态"""
    print("\n" + "=" * 60)
    print("示例4：制备3量子比特W态")
    print("=" * 60)
    
    env = WStatePreparationEnv(
        n_qubits=3,
        max_steps=25,
        action_type='discrete',
        fidelity_threshold=0.90
    )
    
    print("目标: W态 = (|100⟩ + |010⟩ + |001⟩)/√3")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    from rl_quantum_control.rl_agents.dqn_agent import DQNAgent
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={
            'lr': 1e-3,
            'epsilon_decay': 0.995
        }
    )
    
    # 训练
    print("\n开始训练（W态较难制备）...")
    best_fidelity = 0
    
    for episode in range(200):
        metrics = agent.train_episode(env)
        
        # 检查当前保真度
        state, _ = env.reset()
        for _ in range(env.max_steps):
            action = agent.select_action(state, deterministic=True)
            state, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        current_fidelity = info.get('fidelity', 0)
        if current_fidelity > best_fidelity:
            best_fidelity = current_fidelity
        
        if (episode + 1) % 40 == 0:
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"best_fidelity={best_fidelity:.4f}")
    
    print(f"\n最佳保真度: {best_fidelity:.4f}")
    
    return env, agent


def example_random_state():
    """示例：制备随机量子态"""
    print("\n" + "=" * 60)
    print("示例5：制备随机量子态")
    print("=" * 60)
    
    env = RandomStatePreparationEnv(
        n_qubits=1,
        max_steps=15,
        action_type='continuous',
        fidelity_threshold=0.95
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    from rl_quantum_control.rl_agents.sac_agent import SACAgent
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config={'lr_actor': 3e-4, 'lr_critic': 3e-4}
    )
    
    print("训练智能体制备随机目标态...")
    
    results = []
    for episode in range(100):
        metrics = agent.train_episode(env)
        
        if (episode + 1) % 20 == 0:
            # 评估几个随机态
            successes = 0
            for seed in range(5):
                state, _ = env.reset(seed=seed)
                for _ in range(env.max_steps):
                    action = agent.select_action(state, deterministic=True)
                    state, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                if info.get('success', False):
                    successes += 1
            
            print(f"Episode {episode + 1}: "
                  f"reward={metrics['episode_reward']:.2f}, "
                  f"random_success={successes}/5")
            results.append(successes / 5)
    
    print(f"\n平均随机态制备成功率: {np.mean(results[-5:]):.2%}")
    
    return env, agent


def example_real_device_ready():
    """示例：为真机运行准备态制备策略"""
    print("\n" + "=" * 60)
    print("示例6：为真机准备态制备策略")
    print("=" * 60)
    
    # 在模拟器上训练（带噪声）
    print("步骤1：在带噪声的模拟器上训练...")
    
    system = QuantumControlSystem(
        backend='simulator',
        n_qubits=1,
        with_noise=True,
        noise_params={
            'depolarizing_rate': 0.005,
            'readout_error': 0.01
        }
    )
    
    system.setup_task(
        task='state_preparation',
        target='plus',
        action_type='discrete',
        max_steps=10,
        fidelity_threshold=0.95
    )
    
    system.setup_agent('dqn', config={
        'lr': 1e-3,
        'double_dqn': True
    })
    
    # 训练
    history = system.train(n_episodes=100, eval_freq=25, verbose=True)
    
    # 在无噪声环境验证
    print("\n步骤2：在理想环境验证...")
    system.use_simulator(with_noise=False)
    eval_ideal = system.evaluate(n_episodes=10)
    
    # 准备切换到真机（这里只是示例）
    print("\n步骤3：准备真机部署...")
    print("要连接真机，请调用:")
    print("  system.connect_real_device(api_token='your_token', backend_name='ibm_brisbane')")
    
    # 获取动作序列
    result = system.run_single()
    print(f"\n学习到的动作序列: {result['action_sequence']}")
    print(f"在理想环境的保真度: {result['achieved_fidelity']:.6f}")
    
    return system, history


def analyze_state_preparation():
    """分析不同态的制备难度"""
    print("\n" + "=" * 60)
    print("分析：不同量子态的制备难度")
    print("=" * 60)
    
    targets = ['zero', 'one', 'plus', 'minus', 'plus_i', 'minus_i']
    results = {}
    
    for target in targets:
        print(f"\n制备 {target} 态...")
        
        system = create_system(
            task='state_preparation',
            target=target,
            agent='dqn',
            n_qubits=1,
            max_steps=10,
            action_type='discrete'
        )
        
        history = system.train(n_episodes=50, eval_freq=25, verbose=False)
        eval_result = system.evaluate(n_episodes=10, verbose=False)
        
        results[target] = {
            'success_rate': eval_result['success_rate'],
            'mean_fidelity': eval_result['mean_fidelity'],
            'mean_steps': np.mean([r['episode_length'] for r in eval_result['results']])
        }
        
        print(f"  {target}: fidelity={eval_result['mean_fidelity']:.4f}, "
              f"success={eval_result['success_rate']:.2%}")
    
    print("\n制备难度排名（按成功率）:")
    sorted_results = sorted(results.items(), 
                           key=lambda x: x[1]['success_rate'], 
                           reverse=True)
    for i, (target, res) in enumerate(sorted_results, 1):
        print(f"  {i}. {target}: {res['success_rate']:.2%} 成功率")
    
    return results


if __name__ == "__main__":
    print("强化学习量子态制备示例\n")
    
    try:
        # 示例1：|+⟩态
        system1, history1 = example_plus_state()
        
        # 示例2：Bell态
        system2, history2 = example_bell_state()
        
        # 示例3：GHZ态
        env3, agent3 = example_ghz_state()
        
        # 示例4：W态
        env4, agent4 = example_w_state()
        
        # 示例5：随机态
        env5, agent5 = example_random_state()
        
        # 示例6：真机准备
        system6, history6 = example_real_device_ready()
        
        # 分析
        analysis = analyze_state_preparation()
        
        print("\n所有示例运行完成!")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
