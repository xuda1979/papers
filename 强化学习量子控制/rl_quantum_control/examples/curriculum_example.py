"""
课程学习示例
===========

演示如何使用课程学习策略加速量子门合成训练
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_quantum_control.main import create_system


def example_fidelity_curriculum():
    """演示基于保真度的课程学习"""
    print("=" * 60)
    print("示例：基于保真度的课程学习")
    print("=" * 60)
    
    # 创建系统
    system = create_system(
        task='gate_synthesis',
        target='T',  # T门通常比H门难一点
        agent='dqn',
        n_qubits=1,
        max_steps=15
    )
    
    # 设置课程学习
    # 从0.8的保真度阈值开始，逐步提高到0.99
    system.setup_curriculum(
        strategy_type='fidelity',
        start_threshold=0.8,
        end_threshold=0.99,
        steps=5,
        update_freq=50  # 每50个episode检查一次
    )
    
    print("开始训练（带课程学习）...")
    history = system.train(n_episodes=300, eval_freq=50, verbose=True)
    
    # 最终评估
    print("\n最终评估:")
    result = system.evaluate(n_episodes=10)
    print(f"  成功率: {result['success_rate']:.2%}")
    print(f"  平均保真度: {result['mean_fidelity']:.4f}")
    
    return system


def example_steps_curriculum():
    """演示基于步数的课程学习"""
    print("\n" + "=" * 60)
    print("示例：基于步数的课程学习")
    print("=" * 60)
    
    # 创建系统
    system = create_system(
        task='state_preparation',
        target='random',  # 随机态
        agent='ppo',
        n_qubits=1,
        max_steps=5  # 初始步数很少
    )
    
    # 设置课程学习
    # 逐步增加允许的最大步数，让代理先学会简单的，再学复杂的
    system.setup_curriculum(
        strategy_type='steps',
        start_steps=5,
        end_steps=20,
        step_inc=5,
        mode='increase',
        update_freq=50
    )
    
    print("开始训练（步数递增）...")
    history = system.train(n_episodes=250, eval_freq=50, verbose=True)
    
    return system


if __name__ == "__main__":
    try:
        example_fidelity_curriculum()
        example_steps_curriculum()
        print("\n课程学习示例完成!")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
