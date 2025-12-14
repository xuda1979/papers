"""
快速开始脚本
============

运行此脚本快速体验强化学习量子控制系统
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_quantum_control.main import create_system, QuantumControlSystem


def quick_demo():
    """快速演示"""
    print("=" * 70)
    print(" 强化学习量子控制系统 - 快速演示")
    print("=" * 70)
    
    # 演示1：门合成
    print("\n【演示1】量子门合成 - 学习Hadamard门")
    print("-" * 50)
    
    system = create_system(
        task='gate_synthesis',
        target='H',
        agent='dqn',
        n_qubits=1,
        max_steps=10
    )
    
    print("训练中...")
    history = system.train(n_episodes=100, eval_freq=25, verbose=True)
    
    print("\n评估结果:")
    result = system.evaluate(n_episodes=10)
    print(f"  成功率: {result['success_rate']:.2%}")
    print(f"  平均保真度: {result['mean_fidelity']:.4f}")
    
    # 演示2：态制备
    print("\n" + "=" * 70)
    print("【演示2】量子态制备 - 制备|+⟩态")
    print("-" * 50)
    
    system2 = create_system(
        task='state_preparation',
        target='plus',
        agent='dqn',
        n_qubits=1,
        max_steps=10
    )
    
    print("训练中...")
    history2 = system2.train(n_episodes=100, eval_freq=25, verbose=True)
    
    single_result = system2.run_single()
    print(f"\n单次运行:")
    print(f"  动作序列: {single_result['action_sequence'][:5]}...")
    print(f"  最终保真度: {single_result['achieved_fidelity']:.6f}")
    
    # 演示3：噪声环境
    print("\n" + "=" * 70)
    print("【演示3】噪声环境训练")
    print("-" * 50)
    
    system3 = QuantumControlSystem(
        backend='simulator',
        n_qubits=1,
        with_noise=True,
        noise_params={'depolarizing_rate': 0.01}
    )
    
    system3.setup_task('state_preparation', target='plus', max_steps=10)
    system3.setup_agent('ppo')
    
    print("在噪声环境训练...")
    history3 = system3.train(n_episodes=80, verbose=True)
    
    result3 = system3.evaluate(n_episodes=10)
    print(f"\n噪声环境结果:")
    print(f"  成功率: {result3['success_rate']:.2%}")
    print(f"  平均保真度: {result3['mean_fidelity']:.4f}")
    
    print("\n" + "=" * 70)
    print(" 演示完成！")
    print("=" * 70)
    print("""
下一步：
1. 运行示例脚本获取更多细节:
   python examples/gate_synthesis_example.py
   python examples/state_preparation_example.py
   python examples/pulse_optimization_example.py
   
2. 使用命令行接口:
   python cli.py train --task gate_synthesis --target H --agent dqn
   
3. 连接真机:
   参考 examples/real_device_example.py
""")


if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保已安装所需依赖:")
        print("  pip install -r requirements.txt")
