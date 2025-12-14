"""
命令行接口
=========

强化学习量子控制系统的命令行工具
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_quantum_control.main import QuantumControlSystem, create_system


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='强化学习量子控制系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练门合成
  python cli.py train --task gate_synthesis --target H --agent dqn
  
  # 训练态制备
  python cli.py train --task state_preparation --target bell_00 --agent ppo
  
  # 脉冲优化
  python cli.py train --task pulse_optimization --target X --agent sac
  
  # 使用噪声模拟器
  python cli.py train --task gate_synthesis --target CNOT --noise
  
  # 评估模型
  python cli.py evaluate --model checkpoint.pt --task gate_synthesis
  
  # 连接真机
  python cli.py run --backend ibm --token YOUR_TOKEN --backend-name ibm_brisbane
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练强化学习代理')
    train_parser.add_argument('--task', type=str, required=True,
                             choices=['gate_synthesis', 'state_preparation', 'pulse_optimization'],
                             help='任务类型')
    train_parser.add_argument('--target', type=str, required=True,
                             help='目标（门名/态名）')
    train_parser.add_argument('--agent', type=str, default='dqn',
                             choices=['dqn', 'ppo', 'sac'],
                             help='强化学习算法')
    train_parser.add_argument('--n-qubits', type=int, default=1,
                             help='量子比特数')
    train_parser.add_argument('--max-steps', type=int, default=20,
                             help='最大步数')
    train_parser.add_argument('--episodes', type=int, default=500,
                             help='训练回合数')
    train_parser.add_argument('--noise', action='store_true',
                             help='使用噪声模拟器')
    train_parser.add_argument('--noise-rate', type=float, default=0.01,
                             help='噪声率')
    train_parser.add_argument('--save', type=str, default='model.pt',
                             help='保存模型路径')
    train_parser.add_argument('--eval-freq', type=int, default=50,
                             help='评估频率')
    train_parser.add_argument('--verbose', action='store_true',
                             help='详细输出')
    train_parser.add_argument('--npu', action='store_true',
                             help='使用NPU加速')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估训练好的模型')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='模型文件路径')
    eval_parser.add_argument('--task', type=str, required=True,
                            choices=['gate_synthesis', 'state_preparation', 'pulse_optimization'],
                            help='任务类型')
    eval_parser.add_argument('--target', type=str, required=True,
                            help='目标（门名/态名）')
    eval_parser.add_argument('--episodes', type=int, default=100,
                            help='评估回合数')
    eval_parser.add_argument('--output', type=str, default='eval_results.json',
                            help='结果输出文件')
    eval_parser.add_argument('--npu', action='store_true',
                            help='使用NPU加速')
    
    # 运行命令（单次执行）
    run_parser = subparsers.add_parser('run', help='运行单次量子控制')
    run_parser.add_argument('--model', type=str, required=True,
                           help='模型文件路径')
    run_parser.add_argument('--task', type=str, required=True,
                           help='任务类型')
    run_parser.add_argument('--target', type=str, required=True,
                           help='目标')
    run_parser.add_argument('--backend', type=str, default='simulator',
                           choices=['simulator', 'ibm'],
                           help='后端类型')
    run_parser.add_argument('--token', type=str,
                           help='IBM Quantum API Token')
    run_parser.add_argument('--backend-name', type=str,
                           help='IBM量子后端名称')
    run_parser.add_argument('--visualize', action='store_true',
                           help='可视化结果')
    run_parser.add_argument('--npu', action='store_true',
                           help='使用NPU加速')
    
    # 列出可用选项
    list_parser = subparsers.add_parser('list', help='列出可用选项')
    list_parser.add_argument('--type', type=str, required=True,
                            choices=['gates', 'states', 'backends', 'agents'],
                            help='列出类型')
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='生成配置文件')
    config_parser.add_argument('--output', type=str, default='config.json',
                              help='输出文件路径')
    config_parser.add_argument('--template', type=str,
                              choices=['default', 'advanced', 'noise'],
                              default='default',
                              help='配置模板')
    
    return parser.parse_args()


def cmd_train(args):
    """执行训练命令"""
    print("=" * 60)
    print(f"强化学习量子控制训练")
    print("=" * 60)
    print(f"任务: {args.task}")
    print(f"目标: {args.target}")
    print(f"算法: {args.agent}")
    print(f"量子比特: {args.n_qubits}")
    print(f"噪声: {'启用' if args.noise else '禁用'}")
    print("=" * 60)
    
    # 确定动作类型
    action_type = 'continuous' if args.task == 'pulse_optimization' else 'discrete'
    
    # 创建系统
    system = create_system(
        task=args.task,
        target=args.target,
        agent=args.agent,
        n_qubits=args.n_qubits,
        max_steps=args.max_steps,
        action_type=action_type,
        use_npu=args.npu
    )
    
    # 设置噪声
    if args.noise:
        system.use_simulator(
            with_noise=True,
            noise_params={'depolarizing_rate': args.noise_rate}
        )
    
    # 训练
    history = system.train(
        n_episodes=args.episodes,
        eval_freq=args.eval_freq,
        verbose=args.verbose
    )
    
    # 保存模型
    system.save_model(args.save)
    print(f"\n模型已保存到: {args.save}")
    
    # 最终评估
    eval_result = system.evaluate(n_episodes=20)
    print(f"\n最终评估:")
    print(f"  成功率: {eval_result['success_rate']:.2%}")
    print(f"  平均保真度: {eval_result['mean_fidelity']:.4f}")
    
    return history, eval_result


def cmd_evaluate(args):
    """执行评估命令"""
    print("=" * 60)
    print(f"模型评估")
    print("=" * 60)
    
    # 创建系统
    action_type = 'continuous' if args.task == 'pulse_optimization' else 'discrete'
    
    system = create_system(
        task=args.task,
        target=args.target,
        agent='dqn',  # 将从文件加载
        n_qubits=1,
        max_steps=20,
        action_type=action_type,
        use_npu=args.npu
    )
    
    # 加载模型
    system.load_model(args.model)
    print(f"模型已加载: {args.model}")
    
    # 评估
    print(f"\n运行 {args.episodes} 回合评估...")
    results = system.evaluate(n_episodes=args.episodes, verbose=True)
    
    # 保存结果
    with open(args.output, 'w') as f:
        # 转换为可序列化格式
        output_results = {
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'success_rate': results['success_rate'],
            'mean_fidelity': results['mean_fidelity']
        }
        json.dump(output_results, f, indent=2)
    
    print(f"\n结果已保存到: {args.output}")
    
    return results


def cmd_run(args):
    """执行单次运行命令"""
    print("=" * 60)
    print(f"单次量子控制执行")
    print("=" * 60)
    
    action_type = 'continuous' if args.task == 'pulse_optimization' else 'discrete'
    
    system = create_system(
        task=args.task,
        target=args.target,
        agent='dqn',
        n_qubits=1,
        max_steps=20,
        action_type=action_type,
        use_npu=args.npu
    )
    
    # 加载模型
    system.load_model(args.model)
    
    # 设置后端
    if args.backend == 'ibm':
        if not args.token:
            print("错误: 需要提供 --token 参数")
            return None
        system.connect_real_device(
            api_token=args.token,
            backend_name=args.backend_name or 'ibm_brisbane'
        )
        print(f"已连接到IBM量子后端: {args.backend_name}")
    
    # 运行
    result = system.run_single()
    
    print(f"\n执行结果:")
    print(f"  动作序列: {result['action_sequence']}")
    print(f"  保真度: {result['achieved_fidelity']:.6f}")
    print(f"  成功: {result['success']}")
    
    # 可视化
    if args.visualize:
        from rl_quantum_control.utils.visualization import Visualizer
        viz = Visualizer()
        # 生成可视化
        print("\n生成可视化图像...")
    
    return result


def cmd_list(args):
    """列出可用选项"""
    if args.type == 'gates':
        print("可用量子门:")
        gates = ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'RX', 'RY', 'RZ', 'CNOT', 'CZ', 'SWAP']
        for gate in gates:
            print(f"  - {gate}")
    
    elif args.type == 'states':
        print("可用目标态:")
        states = [
            ('zero', '|0⟩'),
            ('one', '|1⟩'),
            ('plus', '|+⟩ = (|0⟩ + |1⟩)/√2'),
            ('minus', '|-⟩ = (|0⟩ - |1⟩)/√2'),
            ('plus_i', '|+i⟩ = (|0⟩ + i|1⟩)/√2'),
            ('minus_i', '|-i⟩ = (|0⟩ - i|1⟩)/√2'),
            ('bell_00', '(|00⟩ + |11⟩)/√2'),
            ('bell_01', '(|01⟩ + |10⟩)/√2'),
            ('bell_10', '(|00⟩ - |11⟩)/√2'),
            ('bell_11', '(|01⟩ - |10⟩)/√2'),
            ('ghz', 'GHZ态'),
            ('w', 'W态'),
        ]
        for name, desc in states:
            print(f"  - {name}: {desc}")
    
    elif args.type == 'backends':
        print("可用后端:")
        print("  - simulator: 本地量子模拟器")
        print("  - simulator (with_noise=True): 带噪声模拟器")
        print("  - ibm: IBM Quantum真机")
        print("\nIBM量子后端:")
        print("  - ibm_brisbane (127 qubits)")
        print("  - ibm_osaka (127 qubits)")
        print("  - ibm_kyoto (127 qubits)")
        print("  - ibmq_qasm_simulator")
    
    elif args.type == 'agents':
        print("可用强化学习算法:")
        agents = [
            ('dqn', 'Deep Q-Network', '离散动作空间，门合成/态制备'),
            ('ppo', 'Proximal Policy Optimization', '离散/连续，通用'),
            ('sac', 'Soft Actor-Critic', '连续动作空间，脉冲优化'),
        ]
        for name, full_name, desc in agents:
            print(f"  - {name} ({full_name})")
            print(f"    适用: {desc}")


def cmd_config(args):
    """生成配置文件"""
    templates = {
        'default': {
            'task': 'gate_synthesis',
            'target': 'H',
            'agent': 'dqn',
            'n_qubits': 1,
            'max_steps': 20,
            'episodes': 500,
            'backend': 'simulator',
            'noise': False,
            'agent_config': {
                'lr': 1e-3,
                'gamma': 0.99,
                'buffer_size': 10000
            }
        },
        'advanced': {
            'task': 'pulse_optimization',
            'target': 'X',
            'agent': 'sac',
            'n_qubits': 1,
            'max_steps': 20,
            'episodes': 1000,
            'backend': 'simulator',
            'noise': True,
            'noise_params': {
                'depolarizing_rate': 0.01,
                'amplitude_damping_rate': 0.005
            },
            'agent_config': {
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'gamma': 0.99,
                'alpha': 0.2,
                'buffer_size': 100000
            }
        },
        'noise': {
            'task': 'state_preparation',
            'target': 'bell_00',
            'agent': 'ppo',
            'n_qubits': 2,
            'max_steps': 15,
            'episodes': 800,
            'backend': 'simulator',
            'noise': True,
            'noise_params': {
                'depolarizing_rate': 0.008,
                'amplitude_damping_rate': 0.003,
                'phase_damping_rate': 0.002,
                'readout_error': 0.015
            },
            'agent_config': {
                'lr': 3e-4,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01
            }
        }
    }
    
    config = templates[args.template]
    
    with open(args.output, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"配置文件已生成: {args.output}")
    print(f"模板: {args.template}")
    print(f"\n配置内容:")
    print(json.dumps(config, indent=2))


def main():
    """主函数"""
    args = parse_args()
    
    if args.command is None:
        print("强化学习量子控制系统")
        print("使用 --help 查看帮助")
        return
    
    try:
        if args.command == 'train':
            cmd_train(args)
        elif args.command == 'evaluate':
            cmd_evaluate(args)
        elif args.command == 'run':
            cmd_run(args)
        elif args.command == 'list':
            cmd_list(args)
        elif args.command == 'config':
            cmd_config(args)
        else:
            print(f"未知命令: {args.command}")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
