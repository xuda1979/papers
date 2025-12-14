"""
测试模块
=======

强化学习量子控制系统的单元测试
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQuantumSimulator(unittest.TestCase):
    """量子模拟器测试"""
    
    def setUp(self):
        from rl_quantum_control.quantum_backend.simulator import (
            QuantumSimulator, QuantumState, QuantumGates
        )
        self.QuantumSimulator = QuantumSimulator
        self.QuantumState = QuantumState
        self.QuantumGates = QuantumGates
    
    def test_initial_state(self):
        """测试初始态"""
        state = self.QuantumState(n_qubits=1)
        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state.state_vector, expected)
    
    def test_two_qubit_initial(self):
        """测试两量子比特初始态"""
        state = self.QuantumState(n_qubits=2)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(state.state_vector, expected)
    
    def test_x_gate(self):
        """测试X门"""
        state = self.QuantumState(n_qubits=1)
        state.apply_gate('X', 0)
        expected = np.array([0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(state.state_vector, expected)
    
    def test_h_gate(self):
        """测试H门"""
        state = self.QuantumState(n_qubits=1)
        state.apply_gate('H', 0)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(state.state_vector, expected)
    
    def test_cnot_gate(self):
        """测试CNOT门"""
        # |10⟩ -> |11⟩
        state = self.QuantumState(n_qubits=2)
        state.apply_gate('X', 0)  # |10⟩
        state.apply_gate('CNOT', 0, 1)  # |11⟩
        expected = np.array([0, 0, 0, 1], dtype=complex)
        np.testing.assert_array_almost_equal(state.state_vector, expected)
    
    def test_bell_state(self):
        """测试Bell态制备"""
        state = self.QuantumState(n_qubits=2)
        state.apply_gate('H', 0)
        state.apply_gate('CNOT', 0, 1)
        # (|00⟩ + |11⟩)/√2
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_array_almost_equal(state.state_vector, expected)
    
    def test_fidelity(self):
        """测试保真度计算"""
        state1 = self.QuantumState(n_qubits=1)
        state2 = self.QuantumState(n_qubits=1)
        fidelity = state1.fidelity(state2.state_vector)
        self.assertAlmostEqual(fidelity, 1.0)
    
    def test_simulator_execute(self):
        """测试模拟器执行"""
        sim = self.QuantumSimulator(n_qubits=1)
        result = sim.execute_circuit([('H', 0)], shots=100)
        self.assertIn('measurement_results', result)
        self.assertIn('state_vector', result)


class TestQuantumUtils(unittest.TestCase):
    """量子工具测试"""
    
    def setUp(self):
        from rl_quantum_control.utils.quantum_utils import (
            state_fidelity, process_fidelity, random_unitary
        )
        self.state_fidelity = state_fidelity
        self.process_fidelity = process_fidelity
        self.random_unitary = random_unitary
    
    def test_state_fidelity_same(self):
        """测试相同态的保真度"""
        state = np.array([1, 0], dtype=complex)
        fidelity = self.state_fidelity(state, state)
        self.assertAlmostEqual(fidelity, 1.0)
    
    def test_state_fidelity_orthogonal(self):
        """测试正交态的保真度"""
        state1 = np.array([1, 0], dtype=complex)
        state2 = np.array([0, 1], dtype=complex)
        fidelity = self.state_fidelity(state1, state2)
        self.assertAlmostEqual(fidelity, 0.0)
    
    def test_random_unitary_is_unitary(self):
        """测试随机酉矩阵"""
        U = self.random_unitary(2)
        # U†U = I
        product = U.conj().T @ U
        np.testing.assert_array_almost_equal(product, np.eye(2))


class TestEnvironments(unittest.TestCase):
    """环境测试"""
    
    def test_gate_synthesis_env(self):
        """测试门合成环境"""
        from rl_quantum_control.environments.gate_synthesis import GateSynthesisEnv
        
        env = GateSynthesisEnv(
            target_gate='H',
            n_qubits=1,
            max_steps=10,
            action_type='discrete'
        )
        
        # 测试重置
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], env.observation_space.shape[0])
        
        # 测试步进
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(reward, float)
        
        env.close()
    
    def test_state_preparation_env(self):
        """测试态制备环境"""
        from rl_quantum_control.environments.state_preparation import StatePreparationEnv
        
        env = StatePreparationEnv(
            target_state='plus',
            n_qubits=1,
            max_steps=10,
            action_type='discrete'
        )
        
        obs, info = env.reset()
        self.assertIsNotNone(obs)
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIn('fidelity', info)
        
        env.close()
    
    def test_pulse_optimization_env(self):
        """测试脉冲优化环境"""
        from rl_quantum_control.environments.pulse_optimization import PulseOptimizationEnv
        
        env = PulseOptimizationEnv(
            target_gate='X',
            n_time_steps=10,
            max_amplitude=1.0
        )
        
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], env.observation_space.shape[0])
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()


class TestAgents(unittest.TestCase):
    """代理测试"""
    
    def test_dqn_agent(self):
        """测试DQN代理"""
        from rl_quantum_control.rl_agents.dqn_agent import DQNAgent
        
        agent = DQNAgent(state_dim=4, action_dim=3)
        
        # 测试动作选择
        state = np.random.randn(4)
        action = agent.select_action(state)
        self.assertIn(action, [0, 1, 2])
        
        # 测试学习
        next_state = np.random.randn(4)
        agent.store_transition(state, action, 1.0, next_state, False)
        # 需要足够的样本才能学习
    
    def test_ppo_agent(self):
        """测试PPO代理"""
        from rl_quantum_control.rl_agents.ppo_agent import PPOAgent
        
        # 离散版本
        agent = PPOAgent(state_dim=4, action_dim=3, continuous=False)
        state = np.random.randn(4)
        action = agent.select_action(state)
        self.assertIn(action, [0, 1, 2])
        
        # 连续版本
        agent_cont = PPOAgent(state_dim=4, action_dim=2, continuous=True)
        action_cont = agent_cont.select_action(state)
        self.assertEqual(len(action_cont), 2)
    
    def test_sac_agent(self):
        """测试SAC代理"""
        from rl_quantum_control.rl_agents.sac_agent import SACAgent
        
        agent = SACAgent(state_dim=4, action_dim=2)
        state = np.random.randn(4)
        action = agent.select_action(state)
        self.assertEqual(len(action), 2)


class TestBackendManager(unittest.TestCase):
    """后端管理器测试"""
    
    def test_backend_switch(self):
        """测试后端切换"""
        from rl_quantum_control.quantum_backend.backend_manager import (
            BackendManager, BackendType
        )
        
        manager = BackendManager(n_qubits=1)
        
        # 默认是模拟器
        self.assertEqual(manager.backend_type, BackendType.SIMULATOR)
        
        # 切换到带噪声模拟器
        manager.set_backend(BackendType.SIMULATOR, with_noise=True)
        info = manager.get_backend_info()
        self.assertTrue(info.get('with_noise', False))
    
    def test_execute_circuit(self):
        """测试电路执行"""
        from rl_quantum_control.quantum_backend.backend_manager import BackendManager
        
        manager = BackendManager(n_qubits=1)
        result = manager.execute_circuit([('H', 0)])
        
        self.assertIn('measurement_results', result)
        self.assertIn('state_vector', result)


class TestMainSystem(unittest.TestCase):
    """主系统测试"""
    
    def test_create_system(self):
        """测试系统创建"""
        from rl_quantum_control.main import create_system
        
        system = create_system(
            task='gate_synthesis',
            target='H',
            agent='dqn',
            n_qubits=1,
            max_steps=5
        )
        
        self.assertIsNotNone(system)
        self.assertIsNotNone(system.env)
        self.assertIsNotNone(system.agent)
    
    def test_quick_train(self):
        """测试快速训练"""
        from rl_quantum_control.main import create_system
        
        system = create_system(
            task='gate_synthesis',
            target='X',  # 简单目标
            agent='dqn',
            n_qubits=1,
            max_steps=5
        )
        
        # 快速训练几个回合
        history = system.train(n_episodes=5, verbose=False)
        self.assertIsInstance(history, list)
    
    def test_evaluate(self):
        """测试评估"""
        from rl_quantum_control.main import create_system
        
        system = create_system(
            task='state_preparation',
            target='zero',  # 简单：保持初态
            agent='dqn',
            n_qubits=1,
            max_steps=5
        )
        
        result = system.evaluate(n_episodes=3, verbose=False)
        self.assertIn('mean_reward', result)
        self.assertIn('success_rate', result)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        from rl_quantum_control.main import QuantumControlSystem
        
        # 创建系统
        system = QuantumControlSystem(
            backend='simulator',
            n_qubits=1,
            with_noise=False
        )
        
        # 设置任务
        system.setup_task(
            task='state_preparation',
            target='one',
            action_type='discrete',
            max_steps=5
        )
        
        # 设置代理
        system.setup_agent('dqn')
        
        # 训练
        history = system.train(n_episodes=10, verbose=False)
        
        # 评估
        result = system.evaluate(n_episodes=3, verbose=False)
        
        # 单次运行
        single_result = system.run_single()
        
        self.assertIn('achieved_fidelity', single_result)
    
    def test_noise_model(self):
        """测试噪声模型"""
        from rl_quantum_control.main import QuantumControlSystem
        
        system = QuantumControlSystem(
            backend='simulator',
            n_qubits=1,
            with_noise=True,
            noise_params={'depolarizing_rate': 0.1}
        )
        
        system.setup_task('state_preparation', target='plus')
        system.setup_agent('dqn')
        
        # 高噪声应该降低保真度
        result = system.run_single()
        # 允许噪声导致保真度下降


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironments))
    suite.addTests(loader.loadTestsFromTestCase(TestAgents))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMainSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
