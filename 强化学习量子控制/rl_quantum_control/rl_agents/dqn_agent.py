"""
DQN代理
======

实现Deep Q-Network算法用于离散动作空间的量子控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os

from .base_agent import BaseAgent, ReplayBuffer, PrioritizedReplayBuffer


class QNetwork(nn.Module):
    """Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: Tuple[int] = (256, 256)):
        """
        初始化Q网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)


class DuelingQNetwork(nn.Module):
    """Dueling Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int] = (256, 256)):
        super().__init__()
        
        # 共享特征层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class DQNAgent(BaseAgent):
    """
    DQN代理
    
    支持多种DQN变体：
    - 标准DQN
    - Double DQN
    - Dueling DQN
    - 优先经验回放
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Optional[Dict] = None):
        """
        初始化DQN代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            config: 配置参数
        """
        super().__init__(state_dim, action_dim, config)
        
        # 默认配置
        self.config.setdefault('lr', 1e-4)
        self.config.setdefault('gamma', 0.99)
        self.config.setdefault('batch_size', 64)
        self.config.setdefault('buffer_size', 100000)
        self.config.setdefault('target_update_freq', 100)
        self.config.setdefault('epsilon_start', 1.0)
        self.config.setdefault('epsilon_end', 0.01)
        self.config.setdefault('epsilon_decay', 0.995)
        self.config.setdefault('hidden_dims', (256, 256))
        self.config.setdefault('double_dqn', True)
        self.config.setdefault('dueling', True)
        self.config.setdefault('prioritized', False)
        
        # 设备
        if 'device' in self.config:
            self.device = torch.device(self.config['device'])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        NetworkClass = DuelingQNetwork if self.config['dueling'] else QNetwork
        self.q_network = NetworkClass(
            state_dim, action_dim, self.config['hidden_dims']
        ).to(self.device)
        
        self.target_network = NetworkClass(
            state_dim, action_dim, self.config['hidden_dims']
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=self.config['lr']
        )
        
        # 回放缓冲区
        if self.config['prioritized']:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.config['buffer_size'], state_dim, 1
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.config['buffer_size'], state_dim, 1
            )
        
        # 探索参数
        self.epsilon = self.config['epsilon_start']
        
        # 更新计数
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, 
                     deterministic: bool = False) -> int:
        """选择动作"""
        if not deterministic and np.random.rand() < self.epsilon:
            # 随机探索
            return np.random.randint(self.action_dim)
        
        # 贪婪选择
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """更新Q网络"""
        # 存储经验
        for i in range(len(batch['states'])):
            self.replay_buffer.add(
                batch['states'][i],
                np.array([batch['actions'][i]]),
                batch['rewards'][i],
                batch['next_states'][i],
                batch['dones'][i]
            )
        
        # 检查缓冲区大小
        if len(self.replay_buffer) < self.config['batch_size']:
            return {'loss': 0.0}
        
        # 采样批次
        if self.config['prioritized']:
            batch_data, weights, indices = self.replay_buffer.sample(
                self.config['batch_size']
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch_data = self.replay_buffer.sample(self.config['batch_size'])
            weights = torch.ones(self.config['batch_size']).to(self.device)
            indices = None
        
        # 转换为张量
        states = torch.FloatTensor(batch_data['states']).to(self.device)
        actions = torch.LongTensor(batch_data['actions']).to(self.device).squeeze()
        rewards = torch.FloatTensor(batch_data['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch_data['next_states']).to(self.device)
        dones = torch.FloatTensor(batch_data['dones']).to(self.device)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN: 用当前网络选择动作，用目标网络评估
                next_actions = self.q_network(next_states).argmax(1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze()
            else:
                next_q = self.target_network(next_states).max(1)[0]
            
            target_q = rewards + self.config['gamma'] * next_q * (1 - dones)
        
        # 计算损失
        td_errors = current_q - target_q
        loss = (weights * td_errors.pow(2)).mean()
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # 更新优先级
        if self.config['prioritized'] and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )
        
        self.training_step += 1
        
        return {
            'loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'epsilon': self.epsilon
        }
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)


def train_dqn(env, agent: DQNAgent, n_episodes: int = 1000,
              eval_freq: int = 100, verbose: bool = True) -> Dict:
    """
    训练DQN代理
    
    Args:
        env: 环境
        agent: DQN代理
        n_episodes: 训练episode数
        eval_freq: 评估频率
        verbose: 是否打印日志
        
    Returns:
        训练历史
    """
    history = {
        'rewards': [],
        'lengths': [],
        'losses': [],
        'eval_rewards': []
    }
    
    for episode in range(n_episodes):
        metrics = agent.train_episode(env)
        
        history['rewards'].append(metrics['episode_reward'])
        history['lengths'].append(metrics['episode_length'])
        history['losses'].append(metrics.get('loss', 0))
        
        if verbose and (episode + 1) % 10 == 0:
            mean_reward = np.mean(history['rewards'][-10:])
            print(f"Episode {episode + 1}: reward={mean_reward:.2f}, "
                  f"epsilon={agent.epsilon:.3f}")
        
        # 评估
        if (episode + 1) % eval_freq == 0:
            eval_stats = agent.evaluate(env)
            history['eval_rewards'].append(eval_stats['mean_reward'])
            
            if verbose:
                print(f"  Eval: reward={eval_stats['mean_reward']:.2f}, "
                      f"success_rate={eval_stats['success_rate']:.2%}")
    
    return history


if __name__ == "__main__":
    print("=== DQN代理测试 ===\n")
    
    # 创建简单的测试环境
    import gymnasium as gym
    
    # 使用CartPole测试
    try:
        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        print(f"环境: CartPole-v1")
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        
        # 创建代理
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config={
                'lr': 1e-3,
                'double_dqn': True,
                'dueling': True,
                'batch_size': 32
            }
        )
        
        # 简短训练测试
        print("\n训练10个episode...")
        for ep in range(10):
            metrics = agent.train_episode(env)
            print(f"  Episode {ep+1}: reward={metrics['episode_reward']:.0f}")
        
        # 评估
        print("\n评估...")
        eval_stats = agent.evaluate(env, n_episodes=5)
        print(f"平均奖励: {eval_stats['mean_reward']:.2f}")
        
        env.close()
        
    except Exception as e:
        print(f"CartPole测试跳过: {e}")
    
    # 测试量子控制环境
    print("\n测试量子门合成环境...")
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from environments.gate_synthesis import GateSynthesisEnv
        
        qenv = GateSynthesisEnv(target_gate='H', max_steps=20, action_type='discrete')
        state_dim = qenv.observation_space.shape[0]
        action_dim = qenv.action_space.n
        
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        
        q_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config={
                'lr': 1e-3,
                'epsilon_decay': 0.99
            }
        )
        
        print("训练5个episode...")
        for ep in range(5):
            metrics = q_agent.train_episode(qenv)
            print(f"  Episode {ep+1}: reward={metrics['episode_reward']:.2f}")
        
    except Exception as e:
        print(f"量子环境测试失败: {e}")
