"""
SAC代理
======

实现Soft Actor-Critic算法用于连续动作空间的量子控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Optional, Tuple
import os

from .base_agent import BaseAgent, ReplayBuffer


class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int] = (256, 256),
                 action_scale: float = np.pi):
        super().__init__()
        
        self.action_scale = action_scale
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作和计算log概率"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale
        
        # 计算log概率（考虑tanh变换）
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - (action / self.action_scale).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """获取动作"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean) * self.action_scale
        
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        return torch.tanh(action) * self.action_scale


class QNetwork(nn.Module):
    """Q网络（Double Q）"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int] = (256, 256)):
        super().__init__()
        
        # Q1
        layers1 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers1.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers1.append(nn.Linear(prev_dim, 1))
        self.q1 = nn.Sequential(*layers1)
        
        # Q2
        layers2 = []
        prev_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers2.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers2.append(nn.Linear(prev_dim, 1))
        self.q2 = nn.Sequential(*layers2)
    
    def forward(self, state: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent(BaseAgent):
    """
    SAC代理
    
    实现Soft Actor-Critic算法
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Optional[Dict] = None):
        """
        初始化SAC代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置参数
        """
        super().__init__(state_dim, action_dim, config)
        
        # 默认配置
        self.config.setdefault('lr_actor', 3e-4)
        self.config.setdefault('lr_critic', 3e-4)
        self.config.setdefault('lr_alpha', 3e-4)
        self.config.setdefault('gamma', 0.99)
        self.config.setdefault('tau', 0.005)
        self.config.setdefault('batch_size', 256)
        self.config.setdefault('buffer_size', 1000000)
        self.config.setdefault('hidden_dims', (256, 256))
        self.config.setdefault('auto_alpha', True)
        self.config.setdefault('alpha', 0.2)
        self.config.setdefault('action_scale', np.pi)
        
        # 设备
        if 'device' in self.config:
            self.device = torch.device(self.config['device'])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = GaussianPolicy(
            state_dim, action_dim,
            self.config['hidden_dims'],
            self.config['action_scale']
        ).to(self.device)
        
        self.critic = QNetwork(
            state_dim, action_dim,
            self.config['hidden_dims']
        ).to(self.device)
        
        self.critic_target = QNetwork(
            state_dim, action_dim,
            self.config['hidden_dims']
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config['lr_actor']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config['lr_critic']
        )
        
        # 自动温度调整
        if self.config['auto_alpha']:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config['lr_alpha'])
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config['alpha']
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(
            self.config['buffer_size'],
            state_dim,
            action_dim
        )
    
    def select_action(self, state: np.ndarray,
                     deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_tensor, deterministic)
            return action.cpu().numpy().squeeze()
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """SAC更新"""
        # 存储经验
        for i in range(len(batch['states'])):
            self.replay_buffer.add(
                batch['states'][i],
                batch['actions'][i],
                batch['rewards'][i],
                batch['next_states'][i],
                batch['dones'][i]
            )
        
        # 检查缓冲区大小
        if len(self.replay_buffer) < self.config['batch_size']:
            return {'loss': 0.0}
        
        # 采样批次
        batch_data = self.replay_buffer.sample(self.config['batch_size'])
        
        states = torch.FloatTensor(batch_data['states']).to(self.device)
        actions = torch.FloatTensor(batch_data['actions']).to(self.device)
        rewards = torch.FloatTensor(batch_data['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch_data['next_states']).to(self.device)
        dones = torch.FloatTensor(batch_data['dones']).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target_value = rewards + self.config['gamma'] * (1 - dones) * (q_target - self.alpha * next_log_probs)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度
        alpha_loss = 0.0
        if self.config['auto_alpha']:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), 
                                       self.critic_target.parameters()):
            target_param.data.copy_(
                self.config['tau'] * param.data + 
                (1 - self.config['tau']) * target_param.data
            )
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            'alpha': self.alpha,
            'q_mean': q1.mean().item()
        }
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }
        
        if self.config['auto_alpha']:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.config['auto_alpha'] and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp().item()
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)


def train_sac(env, agent: SACAgent, n_episodes: int = 1000,
              eval_freq: int = 100, verbose: bool = True) -> Dict:
    """
    训练SAC代理
    """
    history = {
        'rewards': [],
        'lengths': [],
        'critic_losses': [],
        'actor_losses': [],
        'eval_rewards': []
    }
    
    for episode in range(n_episodes):
        metrics = agent.train_episode(env)
        
        history['rewards'].append(metrics['episode_reward'])
        history['lengths'].append(metrics['episode_length'])
        history['critic_losses'].append(metrics.get('critic_loss', 0))
        history['actor_losses'].append(metrics.get('actor_loss', 0))
        
        if verbose and (episode + 1) % 10 == 0:
            mean_reward = np.mean(history['rewards'][-10:])
            print(f"Episode {episode + 1}: reward={mean_reward:.2f}, "
                  f"alpha={agent.alpha:.3f}")
        
        if (episode + 1) % eval_freq == 0:
            eval_stats = agent.evaluate(env)
            history['eval_rewards'].append(eval_stats['mean_reward'])
            
            if verbose:
                print(f"  Eval: reward={eval_stats['mean_reward']:.2f}, "
                      f"success_rate={eval_stats['success_rate']:.2%}")
    
    return history


if __name__ == "__main__":
    print("=== SAC代理测试 ===\n")
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from environments.pulse_optimization import PulseOptimizationEnv
        from quantum_backend.simulator import QuantumGates
        
        print("测试脉冲优化环境...")
        env = PulseOptimizationEnv(
            n_qubits=1,
            n_time_steps=10,
            target_unitary=QuantumGates.X,
            fidelity_threshold=0.95
        )
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            config={
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'batch_size': 32
            }
        )
        
        print("训练5个episode...")
        for ep in range(5):
            metrics = agent.train_episode(env)
            print(f"  Episode {ep+1}: reward={metrics['episode_reward']:.2f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
