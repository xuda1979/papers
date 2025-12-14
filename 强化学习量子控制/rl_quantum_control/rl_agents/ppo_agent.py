"""
PPO代理
======

实现Proximal Policy Optimization算法用于连续动作空间的量子控制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, Optional, Tuple, List
import os

from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int] = (256, 256),
                 continuous: bool = True):
        """
        初始化Actor-Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            continuous: 是否为连续动作空间
        """
        super().__init__()
        
        self.continuous = continuous
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh()
        )
        
        # Actor头
        if continuous:
            self.actor_mean = nn.Linear(hidden_dims[1], action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dims[1], action_dim)
        
        # Critic头
        self.critic = nn.Linear(hidden_dims[1], 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.shared(state)
        value = self.critic(features)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            return mean, std, value
        else:
            logits = self.actor(features)
            return logits, value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和log概率"""
        if self.continuous:
            mean, std, value = self.forward(state)
            
            if deterministic:
                action = mean
                log_prob = torch.zeros_like(action[:, 0])
            else:
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob, value
        else:
            logits, value = self.forward(state)
            
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = torch.zeros_like(value.squeeze())
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob, value
    
    def evaluate_actions(self, states: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作"""
        if self.continuous:
            mean, std, value = self.forward(states)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits, value = self.forward(states)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return log_prob, value.squeeze(), entropy


class PPOAgent(BaseAgent):
    """
    PPO代理
    
    实现PPO算法，支持连续和离散动作空间
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = True,
                 config: Optional[Dict] = None):
        """
        初始化PPO代理
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            continuous: 是否为连续动作空间
            config: 配置参数
        """
        super().__init__(state_dim, action_dim, config)
        
        self.continuous = continuous
        
        # 默认配置
        self.config.setdefault('lr', 3e-4)
        self.config.setdefault('gamma', 0.99)
        self.config.setdefault('gae_lambda', 0.95)
        self.config.setdefault('clip_ratio', 0.2)
        self.config.setdefault('value_coef', 0.5)
        self.config.setdefault('entropy_coef', 0.01)
        self.config.setdefault('max_grad_norm', 0.5)
        self.config.setdefault('n_epochs', 10)
        self.config.setdefault('batch_size', 64)
        self.config.setdefault('hidden_dims', (256, 256))
        
        # 设备
        if 'device' in self.config:
            self.device = torch.device(self.config['device'])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.policy = ActorCritic(
            state_dim, action_dim,
            self.config['hidden_dims'],
            continuous
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config['lr']
        )
        
        # 轨迹缓冲区
        self.trajectory_buffer = []
    
    def select_action(self, state: np.ndarray,
                     deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(
                state_tensor, deterministic
            )
            
            if self.continuous:
                # 限制动作范围
                action = torch.tanh(action) * np.pi
                return action.cpu().numpy().squeeze()
            else:
                return action.cpu().numpy().item()
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """PPO更新"""
        # 存储轨迹
        for i in range(len(batch['states'])):
            self.trajectory_buffer.append({
                'state': batch['states'][i],
                'action': batch['actions'][i],
                'reward': batch['rewards'][i],
                'next_state': batch['next_states'][i],
                'done': batch['dones'][i]
            })
        
        # 检查是否有完整episode
        if not batch['dones'][-1]:
            return {'loss': 0.0}
        
        # 计算优势和回报
        states, actions, returns, advantages = self._compute_gae()
        
        if len(states) == 0:
            return {'loss': 0.0}
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(actions).to(self.device)
        else:
            actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算旧的log概率
        with torch.no_grad():
            old_log_probs, _, _ = self.policy.evaluate_actions(states, actions)
        
        # PPO更新
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        for epoch in range(self.config['n_epochs']):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.config['batch_size']):
                end = min(start + self.config['batch_size'], n_samples)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # 评估动作
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config['clip_ratio'],
                    1 + self.config['clip_ratio']
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = (policy_loss + 
                       self.config['value_coef'] * value_loss +
                       self.config['entropy_coef'] * entropy_loss)
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config['max_grad_norm']
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
        
        # 清空缓冲区
        self.trajectory_buffer = []
        
        self.training_step += 1
        
        n_updates = self.config['n_epochs'] * (n_samples // self.config['batch_size'] + 1)
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': policy_loss_sum / n_updates,
            'value_loss': value_loss_sum / n_updates,
            'entropy': entropy_sum / n_updates
        }
    
    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算广义优势估计"""
        if len(self.trajectory_buffer) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        states = [t['state'] for t in self.trajectory_buffer]
        actions = [t['action'] for t in self.trajectory_buffer]
        rewards = [t['reward'] for t in self.trajectory_buffer]
        dones = [t['done'] for t in self.trajectory_buffer]
        next_states = [t['next_state'] for t in self.trajectory_buffer]
        
        # 计算价值
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            
            _, _, values = self.policy.get_action(states_tensor)
            _, _, next_values = self.policy.get_action(next_states_tensor)
            
            values = values.cpu().numpy().squeeze()
            next_values = next_values.cpu().numpy().squeeze()
        
        # GAE计算
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        gae = 0
        for t in reversed(range(n)):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.config['gamma'] * next_values[t] - values[t]
                gae = delta + self.config['gamma'] * self.config['gae_lambda'] * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return np.array(states), np.array(actions), returns, advantages
    
    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)


# 需要导入F
import torch.nn.functional as F


def train_ppo(env, agent: PPOAgent, n_episodes: int = 1000,
              eval_freq: int = 100, verbose: bool = True) -> Dict:
    """
    训练PPO代理
    
    Args:
        env: 环境
        agent: PPO代理
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
            print(f"Episode {episode + 1}: reward={mean_reward:.2f}")
        
        # 评估
        if (episode + 1) % eval_freq == 0:
            eval_stats = agent.evaluate(env)
            history['eval_rewards'].append(eval_stats['mean_reward'])
            
            if verbose:
                print(f"  Eval: reward={eval_stats['mean_reward']:.2f}, "
                      f"success_rate={eval_stats['success_rate']:.2%}")
    
    return history


if __name__ == "__main__":
    print("=== PPO代理测试 ===\n")
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 测试连续动作空间
    try:
        from environments.gate_synthesis import GateSynthesisEnv
        
        print("测试连续动作空间 (量子门合成)...")
        env = GateSynthesisEnv(
            target_gate='H',
            max_steps=20,
            action_type='continuous'
        )
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=True,
            config={
                'lr': 3e-4,
                'n_epochs': 4
            }
        )
        
        print("训练5个episode...")
        for ep in range(5):
            metrics = agent.train_episode(env)
            print(f"  Episode {ep+1}: reward={metrics['episode_reward']:.2f}")
        
    except Exception as e:
        print(f"连续动作空间测试失败: {e}")
    
    # 测试离散动作空间
    print("\n测试离散动作空间...")
    try:
        env_discrete = GateSynthesisEnv(
            target_gate='X',
            max_steps=20,
            action_type='discrete'
        )
        
        state_dim = env_discrete.observation_space.shape[0]
        action_dim = env_discrete.action_space.n
        
        agent_discrete = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=False
        )
        
        print("训练5个episode...")
        for ep in range(5):
            metrics = agent_discrete.train_episode(env_discrete)
            print(f"  Episode {ep+1}: reward={metrics['episode_reward']:.2f}")
        
    except Exception as e:
        print(f"离散动作空间测试失败: {e}")
