"""
强化学习代理基类
===============

定义RL代理的通用接口
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import os
import json


class BaseAgent(ABC):
    """强化学习代理基类"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Optional[Dict] = None):
        """
        初始化代理
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: 配置参数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_step = 0
        self.episode_count = 0
    
    @abstractmethod
    def select_action(self, state: np.ndarray, 
                     deterministic: bool = False) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            动作
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict) -> Dict[str, float]:
        """
        更新代理
        
        Args:
            batch: 经验批次
            
        Returns:
            训练指标
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """
        训练一个episode
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            训练统计
        """
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        transitions = []
        
        for step in range(max_steps):
            # 选择动作
            action = self.select_action(state, deterministic=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 存储转换
            transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # 更新代理
        batch = self._prepare_batch(transitions)
        metrics = self.update(batch)
        
        # 更新统计
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        
        metrics.update({
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode': self.episode_count
        })
        
        return metrics
    
    def _prepare_batch(self, transitions: List[Dict]) -> Dict:
        """准备训练批次"""
        batch = {
            'states': np.array([t['state'] for t in transitions]),
            'actions': np.array([t['action'] for t in transitions]),
            'rewards': np.array([t['reward'] for t in transitions]),
            'next_states': np.array([t['next_state'] for t in transitions]),
            'dones': np.array([t['done'] for t in transitions])
        }
        return batch
    
    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """
        评估代理
        
        Args:
            env: 环境
            n_episodes: 评估episode数
            
        Returns:
            评估统计
        """
        rewards = []
        lengths = []
        successes = []
        
        for _ in range(n_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(info.get('success', False))
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'success_rate': np.mean(successes),
            'n_episodes': n_episodes
        }
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.training_step,
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'max_reward': np.max(self.episode_rewards),
            'recent_rewards': self.episode_rewards[-10:]
        }


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """添加转换"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict:
        """采样批次"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            alpha: 优先级指数
            beta: 重要性采样指数
        """
        super().__init__(capacity, state_dim, action_dim)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """添加带优先级的转换"""
        self.priorities[self.ptr] = self.max_priority
        super().add(state, action, reward, next_state, done)
    
    def sample(self, batch_size: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """优先级采样"""
        # 计算采样概率
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # 采样
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        self.priorities[indices] = priorities + 1e-6
        self.max_priority = max(self.max_priority, priorities.max())


if __name__ == "__main__":
    print("=== 基础代理测试 ===\n")
    
    # 测试回放缓冲区
    print("1. 回放缓冲区测试")
    buffer = ReplayBuffer(capacity=1000, state_dim=4, action_dim=2)
    
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = i % 10 == 0
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    batch = buffer.sample(32)
    print(f"采样批次形状: states={batch['states'].shape}, actions={batch['actions'].shape}")
    
    # 测试优先级回放
    print("\n2. 优先级回放缓冲区测试")
    per_buffer = PrioritizedReplayBuffer(capacity=1000, state_dim=4, action_dim=2)
    
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randn(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = i % 10 == 0
        
        per_buffer.add(state, action, reward, next_state, done)
    
    batch, weights, indices = per_buffer.sample(32)
    print(f"采样权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # 更新优先级
    new_priorities = np.random.rand(32) * 10
    per_buffer.update_priorities(indices, new_priorities)
    print(f"最大优先级: {per_buffer.max_priority:.3f}")
