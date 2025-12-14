"""
课程学习模块
===========

用于强化学习量子控制的课程学习策略。
通过逐步增加任务难度来加速训练收敛。
"""

import numpy as np

class CurriculumStrategy:
    """课程学习基类"""
    def __init__(self, env, update_freq=100):
        self.env = env
        self.update_freq = update_freq
        self.episode_count = 0
        self.current_level = 0
        
    def update(self, metrics):
        """根据训练指标更新课程"""
        self.episode_count += 1
        if self.episode_count % self.update_freq == 0:
            return self._check_and_advance(metrics)
        return False
        
    def _check_and_advance(self, metrics):
        raise NotImplementedError


class FidelityCurriculum(CurriculumStrategy):
    """
    基于保真度的课程学习
    
    从较低的保真度阈值开始，逐步提高要求。
    """
    def __init__(self, env, start_threshold=0.8, end_threshold=0.99, steps=5, update_freq=100):
        super().__init__(env, update_freq)
        self.thresholds = np.linspace(start_threshold, end_threshold, steps)
        self.current_threshold = self.thresholds[0]
        
        # 设置初始阈值
        if hasattr(self.env, 'fidelity_threshold'):
            self.env.fidelity_threshold = self.current_threshold
            
    def _check_and_advance(self, metrics):
        # 如果最近的平均成功率很高，则提升难度
        success_rate = metrics.get('success_rate', 0)
        mean_reward = metrics.get('mean_reward', 0)
        
        if success_rate > 0.8 and self.current_level < len(self.thresholds) - 1:
            self.current_level += 1
            self.current_threshold = self.thresholds[self.current_level]
            
            # 更新环境参数
            if hasattr(self.env, 'fidelity_threshold'):
                self.env.fidelity_threshold = self.current_threshold
                print(f"\n[课程学习] 提升难度: 保真度阈值 -> {self.current_threshold:.4f}")
                return True
                
        return False


class MaxStepsCurriculum(CurriculumStrategy):
    """
    基于步数的课程学习
    
    从较少的步数开始（强迫寻找短序列），逐步增加允许的步数。
    或者反过来，从长步数开始（容易找到解），逐步减少（优化长度）。
    """
    def __init__(self, env, start_steps=5, end_steps=20, step_inc=5, mode='increase', update_freq=100):
        super().__init__(env, update_freq)
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.step_inc = step_inc
        self.mode = mode
        
        self.current_steps = start_steps
        if hasattr(self.env, 'max_steps'):
            self.env.max_steps = self.current_steps
            
    def _check_and_advance(self, metrics):
        success_rate = metrics.get('success_rate', 0)
        
        should_advance = False
        if success_rate > 0.7:
            if self.mode == 'increase' and self.current_steps < self.end_steps:
                self.current_steps = min(self.end_steps, self.current_steps + self.step_inc)
                should_advance = True
            elif self.mode == 'decrease' and self.current_steps > self.end_steps:
                self.current_steps = max(self.end_steps, self.current_steps - self.step_inc)
                should_advance = True
        
        if should_advance:
            if hasattr(self.env, 'max_steps'):
                self.env.max_steps = self.current_steps
                print(f"\n[课程学习] 调整难度: 最大步数 -> {self.current_steps}")
            return True
            
        return False


class NoiseCurriculum(CurriculumStrategy):
    """
    基于噪声的课程学习
    
    从无噪声环境开始学习，逐步增加噪声强度，提高鲁棒性。
    """
    def __init__(self, env, backend_manager, target_noise_params, steps=5, update_freq=200):
        super().__init__(env, update_freq)
        self.backend_manager = backend_manager
        self.target_noise_params = target_noise_params
        self.steps = steps
        self.current_noise_scale = 0.0
        
    def _check_and_advance(self, metrics):
        success_rate = metrics.get('success_rate', 0)
        
        if success_rate > 0.85 and self.current_level < self.steps:
            self.current_level += 1
            self.current_noise_scale = self.current_level / self.steps
            
            # 计算当前噪声参数
            current_params = {}
            for k, v in self.target_noise_params.items():
                current_params[k] = v * self.current_noise_scale
            
            # 更新后端噪声
            from rl_quantum_control.quantum_backend.backend_manager import BackendType
            self.backend_manager.set_backend(
                BackendType.SIMULATOR,
                with_noise=True,
                noise_params=current_params
            )
            
            print(f"\n[课程学习] 增加噪声: 强度 -> {self.current_noise_scale:.1%}")
            return True
            
        return False
