"""
Deep Deterministic Policy Gradient (DDPG) 算法实现
用于34节点配电网中抽水储能和电池储能的协同调度

基于文献[11-13]的DDPG算法

参考文献:
[11] Lillicrap T P, et al. Continuous control with deep reinforcement learning. 
    ICLR, 2016.
[12] Hou S, et al. RL-ADN: A high-performance Deep Reinforcement Learning 
    environment for optimal Energy Storage Systems dispatch in active 
    distribution networks. Energy and AI, 2025.
[13] Yang J, et al. Data-driven Optimal Dynamic Dispatch for Hydro-PV-PHS 
    Integrated Power Systems Using Deep Reinforcement Learning Approach. 
    IEEE Transactions on Sustainable Energy, 2022.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import copy
import random
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor网络: 策略网络"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        init_w: float = 3e-3
    ):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
            init_w: 权重初始化范围
        """
        super(Actor, self).__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化权重
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        # 使用tanh将输出限制在[-1, 1]
        action = torch.tanh(self.output_layer(x))
        return action


class Critic(nn.Module):
    """Critic网络: 价值网络"""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: List[int] = [256, 256],
        init_w: float = 3e-3
    ):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
            init_w: 权重初始化范围
        """
        super(Critic, self).__init__()
        
        # 第一层: 状态输入
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        
        # 后续层
        self.layers = nn.ModuleList()
        prev_dim = hidden_dims[0] + action_dim
        for hidden_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化权重
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(state))
        # 将动作与状态特征拼接
        x = torch.cat([x, action], dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
        q_value = self.output_layer(x)
        return q_value


class DDPGAgent:
    """DDPG智能体"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.995,
        tau: float = 0.005,
        buffer_capacity: int = 100000,
        batch_size: int = 256,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化DDPG智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            buffer_capacity: 回放缓冲区容量
            batch_size: 批量大小
            hidden_dims: 隐藏层维度
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 噪声参数
        self.noise_std = 0.1
        self.noise_decay = 0.995
        self.min_noise_std = 0.01
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态
            add_noise: 是否添加探索噪声
        
        Returns:
            action: 动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            # 添加Ornstein-Uhlenbeck噪声或高斯噪声
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise
            # 裁剪到[-1, 1]
            action = np.clip(action, -1.0, 1.0)
            
        return action
    
    def update(self) -> Dict:
        """
        更新网络
        
        Returns:
            info: 训练信息
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 更新Critic
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 更新Actor
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # 记录损失
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.q_values.append(current_q.mean().item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_value': current_q.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def decay_noise(self):
        """衰减探索噪声"""
        self.noise_std = max(self.min_noise_std, self.noise_std * self.noise_decay)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_std': self.noise_std
        }, filepath)
        
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.noise_std = checkpoint['noise_std']
        
        # 同步目标网络
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


class DDPGTrainer:
    """DDPG训练器"""
    
    def __init__(
        self,
        env,
        agent: DDPGAgent,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 96,
        eval_interval: int = 50,
        save_interval: int = 100,
        log_interval: int = 10
    ):
        """
        初始化训练器
        
        Args:
            env: 环境
            agent: DDPG智能体
            max_episodes: 最大训练回合数
            max_steps_per_episode: 每回合最大步数
            eval_interval: 评估间隔
            save_interval: 保存间隔
            log_interval: 日志间隔
        """
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        # 训练历史
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        
    def train(self):
        """训练智能体"""
        print("开始训练...")
        
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                # 选择动作
                action = self.agent.select_action(state, add_noise=True)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # 更新网络
                update_info = self.agent.update()
                
                episode_reward += reward
                episode_steps += 1
                
                state = next_state
                
                if done:
                    break
                    
            # 衰减噪声
            self.agent.decay_noise()
            
            # 记录
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # 日志
            if (episode + 1) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                print(f"Episode {episode + 1}/{self.max_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Noise Std: {self.agent.noise_std:.4f}")
                
            # 评估
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                print(f"Evaluation at Episode {episode + 1}: {eval_reward:.2f}")
                
            # 保存
            if (episode + 1) % self.save_interval == 0:
                self.agent.save(f"ddpg_checkpoint_episode_{episode + 1}.pth")
                
        print("训练完成!")
        
    def evaluate(self, num_episodes: int = 5) -> float:
        """评估智能体"""
        total_reward = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state, add_noise=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                    
            total_reward += episode_reward
            
        return total_reward / num_episodes
    
    def plot_training_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 回合奖励
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 评估奖励
        if len(self.eval_rewards) > 0:
            axes[0, 1].plot(
                range(self.eval_interval, len(self.eval_rewards) * self.eval_interval + 1, self.eval_interval),
                self.eval_rewards
            )
            axes[0, 1].set_title('Evaluation Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
        
        # Actor损失
        if len(self.agent.actor_losses) > 0:
            axes[1, 0].plot(self.agent.actor_losses)
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
        
        # Critic损失
        if len(self.agent.critic_losses) > 0:
            axes[1, 1].plot(self.agent.critic_losses)
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()
