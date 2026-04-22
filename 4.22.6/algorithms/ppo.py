"""
Proximal Policy Optimization (PPO) 算法实现 - 版本5.0 (重构版)

核心改进:
1. 混合动作空间: 离散头(启停决策) + 连续头(功率设定)
2. 移除观察空间泄露 (不再接收valid_actions作为状态输入)
3. 降低update_interval适配长时序依赖
4. 改进网络结构: LayerNorm + 更稳定的初始化
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from torch.distributions import Categorical, Normal


class HybridActorNetwork(nn.Module):
    """
    混合Actor网络
    - 离散头: 输出启停动作概率 (3个动作: PUMP, GENERATE, STOP)
    - 连续头: 输出功率设定值 (均值和标准差)
    
    离散动作: 0=PUMP, 1=GENERATE, 2=STOP
    连续动作: PSH功率比例 (0-1), BESS1 (-1~1), BESS2 (-1~1)
    """
    
    def __init__(self, state_dim: int, n_discrete: int = 4, n_continuous: int = 3,
                 hidden_dims: List[int] = [256, 256]):
        super(HybridActorNetwork, self).__init__()
        
        self.n_discrete = n_discrete
        self.n_continuous = n_continuous
        
        # 共享特征提取网络
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())  # Tanh更稳定
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # 离散动作头 (启停决策)
        self.discrete_head = nn.Linear(prev_dim, n_discrete)
        
        # 连续动作头 - 均值
        self.continuous_mean = nn.Linear(prev_dim, n_continuous)
        
        # 连续动作头 - log_std (可学习)
        self.continuous_log_std = nn.Parameter(torch.zeros(n_continuous))
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        discrete_logits = self.discrete_head(features)
        continuous_raw = torch.sigmoid(self.continuous_mean(features))  # (0, 1)
        # 调整BESS输出范围为(-1, 1) - 避免inplace操作
        continuous_mean = continuous_raw.clone()
        if continuous_mean.shape[-1] >= 3:
            continuous_mean = torch.cat([
                continuous_mean[:, 0:1],  # PSH功率保持(0, 1)
                continuous_mean[:, 1:] * 2.0 - 1.0  # BESS: (-1, 1)
            ], dim=-1)
        continuous_std = torch.exp(self.continuous_log_std).clamp(min=0.01, max=1.0)
        return discrete_logits, continuous_mean, continuous_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """获取混合动作"""
        discrete_logits, continuous_mean, continuous_std = self.forward(state)
        
        batch_size = state.shape[0]
        
        # 检查NaN
        if torch.isnan(discrete_logits).any():
            discrete_logits = torch.zeros_like(discrete_logits)
        if torch.isnan(continuous_mean).any():
            continuous_mean = torch.zeros_like(continuous_mean)
        
        # 离散动作
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_probs = discrete_probs / (discrete_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        if deterministic:
            discrete_action = torch.argmax(discrete_probs, dim=-1)
        else:
            discrete_dist = Categorical(discrete_probs)
            discrete_action = discrete_dist.sample()
        
        # 连续动作
        if deterministic:
            continuous_action = continuous_mean
        else:
            continuous_dist = Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.sample()
        
        # 组合动作
        actions = torch.zeros(batch_size, 1 + self.n_continuous, device=state.device)
        actions[:, 0] = discrete_action.float()
        actions[:, 1:] = continuous_action
        
        # 计算log_prob
        discrete_dist = Categorical(discrete_probs)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        
        total_log_prob = discrete_log_prob + continuous_log_prob
        
        return actions.detach().cpu().numpy()[0], total_log_prob, discrete_probs
    
    def evaluate(self, state: torch.Tensor, discrete_action: torch.Tensor, 
                 continuous_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估动作"""
        discrete_logits, continuous_mean, continuous_std = self.forward(state)
        
        # 检查NaN
        if torch.isnan(discrete_logits).any() or torch.isnan(continuous_mean).any():
            batch_size = state.shape[0]
            log_probs = torch.zeros(batch_size, device=state.device)
            entropy = torch.zeros(batch_size, device=state.device)
            return log_probs, entropy
        
        # 离散部分
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_probs = discrete_probs / (discrete_probs.sum(dim=-1, keepdim=True) + 1e-8)
        discrete_dist = Categorical(discrete_probs)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        discrete_entropy = discrete_dist.entropy()
        
        # 连续部分
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)
        
        # 总log_prob和entropy
        log_probs = discrete_log_prob + continuous_log_prob
        entropy = discrete_entropy + continuous_entropy
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """Critic网络 - 输出状态价值"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)
        value = self.value_head(features)
        return value


class RolloutBuffer:
    """经验回放缓冲区 - 支持混合动作"""
    
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, discrete_action, continuous_action, reward, value, log_prob, done):
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.discrete_actions.clear()
        self.continuous_actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.discrete_actions, dtype=np.int64),
            np.array(self.continuous_actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO智能体 - 版本5.0 (混合动作空间)
    
    动作空间:
    - 离散: PSH启停 (0=PUMP, 1=GENERATE, 2=STOP)
    - 连续: [PSH功率(0-1), BESS1(-1~1), BESS2(-1~1)]
    """
    
    def __init__(
        self,
        state_dim: int,
        n_discrete_actions: int = 4,
        n_continuous_actions: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.1,       # V6.1: 提高 (原0.01)，防止策略坍缩
        max_grad_norm: float = 0.5,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cpu',
        epsilon_start: float = 0.15,      # V6.1: epsilon-greedy初始值
        epsilon_end: float = 0.01,        # V6.1: epsilon-greedy最终值
        epsilon_decay_episodes: int = 30  # V6.1: epsilon衰减轮数
    ):
        self.state_dim = state_dim
        self.n_discrete = n_discrete_actions
        self.n_continuous = n_continuous_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # V6.1: Epsilon-greedy探索参数
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.current_epsilon = epsilon_start
        self.current_episode = 0
        
        self.actor = HybridActorNetwork(
            state_dim, n_discrete_actions, n_continuous_actions, hidden_dims
        ).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            eps=1e-5
        )
        
        self.buffer = RolloutBuffer()
        
        # 损失记录
        self.episode_actor_losses = []
        self.episode_critic_losses = []
        self.episode_entropy_losses = []
        self.episode_total_losses = []
        
        self.current_actor_loss = 0.0
        self.current_critic_loss = 0.0
        self.current_entropy_loss = 0.0
        self.current_total_loss = 0.0
        self.current_update_count = 0
    
    def set_episode(self, episode: int):
        """V6.1: 设置当前轮次，更新epsilon-greedy参数"""
        self.current_episode = episode
        # 线性衰减epsilon
        if episode < self.epsilon_decay_episodes:
            self.current_epsilon = self.epsilon_start + \
                (self.epsilon_end - self.epsilon_start) * (episode / self.epsilon_decay_episodes)
        else:
            self.current_epsilon = self.epsilon_end
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """选择混合动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self._get_action(state_tensor, deterministic)
        
        return action, log_prob, value
    
    def _get_action(self, state_tensor: torch.Tensor, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """内部动作选择 - V6.1: 添加epsilon-greedy探索"""
        discrete_logits, continuous_mean, continuous_std = self.actor(state_tensor)
        
        batch_size = state_tensor.shape[0]
        
        # 检查NaN
        if torch.isnan(discrete_logits).any():
            discrete_logits = torch.zeros_like(discrete_logits)
        if torch.isnan(continuous_mean).any():
            continuous_mean = torch.zeros_like(continuous_mean)
        
        # 离散动作
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_probs = discrete_probs / (discrete_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # V6.4: Epsilon-greedy探索 - PUMP(0)/GENERATE(1)/STOP(2)
        use_random = (not deterministic) and (np.random.random() < self.current_epsilon)
        
        if deterministic:
            discrete_action = torch.argmax(discrete_probs, dim=-1)
        elif use_random:
            # Epsilon-greedy: 偏向PUMP和GENERATE，较少STOP
            random_probs = torch.tensor([0.4, 0.4, 0.2], device=self.device)
            discrete_action = torch.multinomial(random_probs, 1).squeeze(-1)
        else:
            discrete_dist = Categorical(discrete_probs)
            discrete_action = discrete_dist.sample()
        
        # 连续动作
        if deterministic:
            continuous_action = continuous_mean.clone()
        else:
            noise = torch.randn_like(continuous_mean)
            continuous_action = continuous_mean + continuous_std * noise
            continuous_action[:, 0] = torch.sigmoid(continuous_action[:, 0])  # PSH功率 (0-1)
            continuous_action[:, 1:] = torch.tanh(continuous_action[:, 1:])  # BESS (-1, 1)
        
        # 组合动作
        action = np.zeros(1 + self.n_continuous)
        action[0] = discrete_action.item()
        action[1:] = continuous_action[0].cpu().numpy()
        
        # 计算log_prob
        discrete_dist = Categorical(discrete_probs)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        
        # 连续log_prob
        continuous_std_clamped = continuous_std.clamp(min=0.01)
        continuous_log_prob = -0.5 * (((continuous_action - continuous_mean) / continuous_std_clamped) ** 2 + 
                                       2 * torch.log(continuous_std_clamped) + np.log(2 * np.pi))
        continuous_log_prob = continuous_log_prob.sum(dim=-1)
        
        total_log_prob = (discrete_log_prob + continuous_log_prob).item()
        
        # 价值
        value = self.critic(state_tensor).item()
        
        return action, total_log_prob, value
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算GAE优势"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_state: np.ndarray, n_epochs: int = 4, batch_size: int = 64):
        """更新网络"""
        if len(self.buffer) == 0:
            return {}
        
        # 获取数据
        states, discrete_actions, continuous_actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # 检查奖励
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print("警告: 奖励中有NaN或Inf，跳过本次更新")
            self.buffer.clear()
            return {}
        
        # 计算下一个状态的价值
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).item()
        
        # 计算GAE优势
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        if np.isnan(advantages).any() or np.isinf(advantages).any():
            print("警告: 优势中有NaN或Inf，跳过本次更新")
            self.buffer.clear()
            return {}
        
        # 转换为tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        discrete_actions_tensor = torch.LongTensor(discrete_actions).to(self.device)
        continuous_actions_tensor = torch.FloatTensor(continuous_actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        if len(advantages) > 1:
            adv_mean = advantages_tensor.mean()
            adv_std = advantages_tensor.std()
            if adv_std > 1e-8:
                advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
        
        # 多轮更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        nan_detected = False
        
        for epoch in range(n_epochs):
            if nan_detected:
                break
            
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_discrete = discrete_actions_tensor[batch_indices]
                batch_continuous = continuous_actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 评估动作
                log_probs, entropy = self.actor.evaluate(
                    batch_states, batch_discrete, batch_continuous
                )
                values_pred = self.critic(batch_states).squeeze(-1)
                
                # 检查NaN
                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                    nan_detected = True
                    break
                if torch.isnan(values_pred).any() or torch.isinf(values_pred).any():
                    nan_detected = True
                    break
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    nan_detected = True
                    break
                
                # 计算裁剪后的目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                critic_loss = F.mse_loss(values_pred, batch_returns)
                
                # 计算熵损失
                entropy_loss = -entropy.mean()
                
                # 检查损失
                if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                    nan_detected = True
                    break
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 记录损失
        if n_updates > 0 and not nan_detected:
            avg_actor_loss = total_actor_loss / n_updates
            avg_critic_loss = total_critic_loss / n_updates
            avg_entropy_loss = total_entropy_loss / n_updates
            avg_total_loss = avg_actor_loss + self.value_coef * avg_critic_loss + self.entropy_coef * avg_entropy_loss
            
            self.current_actor_loss += avg_actor_loss
            self.current_critic_loss += avg_critic_loss
            self.current_entropy_loss += avg_entropy_loss
            self.current_total_loss += avg_total_loss
            self.current_update_count += 1
        
        return {
            'actor_loss': self.current_actor_loss / max(self.current_update_count, 1),
            'critic_loss': self.current_critic_loss / max(self.current_update_count, 1),
            'entropy_loss': self.current_entropy_loss / max(self.current_update_count, 1),
            'total_loss': self.current_total_loss / max(self.current_update_count, 1)
        }
    
    def end_episode(self):
        """结束回合 - 记录本回合的平均损失"""
        if self.current_update_count > 0:
            self.episode_actor_losses.append(self.current_actor_loss / self.current_update_count)
            self.episode_critic_losses.append(self.current_critic_loss / self.current_update_count)
            self.episode_entropy_losses.append(self.current_entropy_loss / self.current_update_count)
            self.episode_total_losses.append(self.current_total_loss / self.current_update_count)
        else:
            if len(self.episode_actor_losses) > 0:
                self.episode_actor_losses.append(self.episode_actor_losses[-1])
                self.episode_critic_losses.append(self.episode_critic_losses[-1])
                self.episode_entropy_losses.append(self.episode_entropy_losses[-1])
                self.episode_total_losses.append(self.episode_total_losses[-1])
            else:
                self.episode_actor_losses.append(0.0)
                self.episode_critic_losses.append(0.0)
                self.episode_entropy_losses.append(0.0)
                self.episode_total_losses.append(0.0)
        
        self.current_actor_loss = 0.0
        self.current_critic_loss = 0.0
        self.current_entropy_loss = 0.0
        self.current_total_loss = 0.0
        self.current_update_count = 0
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_actor_losses': self.episode_actor_losses,
            'episode_critic_losses': self.episode_critic_losses,
            'episode_entropy_losses': self.episode_entropy_losses,
            'episode_total_losses': self.episode_total_losses,
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


class PPOTrainer:
    """PPO训练器 - 版本5.0"""
    
    def __init__(
        self,
        env,
        agent: PPOAgent,
        max_episodes: int = 200,
        max_steps_per_episode: int = 96,
        update_interval: int = 96,  # 每轮(1天)更新一次
        eval_interval: int = 10,
        save_interval: int = 20,
        log_interval: int = 1,
        log_save_path: str = "training_log.csv",
        plot_save_path: str = "training_plots.png",
        max_constraint_violations: int = 7,
        patience: int = 30,
        reset_psh_storage: bool = False  # PSH不重置库容状态
    ):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_interval = update_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        self.log_save_path = log_save_path
        self.plot_save_path = plot_save_path
        
        self.max_constraint_violations = max_constraint_violations
        self.patience = patience
        self.reset_psh_storage = reset_psh_storage
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.episode_voltage_violations = []
        self.episode_constraint_violations = []
        self.psh_action_counts = {0: 0, 1: 0, 2: 0}  # V6.2: 3动作模式
        self.renewable_consumption_rates = []
        
        self.detailed_logs = []
        
        self.best_eval_reward = -np.inf
        self.patience_counter = 0
        
        self.should_stop = False
        self.stop_reason = ""
    
    def train(self):
        """训练智能体"""
        print("=" * 70)
        print("PPO训练开始 - 版本5.0 (混合动作空间)")
        print(f"训练设置: {self.max_episodes}轮, 每轮{self.max_steps_per_episode}步")
        print(f"PSH约束违反目标: <{self.max_constraint_violations}次/轮")
        print("=" * 70)
        
        total_steps = 0
        
        for episode in range(self.max_episodes):
            # V6.1: 更新epsilon-greedy探索率
            self.agent.set_episode(episode)
            
            if self.should_stop:
                print(f"\n{'='*70}")
                print(f"训练提前终止: {self.stop_reason}")
                print(f"{'='*70}")
                break
            
            state = self.env.reset(reset_psh_storage=self.reset_psh_storage)
            episode_reward_sum = 0.0
            episode_steps = 0
            
            episode_violations = 0
            episode_constraint_violations = 0
            psh_powers = []
            psh_upper_socs = []
            psh_lower_socs = []
            voltages_min = []
            voltages_max = []
            episode_psh_actions = {0: 0, 1: 0, 2: 0}  # V6.2: 3动作模式
            episode_renewable_total = 0.0
            episode_renewable_consumed = 0.0
            
            for step in range(self.max_steps_per_episode):
                action, log_prob, value = self.agent.select_action(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                
                # 检查奖励
                if np.isnan(reward) or np.isinf(reward):
                    reward = -1.0
                
                reward = np.clip(reward, -100, 100)
                
                # 存储经验 - 分离离散和连续动作
                discrete_action = int(action[0])
                continuous_action = action[1:]
                
                self.agent.buffer.push(state, discrete_action, continuous_action, reward, value, log_prob, done)
                
                episode_reward_sum += reward
                episode_steps += 1
                total_steps += 1
                
                episode_violations += info.get('v_violation_count', 0)
                
                if info.get('psh_constraint_violated', False):
                    episode_constraint_violations += 1
                
                if 'psh_discrete_action' in info:
                    episode_psh_actions[info['psh_discrete_action']] += 1
                
                if 'voltage' in info:
                    voltages_min.append(info['voltage_min'])
                    voltages_max.append(info['voltage_max'])
                
                psh_powers.append(info.get('psh_power', 0))
                psh_upper_socs.append(info.get('psh_upper_soc', 0.5))
                psh_lower_socs.append(info.get('psh_lower_soc', 0.5))
                
                episode_renewable_total += info.get('renewable_total', 0)
                episode_renewable_consumed += info.get('renewable_consumed', 0)
                
                state = next_state
                
                # 定期更新
                if len(self.agent.buffer) >= self.update_interval:
                    update_info = self.agent.update(next_state)
                
                if done:
                    break
            
            # 如果缓冲区还有数据，更新
            if len(self.agent.buffer) > 0:
                update_info = self.agent.update(next_state)
            
            # 结束回合 - 记录损失
            self.agent.end_episode()
            
            # 计算平均奖励
            avg_episode_reward = episode_reward_sum / max(episode_steps, 1)
            
            # 计算可再生能源消纳率
            renewable_rate = episode_renewable_consumed / max(episode_renewable_total, 1e-6)
            
            self.episode_rewards.append(avg_episode_reward)
            self.episode_lengths.append(episode_steps)
            self.episode_voltage_violations.append(episode_violations)
            self.episode_constraint_violations.append(episode_constraint_violations)
            self.renewable_consumption_rates.append(renewable_rate)
            
            for k, v in episode_psh_actions.items():
                self.psh_action_counts[k] += v
            
            soc_stats = {
                'psh_upper_mean': np.mean(psh_upper_socs) if psh_upper_socs else 0.5,
                'psh_upper_min': np.min(psh_upper_socs) if psh_upper_socs else 0.5,
                'psh_upper_max': np.max(psh_upper_socs) if psh_upper_socs else 0.5,
                'psh_lower_mean': np.mean(psh_lower_socs) if psh_lower_socs else 0.5,
                'psh_lower_min': np.min(psh_lower_socs) if psh_lower_socs else 0.5,
                'psh_lower_max': np.max(psh_lower_socs) if psh_lower_socs else 0.5,
            }
            
            psh_hold_count = episode_psh_actions[0]
            psh_gen_count = episode_psh_actions[1]
            psh_pump_count = episode_psh_actions[2]
            
            # 输出日志
            if (episode + 1) % self.log_interval == 0:
                print(f"\n{'='*70}")
                print(f"训练轮次 {episode + 1}/{self.max_episodes}")
                print(f"{'='*70}")
                print(f"平均奖励: {avg_episode_reward:.4f}")
                print(f"电压越限次数: {episode_violations}")
                print(f"PSH约束违反: {episode_constraint_violations}")
                print(f"可再生能源消纳率: {renewable_rate*100:.1f}%")
                print(f"PSH动作: 抽水={psh_hold_count}, 发电={psh_gen_count}, 停止={psh_pump_count}")
                print(f"PSH上水库SOC: {soc_stats['psh_upper_mean']:.3f} (范围: {soc_stats['psh_upper_min']:.3f}-{soc_stats['psh_upper_max']:.3f})")
                if voltages_min:
                    print(f"电压范围: [{min(voltages_min):.4f}, {max(voltages_max):.4f}]")
                
                if len(self.agent.episode_actor_losses) > 0:
                    actor_loss = self.agent.episode_actor_losses[-1]
                    critic_loss = self.agent.episode_critic_losses[-1]
                    print(f"Actor损失: {actor_loss:.6f}" if not np.isnan(actor_loss) else "Actor损失: NaN")
                    print(f"Critic损失: {critic_loss:.6f}" if not np.isnan(critic_loss) else "Critic损失: NaN")
                
                if episode_constraint_violations < self.max_constraint_violations:
                    print(f">>> PSH约束违反达标! ({episode_constraint_violations} < {self.max_constraint_violations})")
            
            # 保存日志
            log_entry = {
                '训练轮次': episode + 1,
                '平均奖励': round(avg_episode_reward, 4),
                '电压越限次数': episode_violations,
                'PSH约束违反': episode_constraint_violations,
                '可再生能源消纳率': round(renewable_rate * 100, 1),
                'PSH保持次数': psh_hold_count,
                'PSH发电次数': psh_gen_count,
                'PSH抽水次数': psh_pump_count,
                '上水库SOC': round(soc_stats['psh_upper_mean'], 3),
                '下水库SOC': round(soc_stats['psh_lower_mean'], 3),
                '电压最小值': round(min(voltages_min), 4) if voltages_min else 0.95,
                '电压最大值': round(max(voltages_max), 4) if voltages_max else 1.05,
                '总损失': round(self.agent.episode_total_losses[-1], 6) if len(self.agent.episode_total_losses) > 0 else 0,
                'Actor损失': round(self.agent.episode_actor_losses[-1], 6) if len(self.agent.episode_actor_losses) > 0 else 0,
                'Critic损失': round(self.agent.episode_critic_losses[-1], 6) if len(self.agent.episode_critic_losses) > 0 else 0,
                '评估奖励': ''
            }
            
            self.detailed_logs.append(log_entry)
            self._save_training_log()
            
            # 检查训练状态
            self._check_training_status(episode, episode_constraint_violations)
            
            # 评估
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                
                if self.detailed_logs:
                    self.detailed_logs[-1]['评估奖励'] = round(eval_reward, 4)
                    self._save_training_log()
                
                print(f"\n>>> 第 {episode + 1} 轮评估平均奖励: {eval_reward:.4f}")
                
                # 保存最佳模型
                if eval_reward > self.best_eval_reward + 0.001:
                    self.best_eval_reward = eval_reward
                    self.patience_counter = 0
                    self.agent.save("best_model.pth")
                    print(">>> 保存最佳模型")
                else:
                    self.patience_counter += 1
                    print(f">>> 未改进 ({self.patience_counter}/{self.patience})")
                
                # 早停检查
                if self.patience_counter >= self.patience:
                    print(">>> 触发早停")
                    self.should_stop = True
                    self.stop_reason = "评估奖励长期未改进"
            
            # 保存检查点
            if (episode + 1) % self.save_interval == 0:
                self.agent.save(f"ppo_checkpoint_episode_{episode + 1}.pth")
        
        print("\n" + "=" * 70)
        print("训练完成!")
        print("=" * 70)
        
        self._save_training_log()
        self.plot_training_history()
    
    def _check_training_status(self, episode: int, constraint_violations: int):
        """检查训练状态"""
        if len(self.episode_constraint_violations) >= 15:
            recent_violations = self.episode_constraint_violations[-15:]
            avg_recent = np.mean(recent_violations)
            
            if avg_recent > self.max_constraint_violations * 3 and episode > 30:
                self.should_stop = True
                self.stop_reason = f"PSH约束违反持续恶化 (最近15轮平均: {avg_recent:.1f})"
                return
        
        if len(self.episode_rewards) >= 5:
            recent_rewards = self.episode_rewards[-5:]
            if any(np.isnan(r) or np.isinf(r) for r in recent_rewards):
                self.should_stop = True
                self.stop_reason = "检测到NaN或Inf奖励"
                return
        
        if len(self.agent.episode_actor_losses) >= 5:
            recent_losses = self.agent.episode_actor_losses[-5:]
            if any(np.isnan(l) or np.isinf(l) for l in recent_losses):
                self.should_stop = True
                self.stop_reason = "检测到NaN或Inf损失"
                return
    
    def _save_training_log(self):
        """保存训练日志"""
        try:
            df = pd.DataFrame(self.detailed_logs)
            columns = ['训练轮次', '平均奖励', '电压越限次数', 'PSH约束违反', '可再生能源消纳率',
                      'PSH保持次数', 'PSH发电次数', 'PSH抽水次数', 'PSH停止次数',
                      '上水库SOC', '下水库SOC', '电压最小值', '电压最大值',
                      '总损失', 'Actor损失', 'Critic损失', '评估奖励']
            
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
            
            df = df[columns]
            df.to_csv(self.log_save_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"保存日志出错: {e}")
    
    def evaluate(self, num_episodes: int = 3) -> float:
        """评估智能体"""
        total_reward = 0.0
        total_steps = 0
        
        for _ in range(num_episodes):
            state = self.env.reset(reset_psh_storage=self.reset_psh_storage)
            episode_reward_sum = 0.0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                action, _, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                episode_reward_sum += reward
                episode_steps += 1
                
                state = next_state
                if done:
                    break
            
            total_reward += episode_reward_sum
            total_steps += episode_steps
        
        avg_reward = total_reward / max(total_steps, 1)
        return avg_reward
    
    def plot_training_history(self):
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            
            # 1. 平均奖励
            axes[0, 0].plot(self.episode_rewards, alpha=0.5, color='blue', label='Raw')
            if len(self.episode_rewards) >= 10:
                ma10 = np.convolve(self.episode_rewards, np.ones(10) / 10, mode='valid')
                axes[0, 0].plot(range(9, len(self.episode_rewards)), ma10, 'r-', linewidth=2, label='MA(10)')
            axes[0, 0].set_title('Episode Reward', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Avg Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 评估奖励
            if len(self.eval_rewards) > 0:
                eval_episodes = range(self.eval_interval, len(self.eval_rewards) * self.eval_interval + 1,
                                      self.eval_interval)
                axes[0, 1].plot(eval_episodes, self.eval_rewards, 'bo-', markersize=6, linewidth=2)
                axes[0, 1].set_title('Eval Reward', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Avg Reward')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 可再生能源消纳率
            if len(self.renewable_consumption_rates) > 0:
                axes[0, 2].plot(np.array(self.renewable_consumption_rates) * 100, alpha=0.6, color='green')
                if len(self.renewable_consumption_rates) >= 10:
                    ma = np.convolve(self.renewable_consumption_rates, np.ones(10) / 10, mode='valid')
                    axes[0, 2].plot(range(9, len(self.renewable_consumption_rates)), np.array(ma) * 100, 'r-', linewidth=2)
                axes[0, 2].set_title('Renewable Consumption Rate (%)', fontsize=12, fontweight='bold')
                axes[0, 2].set_xlabel('Episode')
                axes[0, 2].set_ylabel('Rate (%)')
                axes[0, 2].grid(True, alpha=0.3)
            
            # 4. PSH约束违反
            axes[1, 0].plot(self.episode_constraint_violations, alpha=0.6, color='orange')
            axes[1, 0].axhline(y=self.max_constraint_violations, color='r', linestyle='--', 
                              linewidth=2, label=f'Target: {self.max_constraint_violations}')
            if len(self.episode_constraint_violations) >= 10:
                ma = np.convolve(self.episode_constraint_violations, np.ones(10) / 10, mode='valid')
                axes[1, 0].plot(range(9, len(self.episode_constraint_violations)), ma, 'b-', linewidth=2, label='MA(10)')
            axes[1, 0].set_title('PSH Constraint Violations', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Violations')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Actor损失
            if len(self.agent.episode_actor_losses) > 0:
                actor_losses = np.array(self.agent.episode_actor_losses)
                valid_indices = ~np.isnan(actor_losses)
                if valid_indices.any():
                    axes[1, 1].plot(np.where(valid_indices)[0], actor_losses[valid_indices], 
                                   alpha=0.6, color='blue', label='Actor Loss')
                axes[1, 1].set_title('Actor Loss', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Critic损失
            if len(self.agent.episode_critic_losses) > 0:
                critic_losses = np.array(self.agent.episode_critic_losses)
                valid_indices = ~np.isnan(critic_losses)
                if valid_indices.any():
                    axes[1, 2].plot(np.where(valid_indices)[0], critic_losses[valid_indices], 
                                   alpha=0.6, color='green', label='Critic Loss')
                axes[1, 2].set_title('Critic Loss', fontsize=12, fontweight='bold')
                axes[1, 2].set_xlabel('Episode')
                axes[1, 2].set_ylabel('Loss')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            # 7. 总损失
            if len(self.agent.episode_total_losses) > 0:
                total_losses = np.array(self.agent.episode_total_losses)
                valid_indices = ~np.isnan(total_losses)
                if valid_indices.any():
                    axes[2, 0].plot(np.where(valid_indices)[0], total_losses[valid_indices], 
                                   alpha=0.6, color='purple', label='Total Loss')
                axes[2, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
                axes[2, 0].set_xlabel('Episode')
                axes[2, 0].set_ylabel('Loss')
                axes[2, 0].legend()
                axes[2, 0].grid(True, alpha=0.3)
            
            # 8. 电压越限
            axes[2, 1].plot(self.episode_voltage_violations, alpha=0.6, color='red')
            if len(self.episode_voltage_violations) >= 10:
                ma = np.convolve(self.episode_voltage_violations, np.ones(10) / 10, mode='valid')
                axes[2, 1].plot(range(9, len(self.episode_voltage_violations)), ma, 'b-', linewidth=2)
            axes[2, 1].set_title('Voltage Violations', fontsize=12, fontweight='bold')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].grid(True, alpha=0.3)
            
            # 9. 奖励分布
            if len(self.episode_rewards) > 0:
                valid_rewards = [r for r in self.episode_rewards if not np.isnan(r) and not np.isinf(r)]
                if valid_rewards:
                    axes[2, 2].hist(valid_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[2, 2].axvline(np.mean(valid_rewards), color='r', linestyle='--', 
                                      linewidth=2, label=f'Mean: {np.mean(valid_rewards):.2f}')
                    axes[2, 2].set_title('Reward Distribution', fontsize=12, fontweight='bold')
                    axes[2, 2].set_xlabel('Avg Reward')
                    axes[2, 2].set_ylabel('Frequency')
                    axes[2, 2].legend()
                    axes[2, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plot_save_path, dpi=150, bbox_inches='tight')
            print(f"训练可视化已保存到: {self.plot_save_path}")
            plt.close()
        except Exception as e:
            print(f"绘制可视化出错: {e}")
