"""
主运行脚本
用于训练DDPG智能体在34节点配电网中调度抽水储能和电池储能

运行方式:
1. 生成数据: python main.py --mode generate_data
2. 训练模型: python main.py --mode train
3. 评估模型: python main.py --mode eval --model_path models/ddpg_model.pth
4. 完整流程: python main.py --mode all
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 修复Windows OpenMP库冲突

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
sys.path.append('/mnt/okcomputer/output/psh_drl_project')

from utils.data_generator import (
    generate_ieee34_node_data,
    create_ieee34_node_topology,
    save_data
)
from envs.distribution_network import DistributionNetworkEnv
from algorithms.ddpg import DDPGAgent, DDPGTrainer
from configs.config import (
    psh_config, bess_configs, env_config,
    ddpg_config, training_config, eval_config
)


def set_random_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_data():
    """生成训练和测试数据"""
    print("=" * 60)
    print("步骤 1: 生成IEEE 34节点配电网数据")
    print("=" * 60)

    # 生成数据
    load_data, pv_data, price_data = generate_ieee34_node_data()
    node_data, line_data = create_ieee34_node_topology()

    # 保存数据
    data_dir = "/mnt/okcomputer/output/psh_drl_project/data"
    save_data(
        load_data, pv_data, price_data,
        node_data, line_data,
        data_dir
    )

    print(f"\n数据生成完成!")
    print(f"  - 负荷数据: {load_data.shape}")
    print(f"  - 光伏数据: {pv_data.shape}")
    print(f"  - 电价数据: {price_data.shape}")
    print(f"  - 节点数据: {node_data.shape}")
    print(f"  - 线路数据: {line_data.shape}")

    # 可视化部分数据
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 负荷曲线(第一天)
    day1_load = load_data.iloc[:96, :5]  # 前5个节点
    axes[0, 0].plot(day1_load.index, day1_load.values)
    axes[0, 0].set_title('Day 1 Load Profiles (First 5 Nodes)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Load (kW)')
    axes[0, 0].legend([f'Node {i + 1}' for i in range(5)])
    axes[0, 0].grid(True)

    # 光伏曲线(第一天)
    day1_pv = pv_data.iloc[:96, :5]
    axes[0, 1].plot(day1_pv.index, day1_pv.values)
    axes[0, 1].set_title('Day 1 PV Profiles (First 5 Nodes)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('PV Output (kW)')
    axes[0, 1].legend([f'Node {i + 1}' for i in range(5)])
    axes[0, 1].grid(True)

    # 电价曲线(第一周)
    week1_price = price_data.iloc[:96 * 7]
    axes[1, 0].plot(week1_price.index, week1_price.values)
    axes[1, 0].set_title('Week 1 Price Profile')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Price (€/MWh)')
    axes[1, 0].grid(True)

    # 网络拓扑(简化显示)
    axes[1, 1].text(0.5, 0.5, 'IEEE 34-Bus\nDistribution Network\n\n1 PSH + 2 BESS',
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Network Configuration')

    plt.tight_layout()
    plt.savefig(f'{data_dir}/data_overview.png', dpi=150)
    print(f"\n数据可视化已保存到 {data_dir}/data_overview.png")


def create_environment():
    """创建环境"""
    print("\n" + "=" * 60)
    print("步骤 2: 创建配电网环境")
    print("=" * 60)

    # 加载数据
    data_dir = "/mnt/okcomputer/output/psh_drl_project/data"
    load_data = pd.read_csv(f"{data_dir}/load_data.csv", index_col=0, parse_dates=True)
    pv_data = pd.read_csv(f"{data_dir}/pv_data.csv", index_col=0, parse_dates=True)
    price_data = pd.read_csv(f"{data_dir}/price_data.csv", index_col=0, parse_dates=True)

    # 转换配置
    psh_cfg = {
        'unit_id': psh_config.unit_id,
        'node_id': psh_config.node_id,
        'max_generation_power': psh_config.max_generation_power,
        'min_generation_power': psh_config.min_generation_power,
        'max_pumping_power': psh_config.max_pumping_power,
        'min_pumping_power': psh_config.min_pumping_power,
        'max_reservoir_capacity': psh_config.max_reservoir_capacity,
        'min_reservoir_capacity': psh_config.min_reservoir_capacity,
        'generation_efficiency': psh_config.generation_efficiency,
        'pumping_efficiency': psh_config.pumping_efficiency,
        'initial_soc': psh_config.initial_soc,
        'ramp_rate_limit': psh_config.ramp_rate_limit,
        'is_variable_speed': psh_config.is_variable_speed
    }

    bess_cfgs = []
    for cfg in bess_configs:
        bess_cfgs.append({
            'unit_id': cfg.unit_id,
            'node_id': cfg.node_id,
            'max_power': cfg.max_power,
            'capacity': cfg.capacity,
            'min_soc': cfg.min_soc,
            'max_soc': cfg.max_soc,
            'charge_efficiency': cfg.charge_efficiency,
            'discharge_efficiency': cfg.discharge_efficiency,
            'initial_soc': cfg.initial_soc,
            'ramp_rate_limit': cfg.ramp_rate_limit
        })

    # 创建环境
    env = DistributionNetworkEnv(
        node_file=f"{data_dir}/node_data.csv",
        line_file=f"{data_dir}/line_data.csv",
        load_data=load_data,
        price_data=price_data,
        pv_data=pv_data,
        psh_config=psh_cfg,
        bess_configs=bess_cfgs,
        time_step=env_config.time_step,
        episode_length=env_config.episode_length
    )

    print("环境创建成功!")
    print(f"  - 状态维度: {env.state_dim}")
    print(f"  - 动作维度: {env.action_dim}")
    print(f"  - 时间步长: {env.time_step} 小时")
    print(f"  - 回合长度: {env.episode_length} 步")
    print(f"\n储能系统配置:")
    print(f"  - PSH (节点{psh_config.node_id}): {psh_config.max_generation_power}MW, "
          f"{psh_config.max_reservoir_capacity}MWh")
    for cfg in bess_configs:
        print(f"  - BESS{cfg.unit_id - 1} (节点{cfg.node_id}): {cfg.max_power}MW, "
              f"{cfg.capacity}MWh")

    return env


def train():
    """训练DDPG智能体"""
    print("\n" + "=" * 60)
    print("步骤 3: 训练DDPG智能体")
    print("=" * 60)

    # 设置随机种子
    set_random_seed(training_config.random_seed)

    # 创建环境
    env = create_environment()

    # 创建智能体
    agent = DDPGAgent(
        state_dim=ddpg_config.state_dim,
        action_dim=ddpg_config.action_dim,
        actor_lr=ddpg_config.actor_lr,
        critic_lr=ddpg_config.critic_lr,
        gamma=ddpg_config.gamma,
        tau=ddpg_config.tau,
        buffer_capacity=ddpg_config.buffer_capacity,
        batch_size=ddpg_config.batch_size,
        hidden_dims=ddpg_config.hidden_dims,
        device=ddpg_config.device
    )

    print(f"\nDDPG智能体配置:")
    print(f"  - Actor学习率: {ddpg_config.actor_lr}")
    print(f"  - Critic学习率: {ddpg_config.critic_lr}")
    print(f"  - 折扣因子: {ddpg_config.gamma}")
    print(f"  - 软更新系数: {ddpg_config.tau}")
    print(f"  - 回放缓冲区: {ddpg_config.buffer_capacity}")
    print(f"  - 批量大小: {ddpg_config.batch_size}")
    print(f"  - 隐藏层: {ddpg_config.hidden_dims}")
    print(f"  - 设备: {ddpg_config.device}")

    # 创建训练器
    trainer = DDPGTrainer(
        env=env,
        agent=agent,
        max_episodes=training_config.max_episodes,
        max_steps_per_episode=training_config.max_steps_per_episode,
        eval_interval=training_config.eval_interval,
        save_interval=training_config.save_interval,
        log_interval=training_config.log_interval
    )

    # 开始训练
    print(f"\n开始训练 {training_config.max_episodes} 个回合...")
    start_time = datetime.now()

    trainer.train()

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds() / 3600

    print(f"\n训练完成! 用时: {training_time:.2f} 小时")

    # 保存最终模型
    os.makedirs("models", exist_ok=True)
    final_model_path = training_config.model_save_path
    agent.save(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")

    # 绘制训练历史
    print("\n绘制训练历史...")
    trainer.plot_training_history()

    # 保存训练统计
    results_dir = eval_config.results_dir
    os.makedirs(results_dir, exist_ok=True)

    np.save(f"{results_dir}/episode_rewards.npy", trainer.episode_rewards)
    np.save(f"{results_dir}/eval_rewards.npy", trainer.eval_rewards)

    print(f"训练统计已保存到 {results_dir}")

    return agent, env


def evaluate(model_path: str, num_episodes: int = 10):
    """评估训练好的模型"""
    print("\n" + "=" * 60)
    print("步骤 4: 评估模型")
    print("=" * 60)

    # 创建环境
    env = create_environment()

    # 创建智能体并加载模型
    agent = DDPGAgent(
        state_dim=ddpg_config.state_dim,
        action_dim=ddpg_config.action_dim,
        device=ddpg_config.device
    )

    print(f"加载模型: {model_path}")
    agent.load(model_path)

    # 评估
    print(f"\n评估 {num_episodes} 个回合...")

    all_rewards = []
    all_voltages = []
    all_psh_powers = []
    all_psh_socs = []
    all_bess_powers = [[] for _ in range(len(env.bess_units))]
    all_bess_socs = [[] for _ in range(len(env.bess_units))]

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        episode_voltages = []
        episode_psh_powers = []
        episode_psh_socs = []
        episode_bess_powers = [[] for _ in range(len(env.bess_units))]
        episode_bess_socs = [[] for _ in range(len(env.bess_units))]

        for step in range(env.episode_length):
            # 选择动作(不加噪声)
            action = agent.select_action(state, add_noise=False)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            episode_reward += reward

            # 记录数据
            episode_voltages.append(info['voltage'])
            episode_psh_powers.append(info['psh_power'])
            episode_psh_socs.append(info['psh_soc'])
            for i, (power, soc) in enumerate(zip(info['bess_powers'], info['bess_socs'])):
                episode_bess_powers[i].append(power)
                episode_bess_socs[i].append(soc)

            state = next_state

            if done:
                break

        all_rewards.append(episode_reward)
        all_voltages.append(episode_voltages)
        all_psh_powers.append(episode_psh_powers)
        all_psh_socs.append(episode_psh_socs)
        for i in range(len(env.bess_units)):
            all_bess_powers[i].append(episode_bess_powers[i])
            all_bess_socs[i].append(episode_bess_socs[i])

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # 统计结果
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    print(f"\n评估结果:")
    print(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

    # 可视化结果
    print("\n生成评估可视化...")

    fig = plt.figure(figsize=(16, 12))

    # 选择一个典型回合进行详细可视化
    typical_episode = 0

    # 1. 电压曲线
    ax1 = plt.subplot(3, 2, 1)
    voltages = np.array(all_voltages[typical_episode])
    for i in range(min(5, voltages.shape[1])):
        ax1.plot(voltages[:, i], label=f'Node {i + 1}', alpha=0.7)
    ax1.axhline(y=1.05, color='r', linestyle='--', label='Vmax')
    ax1.axhline(y=0.95, color='r', linestyle='--', label='Vmin')
    ax1.set_title('Voltage Profiles (Typical Episode)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Voltage (p.u.)')
    ax1.legend()
    ax1.grid(True)

    # 2. PSH功率和SOC
    ax2 = plt.subplot(3, 2, 2)
    ax2_twin = ax2.twinx()

    psh_powers = all_psh_powers[typical_episode]
    psh_socs = all_psh_socs[typical_episode]

    ax2.plot(psh_powers, 'b-', label='Power')
    ax2_twin.plot(psh_socs, 'r-', label='SOC')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Power (MW)', color='b')
    ax2_twin.set_ylabel('SOC', color='r')
    ax2.set_title('PSH Operation')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

    # 3. BESS功率和SOC
    for i in range(len(env.bess_units)):
        ax = plt.subplot(3, 2, 3 + i)
        ax_twin = ax.twinx()

        bess_powers = all_bess_powers[i][typical_episode]
        bess_socs = all_bess_socs[i][typical_episode]

        ax.plot(bess_powers, 'b-', label='Power')
        ax_twin.plot(bess_socs, 'r-', label='SOC')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Power (MW)', color='b')
        ax_twin.set_ylabel('SOC', color='r')
        ax.set_title(f'BESS{i + 1} Operation (Node {env.bess_units[i].node_id})')
        ax.grid(True)
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    # 4. 奖励分布
    ax6 = plt.subplot(3, 2, 5)
    ax6.hist(all_rewards, bins=20, edgecolor='black')
    ax6.axvline(mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    ax6.set_title('Reward Distribution')
    ax6.set_xlabel('Episode Reward')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True)

    # 5. 储能协同运行
    ax7 = plt.subplot(3, 2, 6)
    total_storage_power = np.array(psh_powers)
    for i in range(len(env.bess_units)):
        total_storage_power += np.array(all_bess_powers[i][typical_episode])
    ax7.plot(total_storage_power, 'g-', linewidth=2, label='Total Storage Power')
    ax7.plot(psh_powers, 'b--', alpha=0.7, label='PSH')
    for i in range(len(env.bess_units)):
        ax7.plot(all_bess_powers[i][typical_episode], alpha=0.5, label=f'BESS{i + 1}')
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Power (MW)')
    ax7.set_title('Coordinated Storage Operation')
    ax7.legend()
    ax7.grid(True)

    plt.tight_layout()

    results_dir = eval_config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(f'{results_dir}/evaluation_results.png', dpi=150)
    print(f"评估结果已保存到 {results_dir}/evaluation_results.png")

    return all_rewards


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='DDPG for PSH and BESS Dispatch in 34-Bus Distribution Network'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['generate_data', 'train', 'eval', 'all'],
        help='运行模式'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/ddpg_model.pth',
        help='模型路径(用于评估)'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=10,
        help='评估回合数'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("抽水储能与电池储能协同调度 - DDPG强化学习")
    print("34节点配电网环境")
    print("=" * 60)

    if args.mode == 'generate_data':
        generate_data()
    elif args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate(args.model_path, args.num_eval_episodes)
    elif args.mode == 'all':
        # 完整流程
        generate_data()
        agent, env = train()
        evaluate(training_config.model_save_path, args.num_eval_episodes)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
