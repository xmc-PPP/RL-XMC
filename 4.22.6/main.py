"""
主运行脚本 - 版本5.0 (重构版)

重大改进:
1. 移除KMP_DUPLICATE_LIB_OK硬编码
2. 修复路径问题 - 使用相对路径
3. 混合动作空间支持
4. 添加规则策略基线对比
5. 随机抽取1天测试并分析可再生能源消纳率
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Dict, List

# 设置项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from envs.distribution_network import DistributionNetworkEnv
from algorithms.ppo import PPOAgent, PPOTrainer


def set_random_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_environment(data_dir: str = None):
    """创建环境"""
    if data_dir is None:
        # 默认使用与脚本同级的data目录
        data_dir = os.path.join(current_dir, "data")
    
    print("\n" + "=" * 70)
    print("创建配电网环境 - 版本5.0 (重构版)")
    print("=" * 70)
    
    node_file = os.path.join(data_dir, "Nodes_34.csv")
    line_file = os.path.join(data_dir, "Lines_34.csv")
    time_series_file = os.path.join(data_dir, "34_node_time_series.csv")
    
    # 检查文件是否存在
    for f in [node_file, line_file, time_series_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"数据文件不存在: {f}")
    
    env = DistributionNetworkEnv(
        node_file=node_file,
        line_file=line_file,
        time_series_file=time_series_file,
        time_step=0.25,
        episode_length=96,
        enable_domain_randomization=True,
    )
    
    print("\n环境创建成功!")
    print(f"  - 节点数: {env.n_nodes}")
    print(f"  - 状态维度: {env.state_dim}")
    print(f"  - 动作维度: {env.action_dim}")
    print(f"  - 时间步长: {env.time_step} 小时 (15分钟)")
    print(f"  - 回合长度: {env.episode_length} 步 (1天)")
    print(f"  - 潮流计算: {'pandapower' if env.power_flow.net is not None else 'fallback'}")
    
    print(f"\n储能配置:")
    print(f"  - PSH: 节点{env.psh.node_id}")
    print(f"    * 额定发电功率: {env.psh.rated_gen_power} MW")
    print(f"    * 额定抽水功率: {env.psh.rated_pump_power} MW")
    print(f"    * 上水库容量: {env.psh.upper_capacity} MWh")
    print(f"    * 下水库容量: {env.psh.lower_capacity} MWh")
    print(f"    * 发电最小出力: {env.psh.min_gen_output_ratio*100:.0f}%, 抽水最小出力: {env.psh.min_pump_output_ratio*100:.0f}%")
    
    for bess in env.bess_units:
        print(f"  - BESS{bess.unit_id}: 节点{bess.node_id}, {bess.max_power}MW/{bess.capacity}MWh")
    
    return env


def train(max_episodes: int = 200):
    """训练PPO智能体"""
    print("\n" + "=" * 70)
    print("训练PPO智能体 - 版本5.0 (混合动作空间)")
    print("=" * 70)
    
    set_random_seed(42)
    
    env = create_environment()
    
    # PPO配置 - 混合动作空间
    agent = PPOAgent(
        state_dim=env.state_dim,
        n_discrete_actions=3,  # V6.4: PUMP, GENERATE, STOP
        n_continuous_actions=3,  # PSH功率, BESS1, BESS2
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dims=[256, 256],
        device='cpu'
    )
    
    print(f"\nPPO智能体配置:")
    print(f"  - 学习率: 3e-4")
    print(f"  - 折扣因子: 0.99")
    print(f"  - GAE lambda: 0.95")
    print(f"  - Clip epsilon: 0.2")
    print(f"  - 隐藏层: [256, 256]")
    print(f"  - 混合动作空间:")
    print(f"    * 离散: PSH启停 (3动作: PUMP/GENERATE/STOP)")
    print(f"    * 连续: PSH功率(0-1) + BESS1(-1~1) + BESS2(-1~1)")
    
    # V5.2: 支持通过环境变量自定义保存路径，默认使用脚本所在目录
    save_dir = os.environ.get('SAVE_DIR', os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        max_episodes=max_episodes,
        max_steps_per_episode=96,
        update_interval=96,  # 每轮(1天)更新一次
        eval_interval=10,
        save_interval=20,
        log_interval=1,
        log_save_path=os.path.join(save_dir, "training_log.csv"),
        plot_save_path=os.path.join(save_dir, "training_plots.png"),
        max_constraint_violations=7,
        patience=30,
        reset_psh_storage=False  # PSH不重置库容状态
    )
    
    print("\n开始训练...")
    trainer.train()
    
    # V5.2: 保存最终模型到指定目录
    save_dir = os.environ.get('SAVE_DIR', current_dir)
    model_path = os.path.join(save_dir, "ppo_model_final.pth")
    agent.save(model_path)
    print(f"\n最终模型已保存到 {model_path}")
    
    # 生成最终报告
    generate_final_report(trainer)
    
    return agent, env


def generate_final_report(trainer):
    """生成最终训练报告"""
    report = []
    report.append("\n" + "=" * 80)
    report.append("抽水储能与电池储能协同调度系统 - 版本5.0 (重构版) 最终报告")
    report.append("=" * 80)
    
    report.append("\n## 训练结果摘要")
    report.append(f"\n### 基本信息")
    report.append(f"- 总训练轮次: {len(trainer.episode_rewards)}轮")
    report.append(f"- 每轮步数: 96步 (1天)")
    report.append(f"- 总训练步数: {len(trainer.episode_rewards) * 96:,}步")
    report.append(f"- 算法: PPO with 混合动作空间")
    
    violation_list = trainer.episode_constraint_violations
    compliant_episodes = sum(1 for v in violation_list if v < trainer.max_constraint_violations)
    report.append(f"\n### PSH约束违反统计")
    report.append(f"- 达标轮次 (<7次): {compliant_episodes}轮")
    report.append(f"- 达标率: {compliant_episodes / len(violation_list) * 100:.1f}%")
    report.append(f"- 平均PSH约束违反: {np.mean(violation_list):.1f}次/轮")
    report.append(f"- 最少违反: {min(violation_list)}次")
    report.append(f"- 最多违反: {max(violation_list)}次")
    
    # 可再生能源消纳率
    if len(trainer.renewable_consumption_rates) > 0:
        report.append(f"\n### 可再生能源消纳率")
        report.append(f"- 平均消纳率: {np.mean(trainer.renewable_consumption_rates)*100:.1f}%")
        report.append(f"- 最终消纳率: {trainer.renewable_consumption_rates[-1]*100:.1f}%")
    
    rewards = trainer.episode_rewards
    valid_rewards = [r for r in rewards if not np.isnan(r) and not np.isinf(r)]
    report.append(f"\n### 奖励统计")
    if valid_rewards:
        report.append(f"- 平均奖励: {np.mean(valid_rewards):.2f}")
        report.append(f"- 最高奖励: {max(valid_rewards):.2f} (第{rewards.index(max(valid_rewards))+1}轮)")
        report.append(f"- 最低奖励: {min(valid_rewards):.2f} (第{rewards.index(min(valid_rewards))+1}轮)")
        report.append(f"- 标准差: {np.std(valid_rewards):.2f}")
        
        if len(valid_rewards) >= 40:
            early_mean = np.mean(valid_rewards[:20])
            late_mean = np.mean(valid_rewards[-20:])
            report.append(f"- 前20轮平均: {early_mean:.2f}")
            report.append(f"- 后20轮平均: {late_mean:.2f}")
            if early_mean != 0:
                report.append(f"- 改进幅度: {(late_mean - early_mean) / abs(early_mean) * 100:.1f}%")
    
    total_actions = sum(trainer.psh_action_counts.values())
    if total_actions > 0:
        report.append(f"\n### PSH动作统计 (总次数)")
        report.append(f"- 保持: {trainer.psh_action_counts[0]}次 ({trainer.psh_action_counts[0]/total_actions*100:.1f}%)")
        report.append(f"- 发电: {trainer.psh_action_counts[1]}次 ({trainer.psh_action_counts[1]/total_actions*100:.1f}%)")
        report.append(f"- 抽水: {trainer.psh_action_counts[2]}次 ({trainer.psh_action_counts[2]/total_actions*100:.1f}%)")
    
    report.append(f"\n### 电压统计")
    report.append(f"- 平均电压越限: {np.mean(trainer.episode_voltage_violations):.1f}次/轮")
    
    if len(trainer.agent.episode_actor_losses) > 0:
        actor_losses = [l for l in trainer.agent.episode_actor_losses if not np.isnan(l)]
        critic_losses = [l for l in trainer.agent.episode_critic_losses if not np.isnan(l)]
        if actor_losses and critic_losses:
            report.append(f"\n### 损失统计")
            report.append(f"- 最终Actor损失: {actor_losses[-1]:.6f}")
            report.append(f"- 最终Critic损失: {critic_losses[-1]:.6f}")
    
    report.append(f"\n## 版本5.0 关键改进")
    report.append(f"\n1. 混合动作空间")
    report.append(f"   - 离散动作: PSH启停决策")
    report.append(f"   - 连续动作: PSH功率比例(0-100%) + BESS充放电")
    report.append(f"\n2. 真实潮流计算")
    report.append(f"   - 接入pandapower进行AC潮流计算")
    report.append(f"   - 考虑R/X比，不做PQ解耦假设")
    report.append(f"\n3. 物理成本基础奖励")
    report.append(f"   - 统一€量纲")
    report.append(f"   - 包含BESS退化成本")
    report.append(f"\n4. 水力耦合模型")
    report.append(f"   - 水位差ΔH随SOC变化")
    report.append(f"   - 上下游水流时间延迟")
    report.append(f"\n5. 域随机化")
    report.append(f"   - 电价不确定性 (5%噪声)")
    report.append(f"   - 光伏噪声 (3%)")
    report.append(f"\n6. 消除观察泄露")
    report.append(f"   - 从状态中移除valid_actions信息")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    save_dir = os.environ.get('SAVE_DIR', os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(save_dir, "FINAL_REPORT.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n最终报告已保存到 {report_path}")


def rule_based_baseline(env, num_episodes: int = 5):
    """
    规则策略基线:
    - 光伏大发时(PSH抽水/BESS充电)
    - 晚高峰时(PSH发电/BESS放电)
    - 电价低时充电，电价高时放电
    """
    print("\n" + "=" * 70)
    print("规则策略基线评估")
    print("=" * 70)
    
    all_rewards = []
    all_violations = []
    all_constraint_violations = []
    all_renewable_rates = []
    
    for episode in range(num_episodes):
        state = env.reset(reset_psh_storage=False)
        episode_reward_sum = 0.0
        episode_steps = 0
        episode_violations = 0
        episode_constraint_violations = 0
        episode_renewable_total = 0.0
        episode_renewable_consumed = 0.0
        
        for step in range(env.episode_length):
            # 从状态中解析信息
            net_load = state[:34] * 10.0  # 反归一化
            price_norm = state[34]
            price = price_norm * 100.0
            
            # 获取当前时刻的可再生能源
            t = env.current_time
            renewable_p = env.renewable_data.iloc[t].values[:env.n_nodes]
            total_renewable = np.sum(renewable_p)
            
            # 规则策略
            hour_of_day = (t % 96) / 4.0  # 0-24小时
            
            # V6.4 PSH离散动作: 0=PUMP, 1=GENERATE, 2=STOP
            # PSH功率: 0-1
            # BESS: -1~1
            
            if price > 60 and total_renewable < 2:  # 高电价且可再生能源少 -> 发电
                psh_discrete = 1  # GENERATE
                psh_power = 0.8
                bess1 = 0.8  # 放电
                bess2 = 0.8
            elif price < 30 and total_renewable > 3:  # 低电价且可再生能源多 -> 抽水
                psh_discrete = 0  # PUMP
                psh_power = 0.8
                bess1 = -0.8  # 充电
                bess2 = -0.8
            elif 10 <= hour_of_day <= 16 and total_renewable > 3:  # 白天光伏大发 -> 抽水
                psh_discrete = 0  # PUMP
                psh_power = 0.7
                bess1 = -0.6
                bess2 = -0.6
            elif 18 <= hour_of_day <= 22:  # 晚高峰 -> 发电
                psh_discrete = 1  # GENERATE
                psh_power = 0.9
                bess1 = 0.9
                bess2 = 0.9
            else:
                psh_discrete = 2  # STOP
                psh_power = 0.0
                bess1 = 0.0
                bess2 = 0.0
            
            action = np.array([psh_discrete, psh_power, bess1, bess2])
            next_state, reward, done, info = env.step(action)
            
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
            
            episode_reward_sum += reward
            episode_steps += 1
            episode_violations += info.get('v_violation_count', 0)
            if info.get('psh_constraint_violated', False):
                episode_constraint_violations += 1
            
            episode_renewable_total += info.get('renewable_total', 0)
            episode_renewable_consumed += info.get('renewable_consumed', 0)
            
            state = next_state
            if done:
                break
        
        avg_reward = episode_reward_sum / max(episode_steps, 1)
        renewable_rate = episode_renewable_consumed / max(episode_renewable_total, 1e-6)
        
        all_rewards.append(avg_reward)
        all_violations.append(episode_violations)
        all_constraint_violations.append(episode_constraint_violations)
        all_renewable_rates.append(renewable_rate)
        
        print(f"  Episode {episode + 1}: 奖励={avg_reward:.4f}, 电压越限={episode_violations}, "
              f"PSH约束违反={episode_constraint_violations}, 消纳率={renewable_rate*100:.1f}%")
    
    print(f"\n规则策略平均结果:")
    print(f"  平均奖励: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"  平均电压越限: {np.mean(all_violations):.2f}")
    print(f"  平均PSH约束违反: {np.mean(all_constraint_violations):.2f}")
    print(f"  平均可再生能源消纳率: {np.mean(all_renewable_rates)*100:.1f}%")
    
    return {
        'rewards': all_rewards,
        'violations': all_violations,
        'constraint_violations': all_constraint_violations,
        'renewable_rates': all_renewable_rates
    }


def test_random_days(model_path: str, num_days: int = 7, data_dir: str = None):
    """
    随机抽取指定天数进行测试，分析可再生能源消纳率
    
    Args:
        model_path: 模型路径
        num_days: 测试天数(1天窗口数)
        data_dir: 数据目录
    """
    if data_dir is None:
        data_dir = current_dir
    
    print("\n" + "=" * 70)
    print(f"随机抽取{num_days}个1天窗口进行测试 - 可再生能源消纳率分析")
    print("=" * 70)
    
    env = create_environment(data_dir)
    
    # 加载模型
    agent = PPOAgent(
        state_dim=env.state_dim,
        n_discrete_actions=3,
        n_continuous_actions=3,
        hidden_dims=[256, 256],
        device='cpu'
    )
    
    print(f"\n加载模型: {model_path}")
    agent.load(model_path)
    
    # 计算可用的1天起始点
    total_data_points = len(env.load_data)
    max_start = total_data_points - 96  # 96步=1天
    
    if max_start <= 0:
        print("数据不足以抽取1天窗口!")
        return
    
    # 随机选择num_days个不重叠的1天窗口
    np.random.seed(42)
    possible_starts = list(range(0, max_start, 96))  # 每1天一个窗口
    if len(possible_starts) < num_days:
        num_days = len(possible_starts)
    
    selected_starts = np.random.choice(possible_starts, size=num_days, replace=False)
    selected_starts.sort()
    
    print(f"\n随机选择的{num_days}个测试窗口(1天)起始索引: {selected_starts}")
    
    results = []
    
    for day_idx, start_idx in enumerate(selected_starts):
        print(f"\n--- 测试窗口 {day_idx + 1}/{num_days} (起始索引: {start_idx}) ---")
        
        state = env.reset(start_idx=start_idx, reset_psh_storage=False)
        
        # 记录详细数据
        time_points = []
        prices = []
        loads = []
        renewables = []
        psh_powers = []
        bess1_powers = []
        bess2_powers = []
        psh_upper_socs = []
        psh_lower_socs = []
        bess1_socs = []
        bess2_socs = []
        voltages_min = []
        voltages_max = []
        voltage_violations = []
        rewards = []
        renewable_consumed = []
        renewable_total = []
        
        episode_reward_sum = 0.0
        
        for step in range(96):
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
            
            episode_reward_sum += reward
            
            # 记录数据
            time_points.append(step)
            prices.append(info['price'])
            loads.append(info['load_total'])
            renewables.append(info['renewable_total'])
            psh_powers.append(info['psh_power'])
            bess1_powers.append(info['bess_powers'][0])
            bess2_powers.append(info['bess_powers'][1])
            psh_upper_socs.append(info['psh_upper_soc'])
            psh_lower_socs.append(info['psh_lower_soc'])
            bess1_socs.append(info['bess_socs'][0])
            bess2_socs.append(info['bess_socs'][1])
            voltages_min.append(info['voltage_min'])
            voltages_max.append(info['voltage_max'])
            voltage_violations.append(info['v_violation_count'])
            rewards.append(reward)
            renewable_consumed.append(info['renewable_consumed'])
            renewable_total.append(info['renewable_total'])
            
            state = next_state
            if done:
                break
        
        # 计算指标
        avg_reward = episode_reward_sum / len(time_points)
        total_renewable_gen = sum(renewable_total)
        total_renewable_used = sum(renewable_consumed)
        consumption_rate = total_renewable_used / max(total_renewable_gen, 1e-6) * 100
        total_violations = sum(voltage_violations)
        
        # PSH能量统计
        psh_gen_energy = sum(p for p in psh_powers if p > 0) * env.time_step
        psh_pump_energy = abs(sum(p for p in psh_powers if p < 0)) * env.time_step
        bess1_discharge = sum(p for p in bess1_powers if p > 0) * env.time_step
        bess1_charge = abs(sum(p for p in bess1_powers if p < 0)) * env.time_step
        bess2_discharge = sum(p for p in bess2_powers if p > 0) * env.time_step
        bess2_charge = abs(sum(p for p in bess2_powers if p < 0)) * env.time_step
        
        result = {
            'day': day_idx + 1,
            'start_idx': start_idx,
            'avg_reward': avg_reward,
            'consumption_rate': consumption_rate,
            'total_renewable_gen': total_renewable_gen,
            'total_renewable_used': total_renewable_used,
            'voltage_violations': total_violations,
            'psh_gen_mwh': psh_gen_energy,
            'psh_pump_mwh': psh_pump_energy,
            'bess1_discharge_mwh': bess1_discharge,
            'bess1_charge_mwh': bess1_charge,
            'bess2_discharge_mwh': bess2_discharge,
            'bess2_charge_mwh': bess2_charge,
            'time_points': time_points,
            'prices': prices,
            'loads': loads,
            'renewables': renewables,
            'psh_powers': psh_powers,
            'bess1_powers': bess1_powers,
            'bess2_powers': bess2_powers,
            'psh_upper_socs': psh_upper_socs,
            'psh_lower_socs': psh_lower_socs,
            'bess1_socs': bess1_socs,
            'bess2_socs': bess2_socs,
            'voltages_min': voltages_min,
            'voltages_max': voltages_max,
            'rewards': rewards,
        }
        results.append(result)
        
        print(f"  平均奖励: {avg_reward:.4f}")
        print(f"  可再生能源消纳率: {consumption_rate:.1f}%")
        print(f"  总可再生能源发电: {total_renewable_gen:.2f} MWh")
        print(f"  总可再生能源利用: {total_renewable_used:.2f} MWh")
        print(f"  电压越限次数: {total_violations}")
        print(f"  PSH发电量: {psh_gen_energy:.2f} MWh, 抽水量: {psh_pump_energy:.2f} MWh")
        print(f"  BESS1放电: {bess1_discharge:.2f} MWh, 充电: {bess1_charge:.2f} MWh")
        print(f"  BESS2放电: {bess2_discharge:.2f} MWh, 充电: {bess2_charge:.2f} MWh")
    
    # 汇总分析
    print("\n" + "=" * 70)
    print("1天窗口测试汇总分析")
    print("=" * 70)
    
    avg_consumption = np.mean([r['consumption_rate'] for r in results])
    std_consumption = np.std([r['consumption_rate'] for r in results])
    avg_reward = np.mean([r['avg_reward'] for r in results])
    total_renewable_gen = sum([r['total_renewable_gen'] for r in results])
    total_renewable_used = sum([r['total_renewable_used'] for r in results])
    overall_rate = total_renewable_used / max(total_renewable_gen, 1e-6) * 100
    
    print(f"\n总体可再生能源消纳率: {overall_rate:.1f}%")
    print(f"各窗口平均消纳率: {avg_consumption:.1f}% ± {std_consumption:.1f}%")
    print(f"平均奖励: {avg_reward:.4f}")
    print(f"总可再生能源发电: {total_renewable_gen:.2f} MWh")
    print(f"总可再生能源利用: {total_renewable_used:.2f} MWh")
    print(f"\n各窗口消纳率:")
    for r in results:
        print(f"  窗口{r['day']} (索引{r['start_idx']}): {r['consumption_rate']:.1f}%")
    
    # 保存详细结果
    save_test_results(results)
    
    return results


def save_test_results(results: List[Dict]):
    """保存测试结果到CSV"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 汇总表
        summary_data = []
        for r in results:
            summary_data.append({
                '窗口': r['day'],
                '起始索引': r['start_idx'],
                '平均奖励': round(r['avg_reward'], 4),
                '消纳率(%)': round(r['consumption_rate'], 1),
                '可再生能源发电(MWh)': round(r['total_renewable_gen'], 2),
                '可再生能源利用(MWh)': round(r['total_renewable_used'], 2),
                '电压越限次数': r['voltage_violations'],
                'PSH发电(MWh)': round(r['psh_gen_mwh'], 2),
                'PSH抽水(MWh)': round(r['psh_pump_mwh'], 2),
                'BESS1放电(MWh)': round(r['bess1_discharge_mwh'], 2),
                'BESS1充电(MWh)': round(r['bess1_charge_mwh'], 2),
                'BESS2放电(MWh)': round(r['bess2_discharge_mwh'], 2),
                'BESS2充电(MWh)': round(r['bess2_charge_mwh'], 2),
            })
        
        df_summary = pd.DataFrame(summary_data)
        save_dir = os.environ.get('SAVE_DIR', script_dir)
        summary_path = os.path.join(save_dir, "test_1day_summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n1天窗口测试汇总已保存到: {summary_path}")
        
        # 详细时间序列
        for r in results:
            detail_data = []
            for i in range(len(r['time_points'])):
                detail_data.append({
                    '时间步': r['time_points'][i],
                    '电价': round(r['prices'][i], 2),
                    '负荷': round(r['loads'][i], 4),
                    '可再生能源': round(r['renewables'][i], 4),
                    'PSH功率': round(r['psh_powers'][i], 4),
                    'BESS1功率': round(r['bess1_powers'][i], 4),
                    'BESS2功率': round(r['bess2_powers'][i], 4),
                    'PSH上SOC': round(r['psh_upper_socs'][i], 4),
                    'PSH下SOC': round(r['psh_lower_socs'][i], 4),
                    'BESS1_SOC': round(r['bess1_socs'][i], 4),
                    'BESS2_SOC': round(r['bess2_socs'][i], 4),
                    '电压最小值': round(r['voltages_min'][i], 4),
                    '电压最大值': round(r['voltages_max'][i], 4),
                    '奖励': round(r['rewards'][i], 6),
                })
            
            df_detail = pd.DataFrame(detail_data)
            detail_path = os.path.join(save_dir, f"test_1day_window{r['day']}_detail.csv")
            df_detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
        
        print(f"各窗口详细时间序列已保存")
        
    except Exception as e:
        print(f"保存测试结果出错: {e}")
        import traceback
        traceback.print_exc()


def evaluate(model_path: str, num_episodes: int = 10, data_dir: str = None):
    """评估训练好的模型"""
    if data_dir is None:
        data_dir = current_dir
    
    print("\n" + "=" * 70)
    print("评估模型")
    print("=" * 70)
    
    env = create_environment(data_dir)
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        n_discrete_actions=3,
        n_continuous_actions=3,
        hidden_dims=[256, 256],
        device='cpu'
    )
    
    print(f"加载模型: {model_path}")
    agent.load(model_path)
    
    print(f"\n评估 {num_episodes} 个回合...")
    
    all_rewards = []
    all_violations = []
    all_constraint_violations = []
    all_renewable_rates = []
    
    for episode in range(num_episodes):
        state = env.reset(reset_psh_storage=False)
        episode_reward_sum = 0.0
        episode_steps = 0
        episode_violations = 0
        episode_constraint_violations = 0
        episode_renewable_total = 0.0
        episode_renewable_consumed = 0.0
        
        for step in range(env.episode_length):
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
            
            episode_reward_sum += reward
            episode_steps += 1
            episode_violations += info.get('v_violation_count', 0)
            if info.get('psh_constraint_violated', False):
                episode_constraint_violations += 1
            
            episode_renewable_total += info.get('renewable_total', 0)
            episode_renewable_consumed += info.get('renewable_consumed', 0)
            
            state = next_state
            if done:
                break
        
        avg_reward = episode_reward_sum / max(episode_steps, 1)
        renewable_rate = episode_renewable_consumed / max(episode_renewable_total, 1e-6)
        
        all_rewards.append(avg_reward)
        all_violations.append(episode_violations)
        all_constraint_violations.append(episode_constraint_violations)
        all_renewable_rates.append(renewable_rate)
        
        print(f"  Episode {episode + 1}: 奖励={avg_reward:.4f}, 电压越限={episode_violations}, "
              f"PSH约束违反={episode_constraint_violations}, 消纳率={renewable_rate*100:.1f}%")
    
    print(f"\n评估结果:")
    print(f"  平均奖励: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"  平均电压越限次数: {np.mean(all_violations):.2f}")
    print(f"  平均PSH约束违反次数: {np.mean(all_constraint_violations):.2f}")
    print(f"  平均可再生能源消纳率: {np.mean(all_renewable_rates)*100:.1f}%")
    
    return all_rewards, all_violations


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='抽水储能与电池储能协同调度 - 版本5.0 (重构版)')
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['train', 'eval', 'test', 'baseline', 'all'],
        help='运行模式'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='best_model.pth',
        help='模型路径'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=10,
        help='评估回合数'
    )
    parser.add_argument(
        '--num_test_days',
        type=int,
        default=7,
        help='测试天数(1天窗口数)'
    )
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=200,
        help='最大训练轮数'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='数据目录路径'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("抽水储能与电池储能协同调度 - 版本5.0 (重构版)")
    print("=" * 70)
    print("\n重大改进:")
    print("  1. 混合动作空间 (离散+连续)")
    print("  2. pandapower真实潮流计算")
    print("  3. 物理成本基础奖励函数")
    print("  4. 水力耦合模型 + 水流延迟")
    print("  5. 域随机化 (价格+光伏噪声)")
    print("  6. 消除观察空间泄露")
    print("=" * 70)
    
    if args.mode == 'train':
        train(args.max_episodes)
    elif args.mode == 'eval':
        model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(current_dir, args.model_path)
        evaluate(model_path, args.num_eval_episodes, args.data_dir)
    elif args.mode == 'test':
        model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(current_dir, args.model_path)
        test_random_days(model_path, args.num_test_days, args.data_dir)
    elif args.mode == 'baseline':
        env = create_environment(args.data_dir)
        rule_based_baseline(env, args.num_eval_episodes)
    elif args.mode == 'all':
        # 训练
        agent, env = train(args.max_episodes)
        
        # 评估最佳模型
        save_dir = os.environ.get('SAVE_DIR', current_dir)
        best_model_path = os.path.join(save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print("\n" + "=" * 70)
            print("评估最佳模型")
            print("=" * 70)
            evaluate(best_model_path, args.num_eval_episodes, args.data_dir)
            
            # 1天窗口测试
            print("\n" + "=" * 70)
            print("随机1天窗口测试")
            print("=" * 70)
            test_random_days(best_model_path, args.num_test_days, args.data_dir)
        else:
            print(f"未找到最佳模型: {best_model_path}")
        
        # 基线对比
        print("\n" + "=" * 70)
        print("规则策略基线对比")
        print("=" * 70)
        rule_based_baseline(env, 5)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
