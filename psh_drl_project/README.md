# 抽水储能与电池储能协同调度 - DDPG强化学习

## 项目概述

本项目基于DDPG(Deep Deterministic Policy Gradient)深度强化学习算法，实现34节点配电网中抽水储能(PSH)与电池储能系统(BESS)的协同优化调度。

### 主要特点

- **混合储能系统**: 1个抽水储能机组 + 2个电池储能系统
- **改进的IEEE 34节点配电网**: 包含分布式光伏和时变负荷
- **Tensor Power Flow**: 快速潮流计算，训练速度提升10倍
- **完整的MDP建模**: 详细的状态空间、动作空间、奖励函数设计
- **抽水储能特殊约束**: 运行模式、效率、库容、爬坡等约束
- **基于RL-ADN框架**: 参考开源库RL-ADN的架构设计

## 项目结构

```
psh_drl_project/
├── algorithms/
│   └── ddpg.py              # DDPG算法实现
├── configs/
│   └── config.py            # 配置文件
├── data/                     # 数据文件(运行时生成)
│   ├── load_data.csv        # 负荷数据
│   ├── pv_data.csv          # 光伏数据
│   ├── price_data.csv       # 电价数据
│   ├── node_data.csv        # 节点数据
│   └── line_data.csv        # 线路数据
├── envs/
│   └── distribution_network.py  # 配电网环境
├── models/
│   └── pumped_storage.py    # 抽水储能和电池储能模型
├── utils/
│   └── data_generator.py    # 数据生成工具
├── main.py                   # 主运行脚本
├── MDP_FORMULATION.md        # MDP建模详细说明
├── REFERENCES.md             # 参考文献
└── README.md                 # 本文件
```

## 环境配置

### 系统要求

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- OpenAI Gym

### 安装依赖

```bash
pip install torch numpy pandas matplotlib gym
```

## 快速开始

### 1. 生成数据

```bash
python main.py --mode generate_data
```

这将生成IEEE 34节点配电网的一年数据(15分钟分辨率)，包括：
- 负荷数据
- 光伏出力数据
- 电价数据
- 网络拓扑数据

### 2. 训练模型

```bash
python main.py --mode train
```

训练参数可在 `configs/config.py` 中修改：
- 最大回合数: 1000
- 每回合步数: 96 (24小时)
- Actor/Critic学习率: 3e-4
- 折扣因子: 0.995

### 3. 评估模型

```bash
python main.py --mode eval --model_path models/ddpg_model.pth --num_eval_episodes 10
```

### 4. 完整流程

```bash
python main.py --mode all
```

这将依次执行数据生成、模型训练和评估。

## 储能系统配置

### 抽水储能(PSH)

- **位置**: 节点16
- **最大发电功率**: 50 MW
- **最大抽水功率**: 50 MW
- **库容**: 200 MWh
- **发电效率**: 85%
- **抽水效率**: 85%
- **爬坡限制**: 20 MW/时间步

### 电池储能1(BESS1)

- **位置**: 节点12
- **最大功率**: 50 MW
- **容量**: 100 MWh
- **SOC范围**: [0.2, 0.9]
- **充放电效率**: 95%
- **爬坡限制**: 25 MW/时间步

### 电池储能2(BESS2)

- **位置**: 节点27
- **最大功率**: 50 MW
- **容量**: 100 MWh
- **SOC范围**: [0.2, 0.9]
- **充放电效率**: 95%
- **爬坡限制**: 25 MW/时间步

## MDP建模

详细的状态空间、动作空间、状态转移和奖励函数设计见 `MDP_FORMULATION.md`。

### 状态空间 (43维)

- 34节点净负荷
- 1维电价
- 3维PSH状态(SOC、功率、模式)
- 4维BESS状态(SOC、功率，2个BESS)
- 1维时间特征

### 动作空间 (3维)

- PSH动作: [-1, 1]
- BESS1动作: [-1, 1]
- BESS2动作: [-1, 1]

### 奖励函数

$$r_t = \underbrace{\rho_t \cdot P^{storage}_t \cdot \Delta t}_{\text{能量套利收益}} - \underbrace{\sigma \cdot \sum C(V_{i,t})}_{\text{电压越限惩罚}} - \underbrace{1000 \cdot \mathbb{1}[\text{不收敛}]}_{\text{不收敛惩罚}}$$

## 算法架构

### Actor网络

```
输入 (43) → FC(256, ReLU) → FC(256, ReLU) → 输出 (3, Tanh)
```

### Critic网络

```
状态 (43) → FC(256, ReLU) → 拼接动作 (3) → FC(256, ReLU) → 输出 (1)
```

### 训练超参数

| 参数 | 值 |
|------|-----|
| Actor学习率 | 3e-4 |
| Critic学习率 | 3e-4 |
| 折扣因子 γ | 0.995 |
| 软更新系数 τ | 0.005 |
| 回放缓冲区 | 100,000 |
| 批量大小 | 256 |

## 结果可视化

训练完成后，将生成以下可视化结果：

1. **训练历史**: 回合奖励、评估奖励、Actor/Critic损失
2. **电压曲线**: 各节点电压随时间变化
3. **储能运行**: PSH和BESS的功率和SOC曲线
4. **协同调度**: 储能系统总功率和各自贡献

## 与RL-ADN的对比

| 特性 | RL-ADN | 本项目 |
|------|--------|--------|
| 储能类型 | 4×BESS | 1×PSH + 2×BESS |
| 状态维度 | 38 | 43 |
| 动作维度 | 4 | 3 |
| 特殊约束 | SOC、功率 | +效率、运行模式、爬坡 |
| 应用场景 | 电池储能 | 抽水储能协同调度 |

## 参考文献

主要参考文献见 `REFERENCES.md`。

核心文献：

1. Hou S, et al. RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks. Energy and AI, 2025.

2. Lillicrap T P, et al. Continuous control with deep reinforcement learning. ICLR, 2016.

3. Huang B, et al. A Computational Efficient Pumped Storage Hydro Optimization. IEEE Transactions on Power Systems, 2023.

4. Yang J, et al. Data-driven Optimal Dynamic Dispatch for Hydro-PV-PHS Integrated Power Systems. IEEE Transactions on Sustainable Energy, 2022.

## 许可证

本项目基于学术研究目的，参考RL-ADN开源框架。

## 联系方式

如有问题或建议，欢迎提出Issue或Pull Request。
