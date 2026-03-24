# 马尔可夫决策过程(MDP)建模

## 1. 问题描述

本研究考虑一个改进的IEEE 34节点配电网系统，其中包含：
- **1个抽水储能机组(PSH)**：安装在节点16
- **2个电池储能系统(BESS)**：分别安装在节点12和节点27
- **分布式光伏(PV)**：部分节点安装光伏
- **时变负荷**：各节点有不同的负荷曲线
- **动态电价**：基于时间的电价信号

目标是通过DDPG强化学习算法，实现抽水储能和电池储能的协同优化调度，最小化运行成本同时保证电网安全运行。

## 2. MDP形式化定义

MDP由一个五元组 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ 定义：

### 2.1 状态空间 $\mathcal{S}$

状态向量 $s_t \in \mathbb{R}^{43}$ 包含以下信息：

```
s_t = [P^N_{1,t}, P^N_{2,t}, ..., P^N_{34,t},   # 34节点净负荷 [kW]
       ρ_t,                                      # 电价 [€/MWh]
       SOC^{PSH}_t,                              # PSH荷电状态
       P^{PSH}_t / P^{PSH}_{max},                # PSH功率(归一化)
       Mode^{PSH}_t,                             # PSH运行模式
       SOC^{BESS1}_t,                            # BESS1荷电状态
       P^{BESS1}_t / P^{BESS1}_{max},            # BESS1功率(归一化)
       SOC^{BESS2}_t,                            # BESS2荷电状态
       P^{BESS2}_t / P^{BESS2}_{max},            # BESS2功率(归一化)
       h_t]                                      # 时间特征 [0,1]
```

**状态维度**: 43

**各分量说明**:
- **净负荷** $P^N_{i,t} = P^{load}_{i,t} - P^{PV}_{i,t}$：节点$i$在时刻$t$的净有功功率
- **电价** $\rho_t$：时刻$t$的电力市场价格
- **PSH状态**: SOC、功率、运行模式(发电/抽水/停机)
- **BESS状态**: SOC和功率
- **时间特征** $h_t = (t \mod 96) / 96$：一天中的归一化时间(15分钟分辨率)

### 2.2 动作空间 $\mathcal{A}$

动作向量 $a_t \in \mathbb{R}^3$ 包含三个储能系统的控制指令：

```
a_t = [a^{PSH}_t, a^{BESS1}_t, a^{BESS2}_t]
```

其中每个动作 $a \in [-1, 1]$：
- **$a > 0$**: 放电/发电模式，功率与$a$成正比
- **$a = 0$**: 停机模式
- **$a < 0$**: 充电/抽水模式，功率与$|a|$成正比

**动作映射到功率**:

对于PSH:
```
if a^{PSH}_t > 0:
    P^{PSH}_t = a^{PSH}_t × P^{PSH}_{gen,max}    # 发电模式
elif a^{PSH}_t < 0:
    P^{PSH}_t = a^{PSH}_t × P^{PSH}_{pump,max}   # 抽水模式(负值)
else:
    P^{PSH}_t = 0                                  # 停机
```

对于BESS:
```
if a^{BESS}_t > 0:
    P^{BESS}_t = a^{BESS}_t × P^{BESS}_{max}     # 放电
elif a^{BESS}_t < 0:
    P^{BESS}_t = a^{BESS}_t × P^{BESS}_{max}     # 充电(负值)
else:
    P^{BESS}_t = 0                                 # 停机
```

### 2.3 状态转移函数 $\mathcal{P}$

状态转移包含两个部分：

#### 2.3.1 储能系统动态

**PSH的SOC更新**:

$$E_{t+1} = E_t + \Delta E_t$$

其中:
- 发电模式: $\Delta E_t = -\frac{P^{PSH}_t \cdot \Delta t}{\eta_{gen}}$
- 抽水模式: $\Delta E_t = |P^{PSH}_t| \cdot \Delta t \cdot \eta_{pump}$
- 停机: $\Delta E_t = 0$

$$SOC_{t+1} = \frac{E_{t+1} - E_{min}}{E_{max} - E_{min}}$$

**BESS的SOC更新**:

$$SOC_{t+1} = SOC_t + \frac{\Delta E_t}{E_{cap}}$$

其中:
- 放电: $\Delta E_t = -\frac{P^{BESS}_t \cdot \Delta t}{\eta_{dis}}$
- 充电: $\Delta E_t = |P^{BESS}_t| \cdot \Delta t \cdot \eta_{ch}$

#### 2.3.2 配电网潮流计算

节点注入功率:

$$P_{i,t} = P^{load}_{i,t} - P^{PV}_{i,t} - P^{storage}_{i,t}$$

潮流方程:

$$\mathbf{V}_t = \text{PowerFlow}(\mathbf{P}_t, \mathbf{Q}_t, \mathbf{Y}_{bus})$$

其中:
- $\mathbf{V}_t$: 节点电压向量
- $\mathbf{P}_t, \mathbf{Q}_t$: 节点有功和无功注入
- $\mathbf{Y}_{bus}$: 节点导纳矩阵

**潮流计算采用Tensor Power Flow方法**，相比传统牛顿-拉夫逊法计算速度提升10倍。

#### 2.3.3 外生变量演化

- **负荷**: 从历史数据采样 $P^{load}_t \sim \mathcal{D}_{load}$
- **光伏**: 从历史数据采样 $P^{PV}_t \sim \mathcal{D}_{PV}$
- **电价**: 从历史数据采样 $\rho_t \sim \mathcal{D}_{price}$

### 2.4 奖励函数 $\mathcal{R}$

奖励函数设计为能量套利收益减去各种惩罚项：

$$r_t = r^{arbitrage}_t + r^{voltage}_t + r^{convergence}_t$$

#### 2.4.1 能量套利收益

$$r^{arbitrage}_t = \rho_t \cdot \left( \sum_{m \in \mathcal{B}} P^m_t \right) \cdot \Delta t$$

其中:
- $\mathcal{B}$: 所有储能系统的集合
- $P^m_t > 0$: 储能$m$放电/发电(向电网售电，获得收益)
- $P^m_t < 0$: 储能$m$充电/抽水(从电网购电，付出成本)

#### 2.4.2 电压越限惩罚

$$r^{voltage}_t = -\sigma \cdot \sum_{i \in \mathcal{N}} C(V_{i,t})$$

其中惩罚函数:

$$
C(V_{i,t}) = \begin{cases}
V_{min} - V_{i,t}, & \text{if } V_{i,t} < V_{min} \\
V_{i,t} - V_{max}, & \text{if } V_{i,t} > V_{max} \\
0, & \text{otherwise}
\end{cases}
$$

参数:
- $\sigma = 400$: 惩罚系数
- $V_{min} = 0.95$ p.u., $V_{max} = 1.05$ p.u.

#### 2.4.3 潮流不收敛惩罚

$$r^{convergence}_t = \begin{cases}
-1000, & \text{if 潮流不收敛} \\
0, & \text{otherwise}
\end{cases}$$

### 2.5 折扣因子 $\gamma$

$$\gamma = 0.995$$

表示智能体考虑长期回报，但更注重近期收益。

## 3. 抽水储能特殊约束建模

### 3.1 运行模式约束

PSH有三种互斥的运行模式：

$$Mode_t \in \{0: \text{停机}, 1: \text{发电}, -1: \text{抽水}\}$$

**模式切换约束**:
- 不能直接从发电模式切换到抽水模式
- 必须先经过停机状态
- 爬坡约束限制模式切换速度

### 3.2 功率约束

**发电模式**:

$$P^{PSH}_{gen,min} \leq P^{PSH}_t \leq P^{PSH}_{gen,max}$$

**抽水模式**:

$$-P^{PSH}_{pump,max} \leq P^{PSH}_t \leq -P^{PSH}_{pump,min}$$

**定速机组特性**:
- 抽水功率通常为离散值(满功率抽水)
- 发电功率可连续调节

### 3.3 库容(SOC)约束

$$E_{min} \leq E_t \leq E_{max}$$

$$0 \leq SOC_t \leq 1$$

**SOC依赖的功率限制**:
- 低SOC时限制发电功率
- 高SOC时限制抽水功率

### 3.4 爬坡约束

$$|P^{PSH}_t - P^{PSH}_{t-1}| \leq \Delta P^{PSH}_{ramp}$$

防止功率剧烈变化，保护机组设备。

### 3.5 效率模型

**发电效率** $\eta_{gen} = 0.85$:

$$E_{out} = E_{stored} \cdot \eta_{gen}$$

**抽水效率** $\eta_{pump} = 0.85$:

$$E_{stored} = E_{in} \cdot \eta_{pump}$$

**往返效率**:

$$\eta_{round} = \eta_{gen} \cdot \eta_{pump} = 0.7225$$

## 4. 电池储能约束建模

### 4.1 功率约束

$$-P^{BESS}_{max} \leq P^{BESS}_t \leq P^{BESS}_{max}$$

### 4.2 SOC约束

$$SOC_{min} \leq SOC_t \leq SOC_{max}$$

典型值: $SOC_{min} = 0.2$, $SOC_{max} = 0.9$

### 4.3 充放电效率

$$\eta_{ch} = \eta_{dis} = 0.95$$

### 4.4 爬坡约束

$$|P^{BESS}_t - P^{BESS}_{t-1}| \leq \Delta P^{BESS}_{ramp}$$

## 5. 算法实现细节

### 5.1 DDPG网络结构

**Actor网络**:
```
输入: state (43维)
  ↓
全连接层: 43 → 256 (ReLU)
  ↓
全连接层: 256 → 256 (ReLU)
  ↓
输出层: 256 → 3 (Tanh)
  ↓
输出: action (3维, [-1, 1])
```

**Critic网络**:
```
输入1: state (43维)
  ↓
全连接层: 43 → 256 (ReLU)
  ↓
拼接: 256 + action (3维) = 259维
  ↓
全连接层: 259 → 256 (ReLU)
  ↓
输出层: 256 → 1
  ↓
输出: Q-value
```

### 5.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Actor学习率 | 3e-4 | 策略网络学习率 |
| Critic学习率 | 3e-4 | 价值网络学习率 |
| 折扣因子 $\gamma$ | 0.995 | 长期回报权重 |
| 软更新系数 $\tau$ | 0.005 | 目标网络更新速度 |
| 回放缓冲区 | 100,000 | 经验存储容量 |
| 批量大小 | 256 | 每次更新样本数 |
| 初始噪声 | 0.3 | Ornstein-Uhlenbeck噪声初始标准差 |
| 噪声衰减 | 0.995 | 每回合噪声衰减系数 |
| 最小噪声 | 0.01 | 噪声标准差下限 |

### 5.3 训练流程

```
对于每个回合:
    1. 重置环境，获取初始状态s_0
    2. 对于每个时间步t:
        a. Actor选择动作: a_t = μ(s_t) + noise
        b. 环境执行动作，返回 (r_t, s_{t+1}, done)
        c. 存储经验 (s_t, a_t, r_t, s_{t+1}, done)
        d. 从回放缓冲区采样批量经验
        e. 更新Critic网络 (最小化Bellman误差)
        f. 更新Actor网络 (最大化Q值)
        g. 软更新目标网络
    3. 衰减探索噪声
    4. 记录训练统计
```

## 6. 与RL-ADN框架的对比

| 特性 | RL-ADN原始框架 | 本研究改进 |
|------|---------------|-----------|
| 储能类型 | 4个电池储能 | 1个抽水储能 + 2个电池储能 |
| 储能位置 | 节点12, 16, 27, 34 | 节点16(PSH), 节点12, 27(BESS) |
| 动作空间 | 4维 | 3维 |
| 状态空间 | 38维 | 43维(增加PSH模式) |
| 约束建模 | SOC、功率 | SOC、功率、效率、运行模式、爬坡 |
| 应用场景 | 电池储能调度 | 抽水储能与电池储能协同调度 |

## 7. 参考文献

[1] Hou S, et al. RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks. Energy and AI, 2025.

[2] Lillicrap T P, et al. Continuous control with deep reinforcement learning. ICLR, 2016.

[3] Huang B, et al. A Computational Efficient Pumped Storage Hydro Optimization in the Look-ahead Unit Commitment and Real-time Market Dispatch Under Uncertainty. IEEE Transactions on Power Systems, 2023.

[4] Yang J, et al. Data-driven Optimal Dynamic Dispatch for Hydro-PV-PHS Integrated Power Systems Using Deep Reinforcement Learning Approach. IEEE Transactions on Sustainable Energy, 2022.

[5] Kersting W H. Radial distribution test feeders. IEEE Transactions on Power Systems, 1991.

[6] 考虑抽蓄的风光水火多能互补双层优化调度. 长江科学院院报, 2025.

[7] 抽水蓄能电站运行模拟与优化方法研究. 中国电机工程学报, 2022.

[8] A Configuration Based Pumped Storage Hydro Model in Look-ahead Unit Commitment. arXiv:2009.04944, 2020.
