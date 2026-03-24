# 参考文献

## 核心参考文献

### [1] RL-ADN框架 (本项目基础架构)

Hou S, Gao S, Xia W, et al. **RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks**. Energy and AI, 2025, 19: 100457.

- **链接**: https://github.com/ShengrenHou/RL-ADN
- **贡献**: 提供了DRL在配电网储能调度的开源框架，包括Tensor Power Flow、GMC数据增强等
- **引用内容**: 环境架构、MDP设计、Tensor Power Flow算法、DDPG实现

### [2] DDPG算法

Lillicrap T P, Hunt J J, Pritzel A, et al. **Continuous control with deep reinforcement learning**. International Conference on Learning Representations (ICLR), 2016.

- **链接**: https://arxiv.org/abs/1509.02971
- **贡献**: 提出了DDPG算法，用于连续动作空间的强化学习
- **引用内容**: Actor-Critic网络结构、软更新、经验回放

## 抽水储能建模文献

### [3] 抽水储能优化调度 (LAC和实时市场)

Huang B, Ghesmati A, Chen Y, et al. **A Computational Efficient Pumped Storage Hydro Optimization in the Look-ahead Unit Commitment and Real-time Market Dispatch Under Uncertainty**. IEEE Transactions on Power Systems, 2023, 38(4): 3452-3463.

- **链接**: https://arxiv.org/abs/2304.03821
- **贡献**: 提出了考虑不确定性的抽水储能优化模型
- **引用内容**: PSH约束建模、SOC约束、效率模型、运行模式约束

### [4] 抽水储能在水电-光伏-储能联合系统中的应用

Yang J, Liu J, Xiang Y, et al. **Data-driven Optimal Dynamic Dispatch for Hydro-PV-PHS Integrated Power Systems Using Deep Reinforcement Learning Approach**. IEEE Transactions on Sustainable Energy, 2022, 13(2): 846-858.

- **链接**: https://ieeexplore.ieee.org/document/9862581
- **贡献**: 使用DDPG算法优化抽水储能调度
- **引用内容**: DDPG在PHS调度中的应用、MDP设计、多目标优化

### [5] 抽水蓄能电站运行模拟与优化

**抽水蓄能电站运行模拟与优化方法研究**. 中国电机工程学报, 2022.

- **贡献**: 提供了抽水蓄能电站的详细建模方法
- **引用内容**: 日运行优化、水量平衡、库容约束

### [6] 考虑抽蓄的风光水火多能互补双层优化调度

**考虑抽蓄的风光水火多能互补双层优化调度**. 长江科学院院报, 2025, 42(10): 38-45.

- **链接**: http://ckyyb.crsri.cn/CN/Y2025/V42/I10/38
- **贡献**: 抽水蓄能在多能互补系统中的应用
- **引用内容**: PSH约束条件、水量平衡、库容约束

### [7] 抽水蓄能机组配置模型

**A Configuration Based Pumped Storage Hydro Model in Look-ahead Unit Commitment**. arXiv:2009.04944, 2020.

- **链接**: https://arxiv.org/abs/2009.04944
- **贡献**: 基于配置的抽水储能模型
- **引用内容**: SOC约束、能量平衡、效率模型

### [8] 抽水蓄能电站日前调度模型

**Research on Day-ahead Scheduling of Pumped Storage Power Station Considering Rotating Reserve Requirements**. 武汉大学学报(工学版), 2024.

- **链接**: https://irrigate.whu.edu.cn/CN/10.12396/znsd.231831
- **贡献**: 考虑旋转备用的PSH日前调度
- **引用内容**: 功率约束、旋转备用、日电量平衡

### [9] 变速抽水蓄能机组

**Enhancement of Frequency Regulation in AC-Excited Adjustable-Speed Pumped Storage Units during Pumping Operations**. Energy Engineering, 2025, 122(12): 5175-5197.

- **链接**: https://www.techscience.com/energy/v122n12/64603
- **贡献**: 变速抽水蓄能机组的频率调节
- **引用内容**: 变速机组建模、DRL应用

### [10] 抽水蓄能调度优化

**Short-Term Optimal Scheduling of Pumped-Storage Units via DDPG with AOS-LSTM Flow-Curve Fitting**. Water, 2025, 17(13): 1842.

- **链接**: https://www.mdpi.com/2073-4441/17/13/1842
- **贡献**: 使用DDPG优化抽水蓄能短期调度
- **引用内容**: DDPG算法参数、振动区约束、流量曲线拟合

## 配电网建模文献

### [11] IEEE测试配电网

Kersting W H. **Radial distribution test feeders**. IEEE Transactions on Power Systems, 1991, 6(3): 975-985.

- **贡献**: 提出了IEEE标准测试配电网
- **引用内容**: IEEE 34节点网络拓扑、线路参数

### [12] 移动电池储能在配电网中的应用

Farzin H, et al. **Multi-Objective Optimization of Mobile Battery Energy Storage and Dynamic Feeder Reconfiguration for Enhanced Voltage Profiles in Active Distribution Systems**. Energies, 2025, 18(20): 5515.

- **链接**: https://www.mdpi.com/1996-1073/18/20/5515
- **贡献**: 移动储能在配电网中的应用
- **引用内容**: IEEE 34节点系统应用、储能优化

### [13] 基于功率平衡的规划

**基于功率平衡的风-光-抽水蓄能电站复合系统容量规划仿真研究**. 电力系统保护与控制, 2024.

- **贡献**: 风光抽蓄复合系统规划
- **引用内容**: 功率平衡约束、抽水蓄能调度

## 深度强化学习在电力系统中的应用

### [14] DRL在数据中心绿电集成中的应用

**Deep Reinforcement Learning for Real-Time Green Energy Integration in Data Centers**. arXiv:2507.21153, 2025.

- **贡献**: DRL在能源系统实时优化中的应用
- **引用内容**: DRL框架、优化目标、约束处理

### [15] 多智能体DRL在电压控制中的应用

**Data Driven Real-Time Dynamic Voltage Control Using Decentralized Execution Multi-Agent Deep Reinforcement Learning**. IEEE Transactions on Power Systems, 2025.

- **链接**: https://ieeexplore.ieee.org/document/10679222
- **贡献**: 多智能体DRL在电压控制中的应用
- **引用内容**: DRL在电压控制中的应用、抽水蓄能参与

### [16] 电池储能退化模型

Feng C, et al. **Twin-delayed deep reinforcement learning with learning-rate annealing and hindsight prioritized replay for battery degradation model**. Journal of Energy Storage, 2023.

- **贡献**: 考虑电池退化的DRL优化
- **引用内容**: TD3算法、电池退化建模

## 储能系统建模文献

### [17] 储能系统运行和投资模型

**Operating and Investment Models for Energy Storage Systems**. 

- **链接**: https://pdfs.semanticscholar.org/7e63/913b4fa8d0c149ed1e45a3cdbf16e5259fda.pdf
- **贡献**: 储能系统的运行和投资优化模型
- **引用内容**: 爬坡约束、SOC约束、能量约束

### [18] 云边协同多时间尺度调度

**A Cloud–Edge Collaborative Multi-Timescale Scheduling Strategy for Peak Regulation and Renewable Energy Integration in Distributed Multi-Energy Systems**. Energies, 2024, 17(15): 3764.

- **链接**: https://www.mdpi.com/1996-1073/17/15/3764
- **贡献**: 多时间尺度储能调度
- **引用内容**: 抽水蓄能约束、爬坡约束、库容约束

### [19] 抽水蓄能效率优化

**Boosting Efficiency: Optimizing Pumped-Storage Power Station Operation by a Mixed Integer Linear Programming Approach**. Energies, 2025, 18(18): 4977.

- **链接**: https://www.mdpi.com/1996-1073/18/18/4977
- **贡献**: 抽水蓄能效率优化
- **引用内容**: 效率模型、水-能转换关系、线性化方法

## 多能互补系统文献

### [20] 分布式多能互补系统鲁棒优化

**Robust Optimization Scheduling of Distributed Multi Energy Complementary Power Generation Systems Considering Wind and Solar Uncertainties**. 武汉大学学报(工学版), 2025.

- **链接**: https://irrigate.whu.edu.cn/EN/10.12396/znsd.241909
- **贡献**: 风光储蓄互补系统优化
- **引用内容**: 抽水蓄能约束、风光约束、机会约束

### [21] 梯级水电与抽水蓄能联合运行

**抽水蓄能与常规水电联合运行对下库水位影响研究**. 西安理工大学学报, 2024.

- **贡献**: 梯级水电与抽水蓄能联合运行
- **引用内容**: 水量平衡、库容约束、联合调度

### [22] 源荷储协调滚动调度

**基于灵活性裕度的含风电电力系统源荷储协调滚动调度**. 中国电力, 2024.

- **贡献**: 源荷储协调调度
- **引用内容**: 抽水蓄能建模、虚拟发电机/电动机模型

## 美国电力市场抽水蓄能

### [23] 美国电力市场环境下抽水蓄能调度模式

**美国电力市场环境下抽水蓄能调度模式分析及启示**. 电力系统自动化, 2024.

- **贡献**: 美国电力市场中的PSH调度
- **引用内容**: 发电/抽水窗口、调频市场、备用市场

## 引用格式

如果在研究中使用了本项目，请引用以下文献：

```bibtex
@article{hou2025rladn,
  title={RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks},
  author={Hou, Shengren and Gao, Shuyi and Xia, Weijie and Salazar Duque, Edgar Mauricio and Palensky, Peter and Vergara, Pedro P.},
  journal={Energy and AI},
  volume={19},
  pages={100457},
  year={2025},
  publisher={Elsevier}
}

@inproceedings{lillicrap2016continuous,
  title={Continuous control with deep reinforcement learning},
  author={Lillicrap, Timothy P and Hunt, Jonathan J and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2016}
}

@article{huang2023computational,
  title={A Computational Efficient Pumped Storage Hydro Optimization in the Look-ahead Unit Commitment and Real-time Market Dispatch Under Uncertainty},
  author={Huang, Bing and Ghesmati, Arezou and Chen, Yonghong and Baldick, Ross},
  journal={IEEE Transactions on Power Systems},
  volume={38},
  number={4},
  pages={3452--3463},
  year={2023},
  publisher={IEEE}
}

@article{yang2022datadriven,
  title={Data-driven Optimal Dynamic Dispatch for Hydro-PV-PHS Integrated Power Systems Using Deep Reinforcement Learning Approach},
  author={Yang, Jingxian and Liu, Jichun and Xiang, Yue and Zhang, Shuai and Liu, Junyong},
  journal={IEEE Transactions on Sustainable Energy},
  volume={13},
  number={2},
  pages={846--858},
  year={2022},
  publisher={IEEE}
}
```

## 文献分类索引

### 按主题分类

**抽水储能建模**: [3], [5], [6], [7], [8], [9], [10], [19], [21], [22], [23]

**DDPG算法**: [2], [4], [10], [14], [15], [16]

**配电网环境**: [1], [11], [12], [13]

**储能系统约束**: [17], [18], [20]

**多能互补**: [6], [13], [20], [21]

### 按应用领域分类

**电力市场**: [3], [8], [23]

**频率调节**: [9]

**电压控制**: [1], [12], [15]

**经济调度**: [4], [10]

## 更新日志

- **2025-03-23**: 初始版本，整理核心参考文献
