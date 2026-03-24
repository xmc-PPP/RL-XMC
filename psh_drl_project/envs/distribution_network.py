"""
改进的IEEE 34节点配电网环境
包含1个抽水储能机组和2个普通电池储能系统

基于文献[6]的RL-ADN框架和文献[7-9]的配电网建模方法

参考文献:
[6] Hou S, et al. RL-ADN: A high-performance Deep Reinforcement Learning environment
    for optimal Energy Storage Systems dispatch in active distribution networks.
    Energy and AI, 2025.
[7] Kersting W H. Radial distribution test feeders. IEEE Transactions on Power Systems, 1991.
[8] 基于功率平衡的风-光-抽水蓄能电站复合系统容量规划仿真研究. 电力系统保护与控制, 2024.
[9] Farzin H, et al. Multi-Objective Optimization of Mobile Battery Energy Storage and
    Dynamic Feeder Reconfiguration. Energies, 2025.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import gym
from gym import spaces
import sys

sys.path.append('/mnt/okcomputer/output/psh_drl_project')

from models.pumped_storage import PumpedStorageUnit, BatteryEnergyStorageSystem


class TensorPowerFlow:
    """
    基于张量的快速潮流计算
    基于文献[10]的Tensor Power Flow方法

    参考文献:
    [10] Gao S, et al. Tensor Power Flow: A scalable and efficient power flow solver
         for distribution networks. 2023.
    """

    def __init__(self, node_data: pd.DataFrame, line_data: pd.DataFrame):
        """
        初始化潮流计算器

        Args:
            node_data: 节点数据，包含bus_id, type, Pd, Qd, Vmax, Vmin
            line_data: 线路数据，包含from_bus, to_bus, r, x, b
        """
        self.node_data = node_data
        self.line_data = line_data
        self.n_nodes = len(node_data)
        self.n_lines = len(line_data)

        # 构建导纳矩阵
        self._build_admittance_matrix()

        # 节点类型: 0=PQ, 1=PV, 2=Slack
        self.node_types = node_data['type'].values
        self.slack_bus = np.where(self.node_types == 2)[0][0]

        # 电压限制
        self.v_max = node_data['Vmax'].values
        self.v_min = node_data['Vmin'].values

    def _build_admittance_matrix(self):
        """构建节点导纳矩阵"""
        self.Ybus = np.zeros((self.n_nodes, self.n_nodes), dtype=complex)

        for _, line in self.line_data.iterrows():
            i = int(line['from_bus']) - 1
            j = int(line['to_bus']) - 1

            r = line['r']
            x = line['x']
            b = line.get('b', 0)

            z = r + 1j * x
            y = 1 / z

            self.Ybus[i, i] += y + 1j * b / 2
            self.Ybus[j, j] += y + 1j * b / 2
            self.Ybus[i, j] -= y
            self.Ybus[j, i] -= y

    def solve(self, P: np.ndarray, Q: np.ndarray, V0: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        求解潮流

        Args:
            P: 节点有功注入 [MW]
            Q: 节点无功注入 [Mvar]
            V0: 初始电压 [p.u.]

        Returns:
            V: 节点电压幅值 [p.u.]
            theta: 节点电压相角 [rad]
            S_branch: 支路功率 [MW + j Mvar]
            converged: 是否收敛
        """
        if V0 is None:
            V = np.ones(self.n_nodes, dtype=complex)
        else:
            V = V0.astype(complex)

        # 简化计算：使用线性化潮流近似
        # 对于配电网，可以近似认为 V ≈ 1 - (R*P + X*Q)/V0

        # 提取PQ节点
        pq_buses = np.where(self.node_types == 0)[0]

        # 构建简化雅可比矩阵
        G = self.Ybus.real
        B = self.Ybus.imag

        # 迭代求解
        max_iter = 10
        tol = 1e-6
        converged = False

        for iteration in range(max_iter):
            # 计算功率不匹配
            V_mag = np.abs(V)
            V_angle = np.angle(V)

            # 计算注入功率
            I = self.Ybus @ V
            S_calc = V * np.conj(I)
            P_calc = S_calc.real
            Q_calc = S_calc.imag

            # 功率不匹配
            dP = P - P_calc
            dQ = Q - Q_calc

            # 检查收敛
            if np.max(np.abs(dP[pq_buses])) < tol and np.max(np.abs(dQ[pq_buses])) < tol:
                converged = True
                break

            # 简化更新(仅用于配电网快速计算)
            # 实际应用中应使用完整的牛顿-拉夫逊法
            if iteration < max_iter - 1:
                # 电压更新
                for i in pq_buses:
                    if V_mag[i] > 0:
                        V[i] += (dP[i] + 1j * dQ[i]) / np.conj(V[i]) * 0.1

        # 计算支路功率
        S_branch = self._calculate_branch_power(V)

        return np.abs(V), np.angle(V), S_branch, converged

    def _calculate_branch_power(self, V: np.ndarray) -> np.ndarray:
        """计算支路功率"""
        S_branch = np.zeros(self.n_lines, dtype=complex)

        for idx, (_, line) in enumerate(self.line_data.iterrows()):
            i = int(line['from_bus']) - 1
            j = int(line['to_bus']) - 1

            r = line['r']
            x = line['x']
            z = r + 1j * x

            I_branch = (V[i] - V[j]) / z
            S_branch[idx] = V[i] * np.conj(I_branch)

        return S_branch

    def check_voltage_violations(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检查电压越限

        Returns:
            v_violations: 电压越限节点索引
            v_violation_magnitude: 电压越限幅度
        """
        v_upper_violation = V > self.v_max
        v_lower_violation = V < self.v_min

        v_violations = np.where(v_upper_violation | v_lower_violation)[0]

        v_violation_magnitude = np.zeros(self.n_nodes)
        v_violation_magnitude[v_upper_violation] = V[v_upper_violation] - self.v_max[v_upper_violation]
        v_violation_magnitude[v_lower_violation] = self.v_min[v_lower_violation] - V[v_lower_violation]

        return v_violations, v_violation_magnitude


class DistributionNetworkEnv(gym.Env):
    """
    改进的34节点配电网环境
    包含: 1个抽水储能机组 + 2个普通电池储能系统

    储能配置:
    - 节点16: 抽水储能机组(PSH)
    - 节点12: 电池储能系统1(BESS1)
    - 节点27: 电池储能系统2(BESS2)
    """

    def __init__(
            self,
            node_file: str,
            line_file: str,
            load_data: pd.DataFrame,
            price_data: pd.DataFrame,
            pv_data: Optional[pd.DataFrame] = None,
            psh_config: Optional[Dict] = None,
            bess_configs: Optional[List[Dict]] = None,
            time_step: float = 0.25,  # 15分钟
            episode_length: int = 96  # 24小时 = 96个15分钟
    ):
        """
        初始化环境

        Args:
            node_file: 节点数据文件路径
            line_file: 线路数据文件路径
            load_data: 负荷数据DataFrame
            price_data: 电价数据DataFrame
            pv_data: 光伏数据DataFrame (可选)
            psh_config: 抽水储能配置
            bess_configs: 电池储能配置列表
            time_step: 时间步长 [小时]
            episode_length: 每个episode的长度
        """
        super().__init__()

        # 加载网络数据
        self.node_data = pd.read_csv(node_file)
        self.line_data = pd.read_csv(line_file)

        self.n_nodes = len(self.node_data)
        self.time_step = time_step
        self.episode_length = episode_length

        # 初始化潮流计算器
        self.power_flow = TensorPowerFlow(self.node_data, self.line_data)

        # 数据
        self.load_data = load_data
        self.price_data = price_data
        self.pv_data = pv_data if pv_data is not None else pd.DataFrame()

        # 初始化储能系统
        self._init_energy_storages(psh_config, bess_configs)

        # 状态空间和动作空间
        self._setup_spaces()

        # 当前状态
        self.current_step = 0
        self.current_time = 0
        self.episode_start_idx = 0

        # 历史记录
        self.voltage_history = []
        self.reward_history = []
        self.action_history = []

        # 归一化参数（新增）
        self.max_load = 500.0  # kW，用于归一化
        self.max_price = 100.0  # €/MWh，用于归一化

    def _init_energy_storages(self, psh_config: Optional[Dict], bess_configs: Optional[List[Dict]]):
        """初始化储能系统"""

        # 默认抽水储能配置
        if psh_config is None:
            psh_config = {
                'unit_id': 1,
                'node_id': 16,
                'max_generation_power': 50.0,  # MW
                'min_generation_power': 10.0,  # MW
                'max_pumping_power': 50.0,  # MW
                'min_pumping_power': 10.0,  # MW
                'max_reservoir_capacity': 200.0,  # MWh
                'min_reservoir_capacity': 0.0,  # MWh
                'generation_efficiency': 0.85,  # 发电效率
                'pumping_efficiency': 0.85,  # 抽水效率
                'initial_soc': 0.5,  # 初始SOC
                'ramp_rate_limit': 20.0,  # MW/时间步
                'is_variable_speed': False
            }

        # 默认电池储能配置
        if bess_configs is None:
            bess_configs = [
                {
                    'unit_id': 2,
                    'node_id': 12,
                    'max_power': 50.0,  # MW
                    'capacity': 100.0,  # MWh
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                    'charge_efficiency': 0.95,
                    'discharge_efficiency': 0.95,
                    'initial_soc': 0.5,
                    'ramp_rate_limit': 25.0  # MW/时间步
                },
                {
                    'unit_id': 3,
                    'node_id': 27,
                    'max_power': 50.0,  # MW
                    'capacity': 100.0,  # MWh
                    'min_soc': 0.2,
                    'max_soc': 0.9,
                    'charge_efficiency': 0.95,
                    'discharge_efficiency': 0.95,
                    'initial_soc': 0.5,
                    'ramp_rate_limit': 25.0  # MW/时间步
                }
            ]

        # 创建抽水储能
        self.psh = PumpedStorageUnit(**psh_config, time_step=self.time_step)

        # 创建电池储能
        self.bess_units = []
        for config in bess_configs:
            self.bess_units.append(BatteryEnergyStorageSystem(**config, time_step=self.time_step))

        # 储能系统列表
        self.energy_storages = [self.psh] + self.bess_units
        self.n_storages = len(self.energy_storages)

    def _setup_spaces(self):
        """设置状态空间和动作空间"""

        # 状态空间: [净负荷(34节点), 电价, PSH_SOC, PSH功率, PSH模式, BESS1_SOC, BESS1功率, BESS2_SOC, BESS2功率, 时间]
        # 状态维度 = 34 + 1 + 3 + 2 + 2 + 1 = 43
        self.state_dim = 43

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # 动作空间: [PSH动作, BESS1动作, BESS2动作]
        # 每个动作范围[-1, 1]
        self.action_dim = 3

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """
        重置环境

        Args:
            start_idx: 起始时间索引

        Returns:
            state: 初始状态
        """
        if start_idx is None:
            # 随机选择起始点
            max_start = len(self.load_data) - self.episode_length - 1
            self.episode_start_idx = np.random.randint(0, max(max_start, 1))
        else:
            self.episode_start_idx = start_idx

        self.current_step = 0
        self.current_time = self.episode_start_idx

        # 重置储能系统
        self.psh.reset()
        for bess in self.bess_units:
            bess.reset()

        # 清空历史
        self.voltage_history = []
        self.reward_history = []
        self.action_history = []

        # 获取初始状态
        state = self._get_state()

        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一个时间步

        Args:
            action: 动作数组 [PSH动作, BESS1动作, BESS2动作]

        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 裁剪动作到[-1, 1]
        action = np.clip(action, -1.0, 1.0)
        self.action_history.append(action.copy())

        # 执行储能动作
        psh_power, psh_soc, psh_info = self.psh.step(action[0])
        bess_powers = []
        bess_socs = []
        for i, bess in enumerate(self.bess_units):
            power, soc, info = bess.step(action[i + 1])
            bess_powers.append(power)
            bess_socs.append(soc)

        # 获取当前时间步的数据
        # 修改1：单位转换 kW → MW（关键修复）
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes] / 1000.0
        price = self.price_data.iloc[self.current_time]['price']

        # 如果有光伏数据，减去光伏出力
        # 修改2：单位转换 kW → MW（关键修复）
        if not self.pv_data.empty:
            pv_p = self.pv_data.iloc[self.current_time].values[:self.n_nodes] / 1000.0
            net_load = load_p - pv_p
        else:
            net_load = load_p

        # 更新节点注入功率(考虑储能)
        node_p = net_load.copy()

        # 抽水储能在节点16
        psh_node_idx = self.psh.node_id - 1
        node_p[psh_node_idx] -= psh_power  # 发电为正(向电网注入功率)

        # 电池储能
        for bess, power in zip(self.bess_units, bess_powers):
            bess_node_idx = bess.node_id - 1
            node_p[bess_node_idx] -= power  # 放电为正

        # 运行潮流计算
        node_q = np.zeros(self.n_nodes)  # 假设无功为0
        V, theta, S_branch, converged = self.power_flow.solve(node_p, node_q)

        # 检查电压越限
        v_violations, v_violation_mag = self.power_flow.check_voltage_violations(V)

        # 计算奖励
        reward = self._calculate_reward(
            price, psh_power, bess_powers,
            v_violation_mag, converged
        )

        # 记录历史
        self.voltage_history.append(V.copy())
        self.reward_history.append(reward)

        # 更新时间
        self.current_step += 1
        self.current_time += 1

        # 检查是否结束
        done = self.current_step >= self.episode_length

        # 获取下一状态
        next_state = self._get_state()

        # 信息字典
        info = {
            'voltage': V,
            'v_violations': v_violations,
            'v_violation_magnitude': np.sum(v_violation_mag),
            'psh_power': psh_power,
            'psh_soc': psh_soc,
            'bess_powers': bess_powers,
            'bess_socs': bess_socs,
            'converged': converged,
            'price': price
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 获取当前时间步的数据
        # 修改3：单位转换 kW → MW，并进行归一化（关键修复）
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes] / 1000.0

        if not self.pv_data.empty:
            pv_p = self.pv_data.iloc[self.current_time].values[:self.n_nodes] / 1000.0
            net_load = load_p - pv_p
        else:
            net_load = load_p

        price = self.price_data.iloc[self.current_time]['price']

        # 储能状态
        psh_state = self.psh.get_state()
        bess_states = []
        for bess in self.bess_units:
            bess_states.extend(bess.get_state())

        # 时间特征
        hour = (self.current_time % 96) / 96.0  # 归一化到[0, 1]

        # 归一化处理（关键修复）
        net_load_norm = net_load / (self.max_load / 1000.0)  # 转为MW后归一化，假设最大负荷500kW即0.5MW
        price_norm = price / self.max_price  # 电价归一化到[0,1]

        # 组合状态
        state = np.concatenate([
            net_load_norm.astype(np.float32),
            [price_norm],
            psh_state,  # SOC和功率已经归一化
            bess_states,
            [hour]
        ])

        return state

    def _calculate_reward(
            self,
            price: float,
            psh_power: float,
            bess_powers: List[float],
            v_violation_mag: np.ndarray,
            converged: bool
    ) -> float:
        """
        计算奖励

        奖励 = 能量套利收益 - 电压越限惩罚 - 不收敛惩罚

        参考文献:
        [6] Hou S, et al. RL-ADN: A high-performance Deep Reinforcement Learning
            environment for optimal Energy Storage Systems dispatch in active
            distribution networks. Energy and AI, 2025.
        """
        # 能量套利收益
        # PSH: 发电为正(卖电获得收益)，抽水为负(买电付出成本)
        psh_revenue = price * psh_power * self.time_step

        # BESS
        bess_revenue = sum(price * power * self.time_step for power in bess_powers)

        total_revenue = psh_revenue + bess_revenue

        # 电压越限惩罚
        voltage_penalty_coef = 400.0  # 惩罚系数
        voltage_penalty = -voltage_penalty_coef * np.sum(v_violation_mag)

        # 不收敛惩罚
        convergence_penalty = 0.0 if converged else -1000.0

        # 总奖励
        reward = total_revenue + voltage_penalty + convergence_penalty

        # 修改4：奖励缩放，防止数值过大导致梯度爆炸（关键修复）
        reward_scale = 1e-4
        reward = reward * reward_scale

        return reward

    def render(self, mode='human'):
        """渲染环境状态"""
        if len(self.voltage_history) == 0:
            return

        print(f"Step: {self.current_step}")
        print(f"Voltage range: [{np.min(self.voltage_history[-1]):.4f}, {np.max(self.voltage_history[-1]):.4f}]")
        print(f"PSH SOC: {self.psh.current_soc:.3f}, Power: {self.psh.current_power:.2f} MW")
        for i, bess in enumerate(self.bess_units):
            print(f"BESS{i + 1} SOC: {bess.current_soc:.3f}, Power: {bess.current_power:.2f} MW")
        print(f"Reward: {self.reward_history[-1]:.2f}")

    def get_storage_states(self) -> Dict:
        """获取所有储能系统的状态"""
        states = {
            'psh': {
                'soc': self.psh.current_soc,
                'power': self.psh.current_power,
                'mode': self.psh.current_mode
            },
            'bess': []
        }
        for bess in self.bess_units:
            states['bess'].append({
                'soc': bess.current_soc,
                'power': bess.current_power
            })
        return states
