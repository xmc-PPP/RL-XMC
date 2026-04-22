"""
IEEE 34节点配电网环境 - V7.0 (RL-ADN标准版)

V7.0核心修改:
1. 纯RL-ADN奖励函数: economic + voltage_penalty (无reward shaping)
2. PSH动作: PUMP(0)/GENERATE(1)/STOP(2)
3. PSH不重置库容状态
4. 电压控制: [0.95, 1.05] p.u.
5. 状态空间: 34节点电压 + 电价 + PSH状态 + BESS状态 + 时间
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from models.pumped_storage import PumpedStorageUnit, BatteryEnergyStorageSystem, PSHMode, PSHAction
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    sys.path.insert(0, project_dir)
    from models.pumped_storage import PumpedStorageUnit, BatteryEnergyStorageSystem, PSHMode, PSHAction

try:
    import pandapower as pp
    pp.pp_settings = {'numba': False}
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False
    print("警告: pandapower不可用，将使用简化潮流计算")


class PowerFlowCalculator:
    """潮流计算 - 基于pandapower的真实交流潮流"""

    def __init__(self, node_data: pd.DataFrame, line_data: pd.DataFrame):
        self.node_data = node_data
        self.line_data = line_data
        self.n_nodes = len(node_data)
        self.n_lines = len(line_data)
        
        # V7.0: RL-ADN电压限制 [0.95, 1.05] p.u.
        self.v_max = np.ones(self.n_nodes) * 1.05
        self.v_min = np.ones(self.n_nodes) * 0.95
        
        # RL-ADN额定参数
        self.vn_kv = 11.0
        self.s_base_mva = 1.0
        self.z_base = (self.vn_kv ** 2) * 1000.0 / (self.s_base_mva * 1000.0)
        
        self._build_node_connections()
        
        self.net = None
        if PANDAPOWER_AVAILABLE:
            try:
                self._build_pandapower_net()
            except Exception as e:
                print(f"pandapower网络构建失败: {e}，使用回退方法")
                self.net = None
    
    def _build_node_connections(self):
        """构建节点连接关系"""
        self.connections = {i: [] for i in range(self.n_nodes)}
        for _, line in self.line_data.iterrows():
            from_bus = int(line['FROM']) - 1
            to_bus = int(line['TO']) - 1
            r = line['R']
            x = line['X']
            self.connections[from_bus].append((to_bus, r, x))
            self.connections[to_bus].append((from_bus, r, x))
    
    def _build_pandapower_net(self):
        """使用pandapower构建配电网 - RL-ADN标准"""
        self.net = pp.create_empty_network()
        
        for i, row in self.node_data.iterrows():
            bus_idx = int(row['NODES']) - 1
            pp.create_bus(self.net, vn_kv=self.vn_kv, name=f"Bus{bus_idx+1}")
        
        # Slack bus
        slack_bus = 0
        for i, row in self.node_data.iterrows():
            if int(row['Tb']) == 1:
                slack_bus = int(row['NODES']) - 1
                break
        pp.create_ext_grid(self.net, bus=slack_bus, vm_pu=1.0, va_degree=0.0)
        
        # 线路 - R/X直接使用欧姆值
        for _, line in self.line_data.iterrows():
            from_bus = int(line['FROM']) - 1
            to_bus = int(line['TO']) - 1
            r_ohm = line['R']
            x_ohm = line['X']
            
            pp.create_line_from_parameters(
                self.net,
                from_bus=from_bus,
                to_bus=to_bus,
                length_km=1.0,
                r_ohm_per_km=r_ohm,
                x_ohm_per_km=x_ohm,
                c_nf_per_km=0.0,
                max_i_ka=10.0
            )
        
        # 负荷占位符 (无功=0)
        self.load_indices = []
        for i in range(self.n_nodes):
            ld = pp.create_load(self.net, bus=i, p_mw=0.0, q_mvar=0.0, name=f"Load{i+1}")
            self.load_indices.append(ld)
        
        # 储能注入占位符 (无功=0)
        self.gen_indices = []
        for i in range(self.n_nodes):
            sg = pp.create_sgen(self.net, bus=i, p_mw=0.0, q_mvar=0.0, name=f"Storage{i+1}")
            self.gen_indices.append(sg)
    
    def solve(self, P_load: np.ndarray, Q_load: np.ndarray, 
              P_inject: np.ndarray = None, Q_inject: np.ndarray = None,
              max_iter: int = 30, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        """
        求解潮流，返回(V, theta, converged, import_power)
        import_power: 从外部电网注入的总功率(MW)
        """
        if P_inject is None:
            P_inject = np.zeros(self.n_nodes)
        if Q_inject is None:
            Q_inject = np.zeros(self.n_nodes)
        
        if PANDAPOWER_AVAILABLE and self.net is not None:
            return self._solve_pandapower(P_load, Q_load, P_inject, Q_inject)
        else:
            return self._solve_fallback(P_load, Q_load, P_inject, Q_inject, max_iter, tol)
    
    def _solve_pandapower(self, P_load, Q_load, P_inject, Q_inject):
        """使用pandapower求解"""
        try:
            for i in range(self.n_nodes):
                self.net.load.at[self.load_indices[i], 'p_mw'] = max(0, float(P_load[i]))
                self.net.load.at[self.load_indices[i], 'q_mvar'] = max(0, float(Q_load[i]))
            
            for i in range(self.n_nodes):
                net_inject = float(P_inject[i])
                self.net.sgen.at[self.gen_indices[i], 'p_mw'] = max(0, net_inject)
                self.net.sgen.at[self.gen_indices[i], 'q_mvar'] = max(0, float(Q_inject[i]))
            
            pp.runpp(self.net, algorithm='nr', init='flat',
                    max_iteration=30, tolerance_mva=1e-6, verbose=False, numba=False)
            
            V = self.net.res_bus.vm_pu.values.astype(np.float64)
            theta_deg = self.net.res_bus.va_degree.values
            theta = np.deg2rad(theta_deg)
            converged = True
            
            # 计算从外部电网注入的总功率
            import_power = float(self.net.res_ext_grid.p_mw.sum()) if hasattr(self.net, 'res_ext_grid') else 0.0
            
            return V, theta, converged, import_power
        except Exception as e:
            return self._solve_fallback(P_load, Q_load, P_inject, Q_inject)
    
    def _solve_fallback(self, P_load, Q_load, P_inject, Q_inject, max_iter=30, tol=1e-6):
        """回退方法: 改进的牛顿-拉夫逊法"""
        n = self.n_nodes
        
        P_net = P_inject - P_load
        Q_net = Q_inject - Q_load
        
        Y = self._build_ybus()
        G = Y.real
        B = Y.imag
        
        V = np.ones(n, dtype=np.float64)
        theta = np.zeros(n, dtype=np.float64)
        
        slack_bus = 0
        pq_buses = list(range(1, n))
        
        converged = False
        for iteration in range(max_iter):
            dP = np.zeros(n, dtype=np.float64)
            dQ = np.zeros(n, dtype=np.float64)
            
            for i in range(n):
                P_calc = 0.0
                Q_calc = 0.0
                for j in range(n):
                    angle_diff = theta[i] - theta[j]
                    P_calc += V[i] * V[j] * (G[i,j] * np.cos(angle_diff) + B[i,j] * np.sin(angle_diff))
                    Q_calc += V[i] * V[j] * (G[i,j] * np.sin(angle_diff) - B[i,j] * np.cos(angle_diff))
                
                if i != slack_bus:
                    dP[i] = P_net[i] - P_calc
                    dQ[i] = Q_net[i] - Q_calc
            
            max_mismatch = max(np.max(np.abs(dP[1:])), np.max(np.abs(dQ[1:])))
            if max_mismatch < tol:
                converged = True
                break
            
            if len(pq_buses) > 0:
                n_pq = len(pq_buses)
                J_size = n_pq + n_pq
                J = np.zeros((J_size, J_size), dtype=np.float64)
                rhs = np.zeros(J_size, dtype=np.float64)
                
                for idx_i, i in enumerate(pq_buses):
                    rhs[idx_i] = dP[i]
                    rhs[n_pq + idx_i] = dQ[i]
                    
                    for idx_j, j in enumerate(pq_buses):
                        angle_diff = theta[i] - theta[j]
                        
                        if i == j:
                            J[idx_i, idx_i] = -Q_calc - B[i,i] * V[i]**2
                            J[idx_i, n_pq + idx_i] = P_calc / V[i] + G[i,i] * V[i]
                            J[n_pq + idx_i, idx_i] = P_calc - G[i,i] * V[i]**2
                            J[n_pq + idx_i, n_pq + idx_i] = Q_calc / V[i] - B[i,i] * V[i]
                        else:
                            J[idx_i, idx_j] = V[i] * V[j] * (G[i,j] * np.sin(angle_diff) - B[i,j] * np.cos(angle_diff))
                            J[idx_i, n_pq + idx_j] = V[i] * (G[i,j] * np.cos(angle_diff) + B[i,j] * np.sin(angle_diff))
                            J[n_pq + idx_i, idx_j] = -V[i] * V[j] * (G[i,j] * np.cos(angle_diff) + B[i,j] * np.sin(angle_diff))
                            J[n_pq + idx_i, n_pq + idx_j] = V[i] * (G[i,j] * np.sin(angle_diff) - B[i,j] * np.cos(angle_diff))
                
                try:
                    dx = np.linalg.solve(J, rhs)
                    
                    for idx_i, i in enumerate(pq_buses):
                        theta[i] += dx[idx_i]
                        V[i] += dx[n_pq + idx_i]
                        V[i] = max(0.8, min(V[i], 1.2))
                except np.linalg.LinAlgError:
                    converged = False
                    break
        
        V = np.clip(V, 0.85, 1.15)
        
        # 计算import_power (回退方法无法精确计算，返回总负荷)
        import_power = np.sum(np.maximum(0, P_net))
        
        return V, theta, converged, import_power
    
    def _build_ybus(self) -> np.ndarray:
        """构建节点导纳矩阵 - R/X为欧姆值，需转为标幺值"""
        n = self.n_nodes
        Y = np.zeros((n, n), dtype=complex)
        
        for _, line in self.line_data.iterrows():
            from_bus = int(line['FROM']) - 1
            to_bus = int(line['TO']) - 1
            r_ohm = line['R']
            x_ohm = line['X']
            
            if r_ohm == 0 and x_ohm == 0:
                continue
            
            # 欧姆值转标幺值
            r_pu = r_ohm / self.z_base
            x_pu = x_ohm / self.z_base
            
            z = complex(r_pu, x_pu)
            y = 1.0 / z
            
            Y[from_bus, from_bus] += y
            Y[to_bus, to_bus] += y
            Y[from_bus, to_bus] -= y
            Y[to_bus, from_bus] -= y
        
        return Y
    
    def check_voltage_violations(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检查电压越限"""
        v_upper_violation = V > self.v_max
        v_lower_violation = V < self.v_min
        
        v_violations = np.where(v_upper_violation | v_lower_violation)[0]
        
        v_violation_magnitude = np.zeros(self.n_nodes)
        v_violation_magnitude[v_upper_violation] = V[v_upper_violation] - self.v_max[v_upper_violation]
        v_violation_magnitude[v_lower_violation] = self.v_min[v_lower_violation] - V[v_lower_violation]
        
        return v_violations, v_violation_magnitude


class DistributionNetworkEnv:
    """配电网环境 - V7.0 (RL-ADN标准版)"""
    
    def __init__(
        self,
        node_file: str,
        line_file: str,
        time_series_file: str,
        psh_config: Optional[Dict] = None,
        bess_configs: Optional[List[Dict]] = None,
        time_step: float = 0.25,
        episode_length: int = 96,
        price_noise_std: float = 0.05,
        pv_noise_std: float = 0.03,
        enable_domain_randomization: bool = True,
    ):
        self.node_data = pd.read_csv(node_file)
        self.line_data = pd.read_csv(line_file)
        
        self.n_nodes = len(self.node_data)
        self.time_step = time_step
        self.episode_length = episode_length
        
        self.power_flow = PowerFlowCalculator(self.node_data, self.line_data)
        
        self.time_series_data = pd.read_csv(time_series_file, parse_dates=['date_time'])
        self.time_series_data.set_index('date_time', inplace=True)
        
        self.load_data = self.time_series_data[[col for col in self.time_series_data.columns 
                                                 if col.startswith('active_power_node_')]]
        self.load_data = self.load_data / 1000.0
        
        self.renewable_data = self.time_series_data[[col for col in self.time_series_data.columns 
                                                      if col.startswith('renewable_active_power_node_')]]
        self.renewable_data = self.renewable_data / 1000.0
        
        self.renewable_data = self.renewable_data.interpolate(method='linear', limit_direction='both')
        self.renewable_data = self.renewable_data.fillna(0)
        self.load_data = self.load_data.interpolate(method='linear', limit_direction='both')
        self.load_data = self.load_data.fillna(method='ffill').fillna(method='bfill')
        
        self.price_data = self.time_series_data['price']
        self.price_data = self.price_data.interpolate(method='linear', limit_direction='both')
        self.price_data = self.price_data.fillna(self.price_data.median())
        
        self.price_noise_std = price_noise_std
        self.pv_noise_std = pv_noise_std
        self.enable_domain_randomization = enable_domain_randomization
        
        self.psh_config_dict = psh_config
        self.bess_configs_list = bess_configs
        
        self._init_energy_storages(psh_config, bess_configs)
        self._setup_spaces()
        
        self.current_step = 0
        self.current_time = 0
        self.episode_start_idx = 0
        
        self.voltage_history = []
        self.reward_history = []
        self.action_history = []
        self.renewable_consumption_history = []
        self.curtailment_history = []
        
        # V7.0: RL-ADN标准奖励参数
        self.voltage_target_pu = 1.0
        self.voltage_band_pu = 0.05
        self.voltage_penalty_scale = 100.0
    
    def _init_energy_storages(self, psh_config: Optional[Dict], bess_configs: Optional[List[Dict]]):
        """初始化储能系统"""
        if psh_config is None:
            psh_config = {
                'unit_id': 1, 'node_id': 34,
                'rated_generation_power': 0.2, 'rated_pumping_power': 0.2,
                'upper_reservoir_capacity': 1.0, 'lower_reservoir_capacity': 1.0,
                'upper_reservoir_min': 0.1, 'lower_reservoir_min': 0.1,
                'generation_efficiency': 0.90, 'pumping_efficiency': 0.90,
                'initial_upper_soc': 0.5, 'initial_lower_soc': 0.5,
                'max_daily_cycles': 4, 'min_operation_duration': 2,
                'max_operation_duration': 48,
            }
        
        if bess_configs is None:
            bess_configs = [
                {'unit_id': 2, 'node_id': 16, 'max_power': 0.05, 'capacity': 0.2,
                 'min_soc': 0.1, 'max_soc': 0.9, 'charge_efficiency': 0.95,
                 'discharge_efficiency': 0.95, 'initial_soc': 0.5, 'ramp_rate_limit': 1.0, 'degradation_cost': 0.0},
                {'unit_id': 3, 'node_id': 27, 'max_power': 0.05, 'capacity': 0.2,
                 'min_soc': 0.1, 'max_soc': 0.9, 'charge_efficiency': 0.95,
                 'discharge_efficiency': 0.95, 'initial_soc': 0.5, 'ramp_rate_limit': 1.0, 'degradation_cost': 0.0},
            ]
        
        self.psh = PumpedStorageUnit(**psh_config, time_step=self.time_step)
        self.bess_units = []
        for config in bess_configs:
            self.bess_units.append(BatteryEnergyStorageSystem(**config, time_step=self.time_step))
        
        self.n_bess = len(self.bess_units)
    
    def _setup_spaces(self):
        """设置状态空间"""
        self.state_dim = 34 + 1 + 8 + 2 * 2 + 1  # = 48
        self.action_dim = 4  # [psh_discrete, psh_power, bess1, bess2]
    
    def reset(self, start_idx: Optional[int] = None, reset_psh_storage: bool = False) -> np.ndarray:
        """
        重置环境 - V7.0: PSH不重置库容状态（默认）
        """
        if start_idx is None:
            max_start = len(self.load_data) - self.episode_length - 1
            self.episode_start_idx = np.random.randint(0, max(1, max_start)) if max_start > 0 else 0
        else:
            self.episode_start_idx = start_idx
        
        self.current_step = 0
        self.current_time = self.episode_start_idx
        
        # V7.1: PSH不重置库容（SOC保持），但重置日循环计数
        if reset_psh_storage:
            self.psh.reset()  # 完全重置
        else:
            self.psh.reset_daily_cycles()  # 只重置日循环计数，保持SOC
        
        # BESS重置
        for bess in self.bess_units:
            bess.reset()
        
        self.voltage_history = []
        self.reward_history = []
        self.action_history = []
        self.renewable_consumption_history = []
        self.curtailment_history = []
        
        self.last_voltages = self._compute_voltages_no_storage()
        
        return self._get_state()
    
    def _compute_voltages_no_storage(self) -> np.ndarray:
        """运行潮流计算(无储能注入)获取节点电压"""
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes]
        renewable_p = self.renewable_data.iloc[self.current_time].values[:self.n_nodes]
        
        net_load = load_p - renewable_p
        q_load = np.zeros(self.n_nodes)
        q_inject = np.zeros(self.n_nodes)
        p_inject = np.zeros(self.n_nodes)
        
        p_load = np.maximum(0, net_load)
        p_inject_adj = np.maximum(0, -net_load)
        
        V, _, converged, _ = self.power_flow.solve(p_load, q_load, p_inject_adj, q_inject)
        
        if not converged:
            V = self.last_voltages.copy() if hasattr(self, 'last_voltages') else np.ones(self.n_nodes)
        
        return V.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        action = np.array(action).flatten()
        psh_discrete = int(np.clip(action[0], 0, 2)) if len(action) > 0 else 0
        psh_power = np.clip(action[1], 0.0, 1.0) if len(action) > 1 else 1.0
        bess1_action = np.clip(action[2], -1.0, 1.0) if len(action) > 2 else 0.0
        bess2_action = np.clip(action[3], -1.0, 1.0) if len(action) > 3 else 0.0
        
        self.action_history.append(np.array([psh_discrete, psh_power, bess1_action, bess2_action]))
        
        # 执行PSH动作
        psh_power_output, psh_info = self.psh.step(psh_discrete, psh_power, self.current_step)
        # V7.2: 记录PSH实际执行的动作（而非原始传入动作）
        # psh_info['discrete_action'] 已由PSH step设置为实际执行值
        
        # 执行BESS动作
        bess_powers = []
        bess_socs = []
        bess_actions = [bess1_action, bess2_action]
        for i, bess in enumerate(self.bess_units):
            power, soc, info = bess.step(bess_actions[i])
            bess_powers.append(power)
            bess_socs.append(soc)
        
        # 获取当前时刻数据
        load_p = self.load_data.iloc[self.current_time].values[:self.n_nodes]
        renewable_p = self.renewable_data.iloc[self.current_time].values[:self.n_nodes]
        base_price = self.price_data.iloc[self.current_time]
        
        # 域随机化
        if self.enable_domain_randomization:
            pv_noise = np.random.normal(1.0, self.pv_noise_std, size=renewable_p.shape)
            renewable_p = renewable_p * np.clip(pv_noise, 0.7, 1.3)
            price_noise = np.random.normal(1.0, self.price_noise_std)
            price = base_price * max(0.5, price_noise)
        else:
            price = base_price
        
        total_renewable = np.sum(renewable_p)
        
        # 无功功率为0
        node_q_load = np.zeros(self.n_nodes)
        node_q_inject = np.zeros(self.n_nodes)
        
        # 调度前(无储能)
        net_load = load_p - renewable_p
        p_load_no_storage = np.maximum(0, net_load)
        p_inject_no_storage = np.maximum(0, -net_load)
        
        _, _, converged_before, import_before = self.power_flow.solve(
            p_load_no_storage, node_q_load, p_inject_no_storage, node_q_inject
        )
        
        # 调度后(有储能)
        node_p_load = np.maximum(0, net_load)
        node_p_inject = np.maximum(0, -net_load).copy()
        
        # PSH功率注入
        psh_node_idx = self.psh.node_id - 1
        if psh_node_idx < self.n_nodes:
            if psh_power_output > 0:  # 发电 -> 注入电网
                node_p_inject[psh_node_idx] += psh_power_output
            elif psh_power_output < 0:  # 抽水 -> 从电网取电
                node_p_load[psh_node_idx] += abs(psh_power_output)
        
        # BESS功率注入
        for bess, power in zip(self.bess_units, bess_powers):
            bess_node_idx = bess.node_id - 1
            if bess_node_idx < self.n_nodes:
                if power > 0:  # 放电 -> 注入电网
                    node_p_inject[bess_node_idx] += power
                elif power < 0:  # 充电 -> 从电网取电
                    node_p_load[bess_node_idx] += abs(power)
        
        # 运行潮流
        V_after, _, converged_after, import_after = self.power_flow.solve(
            node_p_load, node_q_load, node_p_inject, node_q_inject
        )
        
        self.last_voltages = V_after.copy().astype(np.float32)
        
        # 检查电压越限
        v_violations, v_violation_mag = self.power_flow.check_voltage_violations(V_after)
        
        # 可再生能源消纳率
        total_load = np.sum(load_p)
        renewable_to_load = min(total_renewable, total_load)
        renewable_surplus = max(0, total_renewable - renewable_to_load)
        total_charge_capacity = sum(max(0, -p) for p in bess_powers) + max(0, -psh_power_output)
        renewable_to_storage = min(renewable_surplus, total_charge_capacity)
        curtailment = max(0, renewable_surplus - renewable_to_storage)
        renewable_consumed = renewable_to_load + renewable_to_storage
        
        # V7.0: 纯RL-ADN奖励函数
        reward = self._calculate_reward_rl_adn(
            price, import_before, import_after, V_after, converged_after
        )
        
        self.voltage_history.append(V_after.copy())
        self.reward_history.append(reward)
        self.renewable_consumption_history.append(renewable_consumed)
        self.curtailment_history.append(curtailment)
        
        self.current_step += 1
        self.current_time += 1
        done = self.current_step >= self.episode_length
        
        next_state = self._get_state()
        
        info = {
            'voltage': V_after,
            'v_violations': v_violations,
            'v_violation_magnitude': np.sum(v_violation_mag),
            'v_violation_count': len(v_violations),
            'import_before': import_before,
            'import_after': import_after,
            'saved_power': import_before - import_after,
            'psh_power': psh_power_output,
            'psh_discrete_action': psh_discrete,
            'psh_power_ratio': psh_power,
            'psh_upper_soc': psh_info['upper_soc'],
            'psh_lower_soc': psh_info['lower_soc'],
            'psh_mode': psh_info['mode'],
            'psh_constraint_violated': psh_info.get('is_constraint_violated', False),
            'psh_action_modified': psh_info.get('action_modified', False),
            'bess_powers': bess_powers,
            'bess_socs': bess_socs,
            'converged': converged_after,
            'price': price,
            'base_price': base_price,
            'voltage_mean': np.mean(V_after),
            'voltage_std': np.std(V_after),
            'voltage_min': np.min(V_after),
            'voltage_max': np.max(V_after),
            'renewable_total': total_renewable,
            'renewable_consumed': renewable_consumed,
            'curtailment': curtailment,
            'load_total': np.sum(load_p),
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        voltage_norm = self.last_voltages / 1.1
        
        price = self.price_data.iloc[self.current_time]
        
        psh_state = self.psh.get_state()
        
        bess_states = []
        for bess in self.bess_units:
            bess_states.extend(bess.get_state())
        
        hour = (self.current_time % 96) / 96.0
        
        price_norm = price / 100.0
        
        state = np.concatenate([
            voltage_norm.astype(np.float32),
            [price_norm],
            psh_state,
            bess_states,
            [hour]
        ])
        
        return state
    
    def _calculate_reward_rl_adn(self, price, import_before, import_after, V, converged) -> float:
        """
        V7.0: 纯RL-ADN奖励函数
        
        economic = price * saved_power_kw
        voltage_penalty = sum(min(0, penalty_scale * (voltage_band - |v_target - v|)))
        reward = economic + voltage_penalty
        """
        # 1. 经济奖励: 购电节省 (kW -> MW转换)
        saved_power_mw = import_before - import_after  # MW
        saved_power_kw = saved_power_mw * 1000.0  # kW
        economic = float(price * saved_power_kw)
        
        # 2. 电压惩罚: RL-ADN标准
        voltage_penalty = 0.0
        for v in V:
            voltage_penalty += min(
                0.0,
                self.voltage_penalty_scale * (self.voltage_band_pu - abs(self.voltage_target_pu - float(v)))
            )
        
        # 3. 不收敛惩罚
        convergence_penalty = -500.0 if not converged else 0.0
        
        # 总奖励
        reward = economic + voltage_penalty + convergence_penalty
        
        # 归一化
        reward = reward / 1000.0
        
        if np.isnan(reward) or np.isinf(reward):
            reward = -1.0
        
        reward = np.clip(reward, -10, 10)
        
        return float(reward)
    
    def get_storage_states(self) -> Dict:
        """获取所有储能系统的状态"""
        states = {
            'psh': {
                'upper_soc': self.psh.upper_soc,
                'lower_soc': self.psh.lower_soc,
                'power': self.psh.current_power,
                'mode': self.psh.current_mode,
                'daily_cycles': self.psh.daily_cycle_count,
                'fatigue_factor': getattr(self.psh, 'fatigue_factor', 1.0),
            },
            'bess': []
        }
        for bess in self.bess_units:
            states['bess'].append({
                'soc': bess.current_soc,
                'power': bess.current_power,
                'total_cycled': getattr(bess, 'total_energy_cycled', 0),
            })
        return states
