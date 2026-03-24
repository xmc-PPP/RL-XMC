"""
抽水储能机组(Pumped Storage Hydro Unit, PSH)建模
基于文献[1-5]的建模方法

参考文献:
[1] Huang B, et al. A Computational Efficient Pumped Storage Hydro Optimization in the Look-ahead 
    Unit Commitment and Real-time Market Dispatch Under Uncertainty. IEEE Transactions on 
    Power Systems, 2023.
[2] 考虑抽蓄的风光水火多能互补双层优化调度. 长江科学院院报, 2025.
[3] Yang J, et al. Data-driven Optimal Dynamic Dispatch for Hydro-PV-PHS Integrated Power 
    Systems Using Deep Reinforcement Learning Approach. IEEE Transactions on Sustainable 
    Energy, 2022.
[4] 抽水蓄能电站运行模拟与优化方法研究. 中国电机工程学报, 2022.
[5] A Configuration Based Pumped Storage Hydro Model in Look-ahead Unit Commitment. 
    arXiv:2009.04944, 2020.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class PumpedStorageUnit:
    """
    抽水储能机组模型
    
    特征:
    - 三种运行模式: 发电模式(Generation)、抽水模式(Pumping)、停机模式(Idle)
    - 发电功率可连续调节，抽水功率通常为离散值(定速机组)或可连续调节(变速机组)
    - 考虑发电效率和抽水效率
    - 水库库容约束
    - 运行工况互补约束(不能同时发电和抽水)
    """
    
    def __init__(
        self,
        unit_id: int,
        node_id: int,
        max_generation_power: float,  # 最大发电功率 (MW)
        min_generation_power: float,  # 最小发电功率 (MW)
        max_pumping_power: float,     # 最大抽水功率 (MW)
        min_pumping_power: float,     # 最小抽水功率 (MW)
        max_reservoir_capacity: float, # 最大库容 (MWh)
        min_reservoir_capacity: float, # 最小库容 (MWh)
        generation_efficiency: float,  # 发电效率 (0-1)
        pumping_efficiency: float,     # 抽水效率 (0-1)
        initial_soc: float,            # 初始荷电状态 (0-1)
        ramp_rate_limit: float,        # 爬坡率限制 (MW/时间步)
        is_variable_speed: bool = False,  # 是否为变速机组
        time_step: float = 1.0         # 时间步长 (小时)
    ):
        """
        初始化抽水储能机组
        
        Args:
            unit_id: 机组编号
            node_id: 接入节点编号
            max_generation_power: 最大发电功率 [MW]
            min_generation_power: 最小发电功率 [MW]
            max_pumping_power: 最大抽水功率 [MW]
            min_pumping_power: 最小抽水功率 [MW]
            max_reservoir_capacity: 最大库容 [MWh]
            min_reservoir_capacity: 最小库容 [MWh]
            generation_efficiency: 发电效率
            pumping_efficiency: 抽水效率
            initial_soc: 初始SOC
            ramp_rate_limit: 爬坡率限制 [MW/时间步]
            is_variable_speed: 是否为变速机组
            time_step: 时间步长 [小时]
        """
        self.unit_id = unit_id
        self.node_id = node_id
        self.max_gen_power = max_generation_power
        self.min_gen_power = min_generation_power
        self.max_pump_power = max_pumping_power
        self.min_pump_power = min_pumping_power
        self.max_capacity = max_reservoir_capacity
        self.min_capacity = min_reservoir_capacity
        self.gen_efficiency = generation_efficiency
        self.pump_efficiency = pumping_efficiency
        self.ramp_limit = ramp_rate_limit
        self.is_variable_speed = is_variable_speed
        self.time_step = time_step
        
        # 当前状态
        self.current_soc = initial_soc
        self.current_energy = initial_soc * (max_reservoir_capacity - min_reservoir_capacity) + min_reservoir_capacity
        self.current_power = 0.0  # 正为发电，负为抽水
        self.current_mode = 0  # 0: 停机, 1: 发电, -1: 抽水
        
        # 历史记录
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]
        
    def reset(self, initial_soc: Optional[float] = None):
        """重置机组状态"""
        if initial_soc is not None:
            self.current_soc = initial_soc
        self.current_energy = self.current_soc * (self.max_capacity - self.min_capacity) + self.min_capacity
        self.current_power = 0.0
        self.current_mode = 0
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]
        
    def step(self, action: float) -> Tuple[float, float, Dict]:
        """
        执行一个时间步的动作
        
        Args:
            action: 动作值，范围[-1, 1]
                   -1: 满功率抽水
                    0: 停机
                    1: 满功率发电
        
        Returns:
            actual_power: 实际输出功率 [MW] (发电为正，抽水为负)
            next_soc: 下一时刻SOC
            info: 额外信息
        """
        # 将动作映射到功率
        if action > 0:  # 发电模式
            target_power = action * self.max_gen_power
            target_power = max(self.min_gen_power, min(target_power, self.max_gen_power))
            mode = 1
        elif action < 0:  # 抽水模式
            target_power = action * self.max_pump_power  # 负值
            target_power = min(-self.min_pump_power, max(target_power, -self.max_pump_power))
            mode = -1
        else:  # 停机
            target_power = 0.0
            mode = 0
            
        # 应用爬坡约束
        power_change = target_power - self.current_power
        if abs(power_change) > self.ramp_limit:
            power_change = np.sign(power_change) * self.ramp_limit
            target_power = self.current_power + power_change
            
        # 检查SOC约束
        if mode == 1:  # 发电模式
            # 检查是否有足够能量发电
            energy_required = target_power * self.time_step / self.gen_efficiency
            if self.current_energy - energy_required < self.min_capacity:
                # 调整功率以满足SOC约束
                available_energy = self.current_energy - self.min_capacity
                target_power = available_energy * self.gen_efficiency / self.time_step
                target_power = max(0, target_power)
                if target_power < self.min_gen_power:
                    target_power = 0.0
                    mode = 0
                    
        elif mode == -1:  # 抽水模式
            # 检查是否有足够容量抽水
            energy_stored = abs(target_power) * self.time_step * self.pump_efficiency
            if self.current_energy + energy_stored > self.max_capacity:
                # 调整功率以满足SOC约束
                available_capacity = self.max_capacity - self.current_energy
                target_power = -available_capacity / (self.pump_efficiency * self.time_step)
                target_power = min(0, target_power)
                if abs(target_power) < self.min_pump_power:
                    target_power = 0.0
                    mode = 0
        
        # 更新SOC
        if mode == 1:  # 发电
            energy_change = -target_power * self.time_step / self.gen_efficiency
        elif mode == -1:  # 抽水
            energy_change = abs(target_power) * self.time_step * self.pump_efficiency
        else:
            energy_change = 0.0
            
        next_energy = self.current_energy + energy_change
        next_energy = max(self.min_capacity, min(next_energy, self.max_capacity))
        next_soc = (next_energy - self.min_capacity) / (self.max_capacity - self.min_capacity)
        
        # 更新状态
        self.current_power = target_power
        self.current_mode = mode
        self.current_energy = next_energy
        self.current_soc = next_soc
        
        # 记录历史
        self.power_history.append(self.current_power)
        self.soc_history.append(self.current_soc)
        self.mode_history.append(self.current_mode)
        
        info = {
            'mode': mode,
            'energy_change_mwh': energy_change,
            'is_constraint_violated': False
        }
        
        return self.current_power, next_soc, info
    
    def get_state(self) -> np.ndarray:
        """获取机组当前状态"""
        return np.array([
            self.current_soc,
            self.current_power / self.max_gen_power if self.current_power > 0 
            else self.current_power / self.max_pump_power,
            self.current_mode
        ], dtype=np.float32)
    
    def get_constraints(self) -> Dict:
        """获取机组约束信息"""
        return {
            'max_gen_power': self.max_gen_power,
            'min_gen_power': self.min_gen_power,
            'max_pump_power': self.max_pump_power,
            'min_pump_power': self.min_pump_power,
            'max_soc': 1.0,
            'min_soc': 0.0,
            'ramp_limit': self.ramp_limit
        }
    
    def check_feasibility(self, action: float) -> bool:
        """检查动作是否可行"""
        # 将动作映射到功率
        if action > 0:
            target_power = action * self.max_gen_power
        elif action < 0:
            target_power = action * self.max_pump_power
        else:
            return True
            
        # 检查功率范围
        if target_power > 0 and (target_power < self.min_gen_power or target_power > self.max_gen_power):
            return False
        if target_power < 0 and (abs(target_power) < self.min_pump_power or abs(target_power) > self.max_pump_power):
            return False
            
        # 检查SOC约束
        if target_power > 0:  # 发电
            energy_required = target_power * self.time_step / self.gen_efficiency
            if self.current_energy - energy_required < self.min_capacity:
                return False
        elif target_power < 0:  # 抽水
            energy_stored = abs(target_power) * self.time_step * self.pump_efficiency
            if self.current_energy + energy_stored > self.max_capacity:
                return False
                
        return True


class BatteryEnergyStorageSystem:
    """
    普通电池储能系统(BESS)模型
    基于文献[6]的建模方法
    
    参考文献:
    [6] Hou S, et al. RL-ADN: A high-performance Deep Reinforcement Learning environment 
        for optimal Energy Storage Systems dispatch in active distribution networks. 
        Energy and AI, 2025.
    """
    
    def __init__(
        self,
        unit_id: int,
        node_id: int,
        max_power: float,           # 最大功率 [MW]
        capacity: float,            # 容量 [MWh]
        min_soc: float,             # 最小SOC
        max_soc: float,             # 最大SOC
        charge_efficiency: float,   # 充电效率
        discharge_efficiency: float, # 放电效率
        initial_soc: float,         # 初始SOC
        ramp_rate_limit: float,     # 爬坡率限制 [MW/时间步]
        time_step: float = 1.0      # 时间步长 [小时]
    ):
        """
        初始化电池储能系统
        
        Args:
            unit_id: 机组编号
            node_id: 接入节点编号
            max_power: 最大功率 [MW]
            capacity: 容量 [MWh]
            min_soc: 最小SOC
            max_soc: 最大SOC
            charge_efficiency: 充电效率
            discharge_efficiency: 放电效率
            initial_soc: 初始SOC
            ramp_rate_limit: 爬坡率限制 [MW/时间步]
            time_step: 时间步长 [小时]
        """
        self.unit_id = unit_id
        self.node_id = node_id
        self.max_power = max_power
        self.capacity = capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charge_eff = charge_efficiency
        self.discharge_eff = discharge_efficiency
        self.ramp_limit = ramp_rate_limit
        self.time_step = time_step
        
        # 当前状态
        self.current_soc = initial_soc
        self.current_power = 0.0  # 正为放电，负为充电
        
        # 历史记录
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        
    def reset(self, initial_soc: Optional[float] = None):
        """重置储能状态"""
        if initial_soc is not None:
            self.current_soc = initial_soc
        self.current_power = 0.0
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]
        
    def step(self, action: float) -> Tuple[float, float, Dict]:
        """
        执行一个时间步的动作
        
        Args:
            action: 动作值，范围[-1, 1]
                   -1: 满功率充电
                    0: 停机
                    1: 满功率放电
        
        Returns:
            actual_power: 实际功率 [MW] (放电为正，充电为负)
            next_soc: 下一时刻SOC
            info: 额外信息
        """
        # 将动作映射到功率
        if action > 0:  # 放电
            target_power = action * self.max_power
            target_power = min(target_power, self.max_power)
        elif action < 0:  # 充电
            target_power = action * self.max_power  # 负值
            target_power = max(target_power, -self.max_power)
        else:
            target_power = 0.0
            
        # 应用爬坡约束
        power_change = target_power - self.current_power
        if abs(power_change) > self.ramp_limit:
            power_change = np.sign(power_change) * self.ramp_limit
            target_power = self.current_power + power_change
            
        # 检查SOC约束
        if target_power > 0:  # 放电
            energy_discharged = target_power * self.time_step / self.discharge_eff
            if self.current_soc * self.capacity - energy_discharged < self.min_soc * self.capacity:
                available_energy = (self.current_soc - self.min_soc) * self.capacity
                target_power = available_energy * self.discharge_eff / self.time_step
                target_power = max(0, target_power)
                
        elif target_power < 0:  # 充电
            energy_charged = abs(target_power) * self.time_step * self.charge_eff
            if self.current_soc * self.capacity + energy_charged > self.max_soc * self.capacity:
                available_capacity = (self.max_soc - self.current_soc) * self.capacity
                target_power = -available_capacity / (self.charge_eff * self.time_step)
                target_power = min(0, target_power)
        
        # 更新SOC
        if target_power > 0:  # 放电
            energy_change = -target_power * self.time_step / self.discharge_eff
        elif target_power < 0:  # 充电
            energy_change = abs(target_power) * self.time_step * self.charge_eff
        else:
            energy_change = 0.0
            
        next_soc = self.current_soc + energy_change / self.capacity
        next_soc = max(self.min_soc, min(next_soc, self.max_soc))
        
        # 更新状态
        self.current_power = target_power
        self.current_soc = next_soc
        
        # 记录历史
        self.power_history.append(self.current_power)
        self.soc_history.append(self.current_soc)
        
        info = {
            'energy_change_mwh': energy_change,
            'is_constraint_violated': False
        }
        
        return self.current_power, next_soc, info
    
    def get_state(self) -> np.ndarray:
        """获取储能当前状态"""
        return np.array([
            self.current_soc,
            self.current_power / self.max_power
        ], dtype=np.float32)
    
    def get_constraints(self) -> Dict:
        """获取储能约束信息"""
        return {
            'max_power': self.max_power,
            'min_soc': self.min_soc,
            'max_soc': self.max_soc,
            'capacity': self.capacity,
            'ramp_limit': self.ramp_limit
        }
