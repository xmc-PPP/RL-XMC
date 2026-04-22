"""
抽水储能机组(PSH)与电池储能(BESS)模型 - V8.0

V8.0核心修复：
1. 能量守恒改为水量守恒（上变化 = -下变化）
2. 降低get_valid_actions SOC阈值
3. 3动作: PUMP(0)/GENERATE(1)/STOP(2)
4. Cycle只在IDLE切换时计数
5. 取消水延迟（能量即时守恒）
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from enum import IntEnum


class PSHMode(IntEnum):
    IDLE = 0
    GENERATING = 1
    PUMPING = 2


class PSHAction(IntEnum):
    """V8.0: 3动作 - 抽水/发电/停止"""
    PUMP = 0      # 抽水充电
    GENERATE = 1  # 发电放电
    STOP = 2      # 停止运行


class PumpedStorageUnit:
    """
    PSH模型 V8.0 - 水量守恒版本
    
    能量守恒原理（水量守恒）：
    - 发电: 上库水放下驱动涡轮，上库减少 = 下库增加（等量转移）
    - 抽水: 用电抽水到上库，下库减少 = 上库增加（等量转移）
    
    效率只影响电量↔水量转换比例，不影响水量守恒。
    """

    def __init__(
        self,
        unit_id: int,
        node_id: int,
        rated_generation_power: float,
        rated_pumping_power: float,
        upper_reservoir_capacity: float,
        lower_reservoir_capacity: float,
        upper_reservoir_min: float,
        lower_reservoir_min: float,
        generation_efficiency: float,
        pumping_efficiency: float,
        initial_upper_soc: float,
        initial_lower_soc: float,
        max_daily_cycles: int = 4,
        min_operation_duration: int = 2,
        max_operation_duration: int = 48,
        time_step: float = 0.25
    ):
        self.unit_id = unit_id
        self.node_id = node_id
        self.rated_gen_power = rated_generation_power
        self.rated_pump_power = rated_pumping_power
        self.upper_capacity = upper_reservoir_capacity
        self.lower_capacity = lower_reservoir_capacity
        self.upper_min = upper_reservoir_min
        self.lower_min = lower_reservoir_min
        self.gen_efficiency = generation_efficiency
        self.pump_efficiency = pumping_efficiency
        self.max_daily_cycles = max_daily_cycles
        self.min_duration = min_operation_duration
        self.max_duration = max_operation_duration
        self.time_step = time_step

        self.initial_upper_soc = initial_upper_soc
        self.initial_lower_soc = initial_lower_soc

        # V8.0: 发电30%~100%, 抽水60%~100%
        self.min_gen_output_ratio = 0.30
        self.min_pump_output_ratio = 0.60

        # 状态
        self.current_mode = PSHMode.IDLE
        self.operation_duration = 0
        self.daily_cycle_count = 0
        self.current_day = 0
        self.upper_energy = initial_upper_soc * (upper_reservoir_capacity - upper_reservoir_min) + upper_reservoir_min
        self.lower_energy = initial_lower_soc * (lower_reservoir_capacity - lower_reservoir_min) + lower_reservoir_min
        self.current_power = 0.0
        self.target_power_ratio = 0.0
        self.switch_count_today = 0
        self.cumulative_switches = 0
        self.fatigue_factor = 1.0
        self.constraint_violations = 0

        # 历史
        self.upper_soc_history = [self.upper_soc]
        self.lower_soc_history = [self.lower_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]

    @property
    def upper_soc(self) -> float:
        return (self.upper_energy - self.upper_min) / (self.upper_capacity - self.upper_min)

    @property
    def lower_soc(self) -> float:
        return (self.lower_energy - self.lower_min) / (self.lower_capacity - self.lower_min)

    @property
    def total_energy(self) -> float:
        """V8.0: 上下库总能量（应守恒）"""
        return self.upper_energy + self.lower_energy

    def reset(self):
        """重置PSH状态"""
        self.upper_energy = self.initial_upper_soc * (self.upper_capacity - self.upper_min) + self.upper_min
        self.lower_energy = self.initial_lower_soc * (self.lower_capacity - self.lower_min) + self.lower_min
        self.current_mode = PSHMode.IDLE
        self.operation_duration = 0
        self.daily_cycle_count = 0
        self.current_day = 0
        self.current_power = 0.0
        self.target_power_ratio = 0.0
        self.switch_count_today = 0
        self.cumulative_switches = 0
        self.fatigue_factor = 1.0
        self.constraint_violations = 0
        self.upper_soc_history = [self.upper_soc]
        self.lower_soc_history = [self.lower_soc]
        self.power_history = [self.current_power]
        self.mode_history = [self.current_mode]

    def reset_daily_cycles(self):
        """V8.0: 每轮开始时重置日循环计数（保持SOC不变）"""
        self.daily_cycle_count = 0
        self.switch_count_today = 0
        self.current_day = 0
        self.operation_duration = 0
        self.current_mode = PSHMode.IDLE
        self.current_power = 0.0

    def get_valid_actions(self) -> List[int]:
        """V8.0: 降低SOC阈值，确保低SOC时仍能动作"""
        valid = []
        
        # PUMP(0): 抽水 - 只要下库有能量就能抽
        soc_threshold = 0.105  # V8.0: min_soc(0.10) + 0.005
        if self.lower_soc > soc_threshold and self.daily_cycle_count < self.max_daily_cycles:
            valid.append(PSHAction.PUMP)
        
        # GENERATE(1): 发电 - 只要上库有能量就能发
        if self.upper_soc > soc_threshold and self.daily_cycle_count < self.max_daily_cycles:
            valid.append(PSHAction.GENERATE)
        
        # STOP(2): 停止 - 只要当前在运行中
        if self.current_mode != PSHMode.IDLE:
            valid.append(PSHAction.STOP)
        
        # 如果没有任何动作可用，默认STOP
        if not valid:
            valid = [PSHAction.STOP]
        
        return valid

    def step(self, discrete_action: int, continuous_power: float, current_time: int = 0) -> Tuple[float, Dict]:
        discrete_action = int(discrete_action)
        original_action = discrete_action

        # 更新日期
        day = current_time // 96
        if day > self.current_day:
            self.current_day = day
            self.daily_cycle_count = 0
            self.switch_count_today = 0

        valid_actions = self.get_valid_actions()
        is_valid = discrete_action in valid_actions

        if not is_valid:
            discrete_action = PSHAction.STOP
            action_modified = True
        else:
            action_modified = False

        # V8.0: 执行离散动作 - Cycle只在IDLE切换时计数
        prev_mode = self.current_mode
        
        if discrete_action == PSHAction.PUMP:
            if self.current_mode == PSHMode.IDLE:
                self.current_mode = PSHMode.PUMPING
                self.operation_duration = 0
                self.daily_cycle_count += 1
                self.switch_count_today += 1
                self.cumulative_switches += 1
            elif self.current_mode != PSHMode.PUMPING:
                self.current_mode = PSHMode.PUMPING
                self.operation_duration = 0
                self.switch_count_today += 1
                self.cumulative_switches += 1
        
        elif discrete_action == PSHAction.GENERATE:
            if self.current_mode == PSHMode.IDLE:
                self.current_mode = PSHMode.GENERATING
                self.operation_duration = 0
                self.daily_cycle_count += 1
                self.switch_count_today += 1
                self.cumulative_switches += 1
            elif self.current_mode != PSHMode.GENERATING:
                self.current_mode = PSHMode.GENERATING
                self.operation_duration = 0
                self.switch_count_today += 1
                self.cumulative_switches += 1
        
        elif discrete_action == PSHAction.STOP:
            if self.current_mode != PSHMode.IDLE:
                self.current_mode = PSHMode.IDLE
                self.operation_duration = 0
                self.switch_count_today += 1
                self.cumulative_switches += 1
        
        mode_switched = (prev_mode != self.current_mode)

        # 更新疲劳因子
        self.fatigue_factor = max(0.85, 1.0 - 0.0005 * self.cumulative_switches)

        # 连续功率 (0~1)
        continuous_power = np.clip(continuous_power, 0.0, 1.0)

        constraint_violated = False
        actual_power = 0.0

        if self.current_mode == PSHMode.IDLE:
            self.target_power_ratio = 0.0
            actual_power = 0.0
            self.operation_duration = 0

        elif self.current_mode == PSHMode.GENERATING:
            # 发电30%~100%
            if continuous_power > 0:
                if continuous_power < self.min_gen_output_ratio:
                    continuous_power = self.min_gen_output_ratio
                continuous_power = min(continuous_power, 1.0)
            else:
                continuous_power = self.min_gen_output_ratio

            continuous_power *= self.fatigue_factor
            self.target_power_ratio = continuous_power
            target_power = continuous_power * self.rated_gen_power

            # 检查上水库能量约束
            energy_needed = target_power * self.time_step / self.gen_efficiency
            
            if self.upper_energy - energy_needed < self.upper_min:
                available = self.upper_energy - self.upper_min
                if available > 1e-6:
                    actual_power = available * self.gen_efficiency / self.time_step
                    constraint_violated = True
                else:
                    actual_power = 0.0
                    self.current_mode = PSHMode.IDLE
                    self.operation_duration = 0
                    constraint_violated = True
            else:
                actual_power = target_power
            
            self.operation_duration += 1

            # V8.0: 水量守恒 - 上库减少 = 下库增加（等量转移）
            if actual_power > 0:
                delta = actual_power * self.time_step / self.gen_efficiency
                self.upper_energy = max(self.upper_min, self.upper_energy - delta)
                self.lower_energy = min(self.lower_capacity, self.lower_energy + delta)

        elif self.current_mode == PSHMode.PUMPING:
            # 抽水60%~100%
            if continuous_power > 0:
                if continuous_power < self.min_pump_output_ratio:
                    continuous_power = self.min_pump_output_ratio
                continuous_power = min(continuous_power, 1.0)
            else:
                continuous_power = self.min_pump_output_ratio

            continuous_power *= self.fatigue_factor
            self.target_power_ratio = continuous_power
            target_power = -continuous_power * self.rated_pump_power

            # 检查下水库能量约束
            energy_needed = abs(target_power) * self.time_step / self.pump_efficiency
            
            if self.lower_energy - energy_needed < self.lower_min:
                available = self.lower_energy - self.lower_min
                if available > 1e-6:
                    actual_power = -available * self.pump_efficiency / self.time_step
                    constraint_violated = True
                else:
                    actual_power = 0.0
                    self.current_mode = PSHMode.IDLE
                    self.operation_duration = 0
                    constraint_violated = True
            else:
                actual_power = target_power
            
            self.operation_duration += 1

            # V8.0: 水量守恒 - 下库减少 = 上库增加（等量转移）
            if actual_power < 0:
                delta = abs(actual_power) * self.time_step / self.pump_efficiency
                self.lower_energy = max(self.lower_min, self.lower_energy - delta)
                self.upper_energy = min(self.upper_capacity, self.upper_energy + delta)

        # 确保边界
        self.upper_energy = max(self.upper_min, min(self.upper_energy, self.upper_capacity))
        self.lower_energy = max(self.lower_min, min(self.lower_energy, self.lower_capacity))

        self.current_power = actual_power

        # V8.0: 检查SOC边界 (hardcoded 0.10~0.90)
        soc_violated = (self.upper_soc < 0.10 or self.upper_soc > 0.90 or
                       self.lower_soc < 0.10 or self.lower_soc > 0.90)

        self.power_history.append(actual_power)
        self.upper_soc_history.append(self.upper_soc)
        self.lower_soc_history.append(self.lower_soc)
        self.mode_history.append(self.current_mode)

        return actual_power, {
            'mode': self.current_mode,
            'upper_soc': self.upper_soc,
            'lower_soc': self.lower_soc,
            'operation_duration': self.operation_duration,
            'daily_cycles': self.daily_cycle_count,
            'discrete_action': discrete_action,
            'original_action': original_action,
            'action_modified': action_modified,
            'is_valid': is_valid,
            'valid_actions': valid_actions,
            'is_constraint_violated': constraint_violated,
            'is_soc_violated': soc_violated,
            'target_power_ratio': self.target_power_ratio,
            'actual_power': actual_power,
            'fatigue_factor': self.fatigue_factor,
            'mode_switched': mode_switched,
            'switch_count_today': self.switch_count_today,
        }

    def get_state(self) -> np.ndarray:
        power_norm = 0.0
        if self.current_power > 0:
            power_norm = self.current_power / self.rated_gen_power
        elif self.current_power < 0:
            power_norm = self.current_power / self.rated_pump_power

        return np.array([
            self.upper_soc,
            self.lower_soc,
            power_norm,
            float(self.current_mode) / 2.0,
            min(self.operation_duration / self.max_duration, 1.0),
            self.daily_cycle_count / self.max_daily_cycles,
            self.fatigue_factor,
            self.target_power_ratio,
        ], dtype=np.float32)


class BatteryEnergyStorageSystem:
    """BESS模型 V8.0"""

    def __init__(
        self,
        unit_id: int,
        node_id: int,
        max_power: float,
        capacity: float,
        min_soc: float,
        max_soc: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        initial_soc: float,
        ramp_rate_limit: float,
        degradation_cost: float = 0.0,
        time_step: float = 0.25
    ):
        self.unit_id = unit_id
        self.node_id = node_id
        self.max_power = max_power
        self.capacity = capacity
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.charge_eff = charge_efficiency
        self.discharge_eff = discharge_efficiency
        self.ramp_limit = ramp_rate_limit
        self.degradation_cost = degradation_cost
        self.time_step = time_step

        self.initial_soc = initial_soc
        self.current_soc = initial_soc
        self.current_power = 0.0
        self.total_energy_cycled = 0.0
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]

    def reset(self):
        self.current_soc = self.initial_soc
        self.current_power = 0.0
        self.total_energy_cycled = 0.0
        self.soc_history = [self.current_soc]
        self.power_history = [self.current_power]

    def step(self, action: float) -> Tuple[float, float, Dict]:
        action = np.clip(action, -1.0, 1.0)

        if action > 0:
            target_power = action * self.max_power
        elif action < 0:
            target_power = action * self.max_power
        else:
            target_power = 0.0

        # 爬坡率限制
        power_change = target_power - self.current_power
        if abs(power_change) > self.ramp_limit:
            power_change = np.sign(power_change) * self.ramp_limit
            target_power = self.current_power + power_change

        # SOC约束
        if target_power > 0:  # 放电
            energy_discharged = target_power * self.time_step / self.discharge_eff
            if self.current_soc * self.capacity - energy_discharged < self.min_soc * self.capacity:
                available = (self.current_soc - self.min_soc) * self.capacity
                target_power = available * self.discharge_eff / self.time_step if available > 0 else 0
        elif target_power < 0:  # 充电
            energy_charged = abs(target_power) * self.time_step * self.charge_eff
            if self.current_soc * self.capacity + energy_charged > self.max_soc * self.capacity:
                available = (self.max_soc - self.current_soc) * self.capacity
                target_power = -available / (self.charge_eff * self.time_step) if available > 0 else 0

        # 更新SOC
        if target_power > 0:
            energy_change = -target_power * self.time_step / self.discharge_eff
        elif target_power < 0:
            energy_change = abs(target_power) * self.time_step * self.charge_eff
        else:
            energy_change = 0.0

        next_soc = self.current_soc + energy_change / self.capacity
        next_soc = max(self.min_soc, min(next_soc, self.max_soc))

        self.current_power = target_power
        self.current_soc = next_soc

        self.power_history.append(self.current_power)
        self.soc_history.append(self.current_soc)

        return self.current_power, next_soc, {
            'energy_change_mwh': energy_change,
            'is_constraint_violated': False,
            'total_cycled_mwh': self.total_energy_cycled,
        }

    def get_state(self) -> np.ndarray:
        return np.array([
            self.current_soc,
            self.current_power / self.max_power if self.max_power > 0 else 0
        ], dtype=np.float32)
