"""
配置文件
包含所有可配置的参数
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class PSHConfig:
    """抽水储能配置"""
    unit_id: int = 1
    node_id: int = 16  # 安装在节点16
    max_generation_power: float = 50.0    # MW
    min_generation_power: float = 10.0    # MW
    max_pumping_power: float = 50.0       # MW
    min_pumping_power: float = 10.0       # MW
    max_reservoir_capacity: float = 200.0  # MWh
    min_reservoir_capacity: float = 0.0    # MWh
    generation_efficiency: float = 0.85    # 发电效率
    pumping_efficiency: float = 0.85       # 抽水效率
    initial_soc: float = 0.5               # 初始SOC
    ramp_rate_limit: float = 20.0          # MW/时间步
    is_variable_speed: bool = False        # 定速机组


@dataclass
class BESSConfig:
    """电池储能配置"""
    unit_id: int = 2
    node_id: int = 12  # BESS1在节点12
    max_power: float = 50.0           # MW
    capacity: float = 100.0            # MWh
    min_soc: float = 0.2
    max_soc: float = 0.9
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc: float = 0.5
    ramp_rate_limit: float = 25.0       # MW/时间步


@dataclass
class EnvironmentConfig:
    """环境配置"""
    node_file: str = "data/node_data.csv"
    line_file: str = "data/line_data.csv"
    load_file: str = "data/load_data.csv"
    pv_file: str = "data/pv_data.csv"
    price_file: str = "data/price_data.csv"
    time_step: float = 0.25  # 15分钟
    episode_length: int = 96  # 24小时
    
    # 储能配置
    psh_config: PSHConfig = field(default_factory=PSHConfig)
    bess_configs: List[BESSConfig] = field(default_factory=lambda: [
        BESSConfig(unit_id=2, node_id=12),
        BESSConfig(unit_id=3, node_id=27)
    ])


@dataclass
class DDPGConfig:
    """DDPG算法配置"""
    # 网络结构
    state_dim: int = 43  # 34节点 + 1电价 + 3(PSH) + 2*2(BESS) + 1时间
    action_dim: int = 3   # PSH + 2*BESS
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    
    # 学习率
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    
    # 训练参数
    gamma: float = 0.995
    tau: float = 0.005
    buffer_capacity: int = 100000
    batch_size: int = 256
    
    # 噪声参数
    initial_noise_std: float = 0.3
    noise_decay: float = 0.995
    min_noise_std: float = 0.01
    
    # 设备
    device: str = 'cpu'  # 或 'cpu'


@dataclass
class TrainingConfig:
    """训练配置"""
    max_episodes: int = 10000
    max_steps_per_episode: int = 96
    eval_interval: int = 50
    save_interval: int = 100
    log_interval: int = 10
    
    # 随机种子
    random_seed: int = 42
    
    # 模型保存路径
    model_save_path: str = "models/ddpg_model.pth"
    checkpoint_dir: str = "checkpoints/"


@dataclass
class EvaluationConfig:
    """评估配置"""
    num_eval_episodes: int = 10
    render: bool = True
    save_results: bool = True
    results_dir: str = "results/"


# 全局配置实例
psh_config = PSHConfig()
bess_configs = [
    BESSConfig(unit_id=2, node_id=12),
    BESSConfig(unit_id=3, node_id=27)
]
env_config = EnvironmentConfig()
ddpg_config = DDPGConfig()
training_config = TrainingConfig()
eval_config = EvaluationConfig()
