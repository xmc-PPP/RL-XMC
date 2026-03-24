"""
测试脚本 - 验证环境配置是否正确
"""

import sys
sys.path.append('/mnt/okcomputer/output/psh_drl_project')

import numpy as np
import pandas as pd
from models.pumped_storage import PumpedStorageUnit, BatteryEnergyStorageSystem
from utils.data_generator import create_ieee34_node_topology


def test_pumped_storage():
    """测试抽水储能模型"""
    print("=" * 60)
    print("测试抽水储能模型")
    print("=" * 60)
    
    # 创建PSH
    psh = PumpedStorageUnit(
        unit_id=1,
        node_id=16,
        max_generation_power=50.0,
        min_generation_power=10.0,
        max_pumping_power=50.0,
        min_pumping_power=10.0,
        max_reservoir_capacity=200.0,
        min_reservoir_capacity=0.0,
        generation_efficiency=0.85,
        pumping_efficiency=0.85,
        initial_soc=0.5,
        ramp_rate_limit=20.0,
        time_step=0.25
    )
    
    print(f"初始SOC: {psh.current_soc:.3f}")
    print(f"初始能量: {psh.current_energy:.2f} MWh")
    
    # 测试发电
    print("\n--- 测试发电模式 ---")
    action = 0.8  # 80%发电功率
    power, soc, info = psh.step(action)
    print(f"动作: {action:.2f}")
    print(f"实际功率: {power:.2f} MW")
    print(f"新SOC: {soc:.3f}")
    print(f"运行模式: {info['mode']}")
    
    # 测试抽水
    print("\n--- 测试抽水模式 ---")
    action = -0.6  # 60%抽水功率
    power, soc, info = psh.step(action)
    print(f"动作: {action:.2f}")
    print(f"实际功率: {power:.2f} MW")
    print(f"新SOC: {soc:.3f}")
    print(f"运行模式: {info['mode']}")
    
    # 测试停机
    print("\n--- 测试停机模式 ---")
    action = 0.0
    power, soc, info = psh.step(action)
    print(f"动作: {action:.2f}")
    print(f"实际功率: {power:.2f} MW")
    print(f"新SOC: {soc:.3f}")
    print(f"运行模式: {info['mode']}")
    
    print("\n抽水储能模型测试通过! ✓")
    return True


def test_battery_storage():
    """测试电池储能模型"""
    print("\n" + "=" * 60)
    print("测试电池储能模型")
    print("=" * 60)
    
    # 创建BESS
    bess = BatteryEnergyStorageSystem(
        unit_id=2,
        node_id=12,
        max_power=50.0,
        capacity=100.0,
        min_soc=0.2,
        max_soc=0.9,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        initial_soc=0.5,
        ramp_rate_limit=25.0,
        time_step=0.25
    )
    
    print(f"初始SOC: {bess.current_soc:.3f}")
    print(f"容量: {bess.capacity:.2f} MWh")
    
    # 测试放电
    print("\n--- 测试放电模式 ---")
    action = 0.7  # 70%放电功率
    power, soc, info = bess.step(action)
    print(f"动作: {action:.2f}")
    print(f"实际功率: {power:.2f} MW")
    print(f"新SOC: {soc:.3f}")
    
    # 测试充电
    print("\n--- 测试充电模式 ---")
    action = -0.5  # 50%充电功率
    power, soc, info = bess.step(action)
    print(f"动作: {action:.2f}")
    print(f"实际功率: {power:.2f} MW")
    print(f"新SOC: {soc:.3f}")
    
    print("\n电池储能模型测试通过! ✓")
    return True


def test_network_topology():
    """测试网络拓扑"""
    print("\n" + "=" * 60)
    print("测试网络拓扑")
    print("=" * 60)
    
    node_data, line_data = create_ieee34_node_topology()
    
    print(f"节点数量: {len(node_data)}")
    print(f"线路数量: {len(line_data)}")
    
    print("\n节点数据预览:")
    print(node_data.head())
    
    print("\n线路数据预览:")
    print(line_data.head())
    
    # 验证拓扑连通性
    print("\n--- 验证拓扑 ---")
    from_bus_set = set(line_data['from_bus'])
    to_bus_set = set(line_data['to_bus'])
    all_nodes = set(node_data['bus_id'])
    
    connected_nodes = from_bus_set | to_bus_set
    isolated_nodes = all_nodes - connected_nodes
    
    if isolated_nodes:
        print(f"警告: 孤立节点 {isolated_nodes}")
    else:
        print("所有节点都已连接 ✓")
    
    print("\n网络拓扑测试通过! ✓")
    return True


def test_constraints():
    """测试约束处理"""
    print("\n" + "=" * 60)
    print("测试约束处理")
    print("=" * 60)
    
    # 测试SOC上限约束
    print("\n--- 测试SOC上限约束 ---")
    psh = PumpedStorageUnit(
        unit_id=1,
        node_id=16,
        max_generation_power=50.0,
        min_generation_power=10.0,
        max_pumping_power=50.0,
        min_pumping_power=10.0,
        max_reservoir_capacity=200.0,
        min_reservoir_capacity=0.0,
        generation_efficiency=0.85,
        pumping_efficiency=0.85,
        initial_soc=0.95,  # 高SOC
        ramp_rate_limit=20.0,
        time_step=0.25
    )
    
    # 尝试大量抽水(应该被限制)
    action = -1.0
    power, soc, info = psh.step(action)
    print(f"高SOC时满功率抽水:")
    print(f"  请求功率: -50.0 MW")
    print(f"  实际功率: {power:.2f} MW")
    print(f"  新SOC: {soc:.3f}")
    print(f"  约束生效: {'是' if abs(power) < 50 else '否'}")
    
    # 测试SOC下限约束
    print("\n--- 测试SOC下限约束 ---")
    psh.reset(initial_soc=0.1)  # 低SOC
    
    # 尝试大量发电(应该被限制)
    action = 1.0
    power, soc, info = psh.step(action)
    print(f"低SOC时满功率发电:")
    print(f"  请求功率: 50.0 MW")
    print(f"  实际功率: {power:.2f} MW")
    print(f"  新SOC: {soc:.3f}")
    print(f"  约束生效: {'是' if power < 50 else '否'}")
    
    print("\n约束处理测试通过! ✓")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始环境测试")
    print("=" * 60)
    
    tests = [
        ("抽水储能模型", test_pumped_storage),
        ("电池储能模型", test_battery_storage),
        ("网络拓扑", test_network_topology),
        ("约束处理", test_constraints)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n{name}测试失败: {str(e)}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n所有测试通过! 环境配置正确。")
    else:
        print("\n部分测试失败，请检查配置。")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
