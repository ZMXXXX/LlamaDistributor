#!/usr/bin/env python3
"""
单设备分层策略测试脚本

用于测试和验证新的SINGLE_DEVICE分层策略的功能。
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llamadist.partitioner.strategies import PartitionStrategy, StrategyType
from llamadist.partitioner.analyzer import ModelInfo


def test_single_device_strategy_uniform():
    """测试单设备均匀分层策略"""
    print("测试单设备均匀分层策略...")
    
    # 创建模拟的模型信息
    model_info = ModelInfo(
        model_name="test_llama",
        num_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        total_params=6738415616,
        total_memory=13476831232,
        layer_infos=[],
        layer_memory_costs=[],
        layer_compute_costs=[],
        layer_params=[]
    )
    
    # 创建单设备分层策略 - 4分层
    strategy = PartitionStrategy(
        num_partitions=4,
        strategy_type=StrategyType.SINGLE_DEVICE,
        single_device="cuda:0"
    )
    
    # 生成分层配置
    partitions = strategy.create_partitions(model_info)
    
    print(f"生成了 {len(partitions)} 个分层:")
    for i, partition in enumerate(partitions):
        print(f"  分层 {i}: 层{partition.layer_start}-{partition.layer_end} @ {partition.device}")
    
    # 验证分层的正确性
    assert len(partitions) == 4, "应该生成4个分层"
    assert all(p.device == "cuda:0" for p in partitions), "所有分层应该在同一设备上"
    
    # 验证层覆盖的完整性
    covered_layers = set()
    for partition in partitions:
        for layer_idx in range(partition.layer_start, partition.layer_end + 1):
            assert layer_idx not in covered_layers, f"层 {layer_idx} 被重复覆盖"
            covered_layers.add(layer_idx)
    
    expected_layers = set(range(32))
    assert covered_layers == expected_layers, "未完整覆盖所有层"
    
    print("✅ 单设备均匀分层策略测试通过")


def test_single_device_strategy_custom():
    """测试单设备自定义分层策略"""
    print("\n测试单设备自定义分层策略...")
    
    # 创建模拟的模型信息
    model_info = ModelInfo(
        model_name="test_llama",
        num_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        total_params=6738415616,
        total_memory=13476831232,
        layer_infos=[],
        layer_memory_costs=[],
        layer_compute_costs=[],
        layer_params=[]
    )
    
    # 创建单设备自定义分层策略
    custom_boundaries = [(0, 7), (8, 15), (16, 23), (24, 31)]
    strategy = PartitionStrategy(
        num_partitions=4,
        strategy_type=StrategyType.SINGLE_DEVICE,
        single_device="cuda:1",
        custom_boundaries=custom_boundaries
    )
    
    # 生成分层配置
    partitions = strategy.create_partitions(model_info)
    
    print(f"生成了 {len(partitions)} 个分层:")
    for i, partition in enumerate(partitions):
        expected_start, expected_end = custom_boundaries[i]
        print(f"  分层 {i}: 层{partition.layer_start}-{partition.layer_end} @ {partition.device}")
        assert partition.layer_start == expected_start, f"分层{i}起始层不匹配"
        assert partition.layer_end == expected_end, f"分层{i}结束层不匹配"
        assert partition.device == "cuda:1", f"分层{i}设备不匹配"
    
    print("✅ 单设备自定义分层策略测试通过")


def test_single_device_strategy_validation():
    """测试单设备分层策略的输入验证"""
    print("\n测试单设备分层策略的输入验证...")
    
    model_info = ModelInfo(
        model_name="test_llama",
        num_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        total_params=6738415616,
        total_memory=13476831232,
        layer_infos=[],
        layer_memory_costs=[],
        layer_compute_costs=[],
        layer_params=[]
    )
    
    # 测试无效的分层边界
    try:
        strategy = PartitionStrategy(
            num_partitions=2,
            strategy_type=StrategyType.SINGLE_DEVICE,
            custom_boundaries=[(0, 10), (12, 31)]  # 缺少层11
        )
        partitions = strategy.create_partitions(model_info)
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"正确捕获异常: {e}")
    
    # 测试重叠的分层边界
    try:
        strategy = PartitionStrategy(
            num_partitions=2,
            strategy_type=StrategyType.SINGLE_DEVICE,
            custom_boundaries=[(0, 15), (10, 31)]  # 层10-15重叠
        )
        partitions = strategy.create_partitions(model_info)
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"正确捕获异常: {e}")
    
    # 测试超出范围的分层边界
    try:
        strategy = PartitionStrategy(
            num_partitions=1,
            strategy_type=StrategyType.SINGLE_DEVICE,
            custom_boundaries=[(0, 35)]  # 超出32层
        )
        partitions = strategy.create_partitions(model_info)
        assert False, "应该抛出异常"
    except ValueError as e:
        print(f"正确捕获异常: {e}")
    
    print("✅ 单设备分层策略验证测试通过")


def test_strategy_comparison():
    """测试不同分层数量的策略"""
    print("\n测试不同分层数量的策略...")
    
    model_info = ModelInfo(
        model_name="test_llama",
        num_layers=32,
        hidden_size=4096,
        vocab_size=32000,
        total_params=6738415616,
        total_memory=13476831232,
        layer_infos=[],
        layer_memory_costs=[],
        layer_compute_costs=[],
        layer_params=[]
    )
    
    # 测试不同的分层数量
    for num_partitions in [1, 2, 4, 8, 16]:
        strategy = PartitionStrategy(
            num_partitions=num_partitions,
            strategy_type=StrategyType.SINGLE_DEVICE,
            single_device="cuda:0"
        )
        
        partitions = strategy.create_partitions(model_info)
        
        print(f"  {num_partitions}分层:")
        for i, partition in enumerate(partitions):
            layer_count = partition.layer_end - partition.layer_start + 1
            print(f"    分层{i}: {layer_count}层 (层{partition.layer_start}-{partition.layer_end})")
        
        # 验证
        assert len(partitions) == num_partitions
        total_layers = sum(p.layer_end - p.layer_start + 1 for p in partitions)
        assert total_layers == 32, f"总层数不匹配: {total_layers}"
    
    print("✅ 不同分层数量测试通过")


def test_strategy_summary():
    """测试策略摘要功能"""
    print("\n测试策略摘要功能...")
    
    strategy = PartitionStrategy(
        num_partitions=3,
        strategy_type=StrategyType.SINGLE_DEVICE,
        single_device="cuda:0",
        custom_boundaries=[(0, 10), (11, 21), (22, 31)]
    )
    
    summary = strategy.get_summary()
    
    print("策略摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 验证摘要内容
    assert summary['strategy_type'] == 'single_device'
    assert summary['num_partitions'] == 3
    assert 'custom_boundaries' in summary['additional_params']
    assert 'single_device' in summary['additional_params']
    
    print("✅ 策略摘要测试通过")


def main():
    """主测试函数"""
    print("🧪 单设备分层策略测试套件")
    print("=" * 50)
    
    try:
        test_single_device_strategy_uniform()
        test_single_device_strategy_custom()
        test_single_device_strategy_validation()
        test_strategy_comparison()
        test_strategy_summary()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！")
        print("单设备分层策略功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 