"""
基础分层示例

演示如何使用LlamaDistributor进行基础的模型分层和推理。
"""

import sys
import os
import torch

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llamadist import PartitionStrategy, LlamaPartitioner, DistributedInference, LlamaDistConfig
from llamadist.inference.coordinator import GenerationConfig


def main():
    """基础分层示例"""
    print("LlamaDistributor 基础分层示例")
    print("=" * 50)
    
    # 1. 创建配置
    print("1. 创建分层配置...")
    config = LlamaDistConfig(
        # 模型配置 - 使用默认配置创建示例模型
        model_path=None,  # 将使用默认配置
        
        # 分层配置
        num_partitions=4,
        strategy_type="uniform",  # 均匀分层
        target_devices=["cpu", "cpu", "cpu", "cpu"],  # 使用CPU避免GPU依赖
        
        # 推理配置
        max_length=128,
        temperature=0.8,
        do_sample=True,
        use_cache=True,
        
        # 输出配置
        output_dir="./demo_partitioned_models"
    )
    
    # 验证配置
    if not config.validate():
        print("配置验证失败！")
        return
    
    print(f"配置验证通过，将创建 {config.num_partitions} 个分层")
    
    # 2. 创建分层策略
    print("\n2. 创建分层策略...")
    strategy = PartitionStrategy(**config.get_partition_strategy_config())
    print(f"使用策略: {strategy.strategy_type.value}")
    
    # 3. 初始化分层器
    print("\n3. 初始化模型分层器...")
    try:
        partitioner = LlamaPartitioner(
            model_path=config.model_path,
            config=config.model_config
        )
        print("分层器初始化成功")
    except ImportError as e:
        print(f"无法导入QLLM模块: {e}")
        print("请确保QLLM项目在Python路径中")
        return
    except Exception as e:
        print(f"分层器初始化失败: {e}")
        return
    
    # 4. 分析模型（可选）
    print("\n4. 分析模型结构...")
    try:
        model_info = partitioner.analyze_model(detailed=False)  # 简单分析避免复杂计算
        print(f"模型分析完成:")
        print(f"  - 总层数: {model_info.num_layers}")
        print(f"  - 总参数: {model_info.total_params/1e6:.1f}M")
        print(f"  - 预估内存: {model_info.total_memory/1e9:.2f}GB")
    except Exception as e:
        print(f"模型分析失败: {e}")
        return
    
    # 5. 执行分层
    print("\n5. 执行模型分层...")
    try:
        submodels = partitioner.partition(
            strategy=strategy,
            analyze_first=False,  # 跳过详细分析
            copy_weights=True
        )
        print(f"模型分层完成，创建了 {len(submodels)} 个子模型")
        
        # 显示分层信息
        for submodel in submodels:
            info = submodel.get_info()
            print(f"  - 分层 {info['partition_idx']}: "
                  f"层 {info['layer_start']}-{info['layer_end']} "
                  f"({info['num_layers']} 层) @ {info['device']}")
    except Exception as e:
        print(f"模型分层失败: {e}")
        return
    
    # 6. 验证分层结果
    print("\n6. 验证分层结果...")
    try:
        # 创建测试输入
        test_input = torch.randint(0, 1000, (1, 16))  # 简单的测试输入
        
        is_valid = partitioner.validate_partitioned_models(
            submodels=submodels,
            sample_input=test_input
        )
        
        if is_valid:
            print("分层验证通过！")
        else:
            print("分层验证失败！")
            return
    except Exception as e:
        print(f"分层验证失败: {e}")
        return
    
    # 7. 保存分层模型
    print("\n7. 保存分层模型...")
    try:
        partitioner.save_partitioned_models(
            submodels=submodels,
            output_dir=config.output_dir,
            save_config=config.save_config
        )
        print(f"分层模型已保存到: {config.output_dir}")
    except Exception as e:
        print(f"保存分层模型失败: {e}")
        return
    
    # 8. 创建分布式推理引擎
    print("\n8. 创建分布式推理引擎...")
    try:
        generation_config = GenerationConfig(**config.get_generation_config())
        
        inference_engine = DistributedInference(
            submodels=submodels,
            generation_config=generation_config,
            enable_async=config.enable_async,
            max_workers=config.max_workers
        )
        print("推理引擎创建成功")
    except Exception as e:
        print(f"推理引擎创建失败: {e}")
        return
    
    # 9. 测试推理
    print("\n9. 测试分布式推理...")
    try:
        # 准备测试输入
        test_input_ids = torch.randint(1, 1000, (1, 10))  # 避免使用0（可能是pad_token）
        print(f"测试输入形状: {test_input_ids.shape}")
        
        # 执行前向传播测试
        with torch.no_grad():
            result = inference_engine.forward_pass(
                input_ids=test_input_ids,
                use_cache=True
            )
        
        print("前向传播测试成功！")
        print(f"  - 输出logits形状: {result['logits'].shape}")
        print(f"  - 隐藏状态形状: {result['hidden_states'].shape}")
        print(f"  - KV缓存层数: {len(result['past_key_values']) if result['past_key_values'] else 0}")
        
        # 如果成功，尝试生成几个token
        print("\n尝试生成少量token...")
        generated = inference_engine.generate(
            input_ids=test_input_ids,
            max_new_tokens=5,  # 只生成5个token进行测试
            do_sample=False,   # 使用贪心解码避免随机性
            use_cache=True
        )
        
        print(f"生成成功！输入长度: {test_input_ids.shape[1]}, 输出长度: {generated.shape[1]}")
        print(f"生成的token: {generated[0, test_input_ids.shape[1]:].tolist()}")
        
    except Exception as e:
        print(f"推理测试失败: {e}")
        print("这可能是正常的，因为使用的是随机初始化的权重")
    
    # 10. 显示性能统计
    print("\n10. 性能统计:")
    stats = inference_engine.get_stats()
    print(f"  - 推理次数: {stats['inference_count']}")
    if stats['inference_count'] > 0:
        print(f"  - 平均推理时间: {stats.get('avg_inference_time', 0):.4f}s")
        print(f"  - 状态传递时间: {stats['state_transfer_time']:.4f}s")
        print(f"  - 缓存管理时间: {stats['cache_management_time']:.4f}s")
    
    print("\n" + "=" * 50)
    print("基础分层示例完成！")
    print(f"分层模型保存在: {config.output_dir}")
    print("您可以使用这些子模型进行分布式推理。")


def load_and_test():
    """加载已保存的分层模型并测试"""
    print("\n" + "=" * 50)
    print("加载并测试已保存的分层模型")
    print("=" * 50)
    
    try:
        # 重新加载分层器
        partitioner = LlamaPartitioner()
        
        # 加载分层模型
        submodels = partitioner.load_partitioned_models(
            input_dir="./demo_partitioned_models",
            devices=["cpu", "cpu", "cpu", "cpu"]  # 指定设备
        )
        
        print(f"成功加载 {len(submodels)} 个子模型")
        
        # 创建推理引擎
        inference_engine = DistributedInference(submodels)
        
        # 简单测试
        test_input = torch.randint(1, 1000, (1, 8))
        result = inference_engine.forward_pass(test_input)
        
        print("加载测试成功！")
        print(f"输出形状: {result['logits'].shape}")
        
    except Exception as e:
        print(f"加载测试失败: {e}")


if __name__ == "__main__":
    main()
    
    # 如果基础示例成功，尝试加载测试
    if os.path.exists("./demo_partitioned_models"):
        load_and_test() 