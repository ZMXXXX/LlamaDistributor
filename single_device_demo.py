#!/usr/bin/env python3
"""
单设备分层推理演示

演示如何使用新的SINGLE_DEVICE策略在同一设备上进行分层推理。
这种策略适用于：
1. 测试分层效果
2. 内存优化
3. 分析不同层的计算开销
4. 调试分层逻辑
"""

import torch
import time
from transformers import LlamaTokenizer, LlamaConfig
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llamadist.partitioner.strategies import PartitionStrategy, StrategyType
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.single_device_coordinator import SingleDeviceInference
from llamadist.inference.coordinator import GenerationConfig


def demo_single_device_partitioning():
    """演示单设备分层推理的基本功能"""
    print("🍔 LlamaDistributor - 单设备分层推理演示")
    print("=" * 60)
    
    # 检查设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 模型路径 - 需要根据实际情况调整
    model_path = "/path/to/llama-2-7b-hf"  # 请替换为实际模型路径
    
    # 检查模型路径是否存在
    if not Path(model_path).exists():
        print(f"错误：模型路径不存在: {model_path}")
        print("请下载Llama-2-7B模型并更新model_path变量")
        return
    
    try:
        # 1. 创建分层器
        print("\n1. 初始化分层器...")
        partitioner = LlamaPartitioner(model_path=model_path)
        
        # 2. 分析模型
        print("2. 分析模型结构...")
        model_info = partitioner.analyze_model()
        print(f"模型信息: {model_info.num_layers}层, {model_info.total_params:,}参数")
        
        # 3. 演示不同的单设备分层策略
        demo_strategies = [
            {
                "name": "均匀4分层",
                "strategy": PartitionStrategy(
                    num_partitions=4,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device
                )
            },
            {
                "name": "自定义分层点",
                "strategy": PartitionStrategy(
                    num_partitions=3,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device,
                    custom_boundaries=[(0, 10), (11, 21), (22, 31)]
                )
            },
            {
                "name": "不均匀分层",
                "strategy": PartitionStrategy(
                    num_partitions=3,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device,
                    custom_boundaries=[(0, 7), (8, 23), (24, 31)]
                )
            }
        ]
        
        for demo in demo_strategies:
            print(f"\n{'='*50}")
            print(f"演示策略: {demo['name']}")
            print(f"{'='*50}")
            
            # 创建分层配置
            partitions = demo['strategy'].create_partitions(model_info)
            print(f"分层配置:")
            for i, partition in enumerate(partitions):
                print(f"  分层 {i}: 层{partition.layer_start}-{partition.layer_end} @ {partition.device}")
            
            # 执行分层
            print("执行模型分层...")
            submodels = partitioner.partition(
                strategy=demo['strategy'],
                copy_weights=True
            )
            
            # 创建单设备推理引擎
            print("创建单设备推理引擎...")
            inference_engine = SingleDeviceInference(
                submodels=submodels,
                generation_config=GenerationConfig(
                    max_new_tokens=20,
                    temperature=0.8,
                    do_sample=True,
                    use_cache=True
                ),
                device=device
            )
            
            # 测试推理
            print("测试推理性能...")
            test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            
            # 预热
            for _ in range(3):
                _ = inference_engine.forward_pass(test_input, use_cache=False)
            
            # 性能测试
            inference_times = []
            for i in range(10):
                start_time = time.time()
                result = inference_engine.forward_pass(test_input, use_cache=True)
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            avg_time = sum(inference_times) / len(inference_times)
            print(f"平均推理时间: {avg_time:.4f}秒")
            
            # 显示分层分析
            inference_engine.print_layer_analysis()
            
            # 清理内存
            del submodels
            del inference_engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def demo_text_generation():
    """演示单设备分层推理的文本生成功能"""
    print("\n" + "="*60)
    print("单设备分层文本生成演示")
    print("="*60)
    
    # 设备配置
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 模型路径
    model_path = "/path/to/llama-2-7b-hf"  # 请替换为实际模型路径
    
    if not Path(model_path).exists():
        print(f"错误：模型路径不存在: {model_path}")
        return
    
    try:
        # 加载分词器
        print("加载分词器...")
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 创建分层器
        print("创建分层器...")
        partitioner = LlamaPartitioner(model_path=model_path)
        model_info = partitioner.analyze_model()
        
        # 创建分层策略 - 分成3个子模型
        strategy = PartitionStrategy(
            num_partitions=3,
            strategy_type=StrategyType.SINGLE_DEVICE,
            single_device=device,
            custom_boundaries=[(0, 10), (11, 21), (22, 31)]
        )
        
        # 执行分层
        print("执行模型分层...")
        submodels = partitioner.partition(strategy=strategy, copy_weights=True)
        
        # 创建推理引擎
        print("创建推理引擎...")
        inference_engine = SingleDeviceInference(
            submodels=submodels,
            generation_config=GenerationConfig(
                max_new_tokens=50,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            device=device
        )
        
        # 测试提示
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important lesson I learned is",
            "Once upon a time in a distant galaxy,"
        ]
        
        print("\n文本生成测试:")
        print("-" * 40)
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n提示 {i+1}: {prompt}")
            
            start_time = time.time()
            generated_text = inference_engine.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                return_full_text=True
            )
            generation_time = time.time() - start_time
            
            print(f"生成结果: {generated_text}")
            print(f"生成时间: {generation_time:.2f}秒")
        
        # 显示最终统计
        print("\n最终性能统计:")
        print("-" * 40)
        stats = inference_engine.get_stats()
        print(f"总推理次数: {stats['inference_count']}")
        print(f"平均推理时间: {stats['avg_inference_time']:.4f}秒")
        print(f"生成速度: {stats['tokens_per_second']:.2f} tokens/秒")
        print(f"总生成token数: {stats['total_tokens_generated']}")
        
        # 分层分析
        inference_engine.print_layer_analysis()
        
    except Exception as e:
        print(f"文本生成演示中出现错误: {e}")
        import traceback
        traceback.print_exc()


def compare_strategies():
    """比较不同分层策略的性能"""
    print("\n" + "="*60)
    print("分层策略性能比较")
    print("="*60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "/path/to/llama-2-7b-hf"  # 请替换为实际模型路径
    
    if not Path(model_path).exists():
        print(f"错误：模型路径不存在: {model_path}")
        return
    
    try:
        # 创建分层器
        partitioner = LlamaPartitioner(model_path=model_path)
        model_info = partitioner.analyze_model()
        
        # 测试不同的分层策略
        strategies_to_compare = [
            {
                "name": "2分层",
                "strategy": PartitionStrategy(
                    num_partitions=2,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device
                )
            },
            {
                "name": "4分层",
                "strategy": PartitionStrategy(
                    num_partitions=4,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device
                )
            },
            {
                "name": "8分层",
                "strategy": PartitionStrategy(
                    num_partitions=8,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device
                )
            }
        ]
        
        results = []
        
        for strategy_config in strategies_to_compare:
            print(f"\n测试策略: {strategy_config['name']}")
            
            # 创建子模型
            submodels = partitioner.partition(
                strategy=strategy_config['strategy'],
                copy_weights=True
            )
            
            # 创建推理引擎
            inference_engine = SingleDeviceInference(
                submodels=submodels,
                device=device
            )
            
            # 性能测试
            test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
            
            # 预热
            for _ in range(3):
                _ = inference_engine.forward_pass(test_input)
            
            # 测试推理时间
            inference_times = []
            for _ in range(20):
                start_time = time.time()
                _ = inference_engine.forward_pass(test_input)
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            avg_time = sum(inference_times) / len(inference_times)
            std_time = (sum((t - avg_time) ** 2 for t in inference_times) / len(inference_times)) ** 0.5
            
            results.append({
                "name": strategy_config['name'],
                "avg_time": avg_time,
                "std_time": std_time,
                "num_partitions": len(submodels)
            })
            
            print(f"  平均推理时间: {avg_time:.4f}±{std_time:.4f}秒")
            
            # 清理内存
            del submodels
            del inference_engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 显示比较结果
        print("\n策略比较结果:")
        print("-" * 50)
        print(f"{'策略名称':<10} {'分层数':<8} {'平均时间(秒)':<15} {'标准差':<10}")
        print("-" * 50)
        for result in results:
            print(f"{result['name']:<10} {result['num_partitions']:<8} {result['avg_time']:<15.4f} {result['std_time']:<10.4f}")
        
        # 找出最快的策略
        best_strategy = min(results, key=lambda x: x['avg_time'])
        print(f"\n最佳策略: {best_strategy['name']} (平均时间: {best_strategy['avg_time']:.4f}秒)")
        
    except Exception as e:
        print(f"策略比较中出现错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("选择演示模式:")
    print("1. 基本单设备分层演示")
    print("2. 文本生成演示")
    print("3. 策略性能比较")
    print("4. 运行所有演示")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        demo_single_device_partitioning()
    elif choice == "2":
        demo_text_generation()
    elif choice == "3":
        compare_strategies()
    elif choice == "4":
        demo_single_device_partitioning()
        demo_text_generation()
        compare_strategies()
    else:
        print("无效选择，运行基本演示...")
        demo_single_device_partitioning()


if __name__ == "__main__":
    main() 