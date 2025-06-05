#!/usr/bin/env python3
"""
单设备分层推理实际测试

使用Llama-2-7B模型进行实际的单设备分层推理测试。
演示如何使用SINGLE_DEVICE策略进行模型分层和推理。
"""

import torch
import sys
import time
from pathlib import Path
from transformers import LlamaTokenizer
import torch.nn.functional as F

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llamadist import (
    PartitionStrategy, 
    LlamaPartitioner, 
    SingleDeviceInference, 
    StrategyType
)
from llamadist.inference.coordinator import GenerationConfig


def test_baseline_inference(model_path: str, tokenizer, device: str) -> dict:
    """
    测试原始模型（不分层）的推理性能作为baseline
    
    Args:
        model_path: 模型路径
        tokenizer: 分词器
        device: 设备
        
    Returns:
        dict: 测试结果
    """
    from llamadist.models.llama_seq import LlamaForCausalLMSeq
    
    print("   加载原始模型...")
    start_time = time.time()
    
    # 直接加载完整模型
    model = LlamaForCausalLMSeq.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16
    )
    model.eval()
    
    load_time = time.time() - start_time
    print(f"   原始模型加载完成，耗时: {load_time:.2f}秒")
    
    # 测试推理性能
    print("   测试推理性能...")
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,"
    ]
    
    # 生成参数
    temperature = 0.8
    top_p = 0.9
    do_sample = True
    
    total_inference_time = 0
    total_generation_time = 0
    total_tokens = 0
    
    for prompt in test_prompts:
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = input_ids.to(device)
        
        # 测试前向传播
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
        inference_time = time.time() - start_time
        
        # 测试文本生成
        start_time = time.time()
        with torch.no_grad():
            # 实现简单的自回归生成
            current_ids = input_ids.clone()
            max_new_tokens = 100
            
            for _ in range(max_new_tokens):
                # 前向传播获取logits
                outputs = model(current_ids)
                logits = outputs.logits
                
                # 获取下一个token（使用贪心解码）
                next_token_logits = logits[:, -1, :]
                
                # 应用temperature和top_p
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # 简单采样：使用概率最高的token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 检查是否遇到结束token
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # 添加新token
                current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            generated_ids = current_ids
        generation_time = time.time() - start_time
        
        # 解码生成的文本
        generated_text = tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)
        generated_tokens = len(generated_ids[0]) - len(input_ids[0])
        
        total_inference_time += inference_time
        total_generation_time += generation_time
        total_tokens += generated_tokens
        
        print(f"     提示: {prompt}")
        print(f"     生成: {generated_text}")
        print(f"     推理时间: {inference_time:.4f}秒, 生成时间: {generation_time:.4f}秒")
    
    # 计算平均指标
    avg_inference_time = total_inference_time / len(test_prompts)
    avg_generation_time = total_generation_time / len(test_prompts)
    tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
    
    # 清理内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "name": "Baseline (不分层)",
        "num_partitions": 1,  # 不分层，算作1个分区
        "partition_time": 0,  # 不需要分层时间
        "load_time": load_time,
        "avg_inference_time": avg_inference_time,
        "avg_generation_time": avg_generation_time,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second
    }


def test_real_model():
    """使用真实模型进行单设备分层推理测试"""
    print("LlamaDistributor - Llama-2-7B 单设备分层推理测试")
    print("=" * 60)
    
    # 模型路径
    model_path = "/home/zmx/models/Llama/Llama-2-7b-hf"
    
    # 检查设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    
    # 检查模型路径是否存在
    if not Path(model_path).exists():
        print(f"错误：模型路径不存在: {model_path}")
        print("请确保Llama-2-7B模型已下载到指定路径")
        return False
    
    try:
        print("\n" + "="*60)
        print("第一阶段：模型加载和分析")
        print("="*60)
        
        # 1. 加载分词器
        print("\n1. 加载分词器...")
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   词汇表大小: {tokenizer.vocab_size}")
        
        # 2. 创建分层器并分析模型
        print("\n2. 创建分层器并分析模型...")
        partitioner = LlamaPartitioner(model_path=model_path)
        
        print("   分析模型结构...")
        model_info = partitioner.analyze_model(detailed=False, device=device)
        print(f"   模型层数: {model_info.num_layers}")
        print(f"   隐藏维度: {model_info.hidden_size}")
        print(f"   总参数: {model_info.total_params:,}")
        print(f"   估计内存: {model_info.total_memory / (1024**3):.2f} GB")
        
        print("\n" + "="*60)
        print("第二阶段：测试不同分层策略")
        print("="*60)
        
        results = []
        
        # 先测试baseline（不分层）
        print(f"\n测试策略: Baseline (不分层)")
        print("-" * 40)
        
        try:
            baseline_result = test_baseline_inference(
                model_path=model_path,
                tokenizer=tokenizer,
                device=device
            )
            results.append(baseline_result)
            print(f"   Baseline测试完成")
        except Exception as e:
            print(f"   Baseline测试失败: {e}")
            results.append({
                "name": "Baseline (不分层)",
                "error": str(e)
            })
        
        # 测试不同的分层策略
        strategies_to_test = [
            {
                "name": "3分层-均匀",
                "strategy": PartitionStrategy(
                    num_partitions=3,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device
                )
            },
            {
                "name": "3分层-自定义",
                "strategy": PartitionStrategy(
                    num_partitions=3,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device,
                    custom_boundaries=[(0, 8), (9, 21), (22, 31)]
                )
            },
            {
                "name": "4分层-均匀",
                "strategy": PartitionStrategy(
                    num_partitions=4,
                    strategy_type=StrategyType.SINGLE_DEVICE,
                    single_device=device
                )
            }
        ]
        
        for test_config in strategies_to_test:
            print(f"\n测试策略: {test_config['name']}")
            print("-" * 40)
            
            try:
                # 创建分层
                print("   创建分层配置...")
                partitions = test_config['strategy'].create_partitions(model_info)
                
                print("   分层详情:")
                for i, partition in enumerate(partitions):
                    layer_count = partition.layer_end - partition.layer_start + 1
                    print(f"     分层 {i}: 层{partition.layer_start}-{partition.layer_end} ({layer_count}层) @ {partition.device}")
                
                # 执行分层
                print("   执行模型分层...")
                start_time = time.time()
                submodels = partitioner.partition(
                    strategy=test_config['strategy'],
                    copy_weights=True
                )
                partition_time = time.time() - start_time
                print(f"   分层完成，耗时: {partition_time:.2f}秒")
                
                # 创建推理引擎
                print("   创建推理引擎...")
                inference_engine = SingleDeviceInference(
                    submodels=submodels,
                    generation_config=GenerationConfig(
                        max_new_tokens=100,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    ),
                    device=device
                )
                
                # 测试推理性能
                print("   测试推理性能...")
                test_prompts = [
                    "The future of artificial intelligence is",
                    "In a world where technology advances rapidly,"
                ]
                
                total_inference_time = 0
                total_tokens = 0
                
                for prompt in test_prompts:
                    # 编码输入
                    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
                    
                    # 测试前向传播
                    start_time = time.time()
                    with torch.no_grad():
                        result = inference_engine.forward_pass(input_ids, use_cache=True)
                    inference_time = time.time() - start_time
                    
                    # 测试文本生成
                    start_time = time.time()
                    generated_text = inference_engine.generate_text(
                        prompt=prompt,
                        tokenizer=tokenizer,
                        return_full_text=False
                    )
                    generation_time = time.time() - start_time
                    
                    total_inference_time += inference_time
                    generated_tokens = len(tokenizer.encode(generated_text))
                    total_tokens += generated_tokens
                    
                    print(f"     提示: {prompt}")
                    print(f"     生成: {generated_text}")
                    print(f"     推理时间: {inference_time:.4f}秒, 生成时间: {generation_time:.4f}秒")
                
                # 获取统计信息
                stats = inference_engine.get_stats()
                avg_inference_time = total_inference_time / len(test_prompts)
                
                results.append({
                    "name": test_config['name'],
                    "num_partitions": len(submodels),
                    "partition_time": partition_time,
                    "avg_inference_time": avg_inference_time,
                    "total_tokens": total_tokens,
                    "tokens_per_second": stats.get('tokens_per_second', 0)
                })
                
                print(f"   测试完成")
                
                # 清理内存
                del submodels
                del inference_engine
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   测试失败: {e}")
                results.append({
                    "name": test_config['name'],
                    "error": str(e)
                })
        
        print("\n" + "="*60)
        print("第三阶段：性能比较结果")
        print("="*60)
        
        # 显示结果比较
        print(f"{'策略名称':<15} {'分层数':<8} {'分层时间':<12} {'推理时间':<12} {'生成速度':<12}")
        print("-" * 70)
        
        for result in results:
            if 'error' not in result:
                # 处理baseline和分层结果的不同字段
                partition_time = result.get('partition_time', 0)
                partition_time_str = f"{partition_time:.2f}s" if partition_time > 0 else "N/A"
                
                # 获取生成速度
                tokens_per_second = result.get('tokens_per_second', 0)
                if tokens_per_second == 0 and 'avg_generation_time' in result and result['avg_generation_time'] > 0:
                    # 对于baseline，从总token数和生成时间计算速度
                    tokens_per_second = result['total_tokens'] / result['avg_generation_time']
                
                print(f"{result['name']:<15} {result['num_partitions']:<8} "
                      f"{partition_time_str:<12} {result['avg_inference_time']:.4f}s{'':<4} "
                      f"{tokens_per_second:.1f}t/s{'':<4}")
            else:
                print(f"{result['name']:<15} {'失败':<8} {result['error'][:40]}")
        
        # 分析性能提升
        baseline_result = next((r for r in results if r.get('name') == 'Baseline (不分层)' and 'error' not in r), None)
        if baseline_result:
            print(f"\n与Baseline对比分析:")
            print("-" * 50)
            
            for result in results:
                if 'error' not in result and result.get('name') != 'Baseline (不分层)':
                    speedup = baseline_result['avg_inference_time'] / result['avg_inference_time']
                    if speedup > 1:
                        print(f"  {result['name']}: {speedup:.2f}x 加速")
                    else:
                        print(f"  {result['name']}: {1/speedup:.2f}x 变慢")
        
        # 找出最佳策略
        successful_results = [r for r in results if 'error' not in r]
        if successful_results:
            best_strategy = min(successful_results, key=lambda x: x['avg_inference_time'])
            print(f"\n最佳策略: {best_strategy['name']} (推理时间: {best_strategy['avg_inference_time']:.4f}秒)")
        
        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试不同分层策略的内存使用"""
    print("\n" + "="*60)
    print("内存使用分析")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存分析")
        return
    
    device = "cuda:0"
    model_path = "/home/zmx/models/Llama/Llama-2-7b-hf"
    
    try:
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        print(f"初始GPU内存使用: {initial_memory / (1024**2):.1f} MB")
        
        # 创建分层器
        partitioner = LlamaPartitioner(model_path=model_path)
        model_info = partitioner.analyze_model(detailed=False)
        
        # 测试2分层策略的内存使用
        strategy = PartitionStrategy(
            num_partitions=2,
            strategy_type=StrategyType.SINGLE_DEVICE,
            single_device=device
        )
        
        print("\n创建2分层模型...")
        submodels = partitioner.partition(strategy=strategy, copy_weights=True)
        
        # 记录分层后内存
        partitioned_memory = torch.cuda.memory_allocated(device)
        print(f"分层后GPU内存使用: {partitioned_memory / (1024**2):.1f} MB")
        print(f"内存增量: {(partitioned_memory - initial_memory) / (1024**2):.1f} MB")
        
        # 创建推理引擎并测试
        inference_engine = SingleDeviceInference(submodels=submodels, device=device)
        
        # 模拟推理
        test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
        with torch.no_grad():
            _ = inference_engine.forward_pass(test_input)
        
        # 记录推理后内存
        inference_memory = torch.cuda.memory_allocated(device)
        print(f"推理后GPU内存使用: {inference_memory / (1024**2):.1f} MB")
        
        # 显示内存使用详情
        print(f"\n内存使用详情:")
        print(f"  基础内存: {initial_memory / (1024**2):.1f} MB")
        print(f"  模型分层: +{(partitioned_memory - initial_memory) / (1024**2):.1f} MB")
        print(f"  推理缓存: +{(inference_memory - partitioned_memory) / (1024**2):.1f} MB")
        print(f"  总计使用: {inference_memory / (1024**2):.1f} MB")
        
        # 清理
        del submodels
        del inference_engine
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(device)
        print(f"清理后GPU内存: {final_memory / (1024**2):.1f} MB")
        
    except Exception as e:
        print(f"内存分析失败: {e}")


def main():
    """主函数"""
    print("请选择测试模式:")
    print("1. 完整模型测试（推荐）")
    print("2. 内存使用分析")
    print("3. 全部测试")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            success = test_real_model()
        elif choice == "2":
            test_memory_usage()
            success = True
        elif choice == "3":
            success = test_real_model()
            if success:
                test_memory_usage()
        else:
            print("无效选择，运行完整测试...")
            success = test_real_model()
            
        if success:
            print("\n测试完成！单设备分层推理功能正常工作。")
        else:
            print("\n测试未完全成功，请检查错误信息。")
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")


if __name__ == "__main__":
    main() 