#!/usr/bin/env python3
"""
LlamaDistributor 演示脚本

展示如何使用LlamaDistributor进行模型分层和分布式推理
"""

import torch
from transformers import AutoTokenizer

from llamadist.partitioner.analyzer import LlamaModelAnalyzer
from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig

def main():
    print("🚀 LlamaDistributor 演示")
    print("=" * 50)
    
    # 配置
    model_path = "/home/zmx/models/Llama/Llama-2-7b-hf"
    num_partitions = 2
    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cpu", "cpu"]
    
    print(f"📁 模型路径: {model_path}")
    print(f"🔧 分层数量: {num_partitions}")
    print(f"💻 目标设备: {devices}")
    print()
    
    # 1. 模型分析
    print("1️⃣ 分析模型结构...")
    analyzer = LlamaModelAnalyzer(model_path=model_path)
    model_info = analyzer.analyze_model(detailed=False)
    
    print(f"   ✓ 模型: {model_info.model_name}")
    print(f"   ✓ 层数: {model_info.num_layers}")
    print(f"   ✓ 参数: {model_info.total_params / 1e9:.2f}B")
    print(f"   ✓ 内存: {model_info.total_memory / 1e9:.2f}GB")
    print()
    
    # 2. 创建分层策略
    print("2️⃣ 创建分层策略...")
    strategy = PartitionStrategy(
        num_partitions=num_partitions,
        strategy_type="uniform",
        target_devices=devices
    )
    
    partitions = strategy.create_partitions(model_info)
    print(f"   ✓ 创建了 {len(partitions)} 个分层:")
    for i, partition in enumerate(partitions):
        print(f"     - 分层 {i}: 层 {partition.layer_start}-{partition.layer_end} -> {partition.device}")
    print()
    
    # 3. 执行模型分层
    print("3️⃣ 执行模型分层...")
    partitioner = LlamaPartitioner(model_path=model_path)
    submodels = partitioner.partition(
        strategy=strategy,
        analyze_first=False,
        copy_weights=True
    )
    
    print(f"   ✓ 成功创建 {len(submodels)} 个子模型")
    for sm in submodels:
        info = sm.get_info()
        print(f"     - 子模型 {info['partition_idx']}: {info['memory_usage']/1e6:.1f}MB @ {info['device']}")
    print()
    
    # 4. 创建分布式推理引擎
    print("4️⃣ 创建分布式推理引擎...")
    inference_engine = DistributedInference(
        submodels=submodels,
        generation_config=GenerationConfig(
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False
        )
    )
    print("   ✓ 分布式推理引擎已就绪")
    print()
    
    # 5. 测试推理
    print("5️⃣ 测试分布式推理...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试输入
    test_prompts = [
        "Hello, how are you?",
        "The capital of France is",
        "In machine learning,"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"   测试 {i+1}: '{prompt}'")
        
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        print(f"     输入形状: {input_ids.shape}")
        
        # 执行推理
        import time
        start_time = time.time()
        result = inference_engine.forward_pass(
            input_ids=input_ids,
            use_cache=False
        )
        inference_time = time.time() - start_time
        
        print(f"     推理耗时: {inference_time:.3f}秒")
        print(f"     输出形状: {result['logits'].shape}")
        
        # 获取预测的下一个token
        next_token_logits = result['logits'][0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode([next_token_id])
        print(f"     预测下一个token: '{next_token}'")
        print()
    
    # 6. 性能统计
    print("6️⃣ 性能统计...")
    stats = inference_engine.get_stats()
    print(f"   ✓ 推理次数: {stats['inference_count']}")
    print(f"   ✓ 总推理时间: {stats['total_inference_time']:.3f}秒")
    print(f"   ✓ 平均推理时间: {stats['avg_inference_time']:.3f}秒")
    print(f"   ✓ 状态传输时间: {stats['state_transfer_time']:.3f}秒")
    print()
    
    print("🎉 演示完成！LlamaDistributor成功运行")
    print("=" * 50)
    
    return submodels, inference_engine

if __name__ == "__main__":
    submodels, engine = main() 