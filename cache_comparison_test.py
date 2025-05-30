#!/usr/bin/env python3
"""
KV缓存性能对比测试

对比使用和不使用KV缓存的性能差异
"""

import torch
import time
from transformers import AutoTokenizer

from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig

def benchmark_cache_vs_no_cache(model_path: str = "/home/zmx/models/Llama/Llama-2-7b-hf"):
    """对比使用和不使用KV缓存的性能"""
    
    print("🔥 KV缓存性能对比测试")
    print("=" * 60)
    
    # 配置
    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cpu", "cpu"]
    
    # 创建分层策略
    strategy = PartitionStrategy(
        num_partitions=2,
        strategy_type="uniform", 
        target_devices=devices
    )
    
    # 分层模型
    partitioner = LlamaPartitioner(model_path=model_path)
    submodels = partitioner.partition(strategy=strategy, analyze_first=False, copy_weights=True)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试提示
    prompt = "The benefits of using cache in language models include"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    first_device = submodels[0].get_info()['device'] 
    input_ids = input_ids.to(first_device)
    
    print(f"📝 测试提示: {prompt}")
    print(f"💻 使用设备: {devices}")
    
    # 测试参数
    max_tokens = 20  # 减少token数量以加快测试
    
    # === 测试1: 不使用KV缓存 ===
    print("\n🐌 测试1: 不使用KV缓存")
    inference_engine_no_cache = DistributedInference(
        submodels=submodels,
        generation_config=GenerationConfig(use_cache=False)
    )
    
    times_no_cache = []
    sequence_lengths = []
    all_input_ids = input_ids.clone()
    
    start_time = time.time()
    for step in range(max_tokens):
        step_start = time.time()
        
        # 每次传递完整序列
        result = inference_engine_no_cache.forward_pass(
            input_ids=all_input_ids,
            past_key_values=None,
            use_cache=False
        )
        
        # 简单贪心采样
        next_token = torch.argmax(result['logits'][0, -1, :], dim=-1)
        # 确保next_token在正确的设备上
        next_token = next_token.to(first_device)
        all_input_ids = torch.cat([all_input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        step_time = time.time() - step_start
        times_no_cache.append(step_time)
        sequence_lengths.append(all_input_ids.shape[1])
        
        if step % 3 == 0:
            print(f"   步骤 {step}: {step_time:.3f}s (序列长度: {all_input_ids.shape[1]})")
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    total_time_no_cache = time.time() - start_time
    
    # === 测试2: 使用KV缓存 ===
    print("\n🚀 测试2: 使用KV缓存")
    inference_engine_with_cache = DistributedInference(
        submodels=submodels,
        generation_config=GenerationConfig(use_cache=True)
    )
    
    times_with_cache = []
    generated_ids = input_ids.clone()
    past_key_values = None
    
    start_time = time.time()
    for step in range(max_tokens):
        step_start = time.time()
        
        # 第一步传递完整序列，后续只传递最后一个token
        current_input = generated_ids if step == 0 else generated_ids[:, -1:]
        
        result = inference_engine_with_cache.forward_pass(
            input_ids=current_input,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # 简单贪心采样
        next_token = torch.argmax(result['logits'][0, -1, :], dim=-1)
        # 确保next_token在正确的设备上 
        next_token = next_token.to(first_device)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        past_key_values = result['past_key_values']
        
        step_time = time.time() - step_start
        times_with_cache.append(step_time)
        
        if step % 3 == 0:
            print(f"   步骤 {step}: {step_time:.3f}s (序列长度: {generated_ids.shape[1]})")
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    total_time_with_cache = time.time() - start_time
    
    # === 结果对比 ===
    print("\n📊 性能对比结果")
    print("=" * 60)
    print(f"不使用缓存总时间: {total_time_no_cache:.2f}s")
    print(f"使用缓存总时间:   {total_time_with_cache:.2f}s") 
    speedup = total_time_no_cache/total_time_with_cache if total_time_with_cache > 0 else 0
    print(f"性能提升:         {speedup:.1f}x")
    
    # 平均每步时间
    avg_time_no_cache = sum(times_no_cache) / len(times_no_cache) if times_no_cache else 0
    avg_time_with_cache = sum(times_with_cache) / len(times_with_cache) if times_with_cache else 0
    
    print(f"\n平均每步时间:")
    print(f"不使用缓存: {avg_time_no_cache:.3f}s")
    print(f"使用缓存:   {avg_time_with_cache:.3f}s")
    if avg_time_with_cache > 0:
        print(f"每步提升:   {avg_time_no_cache/avg_time_with_cache:.1f}x")
    
    # 最后几步对比（显示二次增长）
    if len(times_no_cache) >= 5:
        print(f"\n时间增长趋势:")
        for i in range(min(5, len(times_no_cache))):
            no_cache_time = times_no_cache[i]
            with_cache_time = times_with_cache[i] if i < len(times_with_cache) else 0
            seq_len = sequence_lengths[i] if i < len(sequence_lengths) else 0
            ratio = no_cache_time / with_cache_time if with_cache_time > 0 else 0
            print(f"  步骤 {i}: 无缓存={no_cache_time:.3f}s, 有缓存={with_cache_time:.3f}s, 提升={ratio:.1f}x, 序列长度={seq_len}")
    
    # 生成的文本对比
    text_no_cache = tokenizer.decode(all_input_ids[0], skip_special_tokens=True)
    text_with_cache = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"\n📄 生成文本对比:")
    print(f"无缓存结果: {text_no_cache}")
    print(f"有缓存结果: {text_with_cache}")
    
    return {
        'times_no_cache': times_no_cache,
        'times_with_cache': times_with_cache,
        'sequence_lengths': sequence_lengths,
        'total_time_no_cache': total_time_no_cache,
        'total_time_with_cache': total_time_with_cache
    }

def analyze_complexity_growth(results):
    """分析计算复杂度增长"""
    print("\n🔍 计算复杂度分析")
    print("=" * 60)
    
    if len(results['times_no_cache']) < 3:
        print("数据点不足，无法分析")
        return
    
    # 分析无缓存时间增长
    print("不使用缓存的时间增长模式:")
    for i in range(1, min(len(results['times_no_cache']), 10)):
        current_time = results['times_no_cache'][i]
        previous_time = results['times_no_cache'][i-1]
        growth_ratio = current_time / previous_time if previous_time > 0 else 0
        seq_len = results['sequence_lengths'][i] if i < len(results['sequence_lengths']) else 0
        print(f"  步骤 {i}: {current_time:.3f}s (增长 {growth_ratio:.2f}x, 序列长度 {seq_len})")
    
    # 分析有缓存时间稳定性
    if results['times_with_cache']:
        print("\n使用缓存的时间稳定性:")
        cache_times = results['times_with_cache']
        avg_cache_time = sum(cache_times) / len(cache_times)
        variance = sum((t - avg_cache_time) ** 2 for t in cache_times) / len(cache_times)
        std_dev = variance ** 0.5
        
        print(f"  平均时间: {avg_cache_time:.3f}s")
        print(f"  标准差:   {std_dev:.3f}s")
        print(f"  变异系数: {std_dev/avg_cache_time:.3f} (越小越稳定)")

if __name__ == "__main__":
    try:
        results = benchmark_cache_vs_no_cache()
        analyze_complexity_growth(results)
        
        print("\n" + "=" * 60)
        print("🎯 测试结论:")
        print("1. KV缓存显著提升性能，避免重复计算")
        print("2. 不使用缓存时，生成时间随序列长度增长而增长")
        print("3. 在分布式环境中，缓存还能减少跨设备数据传输")
        print("4. 建议在生产环境中总是启用KV缓存")
        print("5. 特别是生成长文本时，性能差异会更加明显")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 