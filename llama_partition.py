#! /usr/bin/env python3
"""
Llama 分层分割推理
"""

import torch
import sys
import time
import argparse
from pathlib import Path
from transformers import LlamaTokenizer, LlamaForCausalLM
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

# 全局测试参数
test_prompts = [
    "Llama is a large language model",
    "In a world full of love,"
]

# 全局生成参数（作为默认值）
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_DO_SAMPLE = True
DEFAULT_MAX_NEW_TOKENS = 100



def sample_next_token(logits, temperature=1.0, top_p=0.9, do_sample=True):
    """
    从logits中采样下一个token
    使用transformers内置的采样逻辑以保持一致性
    
    Args:
        logits: 模型输出的logits tensor, shape: (batch_size, seq_len, vocab_size)
        temperature: 温度参数，控制随机性，默认1.0
        top_p: top-p采样参数，默认0.9
        do_sample: 是否进行采样，如果False则使用贪婪搜索
        
    Returns:
        torch.Tensor: 采样得到的下一个token，shape: (batch_size, 1)
    """
    from transformers.generation.utils import (
        TemperatureLogitsWarper,
        TopPLogitsWarper,
        LogitsProcessorList
    )
    
    # 获取最后一个位置的logits
    logits = logits[:, -1, :]  # (batch_size, vocab_size)
    
    if not do_sample:
        # 贪婪搜索：选择概率最大的token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token
    
    # 使用transformers内置的logits processors
    logits_processors = LogitsProcessorList()
    
    # 添加温度scaling
    if temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    
    # 添加top-p filtering
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p))
    
    # 应用logits processors
    if len(logits_processors) > 0:
        # 创建dummy input_ids用于logits processor接口
        dummy_input_ids = torch.zeros((logits.shape[0], 1), dtype=torch.long, device=logits.device)
        logits = logits_processors(dummy_input_ids, logits)
    
    # 从处理后的logits中采样
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def benchmark_baseline_inference(
    model_path: str, 
    tokenizer, 
    device: str, 
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> dict:
    """
    测试完整模型推理性能作为baseline

    Args:
        model_path: 模型路径
        tokenizer: 分词器
        device: 设备
        temperature: 温度参数，控制随机性，默认0.8
        top_p: top-p采样参数，默认0.9
        do_sample: 是否启用采样，默认True
        max_new_tokens: 最大生成token数，默认100

    Returns:
        dict: 测试结果字典，包含各种性能指标
    """
    # 参数验证
    if temperature <= 0:
        raise ValueError(f"temperature必须大于0，当前值: {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p必须在(0,1]范围内，当前值: {top_p}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens必须大于0，当前值: {max_new_tokens}")
    
    print(f"完整模型推理")
    print("="*60)
    print(f"设备: {device}")
    print(f"生成参数: temperature={temperature}, top_p={top_p}, do_sample={do_sample}, max_new_tokens={max_new_tokens}")

    # 检查设备可用性
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，但指定了CUDA设备")
    
    start_time = time.time()

    try:
        # 加载原始模型 
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

    load_time = time.time() - start_time
    print(f"原始模型加载完成，耗时: {load_time:.2f}秒")

    # 初始化性能指标
    total_inference_time = 0  # 总推理时间
    total_tokens_generated = 0  # 生成的总token数
    total_prefill_time = 0  # 预填充时间（处理prompt的时间）
    total_decode_time = 0  # 解码时间（逐token生成时间）
    peak_memory_usage = 0  # 峰值内存使用
    first_token_latencies = []  # 每个prompt的首token延迟
    total_generation_time = 0 # 总生成时间

    for prompt in test_prompts:
        # 记录每个prompt的开始时间
        prompt_start_time = time.time()
        
        # tokenization
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 解码阶段
        first_token_generated = False
        generation_start_time = time.time() 
        
        with torch.no_grad():

            current_ids = input_ids.clone()
            outputs = model(current_ids)

            for token_idx in range(max_new_tokens):
                # 前向传播获取logits
                token_start_time = time.time()
                outputs = model(current_ids)
                logits = outputs.logits

                # 采样
                next_token = sample_next_token(logits, temperature, top_p, do_sample)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                
                token_end_time = time.time()
                token_decode_time = token_end_time - token_start_time
                total_decode_time += token_decode_time

                # 记录首token延迟（只记录一次）
                if not first_token_generated:
                    first_token_latency = token_end_time - prompt_start_time
                    first_token_latencies.append(first_token_latency)
                    first_token_generated = True

                total_tokens_generated += 1

                # 监控内存使用（每隔几个token检查一次即可）
                if token_idx % 10 == 0:
                    current_memory_usage = torch.cuda.memory_allocated() / 1024**2
                    peak_memory_usage = max(peak_memory_usage, current_memory_usage)

       
        # 解码和显示生成的文本
        generated_tokens = current_ids[0, len(input_ids[0]):].tolist()  # 只取新生成的token
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 记录这个prompt的总生成时间（不包含打印时间）
        generation_end_time = time.time()
        prompt_generation_time = generation_end_time - generation_start_time
        total_generation_time += prompt_generation_time
        
        print(f"\n--- Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Generated tokens: {len(generated_tokens)}")
            
    # 在所有prompt处理完后计算最终指标
    if total_tokens_generated > 0:
        average_token_decode_time = total_decode_time / total_tokens_generated
        average_throughput = total_tokens_generated / total_generation_time
        average_latency = total_generation_time / total_tokens_generated
        average_first_token_latency = sum(first_token_latencies) / len(first_token_latencies) if first_token_latencies else 0
    else:
        average_throughput = 0
        average_latency = 0
        average_first_token_latency = 0

    # 打印结果
    print(f"\n=== 完整模型推理测试结果 ===")
    print(f"模型加载时间: {load_time:.3f}秒")
    print(f"平均每token解码时间: {average_token_decode_time:.3f}秒")
    print(f"总生成时间: {total_generation_time:.3f}秒")
    print(f"生成token总数: {total_tokens_generated}")
    print(f"平均吞吐量: {average_throughput:.3f} tokens/秒")
    print(f"平均每token延迟: {average_latency*1000:.3f}毫秒/token")
    print(f"平均首token延迟(TTFT): {average_first_token_latency*1000:.3f}毫秒")
    print(f"峰值GPU内存使用: {peak_memory_usage:.3f}MB")

    return {
        "load_time": load_time,
        "total_decode_time": total_decode_time,
        "total_generation_time": total_generation_time,
        "total_tokens_generated": total_tokens_generated,
        "average_throughput": average_throughput,
        "average_latency": average_latency,
        "average_first_token_latency": average_first_token_latency,
        "peak_memory_usage": peak_memory_usage
    }




def benchmark_partition_inference(
    strategies_to_test: list[dict],
    model_path: str, 
    tokenizer, 
    device: str, 
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> dict:
    """
    测试分层分割推理性能
    """
    print("按层分割推理")
    print(f"设备: {device}")
    print(f"生成参数: temperature={temperature}, top_p={top_p}, do_sample={do_sample}, max_new_tokens={max_new_tokens}")

    print("="*60)
    
    # 检查设备可用性
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，但指定了CUDA设备")
    
    # 创建分层器
    partitioner = LlamaPartitioner(model_path=model_path)
    model_info = partitioner.analyze_model(detailed=False, device=device)
    print("分割模型信息：")
    print(f"模型层数: {model_info.num_layers}")
    print(f"隐藏维度: {model_info.hidden_size}")
    print(f"总参数: {model_info.total_params:,}")
    print(f"估计内存: {model_info.total_memory / (1024**3):.2f} GB")


    for strategy in strategies_to_test:
        print(f"按测试策略: {strategy['name']}开始分层...")
        print("-" * 30)

        partition_start_time = time.time()
        # 创建分层
        partitions = strategy['strategy'].create_partitions(model_info)
            
        print("当前分层策略：")
        for i, partition in enumerate(partitions):
            layer_count = partition.layer_end - partition.layer_start + 1
            print(f"     分层 {i}: 层{partition.layer_start}-{partition.layer_end} ({layer_count}层) @ {partition.device}")
            
        # 执行分层
        print("执行模型分层...")
        start_time = time.time()
        submodels = partitioner.partition(
            strategy = strategy['strategy'],
            copy_weights = True
        )

        partition_end_time = time.time() - partition_start_time

        print("创建分层引擎...")
        
        inference_engine = SingleDeviceInference(
            submodels=submodels,
            generation_config = GenerationConfig(
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                top_p = top_p,
                do_sample = do_sample,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            ),
            device = device
        )
        
        # 初始化性能指标
        total_tokens_generated = 0  # 生成的总token数
        total_decode_time = 0  # 解码时间（逐token生成时间）
        peak_memory_usage = 0  # 峰值内存使用
        first_token_latencies = []  # 每个prompt的首token延迟
        total_generation_time = 0 # 总生成时间

        for prompt in test_prompts:
            generation_start_time = time.time()
            generated_text = inference_engine.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                return_full_text=False
            )
            generation_time = time.time() - generation_start_time
            total_generation_time += generation_time

            generated_tokens = len(tokenizer.encode(generated_text))
            total_tokens_generated += generated_tokens

            print(f"\n--- Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print(f"Generated tokens: {generated_tokens}")
        
        stats = inference_engine.get_stats()

        

        print(f"\n=== 分割模型推理测试结果 ===")
        print(f"总解码时间: {stats['token_decode_time']:.3f}秒")
        print(f"总生成时间: {total_generation_time:.3f}秒")
        print(f"生成token总数: {total_tokens_generated}")
        print(f"平均吞吐量: {stats['tokens_per_second']:.3f} tokens/秒")
        print(f"平均每token延迟: {1/stats['tokens_per_second']*1000:.3f}毫秒")
        print(f"平均首token延迟(TTFT): {stats['time_to_first_token']*1000:.3f}毫秒")

        del submodels
        del inference_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(
    model_path: str = "/home/zmx/models/Llama/Llama-2-7b-hf",
    device: str = "cuda:0",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    benchmark_baseline_inference(
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens
    )
    # 分层策略列表
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
                custom_boundaries=[(0, 7), (8, 20), (21, 31)]
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

    benchmark_partition_inference(
        strategies_to_test=strategies_to_test,
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama模型分层分割推理性能测试")
    parser.add_argument("--model_path", type=str, 
                       default="/home/zmx/models/Llama/Llama-2-7b-hf",
                       help="模型路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备名称")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help=f"温度参数 (默认: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                       help=f"top-p采样参数 (默认: {DEFAULT_TOP_P})")
    parser.add_argument("--do_sample", action="store_true", default=DEFAULT_DO_SAMPLE,
                       help=f"是否启用采样 (默认: {DEFAULT_DO_SAMPLE})")
    parser.add_argument("--no_sample", dest="do_sample", action="store_false",
                       help="禁用采样，使用贪婪搜索")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                       help=f"最大生成token数 (默认: {DEFAULT_MAX_NEW_TOKENS})")
    
    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens
    )