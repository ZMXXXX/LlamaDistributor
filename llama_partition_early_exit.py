#! /usr/bin/env python3
"""
Llama 分层分割推理
"""

import torch
import sys
import time
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F
import matplotlib
from tabulate import tabulate

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
    "Llama is a large language model,",
    "USA is a country in North America,",
    "The capital of USA is Washington, D.C.,",
    "write a poem about LOVE",
    "The answer of 1+1 is",
    "give me a joke"
]

# 全局生成参数（作为默认值）
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_DO_SAMPLE = True
DEFAULT_MAX_NEW_TOKENS = 100


def load_strategies_from_config(config_file: str = "configs/strategies_config.json", device: str = "cuda:0") -> List[Dict[str, Any]]:
    """
    从JSON配置文件加载策略列表
    
    Args:
        config_file: 配置文件路径，默认为strategies_config.json
        device: 设备名称，用于设置strategy中的single_device参数
        
    Returns:
        List[Dict]: 包含策略对象的字典列表
    """
    # 获取配置文件的完整路径
    config_path = Path(__file__).parent / config_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"策略配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件JSON格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"读取配置文件失败: {e}")
    
    if "strategies" not in config_data:
        raise ValueError("配置文件中缺少'strategies'字段")
    
    strategies_to_test = []
    
    for strategy_config in config_data["strategies"]:
        try:
            # 验证必需字段
            required_fields = ["name", "num_partitions", "strategy_type"]
            for field in required_fields:
                if field not in strategy_config:
                    raise ValueError(f"策略配置缺少必需字段: {field}")
            
            # 转换strategy_type字符串为枚举
            strategy_type_str = strategy_config["strategy_type"]
            if strategy_type_str == "SINGLE_DEVICE":
                strategy_type = StrategyType.SINGLE_DEVICE
            else:
                # 可以在这里添加其他策略类型的支持
                raise ValueError(f"不支持的策略类型: {strategy_type_str}")
            
            # 处理custom_boundaries
            custom_boundaries = strategy_config.get("custom_boundaries")
            if custom_boundaries is not None:
                # 将list转换为tuple以符合PartitionStrategy的要求
                custom_boundaries = [tuple(boundary) for boundary in custom_boundaries]
            
            # 创建PartitionStrategy对象
            strategy = PartitionStrategy(
                num_partitions=strategy_config["num_partitions"],
                strategy_type=strategy_type,
                single_device=device,
                custom_boundaries=custom_boundaries
            )
            
            strategies_to_test.append({
                "name": strategy_config["name"],
                "strategy": strategy,
                "description": strategy_config.get("description", ""),
                "exit_position": strategy_config.get("exit_position")
            })
            
        except Exception as e:
            print(f"警告: 跳过无效的策略配置 '{strategy_config.get('name', 'unknown')}': {e}")
            continue
    
    if not strategies_to_test:
        raise ValueError("没有有效的策略配置")
    
    print(f"成功加载 {len(strategies_to_test)} 个策略配置:")
    for strategy_dict in strategies_to_test:
        description = f" - {strategy_dict['description']}" if strategy_dict['description'] else ""
        print(f"  - {strategy_dict['name']}{description}")
    
    return strategies_to_test



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
    
    load_start_time = time.time()

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

    load_time = time.time() - load_start_time
    print(f"原始模型加载完成，耗时: {load_time:.2f}秒")

    # 初始化性能指标
    total_tokens_generated = 0  # 生成的总token数
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
                    
                    # 立即解码并打印第一个token，用于验证TTFT计算
                    first_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    print(f"\n--- [完整模型] 第一个token '{first_token_text}' 生成完成，TTFT: {first_token_latency*1000:.3f}ms")

                total_tokens_generated += 1

                # 监控内存使用（每隔几个token检查一次即可）
                if token_idx % 10 == 0:
                    current_memory_usage = torch.cuda.memory_allocated() / 1024**2
                    peak_memory_usage = max(peak_memory_usage, current_memory_usage)

       
        # 解码和显示生成的文本
        generated_tokens = current_ids[0, len(input_ids[0]):].tolist()  # 只取新生成的token
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 记录这个prompt的总生成时间（不包含打印时间）
        prompt_generation_time = time.time() - generation_start_time
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
    print(f"总生成时间: {total_generation_time:.3f}秒")
    print(f"总解码时间: {total_decode_time:.3f}秒")
    print(f"生成token总数: {total_tokens_generated}")
    print(f"平均吞吐量: {average_throughput:.3f} tokens/秒")
    print(f"平均每token生成延迟: {average_latency*1000:.3f}毫秒/token")
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

    # 存储所有策略的结果
    all_results = {}

    for strategy in strategies_to_test:
        strategy_exit_position = strategy.get('exit_position')
        exit_info = f" (Early-exit: 第{strategy_exit_position}个submodel后)" if strategy_exit_position is not None else " (正常推理到最后一层)"
        print(f"按测试策略: {strategy['name']}{exit_info}开始分层...")
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
        partition_start_time = time.time()
        submodels = partitioner.partition(
            strategy = strategy['strategy'],
            copy_weights = True
        )

        

        print("创建分层引擎...")
        
        inference_engine = SingleDeviceInference(
            submodels=submodels,
            generation_config = GenerationConfig(
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                top_p = top_p,
                do_sample = do_sample,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                exit_position = strategy_exit_position
            ),
            device = device
        )
        
        # 🔧 修复：为early-exit设置原始模型的权重
        if strategy_exit_position is not None:
            print("检测到early-exit配置，正在获取原始模型权重...")
            # 临时加载原始模型以获取lm_head和norm权重
            try:
                original_model = LlamaForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu",  # 先加载到CPU以节省显存
                    torch_dtype=torch.float16
                )
                
                # 设置lm_head权重
                if hasattr(original_model, 'lm_head') and original_model.lm_head is not None:
                    inference_engine.set_original_lm_head_weights(original_model.lm_head.weight.data)
                    print("✅ 已设置原始lm_head权重")
                
                # 设置norm权重（如果需要early-exit子模型中使用）
                if hasattr(original_model.model, 'norm') and original_model.model.norm is not None:
                    # 为推理引擎设置原始norm权重
                    inference_engine._original_norm_weights = original_model.model.norm.weight.data.clone()
                    print("✅ 已设置原始norm权重")
                
                # 清理原始模型以释放内存
                del original_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("✅ 已清理临时加载的原始模型")
                    
            except Exception as e:
                print(f"⚠️  警告：无法加载原始模型权重，early-exit可能效果不佳: {e}")

        partition_time = time.time() - partition_start_time
        

        for prompt in test_prompts:
            generated_text = inference_engine.generate_text(
                prompt=prompt,
                tokenizer=tokenizer,
                return_full_text=False
            )
            generated_tokens = len(tokenizer.encode(generated_text))

            print(f"\n--- Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print(f"Generated tokens: {generated_tokens}")
        
        stats = inference_engine.get_stats()

        
        print(f"\n=== 分割模型推理测试结果 ===")
        print(f"模型分层时间: {partition_time:.3f}秒")
        print(f"总生成时间: {stats['total_generation_time']:.3f}秒")
        print(f"总解码时间: {stats['token_decode_time']:.3f}秒")
        print(f"生成token总数: {stats['total_tokens_generated']}")
        print(f"平均吞吐量: {stats['tokens_per_second']:.3f} tokens/秒")
        print(f"平均每token生成延迟: {stats['total_generation_time']/stats['total_tokens_generated']*1000:.3f}毫秒/token")
        print(f"平均首token延迟(TTFT): {stats['total_time_to_first_token']/len(test_prompts)*1000:.3f}毫秒")

        # 存储当前策略的结果
        strategy_result = {
            "partition_time": partition_time,
            "total_generation_time": stats['total_generation_time'],
            "total_decode_time": stats['token_decode_time'],
            "total_tokens_generated": stats['total_tokens_generated'],
            "average_throughput": stats['tokens_per_second'],
            "average_latency": stats['total_generation_time']/stats['total_tokens_generated'],
            "average_first_token_latency": stats['total_time_to_first_token']/len(test_prompts),
            "peak_memory_usage": 0  # 这里可以后续添加内存监控
        }
        
        all_results[strategy['name']] = strategy_result

        del submodels
        del inference_engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_results


def create_comparison_table(baseline_result: dict, partition_results: dict, benchmark: bool = False):
    """
    创建完整推理和分割推理的对比表格，并保存到benchmark子文件夹
    
    Args:
        baseline_result: 完整推理的结果字典
        partition_results: 分割推理的结果字典，包含多个策略的结果
        benchmark: 是否生成benchmark图表和详细分析
    """
    # 根据benchmark参数决定是否创建文件夹和时间戳
    if benchmark:
        # 创建benchmark文件夹
        benchmark_dir = Path("benchmark")
        benchmark_dir.mkdir(exist_ok=True)
        
        # 生成时间戳用于文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        benchmark_dir = None
        timestamp = None
    
    try:
        # 导入tabulate用于创建表格
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        print("未安装tabulate库，将使用简单格式输出")
        print("请运行: pip install tabulate")
        use_tabulate = False
    
    print("\n" + "="*80)
    print("性能对比表格")
    print("="*80)
    
    # 定义指标映射
    metrics = [
        ("模型加载/分层时间 (秒)", "load_time", "partition_time"),
        ("总生成时间 (秒)", "total_generation_time", "total_generation_time"),
        ("总解码时间 (秒)", "total_decode_time", "total_decode_time"),
        ("生成token总数", "total_tokens_generated", "total_tokens_generated"),
        ("平均吞吐量 (tokens/秒)", "average_throughput", "average_throughput"),
        ("平均每token延迟 (毫秒)", "average_latency", "average_latency"),
        ("平均首token延迟 (毫秒)", "average_first_token_latency", "average_first_token_latency"),
        ("峰值内存使用 (MB)", "peak_memory_usage", "peak_memory_usage")
    ]
    
    # 准备表格数据
    headers = ["指标", "完整推理"]
    for strategy_name in partition_results.keys():
        headers.append(f"分割推理-{strategy_name}")
    
    table_data = []
    csv_data = [headers]  # 用于保存CSV格式
    
    for metric_name, baseline_key, partition_key in metrics:
        row = [metric_name]
        
        # 添加完整推理的值
        baseline_value = baseline_result.get(baseline_key, 0)
        if "延迟" in metric_name or "毫秒" in metric_name:
            # 延迟相关指标转换为毫秒
            if baseline_key in ["average_latency", "average_first_token_latency"]:
                baseline_value = baseline_value * 1000
            row.append(f"{baseline_value:.3f}")
        elif "吞吐量" in metric_name or "tokens/秒" in metric_name:
            row.append(f"{baseline_value:.3f}")
        elif metric_name == "生成token总数":
            row.append(f"{int(baseline_value)}")
        else:
            row.append(f"{baseline_value:.3f}")
        
        # 添加分割推理的值
        for strategy_name, strategy_result in partition_results.items():
            partition_value = strategy_result.get(partition_key, 0)
            if "延迟" in metric_name or "毫秒" in metric_name:
                # 延迟相关指标转换为毫秒
                if partition_key in ["average_latency", "average_first_token_latency"]:
                    partition_value = partition_value * 1000
                row.append(f"{partition_value:.3f}")
            elif "吞吐量" in metric_name or "tokens/秒" in metric_name:
                row.append(f"{partition_value:.3f}")
            elif metric_name == "生成token总数":
                row.append(f"{int(partition_value)}")
            else:
                row.append(f"{partition_value:.3f}")
        
        table_data.append(row)
        csv_data.append(row)
    
    # 打印表格到控制台
    if use_tabulate:
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
        print(table_str)
    else:
        _print_simple_table(table_data, headers)
    
    # 只有在benchmark模式下才保存文件
    if benchmark:
        # 保存表格到文件
        table_file = benchmark_dir / f"performance_comparison_{timestamp}.txt"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write("性能对比表格\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if use_tabulate:
                f.write(table_str)
            else:
                # 写入简单格式表格
                header_line = " | ".join(f"{h:<20}" for h in headers)
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")
                for row in table_data:
                    row_line = " | ".join(f"{str(cell):<20}" for cell in row)
                    f.write(row_line + "\n")
        
        # 保存CSV格式
        csv_file = benchmark_dir / f"performance_comparison_{timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            for row in csv_data:
                f.write(",".join(str(cell) for cell in row) + "\n")
    
    # 计算并显示性能提升/下降
    print("\n" + "="*80)
    print("性能对比分析 (相对于完整推理)")
    print("="*80)
    
    comparison_data = []
    comparison_headers = ["指标"]
    for strategy_name in partition_results.keys():
        comparison_headers.append(f"{strategy_name} (变化%)")
    
    key_metrics = [
        ("平均吞吐量", "average_throughput", "average_throughput", "higher_is_better"),
        ("平均每token延迟", "average_latency", "average_latency", "lower_is_better"),
        ("平均首token延迟", "average_first_token_latency", "average_first_token_latency", "lower_is_better"),
        ("总生成时间", "total_generation_time", "total_generation_time", "lower_is_better")
    ]
    
    for metric_name, baseline_key, partition_key, direction in key_metrics:
        row = [metric_name]
        baseline_value = baseline_result.get(baseline_key, 0)
        
        for strategy_name, strategy_result in partition_results.items():
            partition_value = strategy_result.get(partition_key, 0)
            
            if baseline_value != 0:
                change_percent = ((partition_value - baseline_value) / baseline_value) * 100
                
                # 根据指标方向确定是改进还是退化
                if direction == "higher_is_better":
                    status = "↑" if change_percent > 0 else "↓"
                else:  # lower_is_better
                    status = "↓" if change_percent > 0 else "↑"
                
                row.append(f"{change_percent:+.2f}% {status}")
            else:
                row.append("N/A")
        
        comparison_data.append(row)
    
    if use_tabulate:
        comparison_table = tabulate(comparison_data, headers=comparison_headers, tablefmt="grid")
        print(comparison_table)
    else:
        _print_simple_table(comparison_data, comparison_headers)
    print("\n说明: ↑ 表示性能提升, ↓ 表示性能下降")
    
    # 只有在benchmark模式下才保存性能分析和生成图表
    if benchmark:
        # 保存性能分析表格
        analysis_file = benchmark_dir / f"performance_analysis_{timestamp}.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("性能对比分析 (相对于完整推理)\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if use_tabulate:
                f.write(comparison_table)
            else:
                header_line = " | ".join(f"{h:<20}" for h in comparison_headers)
                f.write(header_line + "\n")
                f.write("-" * len(header_line) + "\n")
                for row in comparison_data:
                    row_line = " | ".join(f"{str(cell):<20}" for cell in row)
                    f.write(row_line + "\n")
            f.write("\n\n说明: ↑ 表示性能提升, ↓ 表示性能下降")
        
        # 生成性能对比图表
        _create_performance_charts(baseline_result, partition_results, benchmark_dir, timestamp)
        
        print(f"\n📁 结果已保存到benchmark文件夹:")
        print(f"   - 性能表格: {table_file}")
        print(f"   - CSV数据: {csv_file}")
        print(f"   - 性能分析: {analysis_file}")
        print(f"   - 性能图表: benchmark/performance_charts_{timestamp}.png")
    else:
        print(f"\n✅ 性能对比完成 (如需保存图表和详细分析，请使用 --benchmark 参数)")


def _print_simple_table(table_data, headers):
    """打印简单格式表格"""
    # 打印标题行
    header = " | ".join(f"{h:<20}" for h in headers)
    print(header)
    print("-" * len(header))
    
    # 打印数据行
    for row in table_data:
        row_line = " | ".join(f"{str(cell):<20}" for cell in row)
        print(row_line)


def _create_performance_charts(baseline_result: dict, partition_results: dict, 
                              benchmark_dir: Path, timestamp: str):
    """
    创建性能对比图表
    
    Args:
        baseline_result: 完整推理结果
        partition_results: 分割推理结果
        benchmark_dir: 保存目录
        timestamp: 时间戳
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 配置中文字体
        try:
            # 尝试设置中文字体
            import matplotlib.font_manager as fm
            # 查找系统中的中文字体
            chinese_fonts = []
            for font in fm.fontManager.ttflist:
                # 优先选择真正的中文字体
                if any(keyword in font.name for keyword in ['Noto Sans CJK', 'SimHei', 'Microsoft YaHei', 'WenQuanYi']):
                    chinese_fonts.append(font.name)
            
            if chinese_fonts:
                # 去重并选择最佳字体
                unique_fonts = list(set(chinese_fonts))
                # 优先级：Noto Sans CJK > SimHei > Microsoft YaHei > WenQuanYi
                for preferred in ['Noto Sans CJK', 'SimHei', 'Microsoft YaHei', 'WenQuanYi']:
                    for font in unique_fonts:
                        if preferred in font:
                            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                            plt.rcParams['axes.unicode_minus'] = False
                            use_chinese = True
                            print(f"使用中文字体: {font}")
                            break
                    if 'use_chinese' in locals():
                        break
                else:
                    # 如果没有找到优先字体，使用第一个可用的
                    plt.rcParams['font.sans-serif'] = [unique_fonts[0], 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False
                    use_chinese = True
                    print(f"使用中文字体: {unique_fonts[0]}")
            else:
                # 如果没有中文字体，使用英文标签
                use_chinese = False
                print("未找到中文字体，将使用英文标签")
        except:
            use_chinese = False
            
    except ImportError:
        print("未安装matplotlib，跳过图表生成")
        print("请运行: pip install matplotlib")
        return
    
    # 准备数据
    strategies = list(partition_results.keys())
    strategies.insert(0, "Complete Model" if not use_chinese else "完整推理")
    
    # 关键指标
    throughput_data = [baseline_result.get("average_throughput", 0)]
    latency_data = [baseline_result.get("average_latency", 0) * 1000]  # 转换为毫秒
    ttft_data = [baseline_result.get("average_first_token_latency", 0) * 1000]  # 转换为毫秒
    generation_time_data = [baseline_result.get("total_generation_time", 0)]
    
    for strategy_result in partition_results.values():
        throughput_data.append(strategy_result.get("average_throughput", 0))
        latency_data.append(strategy_result.get("average_latency", 0) * 1000)
        ttft_data.append(strategy_result.get("average_first_token_latency", 0) * 1000)
        generation_time_data.append(strategy_result.get("total_generation_time", 0))
    
    # 创建4个子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 设置标题
    main_title = 'LLM推理性能对比' if use_chinese else 'LLM Inference Performance Comparison'
    fig.suptitle(main_title, fontsize=16, fontweight='bold')
    
    # 颜色设置 - 明亮生动但不过分饱和的配色（支持12+种策略）
    colors = [
        '#4A90E2',  # 明亮的蓝色
        '#7ED321',  # 鲜绿色
        '#F5A623',  # 橙黄色
        '#BD10E0',  # 紫色
        '#50E3C2',  # 青绿色
        '#F8E71C',  # 柠檬黄
        '#B8E986',  # 浅绿色
        '#9013FE',  # 蓝紫色
        '#FF6B6B',  # 珊瑚红
        '#4ECDC4',  # 薄荷绿
        '#45B7D1',  # 天蓝色
        '#96CEB4',  # 薄荷绿
        '#FFEAA7',  # 奶油黄
        '#DDA0DD',  # 淡紫色
        '#98D8C8',  # 浅蓝绿
        '#F7DC6F',  # 金黄色
        '#BB8FCE',  # 薰衣草紫
        '#85C1E9',  # 浅蓝色
        '#F8C471',  # 桃色
        '#82E0AA'   # 淡绿色
    ]
    
    # 1. 平均吞吐量对比
    bars1 = ax1.bar(strategies, throughput_data, color=[colors[i % len(colors)] for i in range(len(strategies))])
    title1 = '平均吞吐量 (tokens/秒)' if use_chinese else 'Average Throughput (tokens/sec)'
    ylabel1 = 'Tokens/秒' if use_chinese else 'Tokens/sec'
    ax1.set_title(title1, fontweight='bold')
    ax1.set_ylabel(ylabel1)
    ax1.tick_params(axis='x', rotation=45)
    # 在柱子上添加数值标签
    for bar, value in zip(bars1, throughput_data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')
    
    # 2. 平均每token延迟对比
    bars2 = ax2.bar(strategies, latency_data, color=[colors[i % len(colors)] for i in range(len(strategies))])
    title2 = '平均每token延迟 (毫秒)' if use_chinese else 'Average Token Latency (ms)'
    ylabel2 = '毫秒' if use_chinese else 'Milliseconds'
    ax2.set_title(title2, fontweight='bold')
    ax2.set_ylabel(ylabel2)
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, latency_data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 3. 首token延迟对比
    bars3 = ax3.bar(strategies, ttft_data, color=[colors[i % len(colors)] for i in range(len(strategies))])
    title3 = '平均首token延迟 TTFT (毫秒)' if use_chinese else 'Average TTFT (ms)'
    ylabel3 = '毫秒' if use_chinese else 'Milliseconds'
    ax3.set_title(title3, fontweight='bold')
    ax3.set_ylabel(ylabel3)
    ax3.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars3, ttft_data):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 4. 总生成时间对比
    bars4 = ax4.bar(strategies, generation_time_data, color=[colors[i % len(colors)] for i in range(len(strategies))])
    title4 = '总生成时间 (秒)' if use_chinese else 'Total Generation Time (sec)'
    ylabel4 = '秒' if use_chinese else 'Seconds'
    ax4.set_title(title4, fontweight='bold')
    ax4.set_ylabel(ylabel4)
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars4, generation_time_data):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    chart_file = benchmark_dir / f"performance_charts_{timestamp}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建性能变化百分比图表
    _create_percentage_change_chart(baseline_result, partition_results, benchmark_dir, timestamp, use_chinese)


def _create_percentage_change_chart(baseline_result: dict, partition_results: dict,
                                   benchmark_dir: Path, timestamp: str, use_chinese: bool):
    """
    创建性能变化百分比图表
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    
    strategies = list(partition_results.keys())
    
    # 根据语言设置指标名称
    if use_chinese:
        metrics = ["吞吐量", "每token延迟", "首token延迟", "总生成时间"]
    else:
        metrics = ["Throughput", "Token Latency", "TTFT", "Generation Time"]
    
    # 计算百分比变化
    changes = {strategy: [] for strategy in strategies}
    
    for strategy_name, strategy_result in partition_results.items():
        # 吞吐量 (higher is better)
        baseline_throughput = baseline_result.get("average_throughput", 0)
        partition_throughput = strategy_result.get("average_throughput", 0)
        throughput_change = ((partition_throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput != 0 else 0
        
        # 每token延迟 (lower is better, 所以取负值让改进显示为正)
        baseline_latency = baseline_result.get("average_latency", 0)
        partition_latency = strategy_result.get("average_latency", 0)
        latency_change = -((partition_latency - baseline_latency) / baseline_latency * 100) if baseline_latency != 0 else 0
        
        # 首token延迟 (lower is better)
        baseline_ttft = baseline_result.get("average_first_token_latency", 0)
        partition_ttft = strategy_result.get("average_first_token_latency", 0)
        ttft_change = -((partition_ttft - baseline_ttft) / baseline_ttft * 100) if baseline_ttft != 0 else 0
        
        # 总生成时间 (lower is better)
        baseline_time = baseline_result.get("total_generation_time", 0)
        partition_time = strategy_result.get("total_generation_time", 0)
        time_change = -((partition_time - baseline_time) / baseline_time * 100) if baseline_time != 0 else 0
        
        changes[strategy_name] = [throughput_change, latency_change, ttft_change, time_change]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.25
    # 使用与主图表相同的明亮生动配色
    colors = [
        '#4A90E2',  # 明亮的蓝色
        '#7ED321',  # 鲜绿色
        '#F5A623',  # 橙黄色
        '#BD10E0',  # 紫色
        '#50E3C2',  # 青绿色
        '#F8E71C',  # 柠檬黄
        '#B8E986',  # 浅绿色
        '#9013FE',  # 蓝紫色
        '#FF6B6B',  # 珊瑚红
        '#4ECDC4',  # 薄荷绿
        '#45B7D1',  # 天蓝色
        '#96CEB4',  # 薄荷绿
        '#FFEAA7',  # 奶油黄
        '#DDA0DD',  # 淡紫色
        '#98D8C8',  # 浅蓝绿
        '#F7DC6F',  # 金黄色
        '#BB8FCE',  # 薰衣草紫
        '#85C1E9',  # 浅蓝色
        '#F8C471',  # 桃色
        '#82E0AA'   # 淡绿色
    ]
    
    for i, (strategy_name, strategy_changes) in enumerate(changes.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, strategy_changes, width, label=strategy_name, color=colors[i % len(colors)])
        
        # 添加数值标签
        for bar, value in zip(bars, strategy_changes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.5 if height >= 0 else -1),
                   f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 设置标签和标题
    if use_chinese:
        ax.set_xlabel('性能指标')
        ax.set_ylabel('性能改进 (%)')
        ax.set_title('性能改进对比 (相对于完整推理)\n正值表示性能提升，负值表示性能下降', fontweight='bold')
    else:
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Performance Improvement (%)')
        ax.set_title('Performance Improvement Comparison (vs Complete Model)\nPositive values indicate improvement, negative values indicate degradation', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    change_chart_file = benchmark_dir / f"performance_improvement_{timestamp}.png"
    plt.savefig(change_chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   - 性能改进图: benchmark/performance_improvement_{timestamp}.png")


def main(
    model_path: str = "/home/zmx/models/Llama/layerskip-llama2-7B",
    device: str = "cuda:0",
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    benchmark: bool = False,
    config_file: str = "configs/strategies_config.json"
):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    # 运行完整推理基准测试
    baseline_result = benchmark_baseline_inference(
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens
    )
    
    # 从JSON配置文件加载分层策略列表
    try:
        strategies_to_test = load_strategies_from_config(config_file=config_file, device=device)
    except Exception as e:
        print(f"错误: 加载策略配置失败: {e}")
        print("请检查配置文件是否存在且格式正确")
        return

    # 运行分层推理基准测试
    partition_results = benchmark_partition_inference(
        strategies_to_test=strategies_to_test,
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens
    )
    
    # 创建对比表格
    create_comparison_table(baseline_result, partition_results, benchmark=benchmark)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama模型分层分割推理性能测试")
    parser.add_argument("--model_path", type=str, 
                       default="/home/zmx/models/Llama/layerskip-llama2-7B",
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
    parser.add_argument("--benchmark", action="store_true", default=False,
                       help="是否生成benchmark图表和详细分析（默认: False）")
    parser.add_argument("--config_file", type=str, default="configs/strategies_config.json",
                       help="策略配置文件路径（默认: configs/strategies_config.json）")
    
    args = parser.parse_args()
    
    # 运行测试并自动生成对比表格和图片
    main(
        model_path=args.model_path,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
        benchmark=args.benchmark,
        config_file=args.config_file
    )

