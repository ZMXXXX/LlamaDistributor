#!/usr/bin/env python3
"""
LlamaDistributor 简单文本生成演示

展示如何使用分层推理来生成文本回答
"""

import torch
from transformers import AutoTokenizer

from llamadist.partitioner.analyzer import LlamaModelAnalyzer
from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig

def simple_generate_text(prompt: str, model_path: str = "/home/zmx/models/Llama/Llama-2-7b-hf"):
    """
    简单的文本生成函数
    
    Args:
        prompt: 输入提示
        model_path: 模型路径
    
    Returns:
        str: 生成的文本
    """
    print(f"🚀 开始分布式文本生成")
    print(f"📝 输入提示: '{prompt}'")
    print("=" * 60)
    
    # 配置
    num_partitions = 2
    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cpu", "cpu"]
    
    print(f"💻 使用设备: {devices}")
    
    # 1. 创建分层策略
    print("1️⃣ 创建分层策略...")
    strategy = PartitionStrategy(
        num_partitions=num_partitions,
        strategy_type="uniform",
        target_devices=devices
    )
    
    # 2. 分层模型
    print("2️⃣ 分层模型...")
    partitioner = LlamaPartitioner(model_path=model_path)
    submodels = partitioner.partition(
        strategy=strategy,
        analyze_first=False,
        copy_weights=True
    )
    
    # 3. 创建分布式推理引擎
    print("3️⃣ 创建分布式推理引擎...")
    inference_engine = DistributedInference(
        submodels=submodels,
        generation_config=GenerationConfig(
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            use_cache=True
        )
    )
    
    # 4. 加载分词器
    print("4️⃣ 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置生成配置中的特殊token
    inference_engine.generation_config.eos_token_id = tokenizer.eos_token_id
    inference_engine.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # 5. 生成文本
    print("5️⃣ 开始生成...")
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"   输入token数量: {input_ids.shape[1]}")
    
    # 确保输入在第一个子模型的设备上
    first_device = submodels[0].get_info()['device']
    input_ids = input_ids.to(first_device)
    
    # 逐步生成文本（简化版本，不使用KV缓存）
    generated_tokens = []
    all_input_ids = input_ids
    
    max_new_tokens = 10
    for step in range(max_new_tokens):
        print(f"   生成步骤 {step + 1}...")
        
        # 执行前向传播（不使用缓存，每次都传递完整序列）
        result = inference_engine.forward_pass(
            input_ids=all_input_ids,
            past_key_values=None,
            use_cache=False
        )
        
        # 获取下一个token的logits
        next_token_logits = result['logits'][0, -1, :]
        
        # 应用温度
        next_token_logits = next_token_logits / 0.7
        
        # 简单采样（使用top-k）
        top_k = 50
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # 采样下一个token
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 检查是否是EOS token
        if next_token.item() == tokenizer.eos_token_id:
            print("   遇到EOS token，停止生成")
            break
        
        # 解码token
        next_token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        generated_tokens.append(next_token.item())
        print(f"   生成token: '{next_token_text}' (ID: {next_token.item()})")
        
        # 将新token添加到序列中（确保在同一设备上）
        next_token_device = next_token.to(first_device)
        all_input_ids = torch.cat([all_input_ids, next_token_device.unsqueeze(0)], dim=1)
    
    # 解码完整生成的文本
    generated_text = tokenizer.decode(all_input_ids[0], skip_special_tokens=True)
    
    print("=" * 60)
    print(f"✅ 生成完成！")
    print(f"📄 完整回答: {generated_text}")
    
    return generated_text

def test_qa_examples():
    """测试问答示例"""
    examples = [
        "What is the capital of France?",
        "How does machine learning work?",
        "Tell me a joke about programming.",
    ]
    
    print("🧪 测试多个问答示例")
    print("=" * 60)
    
    for i, question in enumerate(examples):
        print(f"\n📋 示例 {i+1}: {question}")
        print("-" * 40)
        
        try:
            answer = simple_generate_text(question)
            print(f"✅ 成功生成回答")
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 测试单个示例
    prompt = "The future of artificial intelligence is"
    try:
        result = simple_generate_text(prompt)
        print("\n" + "=" * 60)
        print("🎉 单个示例测试成功！")
        
        # 测试多个示例
        print("\n")
        test_qa_examples()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 