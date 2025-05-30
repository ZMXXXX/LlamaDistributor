#!/usr/bin/env python3
"""
LlamaDistributor 交互式问答演示

用户可以输入问题，通过分层推理获得回答
"""

import torch
from transformers import AutoTokenizer
import time

from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig

class LlamaDistributorQA:
    """LlamaDistributor问答系统"""
    
    def __init__(self, model_path: str = "/home/zmx/models/Llama/Llama-2-7b-hf"):
        """初始化问答系统"""
        self.model_path = model_path
        self.inference_engine = None
        self.tokenizer = None
        self.setup()
    
    def setup(self):
        """设置分布式推理环境"""
        print("🚀 初始化LlamaDistributor问答系统...")
        
        # 配置
        num_partitions = 2
        devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cpu", "cpu"]
        
        print(f"💻 使用设备: {devices}")
        
        # 创建分层策略
        strategy = PartitionStrategy(
            num_partitions=num_partitions,
            strategy_type="uniform",
            target_devices=devices
        )
        
        # 分层模型
        print("🔧 分层模型...")
        partitioner = LlamaPartitioner(model_path=self.model_path)
        submodels = partitioner.partition(
            strategy=strategy,
            analyze_first=False,
            copy_weights=True
        )
        
        # 创建分布式推理引擎
        self.inference_engine = DistributedInference(
            submodels=submodels,
            generation_config=GenerationConfig(
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=False  # 暂时不使用缓存以确保稳定性
            )
        )
        
        # 加载分词器
        print("📚 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ 初始化完成！")
        print("=" * 60)
    
    def answer_question(self, question: str, max_new_tokens: int = 20) -> str:
        """回答问题"""
        print(f"❓ 问题: {question}")
        print("💭 思考中...")
        
        start_time = time.time()
        
        # 编码输入
        input_ids = self.tokenizer.encode(question, return_tensors='pt')
        first_device = self.inference_engine.submodels[0].get_info()['device']
        input_ids = input_ids.to(first_device)
        
        # 生成回答
        generated_tokens = []
        all_input_ids = input_ids
        
        for step in range(max_new_tokens):
            # 执行前向传播
            result = self.inference_engine.forward_pass(
                input_ids=all_input_ids,
                past_key_values=None,
                use_cache=False
            )
            
            # 获取下一个token
            next_token_logits = result['logits'][0, -1, :]
            next_token_logits = next_token_logits / 0.7
            
            # Top-k采样
            top_k = 50
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 检查停止条件
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # 检查是否是句号，如果是则可能停止
            if self.tokenizer.decode([next_token.item()]).strip() == '.':
                generated_tokens.append(next_token.item())
                next_token_device = next_token.to(first_device)
                all_input_ids = torch.cat([all_input_ids, next_token_device.unsqueeze(0)], dim=1)
                break
            
            # 添加到序列
            generated_tokens.append(next_token.item())
            next_token_device = next_token.to(first_device)
            all_input_ids = torch.cat([all_input_ids, next_token_device.unsqueeze(0)], dim=1)
        
        # 解码完整答案
        answer = self.tokenizer.decode(all_input_ids[0], skip_special_tokens=True)
        
        elapsed_time = time.time() - start_time
        print(f"⚡ 生成时间: {elapsed_time:.2f}秒")
        print(f"💬 回答: {answer}")
        print("-" * 60)
        
        return answer
    
    def interactive_mode(self):
        """交互模式"""
        print("🎯 进入交互问答模式")
        print("输入问题，输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        
        while True:
            try:
                question = input("🤔 请输入您的问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not question:
                    print("❌ 请输入有效问题")
                    continue
                
                # 回答问题
                self.answer_question(question)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                continue

def main():
    """主函数"""
    print("🎉 欢迎使用LlamaDistributor问答系统!")
    print("=" * 60)
    
    # 创建问答系统
    qa_system = LlamaDistributorQA()
    
    # 预设问题测试
    test_questions = [
        "What is the capital of France?",
        "How does machine learning work?",
        "What is Python programming language?",
    ]
    
    print("📋 预设问题测试:")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        qa_system.answer_question(question, max_new_tokens=15)
    
    print("\n🎯 开始交互模式...")
    qa_system.interactive_mode()

if __name__ == "__main__":
    main() 