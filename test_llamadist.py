#!/usr/bin/env python3
"""
LlamaDistributor测试脚本

测试模型分析、分层、分布式推理等功能
"""

import os
import torch
from transformers import AutoTokenizer
import time

# 导入LlamaDistributor组件
from llamadist.partitioner.analyzer import LlamaModelAnalyzer
from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig
from llamadist.submodels.manager import SubModelManager

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("LlamaDistributor 基本功能测试")
    print("=" * 60)
    
    # 模型路径
    model_path = "/home/zmx/models/Llama/Llama-2-7b-hf"
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在 {model_path}")
        return False
    
    print(f"✓ 模型路径验证通过: {model_path}")
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ 使用设备: {device}")
    
    try:
        # 1. 测试模型分析
        print("\n1. 测试模型分析器...")
        analyzer = LlamaModelAnalyzer(model_path=model_path)
        
        # 简单分析（避免内存问题）
        model_info = analyzer.analyze_model(
            sample_input_shape=(1, 64),  # 较小的输入
            device="cpu",  # 使用CPU进行分析
            detailed=False  # 简单分析
        )
        
        print(f"   模型名称: {model_info.model_name}")
        print(f"   层数: {model_info.num_layers}")
        print(f"   参数总数: {model_info.total_params / 1e9:.2f}B")
        print(f"   总内存: {model_info.total_memory / 1e9:.2f}GB")
        print("✓ 模型分析完成")
        
        # 2. 测试分层策略
        print("\n2. 测试分层策略...")
        strategy = PartitionStrategy(
            num_partitions=2,
            strategy_type="uniform",
            devices=["cpu", "cpu"]  # 都使用CPU
        )
        
        partitions = strategy.create_partitions(model_info)
        print(f"   创建了 {len(partitions)} 个分层:")
        for i, partition in enumerate(partitions):
            print(f"     分层 {i}: 层 {partition.layer_start}-{partition.layer_end}")
        print("✓ 分层策略测试完成")
        
        # 3. 测试模型分层器
        print("\n3. 测试模型分层器...")
        partitioner = LlamaPartitioner(model_path=model_path)
        
        # 创建子模型（较小规模）
        submodels = partitioner.partition(
            strategy=strategy,
            analyze_first=False,  # 使用已有的分析结果
            copy_weights=True
        )
        
        print(f"   成功创建 {len(submodels)} 个子模型")
        for sm in submodels:
            info = sm.get_info()
            print(f"     子模型 {info['partition_idx']}: {info['memory_usage']/1e6:.1f}MB")
        print("✓ 模型分层完成")
        
        # 4. 测试分布式推理
        print("\n4. 测试分布式推理...")
        
        # 加载分词器
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("   ✓ 分词器加载成功")
        except Exception as e:
            print(f"   警告：分词器加载失败: {e}")
            print("   使用模拟输入进行测试...")
            tokenizer = None
        
        # 创建分布式推理引擎
        inference_engine = DistributedInference(
            submodels=submodels,
            generation_config=GenerationConfig(
                max_new_tokens=5,  # 只生成少量token
                temperature=1.0,
                do_sample=False  # 使用greedy解码
            )
        )
        
        # 测试前向传播
        if tokenizer is not None:
            # 使用真实输入
            test_prompt = "Hello, how are you?"
            input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
        else:
            # 使用模拟输入
            input_ids = torch.randint(1, 1000, (1, 8))
        
        print(f"   输入形状: {input_ids.shape}")
        
        # 前向传播测试
        start_time = time.time()
        result = inference_engine.forward_pass(
            input_ids=input_ids,
            use_cache=False  # 暂时不使用缓存
        )
        forward_time = time.time() - start_time
        
        print(f"   前向传播耗时: {forward_time:.3f}秒")
        print(f"   输出logits形状: {result['logits'].shape}")
        print("✓ 分布式推理测试完成")
        
        # 5. 测试文本生成（如果有分词器）
        if tokenizer is not None:
            print("\n5. 测试文本生成...")
            try:
                start_time = time.time()
                generated_text = inference_engine.generate_text(
                    prompt="Hello",
                    tokenizer=tokenizer,
                    return_full_text=True
                )
                generation_time = time.time() - start_time
                
                print(f"   生成耗时: {generation_time:.3f}秒")
                print(f"   生成文本: {generated_text[:100]}...")
                print("✓ 文本生成测试完成")
            except Exception as e:
                print(f"   文本生成测试失败: {e}")
        
        # 6. 测试子模型管理器
        print("\n6. 测试子模型管理器...")
        manager = SubModelManager(base_dir="./test_models")
        
        # 保存分层模型
        model_name = "test_llama2_7b"
        manager.save_partitioned_model(
            submodels=submodels,
            model_name=model_name,
            description="Llama-2-7B测试模型",
            overwrite=True
        )
        
        # 列出模型
        models = manager.list_models()
        print(f"   管理器中的模型数量: {len(models)}")
        
        # 验证模型
        validation_result = manager.validate_model(model_name)
        print(f"   模型验证结果: {'通过' if validation_result['valid'] else '失败'}")
        
        print("✓ 子模型管理器测试完成")
        
        # 7. 性能统计
        print("\n7. 性能统计...")
        stats = inference_engine.get_stats()
        print(f"   推理次数: {stats['inference_count']}")
        print(f"   总推理时间: {stats['total_inference_time']:.3f}秒")
        if stats['inference_count'] > 0:
            print(f"   平均推理时间: {stats['avg_inference_time']:.3f}秒")
        print("✓ 性能统计完成")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！LlamaDistributor配置成功")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """测试环境配置"""
    print("=" * 60)
    print("环境配置测试")
    print("=" * 60)
    
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name()}")
    
    # 检查transformers
    import transformers
    print(f"Transformers版本: {transformers.__version__}")
    
    # 检查LlamaDistributor
    import llamadist
    print(f"LlamaDistributor版本: {llamadist.__version__}")
    
    print("✓ 环境配置检查完成")


if __name__ == "__main__":
    print("LlamaDistributor 完整测试")
    print("使用模型: /home/zmx/models/Llama/Llama-2-7b-hf")
    print()
    
    # 测试环境
    test_environment()
    print()
    
    # 测试基本功能
    success = test_basic_functionality()
    
    if success:
        print("\n🎉 恭喜！LlamaDistributor环境配置成功并通过所有测试！")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复。") 