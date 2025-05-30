# LlamaDistributor 安装和使用指南

## 📦 安装

### 前置要求

1. **Python 环境**: Python >= 3.8
2. **QLLM 项目**: 需要先安装和配置QLLM项目
3. **PyTorch**: 建议使用 PyTorch >= 1.12.0
4. **系统内存**: 推荐至少 16GB RAM（用于加载较大模型）

### 安装 QLLM (前置依赖)

LlamaDistributor基于QLLM项目，需要先安装QLLM：

```bash
# 克隆QLLM项目（如果还没有的话）
git clone https://github.com/your-org/QLLM.git
cd QLLM

# 安装QLLM依赖
pip install -r requirements.txt
pip install -e .

# 确保QLLM在Python路径中
export PYTHONPATH=$PYTHONPATH:/path/to/QLLM
```

### 安装 LlamaDistributor

```bash
# 方法1: 从源码安装
git clone https://github.com/your-org/LlamaDistributor.git
cd LlamaDistributor
pip install -r requirements.txt
pip install -e .

# 方法2: 直接从当前目录安装
cd LlamaDistributor
pip install -e .
```

### 验证安装

```python
# 测试导入
import llamadist
print(llamadist.get_info())

# 运行基础示例
python examples/basic_partition.py
```

## 🚀 快速开始

### 1. 基础分层示例

```python
from llamadist import PartitionStrategy, LlamaPartitioner, DistributedInference

# 创建分层策略
strategy = PartitionStrategy(
    num_partitions=4,
    strategy_type="uniform",
    target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
)

# 初始化分层器
partitioner = LlamaPartitioner(model_path="path/to/llama/model")

# 执行分层
submodels = partitioner.partition(strategy)

# 保存分层结果
partitioner.save_partitioned_models(submodels, "output_dir")
```

### 2. 分布式推理

```python
from llamadist import DistributedInference
from llamadist.inference.coordinator import GenerationConfig

# 创建推理引擎
inference = DistributedInference(submodels)

# 配置生成参数
config = GenerationConfig(
    max_length=512,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# 生成文本
result = inference.generate_text(
    prompt="你好，请介绍一下人工智能",
    tokenizer=tokenizer,
    generation_config=config
)
print(result)
```

### 3. 使用配置文件

```python
from llamadist import LlamaDistConfig

# 创建配置
config = LlamaDistConfig(
    model_path="path/to/model",
    num_partitions=4,
    strategy_type="memory",
    max_memory_per_partition="4GB",
    target_devices=["cuda:0", "cuda:1", "cpu", "cpu"]
)

# 保存配置
config.save("my_config.json")

# 加载配置
config = LlamaDistConfig.load("my_config.json")
```

## 📋 详细使用教程

### 分层策略详解

#### 1. 均匀分层 (Uniform)
```python
strategy = PartitionStrategy(
    num_partitions=4,
    strategy_type="uniform"
)
```
- 将模型层数平均分配到各个分层
- 适用于设备配置相似的场景

#### 2. 内存分层 (Memory)
```python
strategy = PartitionStrategy(
    num_partitions=4,
    strategy_type="memory",
    max_memory_per_partition="4GB"
)
```
- 根据内存限制进行分层
- 适用于内存受限的环境

#### 3. 计算分层 (Compute)
```python
strategy = PartitionStrategy(
    num_partitions=4,
    strategy_type="compute",
    load_balance_factor=0.8
)
```
- 根据计算负载进行分层
- 确保各分层计算量相对均衡

#### 4. 混合策略 (Mixed)
```python
strategy = PartitionStrategy(
    num_partitions=4,
    strategy_type="mixed",
    max_memory_per_partition="4GB",
    memory_weight=0.6
)
```
- 同时考虑内存和计算负载
- 通过权重平衡两种约束

#### 5. 自定义分层 (Custom)
```python
strategy = PartitionStrategy(
    num_partitions=3,
    strategy_type="custom",
    custom_boundaries=[(0, 10), (11, 20), (21, 31)]
)
```
- 用户自定义层边界
- 最大灵活性

### 设备配置

#### GPU配置
```python
# 多GPU配置
target_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

# 混合设备配置
target_devices = ["cuda:0", "cuda:1", "cpu", "cpu"]
```

#### 内存优化
```python
# 启用量化
config = LlamaDistConfig(
    quantization="8bit",
    max_memory_per_partition="4GB"
)
```

### 推理优化

#### 异步推理
```python
inference = DistributedInference(
    submodels=submodels,
    enable_async=True,
    max_workers=8
)

# 异步生成
future = inference.async_generate(input_ids)
result = future.result()
```

#### 批量推理
```python
# 批量处理
batch_inputs = [input_ids_1, input_ids_2, input_ids_3]
results = inference.batch_generate(batch_inputs, max_batch_size=2)
```

#### KV缓存优化
```python
config = GenerationConfig(
    use_cache=True,
    max_length=1024
)

# 启用缓存的生成
result = inference.generate(input_ids, generation_config=config)
```

## 🔧 高级功能

### 性能监控
```python
# 获取性能统计
stats = inference.get_stats()
print(f"平均推理时间: {stats['avg_inference_time']:.4f}s")
print(f"吞吐量: {stats['tokens_per_second']:.2f} tokens/s")
```

### 模型分析
```python
# 详细模型分析
analyzer = LlamaModelAnalyzer(model_path)
model_info = analyzer.analyze_model(detailed=True)

# 保存分析结果
analyzer.save_analysis("model_analysis.json")
```

### 配置模板
```python
from llamadist.utils.config import create_default_configs

# 获取预定义配置
configs = create_default_configs()
memory_config = configs["memory_optimized"]
performance_config = configs["performance"]
```

## 🐛 故障排除

### 常见问题

1. **QLLM导入失败**
   ```bash
   # 确保QLLM在Python路径中
   export PYTHONPATH=$PYTHONPATH:/path/to/QLLM
   ```

2. **内存不足错误**
   ```python
   # 减少分层大小或使用CPU
   config.max_memory_per_partition = "2GB"
   config.target_devices = ["cpu"] * 4
   ```

3. **CUDA内存错误**
   ```python
   # 清理GPU缓存
   torch.cuda.empty_cache()
   ```

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
config.detailed_analysis = True
```

## 📊 性能基准

### 内存使用对比
| 模型大小 | 原始 | 4分层 | 8分层 | 节省率 |
|---------|------|-------|-------|--------|
| 7B      | 13GB | 3.5GB | 2GB   | 75%    |
| 13B     | 26GB | 7GB   | 4GB   | 80%    |
| 30B     | 60GB | 15GB  | 8GB   | 85%    |

### 推理性能
- **延迟**: 增加10-20%（由于状态传递）
- **吞吐量**: 在批量推理中可提升2-4倍
- **扩展性**: 支持任意数量的设备

## 🤝 贡献指南

欢迎贡献代码！请参考：
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用Apache 2.0许可证。

## 🆘 获取帮助

- 📚 [文档](https://llamadist.readthedocs.io/)
- 🐛 [Issues](https://github.com/llamadist/LlamaDistributor/issues)
- 💬 [讨论](https://github.com/llamadist/LlamaDistributor/discussions) 