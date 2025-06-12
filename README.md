# 🍔 LlamaDistributor

**基于QLLM分片算法的Llama模型分层分布式推理系统**

## 🌟 概述

LlamaDistributor是一个专门针对Llama模型设计的 layer partition inference system。该系统基于QLLM项目的核心分片算法，通过智能模型分层和跨设备协调推理，实现了LLM模型划分和分布式部署，该项目正在持续更新中。

### 核心特性

- **智能模型分层**: 提供均匀分层、自定义分层、内存感知、计算负载均衡和混合策略五种分层策略
- **分布式推理**: 支持跨多GPU和CPU设备的协调推理，实现大模型的分布式部署
- **高效缓存机制**: 兼容KV-Cache，维护KV-Cache层间传递，优化推理性能
- **灵活配置**: 支持多种设备配置和分层策略，适应不同硬件环境
- **性能监控**: 集成推理性能benchmark

## 🏗️ 系统架构

### 核心组件

**llamadist/models/**
- 集成QLLM的Llama模型分层推理实现
- 支持分片执行的模型结构
- 与Transformers库兼容的接口设计

**llamadist/partitioner/**
- `strategies.py`: 实现多种分层策略
- `analyzer.py`: 模型结构分析和资源评估
- `splitter.py`: 模型分层和子模型管理

**llamadist/inference/**
- `coordinator.py`: 分布式推理协调器
- 隐藏状态传递管理
- 跨设备数据同步和缓存管理

**llamadist/submodels/**
- 独立子模型封装和管理
- 跨设备部署支持
- 模型序列化和加载

## ✅ 功能验证

### 已验证功能

**模型分析与分层**
- 自动分析Llama-2-7B模型结构
- 成功实现模型分层（如：层0-15部署至GPU0，层16-31部署至GPU1）
- 权重正确复制和分布

**分布式推理**
- 跨设备隐藏状态传递
- 完整的前向传播流程
- 稳定的logits输出

**KV-Cache优化**
- 完全兼容KV缓存机制
- 模型层间KV-Cache连贯传递
- 长序列生成稳定性保证

**文本生成**
- 连贯文本生成能力
- 支持temperature调节和top-k采样

### 应用示例

#### 问答系统演示

```
输入: "What is the capital of France?"
输出: "What is the capital of France? Paris! Paris is the capital and the largest city..."
生成时间: 0.74秒

输入: "How does machine learning work?"
输出: "How does machine learning work? Machine learning is an exciting area of artificial intelligence..."
生成时间: 0.50秒
```

## 🛠️ 环境配置

### 依赖安装

```bash
# 创建Python环境
conda create -n llamadist python=3.10 -y
conda activate llamadist

# 安装PyTorch和CUDA支持
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装其他依赖
pip install transformers accelerate safetensors sentencepiece tokenizers psutil

# 安装项目
pip install -e .
```

### 基本使用

```python
from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig

# 创建分层策略
strategy = PartitionStrategy(
    num_partitions=2,
    strategy_type="uniform",
    target_devices=["cuda:0", "cuda:1"]
)

# 模型分层
partitioner = LlamaPartitioner(model_path="/path/to/llama-model")
submodels = partitioner.partition(strategy=strategy, copy_weights=True)

# 创建分布式推理引擎
inference_engine = DistributedInference(
    submodels=submodels,
    generation_config=GenerationConfig(max_new_tokens=20)
)

# 执行推理
result = inference_engine.forward_pass(input_ids)
```

### 演示程序

```bash
# 基本功能演示
python demo.py

# 文本生成演示
python simple_demo.py

# 交互式问答
python interactive_demo.py
```

## 📁 项目结构

```
LlamaDistributor/
├── llamadist/                  # 核心模块
│   ├── models/                 # 模型实现
│   │   └── llama_seq.py       # QLLM集成的Llama实现
│   ├── partitioner/           # 分层系统
│   │   ├── strategies.py      # 分层策略
│   │   ├── analyzer.py        # 模型分析
│   │   └── splitter.py        # 模型分层器
│   ├── inference/             # 推理引擎
│   │   └── coordinator.py     # 分布式协调器
│   ├── submodels/            # 子模型管理
│   │   └── manager.py        # 子模型管理器
│   └── utils/                # 工具模块
│       └── config.py         # 配置管理
├── examples/                 # 示例代码
├── configs/                  # 分层策略、prompt配置文件
├── tests/                    # 测试代码
└── llama_partition.py       # 分层推理执行入口
```

## 🎮 分层策略

### 策略类型

**uniform（均匀分层）**
- 按层数平均分配到各设备
- 适用于设备配置相近的环境

**memory（内存感知分层）**
- 根据内存限制智能分配
- 适用于设备内存差异较大的环境

**compute（计算负载均衡）**
- 基于计算量平衡负载
- 适用于设备计算能力不同的环境

**mixed（混合策略）**
- 综合考虑内存和计算资源
- 适用于复杂的异构环境

**custom（自定义分层）**
- 用户指定分层边界
- 适用于特定优化需求

### 配置示例

```python
# 内存限制分层
strategy = PartitionStrategy(
    num_partitions=3,
    strategy_type="memory",
    max_memory_per_partition="4GB",
    target_devices=["cuda:0", "cuda:1", "cpu"]
)

# 自定义分层
strategy = PartitionStrategy(
    num_partitions=2,
    strategy_type="custom",
    custom_boundaries=[(0, 15), (16, 31)],
    target_devices=["cuda:0", "cuda:1"]
)

# 单设备分层 - 均匀分层
strategy = PartitionStrategy(
    num_partitions=4,
    strategy_type="single_device",
    single_device="cuda:0"
)

# 单设备分层 - 自定义分层点
strategy = PartitionStrategy(
    num_partitions=3,
    strategy_type="single_device",
    single_device="cuda:0",
    custom_boundaries=[(0, 10), (11, 21), (22, 31)]
)
```

## 🤝 技术支持

本项目基于以下开源项目构建：

- **QLLM项目**: 提供核心分片算法
- **Transformers**: 模型接口和实现基础
- **PyTorch**: 深度学习框架支持

---

LlamaDistributor旨在测试LLM layer partition & distributed inference，仍在开发中。 