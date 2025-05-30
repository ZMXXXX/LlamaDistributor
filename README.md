# LlamaDistributor

**基于QLLM分片算法的Llama模型自定义分层与分布式推理系统**

## ✨ 项目简介

LlamaDistributor是一个完全自包含的分布式推理系统，专门为Llama模型设计。本项目提取并集成了QLLM的核心分片算法，实现了：

- 🔧 **智能模型分层**: 5种分层策略（均匀、内存、计算、混合、自定义）
- 🚀 **分布式推理**: 跨多设备的协调推理，支持GPU和CPU
- 💾 **高效KV-Cache**: 完全兼容KV缓存机制，性能提升3x
- 🎯 **实时文本生成**: 支持问答、对话等多种应用场景
- 📊 **性能监控**: 详细的推理时间和资源使用统计

## 🎉 成功验证的功能

### ✅ 核心功能测试通过

1. **模型分析与分层**
   - 自动分析Llama-2-7B模型结构
   - 成功分层为2个子模型（层0-15 @ GPU0，层16-31 @ GPU1）
   - 权重正确复制和分布

2. **分布式推理**
   - 跨设备的隐藏状态传递
   - 正确的前向传播流程
   - 稳定的logits输出

3. **KV-Cache优化** ⭐ **最新修复**
   - 完全兼容KV缓存机制
   - 性能提升1.8x，每步提升3.0x
   - 时间复杂度从O(n²)优化到O(1)
   - 长序列生成稳定（测试50+ tokens）

4. **文本生成**
   - 成功生成连贯文本
   - 支持温度调节和top-k采样
   - 自动停止条件判断

5. **问答系统**
   - 交互式问答界面
   - 预设问题测试
   - 实时响应能力

### 📊 测试结果示例

#### 🔥 KV-Cache性能测试

```
📊 性能对比结果（启用vs禁用KV-Cache）
============================================================
不使用缓存总时间: 1.55s
使用缓存总时间:   0.84s
性能提升:         1.8x

平均每步时间:
不使用缓存: 0.045s
使用缓存:   0.015s
每步提升:   3.0x

⏱️  长序列稳定性分析（50 tokens）:
早期平均时间: 0.019s
后期平均时间: 0.014s
时间增长比率: 0.73x ✅ 保持稳定
```

#### 💬 问答对话示例

```
❓ 问题: What is the capital of France?
💭 思考中...
⚡ 生成时间: 0.74秒
💬 回答: What is the capital of France? Paris! Paris is the capital and the largest city

❓ 问题: How does machine learning work?
💭 思考中...
⚡ 生成时间: 0.50秒
💬 回答: How does machine learning work?
Machine learning is an exciting area of artificial intelligence

❓ 问题: What is Python programming language?
💭 思考中...
⚡ 生成时间: 0.65秒
💬 回答: What is Python programming language?
Python is an interpreted, object-oriented, high-level programming
```

## 🏗️ 系统架构

### 核心组件

1. **llamadist/models/**: 内置Llama模型实现
   - 从QLLM提取的核心算法
   - 支持分片执行的模型结构
   - 兼容Transformers接口

2. **llamadist/partitioner/**: 分层系统
   - `strategies.py`: 5种智能分层策略
   - `analyzer.py`: 模型结构分析
   - `splitter.py`: 模型切分和子模型管理

3. **llamadist/inference/**: 分布式推理引擎
   - `coordinator.py`: 推理协调器
   - 隐藏状态传递管理
   - 设备间数据同步

4. **llamadist/submodels/**: 子模型管理
   - 独立子模型封装
   - 跨设备部署支持
   - 模型保存和加载

## 🚀 快速开始

### 环境配置

```bash
# 创建conda环境
conda create -n llamadist python=3.10 -y
conda activate llamadist

# 安装依赖
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers accelerate safetensors sentencepiece tokenizers psutil

# 安装项目
pip install -e .
```

### 基本使用

```python
from llamadist.partitioner.strategies import PartitionStrategy
from llamadist.partitioner.splitter import LlamaPartitioner
from llamadist.inference.coordinator import DistributedInference, GenerationConfig

# 1. 创建分层策略
strategy = PartitionStrategy(
    num_partitions=2,
    strategy_type="uniform",
    target_devices=["cuda:0", "cuda:1"]
)

# 2. 分层模型
partitioner = LlamaPartitioner(model_path="/path/to/llama-model")
submodels = partitioner.partition(strategy=strategy, copy_weights=True)

# 3. 创建分布式推理引擎
inference_engine = DistributedInference(
    submodels=submodels,
    generation_config=GenerationConfig(max_new_tokens=20)
)

# 4. 执行推理
result = inference_engine.forward_pass(input_ids)
```

### 运行演示

```bash
# 基本演示
python demo.py

# 简单文本生成
python simple_demo.py

# 交互式问答
python interactive_demo.py
```

## 📁 项目结构

```
LlamaDistributor/
├── llamadist/                  # 核心包
│   ├── models/                 # 模型实现
│   │   ├── __init__.py
│   │   └── llama_seq.py       # 从QLLM提取的Llama实现
│   ├── partitioner/           # 分层系统
│   │   ├── __init__.py
│   │   ├── strategies.py      # 分层策略
│   │   ├── analyzer.py        # 模型分析
│   │   └── splitter.py        # 模型分层器
│   ├── inference/             # 推理引擎
│   │   ├── __init__.py
│   │   └── coordinator.py     # 分布式协调器
│   ├── submodels/            # 子模型管理
│   │   ├── __init__.py
│   │   └── manager.py        # 子模型管理器
│   └── utils/                # 工具函数
│       ├── __init__.py
│       └── config.py         # 配置管理
├── examples/                 # 示例代码
├── configs/                  # 配置文件
├── tests/                    # 测试代码
├── demo.py                   # 基本演示
├── simple_demo.py           # 简单文本生成演示
├── interactive_demo.py      # 交互式问答演示
├── setup.py                 # 安装脚本
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```

## 🔧 分层策略

### 支持的策略类型

1. **uniform**: 均匀分层 - 按层数平均分配
2. **memory**: 内存分层 - 根据内存限制智能分配
3. **compute**: 计算分层 - 基于计算量平衡负载
4. **mixed**: 混合策略 - 综合考虑内存和计算
5. **custom**: 自定义 - 用户指定分层边界

### 示例配置

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
```

## ⚡ 性能特性

- **推理速度**: 0.7-1.2秒/步骤（Llama-2-7B，双GPU）
- **内存效率**: 支持大模型分布式部署
- **设备支持**: GPU/CPU混合部署
- **可扩展性**: 支持任意数量的分层
- **稳定性**: 完整的错误处理和恢复机制

## 🛠️ 技术特点

### 从QLLM提取的核心算法

- **分片执行**: 保留QLLM的分片推理逻辑
- **状态传递**: 高效的隐藏状态传递机制
- **内存优化**: 智能的内存管理策略
- **设备协调**: 跨设备的推理协调

### 自主创新

- **统一接口**: 简化的API设计
- **灵活配置**: 多种分层策略选择
- **实时监控**: 详细的性能统计
- **扩展性**: 模块化架构设计

## 📋 测试覆盖

- ✅ 模型加载和分析
- ✅ 分层策略验证
- ✅ 权重复制和分布
- ✅ 分布式前向传播
- ✅ 文本生成和采样
- ✅ 设备间状态传递
- ✅ 错误处理和恢复
- ✅ 性能监控统计

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 Apache 2.0 许可证开源 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- **QLLM项目**: 核心分片算法来源
- **Transformers**: 模型接口和实现参考
- **PyTorch**: 深度学习框架支持

---

**LlamaDistributor**: 让大模型分布式推理变得简单高效！ 🚀 