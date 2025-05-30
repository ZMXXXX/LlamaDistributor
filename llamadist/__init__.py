"""
LlamaDistributor: 基于QLLM的Llama模型分层与分布式推理系统

这个包提供了一套完整的工具链，用于：
1. 自定义Llama模型分层
2. 子模型管理和保存
3. 分布式推理协调
4. 高效的状态传递和缓存管理

现在包含从QLLM提取的核心分片算法，无需依赖QLLM。
"""

from .partitioner.strategies import PartitionStrategy
from .partitioner.splitter import LlamaPartitioner
from .submodels.manager import SubModelManager
from .inference.coordinator import DistributedInference
from .utils.config import LlamaDistConfig

# 导出核心模型类
from .models.llama_seq import (
    LlamaForCausalLMSeq,
    LlamaModelSeq,
    LlamaDecoderLayer
)

__version__ = "0.1.0"
__author__ = "LlamaDistributor Team"

# 核心API导出
__all__ = [
    # 分层相关
    "PartitionStrategy",
    "LlamaPartitioner",
    
    # 子模型管理
    "SubModelManager",
    
    # 分布式推理
    "DistributedInference",
    
    # 配置管理
    "LlamaDistConfig",
    
    # 核心模型类
    "LlamaForCausalLMSeq",
    "LlamaModelSeq",
    "LlamaDecoderLayer",
]

# 版本信息
def get_version():
    """返回LlamaDistributor版本信息"""
    return __version__

def get_info():
    """返回项目信息"""
    return {
        "name": "LlamaDistributor",
        "version": __version__,
        "description": "基于QLLM分片算法的Llama模型分层与分布式推理系统",
        "author": __author__,
        "dependencies": [
            "torch>=1.12.0",
            "transformers>=4.21.0",
            "numpy>=1.21.0",
            "psutil>=5.8.0",
        ],
        "features": [
            "自包含的分片算法",
            "多种分层策略",
            "高效的状态传递",
            "KV缓存管理",
            "分布式推理",
            "性能监控"
        ]
    } 