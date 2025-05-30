"""
分层策略模块

定义了多种分层策略，用于将Llama模型分割成多个子模型。
每种策略都有不同的优化目标和约束条件。
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import torch
import numpy as np


class StrategyType(Enum):
    """分层策略类型枚举"""
    UNIFORM = "uniform"           # 均匀分层（按层数平均分配）
    MEMORY = "memory"             # 按内存限制分层
    COMPUTE = "compute"           # 按计算量分层
    MIXED = "mixed"               # 混合策略
    CUSTOM = "custom"             # 自定义策略


@dataclass
class PartitionConfig:
    """分层配置"""
    layer_start: int              # 起始层索引
    layer_end: int                # 结束层索引
    memory_limit: Optional[str] = None    # 内存限制 (e.g., "4GB")
    device: Optional[str] = None          # 目标设备
    quantization: Optional[str] = None    # 量化策略


class PartitionStrategy:
    """
    分层策略基类
    
    提供了多种分层策略的统一接口，支持根据不同的约束条件
    (内存、计算量、设备能力等) 对Llama模型进行智能分层。
    """
    
    def __init__(
        self,
        num_partitions: int,
        strategy_type: Union[str, StrategyType] = StrategyType.UNIFORM,
        max_memory_per_partition: Optional[str] = None,
        target_devices: Optional[List[str]] = None,
        load_balance_factor: float = 0.8,
        **kwargs
    ):
        """
        初始化分层策略
        
        Args:
            num_partitions: 分层数量
            strategy_type: 策略类型
            max_memory_per_partition: 每个分层的最大内存限制
            target_devices: 目标设备列表
            load_balance_factor: 负载均衡因子 (0-1)
            **kwargs: 其他策略特定参数
        """
        self.num_partitions = num_partitions
        self.strategy_type = StrategyType(strategy_type) if isinstance(strategy_type, str) else strategy_type
        self.max_memory_per_partition = max_memory_per_partition
        self.target_devices = target_devices or [f"cuda:{i}" for i in range(num_partitions)]
        self.load_balance_factor = load_balance_factor
        self.kwargs = kwargs
        
        # 验证参数
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证参数有效性"""
        if self.num_partitions <= 0:
            raise ValueError("分层数量必须大于0")
        
        if not 0 <= self.load_balance_factor <= 1:
            raise ValueError("负载均衡因子必须在0-1之间")
        
        if len(self.target_devices) != self.num_partitions:
            raise ValueError("目标设备数量必须等于分层数量")
    
    def create_partitions(self, model_info: Dict) -> List[PartitionConfig]:
        """
        根据策略创建分层配置
        
        Args:
            model_info: 模型信息字典，包含层数、参数量等信息
        
        Returns:
            List[PartitionConfig]: 分层配置列表
        """
        if self.strategy_type == StrategyType.UNIFORM:
            return self._create_uniform_partitions(model_info)
        elif self.strategy_type == StrategyType.MEMORY:
            return self._create_memory_based_partitions(model_info)
        elif self.strategy_type == StrategyType.COMPUTE:
            return self._create_compute_based_partitions(model_info)
        elif self.strategy_type == StrategyType.MIXED:
            return self._create_mixed_partitions(model_info)
        elif self.strategy_type == StrategyType.CUSTOM:
            return self._create_custom_partitions(model_info)
        else:
            raise ValueError(f"不支持的策略类型: {self.strategy_type}")
    
    def _create_uniform_partitions(self, model_info) -> List[PartitionConfig]:
        """
        创建均匀分层配置
        
        简单地将模型层数平均分配到各个分层中
        """
        num_layers = model_info.num_layers
        layers_per_partition = num_layers // self.num_partitions
        remainder = num_layers % self.num_partitions
        
        partitions = []
        current_layer = 0
        
        for i in range(self.num_partitions):
            # 分配层数，余数分配给前几个分层
            partition_layers = layers_per_partition + (1 if i < remainder else 0)
            
            partition = PartitionConfig(
                layer_start=current_layer,
                layer_end=current_layer + partition_layers - 1,
                device=self.target_devices[i]
            )
            partitions.append(partition)
            current_layer += partition_layers
        
        return partitions
    
    def _create_memory_based_partitions(self, model_info) -> List[PartitionConfig]:
        """
        基于内存限制创建分层配置
        
        根据每层的内存占用和限制条件智能分配
        """
        if not self.max_memory_per_partition:
            raise ValueError("内存分层策略需要指定max_memory_per_partition")
        
        # 解析内存限制
        memory_limit_bytes = self._parse_memory_string(self.max_memory_per_partition)
        layer_memory_costs = getattr(model_info, 'layer_memory_costs', [])
        
        if not layer_memory_costs:
            # 如果没有详细的内存信息，使用均匀分层作为后备
            return self._create_uniform_partitions(model_info)
        
        partitions = []
        current_layer = 0
        current_memory = 0
        
        for partition_idx in range(self.num_partitions):
            layer_start = current_layer
            
            # 累积分配层到当前分层，直到达到内存限制
            while current_layer < len(layer_memory_costs):
                layer_memory = layer_memory_costs[current_layer]
                
                # 检查是否超过内存限制（但至少要包含一层）
                if current_memory + layer_memory > memory_limit_bytes and current_layer > layer_start:
                    break
                
                current_memory += layer_memory
                current_layer += 1
            
            # 创建分层配置
            partition = PartitionConfig(
                layer_start=layer_start,
                layer_end=current_layer - 1,
                memory_limit=self.max_memory_per_partition,
                device=self.target_devices[partition_idx]
            )
            partitions.append(partition)
            
            # 重置内存计数器
            current_memory = 0
            
            # 如果已经分配完所有层，跳出循环
            if current_layer >= len(layer_memory_costs):
                break
        
        return partitions
    
    def _create_compute_based_partitions(self, model_info) -> List[PartitionConfig]:
        """
        基于计算量创建分层配置
        
        尝试使各分层的计算负载相对均衡
        """
        layer_compute_costs = getattr(model_info, 'layer_compute_costs', [])
        num_layers = model_info.num_layers
        
        if not layer_compute_costs:
            # 如果没有计算成本信息，假设计算量与参数量成正比
            layer_params = getattr(model_info, 'layer_params', [1] * num_layers)
            layer_compute_costs = layer_params
        
        # 计算目标计算量（考虑负载均衡因子）
        total_compute = sum(layer_compute_costs)
        target_compute_per_partition = total_compute / self.num_partitions
        balance_threshold = target_compute_per_partition * self.load_balance_factor
        
        partitions = []
        current_layer = 0
        current_compute = 0
        
        for partition_idx in range(self.num_partitions):
            layer_start = current_layer
            
            # 累积分配层到当前分层，直到达到目标计算量
            while current_layer < len(layer_compute_costs):
                layer_compute = layer_compute_costs[current_layer]
                
                # 检查是否超过目标计算量
                if (current_compute + layer_compute > target_compute_per_partition + balance_threshold 
                    and current_layer > layer_start):
                    break
                
                current_compute += layer_compute
                current_layer += 1
            
            # 创建分层配置
            partition = PartitionConfig(
                layer_start=layer_start,
                layer_end=current_layer - 1,
                device=self.target_devices[partition_idx]
            )
            partitions.append(partition)
            
            # 重置计算量计数器
            current_compute = 0
            
            # 如果已经分配完所有层，跳出循环
            if current_layer >= len(layer_compute_costs):
                break
        
        return partitions
    
    def _create_mixed_partitions(self, model_info) -> List[PartitionConfig]:
        """
        创建混合策略分层配置
        
        同时考虑内存和计算量约束
        """
        # 首先基于内存创建初始分层
        memory_partitions = self._create_memory_based_partitions(model_info)
        
        # 然后基于计算量进行调整
        compute_partitions = self._create_compute_based_partitions(model_info)
        
        # 合并两种策略的结果（简化实现）
        # 实际应用中可以使用更复杂的优化算法
        return self._merge_partition_strategies(memory_partitions, compute_partitions)
    
    def _create_custom_partitions(self, model_info) -> List[PartitionConfig]:
        """
        创建自定义分层配置
        
        允许用户通过kwargs传入自定义的分层边界
        """
        custom_boundaries = self.kwargs.get('custom_boundaries')
        if not custom_boundaries:
            raise ValueError("自定义策略需要提供custom_boundaries参数")
        
        partitions = []
        for i, (start, end) in enumerate(custom_boundaries):
            partition = PartitionConfig(
                layer_start=start,
                layer_end=end,
                device=self.target_devices[i] if i < len(self.target_devices) else "cpu"
            )
            partitions.append(partition)
        
        return partitions
    
    def _merge_partition_strategies(
        self, 
        memory_partitions: List[PartitionConfig], 
        compute_partitions: List[PartitionConfig]
    ) -> List[PartitionConfig]:
        """
        合并内存和计算量策略的结果
        
        使用加权平均的方式合并两种策略
        """
        # 简化实现：选择更保守的分层边界
        merged_partitions = []
        
        memory_weight = self.kwargs.get('memory_weight', 0.6)
        compute_weight = 1.0 - memory_weight
        
        # 这里使用简单的启发式合并
        # 实际应用中可以使用更复杂的优化算法
        for i in range(min(len(memory_partitions), len(compute_partitions))):
            mem_partition = memory_partitions[i]
            comp_partition = compute_partitions[i]
            
            # 选择更保守的边界
            layer_start = max(mem_partition.layer_start, comp_partition.layer_start)
            layer_end = min(mem_partition.layer_end, comp_partition.layer_end)
            
            if layer_start <= layer_end:
                partition = PartitionConfig(
                    layer_start=layer_start,
                    layer_end=layer_end,
                    memory_limit=mem_partition.memory_limit,
                    device=self.target_devices[i]
                )
                merged_partitions.append(partition)
        
        return merged_partitions
    
    def _parse_memory_string(self, memory_str: str) -> int:
        """
        解析内存字符串为字节数
        
        支持格式: "4GB", "512MB", "2048KB", "1024B"
        """
        memory_str = memory_str.upper().strip()
        
        if memory_str.endswith('GB'):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith('MB'):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith('KB'):
            return int(float(memory_str[:-2]) * 1024)
        elif memory_str.endswith('B'):
            return int(memory_str[:-1])
        else:
            # 默认按字节处理
            return int(memory_str)
    
    def estimate_memory_usage(self, partition: PartitionConfig, model_info) -> int:
        """
        估算分层的内存使用量
        
        Args:
            partition: 分层配置
            model_info: 模型信息
        
        Returns:
            int: 估算的内存使用量（字节）
        """
        layer_memory_costs = getattr(model_info, 'layer_memory_costs', [])
        
        if not layer_memory_costs:
            # 如果没有详细信息，使用粗略估算
            num_layers = partition.layer_end - partition.layer_start + 1
            avg_layer_memory = getattr(model_info, 'total_memory', 0) // getattr(model_info, 'num_layers', 1)
            return num_layers * avg_layer_memory
        
        # 计算分层内所有层的内存使用量
        total_memory = 0
        for layer_idx in range(partition.layer_start, partition.layer_end + 1):
            if layer_idx < len(layer_memory_costs):
                total_memory += layer_memory_costs[layer_idx]
        
        return total_memory
    
    def validate_strategy(self, partitions: List[PartitionConfig], model_info) -> bool:
        """
        验证分层策略的有效性
        
        Args:
            partitions: 分层配置列表
            model_info: 模型信息
        
        Returns:
            bool: 策略是否有效
        """
        num_layers = model_info.num_layers
        
        # 检查层覆盖的完整性
        covered_layers = set()
        for partition in partitions:
            for layer_idx in range(partition.layer_start, partition.layer_end + 1):
                if layer_idx in covered_layers:
                    print(f"警告：层 {layer_idx} 被多个分层覆盖")
                    return False
                covered_layers.add(layer_idx)
        
        # 检查是否覆盖了所有层
        expected_layers = set(range(num_layers))
        if covered_layers != expected_layers:
            missing_layers = expected_layers - covered_layers
            print(f"警告：缺少层的覆盖: {missing_layers}")
            return False
        
        # 检查内存限制
        if self.max_memory_per_partition:
            memory_limit_bytes = self._parse_memory_string(self.max_memory_per_partition)
            for i, partition in enumerate(partitions):
                memory_usage = self.estimate_memory_usage(partition, model_info)
                if memory_usage > memory_limit_bytes:
                    print(f"警告：分层 {i} 的内存使用量 ({memory_usage} bytes) 超过限制 ({memory_limit_bytes} bytes)")
                    return False
        
        return True
    
    def get_summary(self) -> Dict:
        """
        获取策略摘要信息
        
        Returns:
            Dict: 策略摘要
        """
        return {
            "strategy_type": self.strategy_type.value,
            "num_partitions": self.num_partitions,
            "max_memory_per_partition": self.max_memory_per_partition,
            "target_devices": self.target_devices,
            "load_balance_factor": self.load_balance_factor,
            "additional_params": self.kwargs
        } 