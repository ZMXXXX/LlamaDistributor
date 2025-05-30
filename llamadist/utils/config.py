"""
配置管理模块

提供统一的配置管理功能，包括分层配置、推理配置等。
"""

import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict, field


@dataclass
class LlamaDistConfig:
    """LlamaDistributor配置类"""
    
    # 模型配置
    model_path: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    
    # 分层配置
    num_partitions: int = 4
    strategy_type: str = "uniform"  # uniform, memory, compute, mixed, custom
    max_memory_per_partition: Optional[str] = None
    target_devices: Optional[List[str]] = None
    load_balance_factor: float = 0.8
    
    # 推理配置
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    use_cache: bool = True
    enable_async: bool = False
    max_workers: int = 4
    
    # 存储配置
    output_dir: str = "./partitioned_models"
    save_config: bool = True
    
    # 高级配置
    custom_boundaries: Optional[List[tuple]] = None
    memory_weight: float = 0.6
    detailed_analysis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LlamaDistConfig':
        """从字典创建配置"""
        return cls(**data)
    
    def save(self, path: str):
        """保存配置到文件"""
        path = Path(path)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    @classmethod
    def load(cls, path: str) -> 'LlamaDistConfig':
        """从文件加载配置"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        return cls.from_dict(data)
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        errors = []
        
        # 验证分层配置
        if self.num_partitions <= 0:
            errors.append("num_partitions必须大于0")
        
        if self.strategy_type not in ["uniform", "memory", "compute", "mixed", "custom"]:
            errors.append(f"不支持的策略类型: {self.strategy_type}")
        
        if self.strategy_type == "memory" and not self.max_memory_per_partition:
            errors.append("内存策略需要指定max_memory_per_partition")
        
        if self.strategy_type == "custom" and not self.custom_boundaries:
            errors.append("自定义策略需要指定custom_boundaries")
        
        # 验证生成配置
        if self.max_length <= 0:
            errors.append("max_length必须大于0")
        
        if not 0 < self.temperature <= 10:
            errors.append("temperature必须在(0, 10]范围内")
        
        if not 0 < self.top_p <= 1:
            errors.append("top_p必须在(0, 1]范围内")
        
        if self.top_k <= 0:
            errors.append("top_k必须大于0")
        
        # 验证设备配置
        if self.target_devices and len(self.target_devices) != self.num_partitions:
            errors.append("target_devices长度必须等于num_partitions")
        
        if errors:
            for error in errors:
                print(f"配置错误: {error}")
            return False
        
        return True
    
    def get_partition_strategy_config(self) -> Dict[str, Any]:
        """获取分层策略配置"""
        config = {
            "num_partitions": self.num_partitions,
            "strategy_type": self.strategy_type,
            "max_memory_per_partition": self.max_memory_per_partition,
            "target_devices": self.target_devices,
            "load_balance_factor": self.load_balance_factor
        }
        
        # 添加策略特定参数
        if self.strategy_type == "custom":
            config["custom_boundaries"] = self.custom_boundaries
        elif self.strategy_type == "mixed":
            config["memory_weight"] = self.memory_weight
        
        return config
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache
        }
    
    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return {
            "enable_async": self.enable_async,
            "max_workers": self.max_workers,
            "generation_config": self.get_generation_config()
        }


def create_default_configs() -> Dict[str, LlamaDistConfig]:
    """创建一些预定义的配置模板"""
    
    configs = {}
    
    # 基础配置
    configs["basic"] = LlamaDistConfig(
        num_partitions=2,
        strategy_type="uniform",
        max_length=256,
        temperature=1.0,
        do_sample=False
    )
    
    # 内存优化配置
    configs["memory_optimized"] = LlamaDistConfig(
        num_partitions=4,
        strategy_type="memory",
        max_memory_per_partition="4GB",
        target_devices=["cuda:0", "cuda:1", "cpu", "cpu"],
        use_cache=True
    )
    
    # 性能优化配置
    configs["performance"] = LlamaDistConfig(
        num_partitions=8,
        strategy_type="compute",
        target_devices=[f"cuda:{i}" for i in range(8)],
        enable_async=True,
        max_workers=8,
        use_cache=True
    )
    
    # 混合配置
    configs["mixed"] = LlamaDistConfig(
        num_partitions=4,
        strategy_type="mixed",
        max_memory_per_partition="6GB",
        memory_weight=0.7,
        target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        temperature=0.8,
        top_p=0.9
    )
    
    return configs


def load_config_from_env() -> LlamaDistConfig:
    """从环境变量加载配置"""
    import os
    
    config = LlamaDistConfig()
    
    # 从环境变量更新配置
    env_mappings = {
        'LLAMADIST_MODEL_PATH': 'model_path',
        'LLAMADIST_NUM_PARTITIONS': ('num_partitions', int),
        'LLAMADIST_STRATEGY_TYPE': 'strategy_type',
        'LLAMADIST_MAX_MEMORY': 'max_memory_per_partition',
        'LLAMADIST_MAX_LENGTH': ('max_length', int),
        'LLAMADIST_TEMPERATURE': ('temperature', float),
        'LLAMADIST_OUTPUT_DIR': 'output_dir',
        'LLAMADIST_ENABLE_ASYNC': ('enable_async', bool),
    }
    
    for env_var, attr_config in env_mappings.items():
        if env_var in os.environ:
            if isinstance(attr_config, tuple):
                attr_name, attr_type = attr_config
                value = attr_type(os.environ[env_var])
            else:
                attr_name = attr_config
                value = os.environ[env_var]
            
            setattr(config, attr_name, value)
    
    # 特殊处理设备列表
    if 'LLAMADIST_DEVICES' in os.environ:
        devices = os.environ['LLAMADIST_DEVICES'].split(',')
        config.target_devices = [device.strip() for device in devices]
    
    return config


def merge_configs(base_config: LlamaDistConfig, override_config: Dict[str, Any]) -> LlamaDistConfig:
    """合并配置"""
    base_dict = base_config.to_dict()
    
    # 递归合并字典
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_config)
    return LlamaDistConfig.from_dict(merged_dict) 