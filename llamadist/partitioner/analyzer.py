"""
模型分析器

分析Llama模型的结构、内存需求、计算成本等信息，
为分层策略提供决策依据。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass
import psutil

# 使用内置的模型实现
from ..models.llama_seq import LlamaForCausalLMSeq, LlamaDecoderLayer
from transformers import LlamaConfig


@dataclass
class LayerInfo:
    """单层信息"""
    layer_idx: int                    # 层索引
    layer_type: str                   # 层类型
    param_count: int                  # 参数数量
    memory_usage: int                 # 内存使用量（字节）
    compute_cost: float               # 计算成本（相对值）
    input_shape: Optional[Tuple] = None   # 输入形状
    output_shape: Optional[Tuple] = None  # 输出形状


@dataclass
class ModelInfo:
    """模型信息"""
    model_name: str                   # 模型名称
    num_layers: int                   # 层数
    total_params: int                 # 总参数数量
    total_memory: int                 # 总内存使用量
    hidden_size: int                  # 隐藏层维度
    vocab_size: int                   # 词汇表大小
    layer_infos: List[LayerInfo]      # 各层详细信息
    layer_memory_costs: List[int]     # 各层内存成本
    layer_compute_costs: List[float]  # 各层计算成本
    layer_params: List[int]           # 各层参数数量


class LlamaModelAnalyzer:
    """
    Llama模型分析器
    
    分析模型结构、内存需求、计算成本等信息，
    为分层策略制定提供数据支持。
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化模型分析器
        
        Args:
            model_path: 模型路径
            config: 模型配置字典
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        self.model_info = None
    
    def load_model(self, device: str = "cpu") -> nn.Module:
        """
        加载模型
        
        Args:
            device: 目标设备
            
        Returns:
            nn.Module: 加载的模型
        """
        if self.model is not None:
            return self.model
        
        try:
            if self.model_path:
                # 从路径加载模型
                self.model = LlamaForCausalLMSeq.from_pretrained(
                    self.model_path,
                    device_map=device,
                    torch_dtype=torch.float16
                )
            elif self.config:
                # 从配置创建模型
                llama_config = LlamaConfig(**self.config)
                self.model = LlamaForCausalLMSeq(llama_config)
                self.model.to(device)
            else:
                raise ValueError("必须提供model_path或config")
            
            print(f"成功加载模型到 {device}")
            return self.model
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 如果加载失败，尝试创建一个默认配置的模型用于分析
            return self._create_default_model(device)
    
    def _create_default_model(self, device: str = "cpu") -> nn.Module:
        """
        创建默认配置的模型用于分析
        
        Args:
            device: 目标设备
            
        Returns:
            nn.Module: 默认模型
        """
        default_config = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": False,
            "hidden_act": "silu",
            "pretraining_tp": 1
        }
        
        llama_config = LlamaConfig(**default_config)
        model = LlamaForCausalLMSeq(llama_config)
        model.to(device)
        
        self.model = model
        print(f"创建默认模型配置并加载到 {device}")
        return model
    
    def analyze_model(self, 
                     sample_input_shape: Tuple[int, int] = (1, 512),
                     device: str = "cpu",
                     detailed: bool = True) -> ModelInfo:
        """
        分析模型结构和资源需求
        
        Args:
            sample_input_shape: 样本输入形状 (batch_size, seq_len)
            device: 分析设备
            detailed: 是否进行详细分析
            
        Returns:
            ModelInfo: 模型分析结果
        """
        print("开始分析模型...")
        
        # 加载模型
        model = self.load_model(device)
        
        # 基础信息分析
        basic_info = self._analyze_basic_info(model)
        
        if detailed:
            # 详细分析（包括内存和计算成本）
            layer_infos = self._analyze_layers_detailed(model, sample_input_shape, device)
        else:
            # 简单分析
            layer_infos = self._analyze_layers_simple(model)
        
        # 组装模型信息
        self.model_info = ModelInfo(
            model_name=getattr(model, '_name_or_path', 'llama'),
            num_layers=basic_info['num_layers'],
            total_params=basic_info['total_params'],
            total_memory=basic_info['total_memory'],
            hidden_size=basic_info['hidden_size'],
            vocab_size=basic_info['vocab_size'],
            layer_infos=layer_infos,
            layer_memory_costs=[info.memory_usage for info in layer_infos],
            layer_compute_costs=[info.compute_cost for info in layer_infos],
            layer_params=[info.param_count for info in layer_infos]
        )
        
        print(f"模型分析完成：{self.model_info.num_layers}层，"
              f"{self.model_info.total_params/1e9:.2f}B参数，"
              f"{self.model_info.total_memory/1e9:.2f}GB内存")
        
        return self.model_info
    
    def _analyze_basic_info(self, model: nn.Module) -> Dict:
        """
        分析模型基础信息
        
        Args:
            model: 模型实例
            
        Returns:
            Dict: 基础信息
        """
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算内存使用量（考虑FP16）
        param_memory = total_params * 2  # FP16，每个参数2字节
        
        # 获取模型配置信息
        config = getattr(model, 'config', None)
        if config:
            num_layers = getattr(config, 'num_hidden_layers', 32)
            hidden_size = getattr(config, 'hidden_size', 4096)
            vocab_size = getattr(config, 'vocab_size', 32000)
        else:
            # 通过模型结构推断
            num_layers = len([m for m in model.modules() if isinstance(m, LlamaDecoderLayer)])
            hidden_size = 4096  # 默认值
            vocab_size = 32000  # 默认值
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_memory': param_memory,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'vocab_size': vocab_size
        }
    
    def _analyze_layers_detailed(self, 
                               model: nn.Module, 
                               sample_input_shape: Tuple[int, int],
                               device: str) -> List[LayerInfo]:
        """
        详细分析各层信息（包括内存和计算成本）
        
        Args:
            model: 模型实例
            sample_input_shape: 样本输入形状
            device: 设备
            
        Returns:
            List[LayerInfo]: 层信息列表
        """
        layer_infos = []
        
        # 获取解码器层
        decoder_layers = []
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            decoder_layers = model.model.layers
        
        # 创建样本输入用于分析
        batch_size, seq_len = sample_input_shape
        sample_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        print(f"使用样本输入形状 {sample_input_shape} 进行详细分析...")
        
        # 分析嵌入层
        embed_info = self._analyze_embedding_layer(model, sample_input_ids)
        if embed_info:
            layer_infos.append(embed_info)
        
        # 分析每个解码器层
        for i, layer in enumerate(decoder_layers):
            layer_info = self._analyze_decoder_layer(layer, i, sample_input_ids, device)
            layer_infos.append(layer_info)
        
        # 分析输出层
        output_info = self._analyze_output_layer(model, sample_input_ids)
        if output_info:
            layer_infos.append(output_info)
        
        return layer_infos
    
    def _analyze_layers_simple(self, model: nn.Module) -> List[LayerInfo]:
        """
        简单分析各层信息（仅基于参数数量估算）
        
        Args:
            model: 模型实例
            
        Returns:
            List[LayerInfo]: 层信息列表
        """
        layer_infos = []
        
        # 获取解码器层
        decoder_layers = []
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            decoder_layers = model.model.layers
        
        # 分析每个解码器层
        for i, layer in enumerate(decoder_layers):
            param_count = sum(p.numel() for p in layer.parameters())
            memory_usage = param_count * 2  # FP16
            compute_cost = param_count / 1e9  # 简单的计算成本估算
            
            layer_info = LayerInfo(
                layer_idx=i,
                layer_type="LlamaDecoderLayer",
                param_count=param_count,
                memory_usage=memory_usage,
                compute_cost=compute_cost
            )
            layer_infos.append(layer_info)
        
        return layer_infos
    
    def _analyze_embedding_layer(self, model: nn.Module, sample_input: torch.Tensor) -> Optional[LayerInfo]:
        """
        分析嵌入层
        
        Args:
            model: 模型实例
            sample_input: 样本输入
            
        Returns:
            Optional[LayerInfo]: 嵌入层信息
        """
        if not (hasattr(model, 'model') and hasattr(model.model, 'embed_tokens')):
            return None
        
        embed_layer = model.model.embed_tokens
        param_count = sum(p.numel() for p in embed_layer.parameters())
        memory_usage = param_count * 2  # FP16
        
        return LayerInfo(
            layer_idx=-1,  # 特殊标记，表示嵌入层
            layer_type="Embedding",
            param_count=param_count,
            memory_usage=memory_usage,
            compute_cost=0.1,  # 嵌入层计算成本较低
            input_shape=sample_input.shape,
            output_shape=(sample_input.shape[0], sample_input.shape[1], embed_layer.embedding_dim)
        )
    
    def _analyze_decoder_layer(self, 
                             layer: nn.Module, 
                             layer_idx: int,
                             sample_input: torch.Tensor,
                             device: str) -> LayerInfo:
        """
        分析单个解码器层
        
        Args:
            layer: 解码器层
            layer_idx: 层索引
            sample_input: 样本输入
            device: 设备
            
        Returns:
            LayerInfo: 层信息
        """
        # 计算参数数量
        param_count = sum(p.numel() for p in layer.parameters())
        memory_usage = param_count * 2  # FP16
        
        # 估算计算成本（基于参数数量和序列长度）
        seq_len = sample_input.shape[1]
        
        # 注意力计算成本：O(seq_len^2 * hidden_size)
        hidden_size = getattr(layer, 'hidden_size', 4096)
        attention_cost = seq_len * seq_len * hidden_size / 1e9
        
        # MLP计算成本：O(seq_len * hidden_size * intermediate_size)
        intermediate_size = getattr(layer.mlp, 'intermediate_size', 11008) if hasattr(layer, 'mlp') else 11008
        mlp_cost = seq_len * hidden_size * intermediate_size / 1e9
        
        total_compute_cost = attention_cost + mlp_cost
        
        return LayerInfo(
            layer_idx=layer_idx,
            layer_type="LlamaDecoderLayer",
            param_count=param_count,
            memory_usage=memory_usage,
            compute_cost=total_compute_cost,
            input_shape=(sample_input.shape[0], sample_input.shape[1], hidden_size),
            output_shape=(sample_input.shape[0], sample_input.shape[1], hidden_size)
        )
    
    def _analyze_output_layer(self, model: nn.Module, sample_input: torch.Tensor) -> Optional[LayerInfo]:
        """
        分析输出层（语言模型头）
        
        Args:
            model: 模型实例
            sample_input: 样本输入
            
        Returns:
            Optional[LayerInfo]: 输出层信息
        """
        if not hasattr(model, 'lm_head'):
            return None
        
        lm_head = model.lm_head
        param_count = sum(p.numel() for p in lm_head.parameters())
        memory_usage = param_count * 2  # FP16
        
        # 输出层计算成本
        hidden_size = lm_head.in_features
        vocab_size = lm_head.out_features
        seq_len = sample_input.shape[1]
        compute_cost = seq_len * hidden_size * vocab_size / 1e9
        
        return LayerInfo(
            layer_idx=9999,  # 特殊标记，表示输出层
            layer_type="LMHead",
            param_count=param_count,
            memory_usage=memory_usage,
            compute_cost=compute_cost,
            input_shape=(sample_input.shape[0], sample_input.shape[1], hidden_size),
            output_shape=(sample_input.shape[0], sample_input.shape[1], vocab_size)
        )
    
    def benchmark_layer_performance(self, 
                                  layer_idx: int, 
                                  num_runs: int = 10,
                                  sample_input_shape: Tuple[int, int] = (1, 512)) -> Dict:
        """
        基准测试特定层的性能
        
        Args:
            layer_idx: 层索引
            num_runs: 运行次数
            sample_input_shape: 样本输入形状
            
        Returns:
            Dict: 性能基准结果
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()")
        
        # 获取目标层
        if not (hasattr(self.model, 'model') and hasattr(self.model.model, 'layers')):
            raise ValueError("无法访问模型层")
        
        if layer_idx >= len(self.model.model.layers):
            raise ValueError(f"层索引 {layer_idx} 超出范围")
        
        layer = self.model.model.layers[layer_idx]
        device = next(layer.parameters()).device
        
        # 创建样本输入
        batch_size, seq_len = sample_input_shape
        hidden_size = layer.hidden_size
        sample_hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float16)
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = layer(sample_hidden_states)
        
        # 性能测试
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = layer(sample_hidden_states)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        
        return {
            'layer_idx': layer_idx,
            'avg_inference_time': avg_time,
            'throughput_tokens_per_second': (batch_size * seq_len) / avg_time,
            'input_shape': sample_input_shape,
            'device': str(device)
        }
    
    def get_memory_profile(self) -> Dict:
        """
        获取当前内存使用情况
        
        Returns:
            Dict: 内存使用情况
        """
        # 系统内存
        memory = psutil.virtual_memory()
        
        profile = {
            'system_memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            }
        }
        
        # GPU内存（如果可用）
        if torch.cuda.is_available():
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'cuda:{i}'] = {
                    'total': torch.cuda.get_device_properties(i).total_memory,
                    'allocated': torch.cuda.memory_allocated(i),
                    'reserved': torch.cuda.memory_reserved(i)
                }
            profile['gpu_memory'] = gpu_memory
        
        return profile
    
    def save_analysis(self, output_path: str):
        """
        保存分析结果到文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.model_info is None:
            raise ValueError("尚未进行模型分析，请先调用analyze_model()")
        
        # 转换为可序列化的格式
        analysis_data = {
            'model_name': self.model_info.model_name,
            'num_layers': self.model_info.num_layers,
            'total_params': self.model_info.total_params,
            'total_memory': self.model_info.total_memory,
            'hidden_size': self.model_info.hidden_size,
            'vocab_size': self.model_info.vocab_size,
            'layer_memory_costs': self.model_info.layer_memory_costs,
            'layer_compute_costs': self.model_info.layer_compute_costs,
            'layer_params': self.model_info.layer_params,
            'layers': []
        }
        
        # 添加层详细信息
        for layer_info in self.model_info.layer_infos:
            layer_data = {
                'layer_idx': layer_info.layer_idx,
                'layer_type': layer_info.layer_type,
                'param_count': layer_info.param_count,
                'memory_usage': layer_info.memory_usage,
                'compute_cost': layer_info.compute_cost,
                'input_shape': layer_info.input_shape,
                'output_shape': layer_info.output_shape
            }
            analysis_data['layers'].append(layer_data)
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {output_path}")
    
    def load_analysis(self, input_path: str) -> ModelInfo:
        """
        从文件加载分析结果
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            ModelInfo: 模型信息
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # 重建层信息
        layer_infos = []
        for layer_data in analysis_data['layers']:
            layer_info = LayerInfo(
                layer_idx=layer_data['layer_idx'],
                layer_type=layer_data['layer_type'],
                param_count=layer_data['param_count'],
                memory_usage=layer_data['memory_usage'],
                compute_cost=layer_data['compute_cost'],
                input_shape=tuple(layer_data['input_shape']) if layer_data['input_shape'] else None,
                output_shape=tuple(layer_data['output_shape']) if layer_data['output_shape'] else None
            )
            layer_infos.append(layer_info)
        
        # 重建模型信息
        self.model_info = ModelInfo(
            model_name=analysis_data['model_name'],
            num_layers=analysis_data['num_layers'],
            total_params=analysis_data['total_params'],
            total_memory=analysis_data['total_memory'],
            hidden_size=analysis_data['hidden_size'],
            vocab_size=analysis_data['vocab_size'],
            layer_infos=layer_infos,
            layer_memory_costs=analysis_data['layer_memory_costs'],
            layer_compute_costs=analysis_data['layer_compute_costs'],
            layer_params=analysis_data['layer_params']
        )
        
        print(f"分析结果已从 {input_path} 加载")
        return self.model_info 