"""
Llama模型分层器

负责根据分层策略将Llama模型拆分成多个子模型，
每个子模型可以独立运行和部署。
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import copy
from pathlib import Path

# 使用内置的模型实现
from ..models.llama_seq import (
    LlamaForCausalLMSeq, 
    LlamaModelSeq, 
    LlamaDecoderLayer,
    LlamaRMSNorm,
    _make_causal_mask,
    _expand_mask
)
from transformers import LlamaConfig

from .strategies import PartitionStrategy, PartitionConfig
from .analyzer import LlamaModelAnalyzer, ModelInfo, LayerInfo


class LlamaSubModel(nn.Module):
    """
    Llama子模型
    
    表示分层后的单个子模型，包含部分解码器层和必要的组件。
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        partition_config: PartitionConfig,
        partition_idx: int,
        total_partitions: int
    ):
        """
        初始化子模型
        
        Args:
            config: Llama配置
            partition_config: 分层配置
            partition_idx: 当前分层索引
            total_partitions: 总分层数量
        """
        super().__init__()
        
        self.config = config
        self.partition_config = partition_config
        self.partition_idx = partition_idx
        self.total_partitions = total_partitions
        
        # 子模型包含的层范围
        self.layer_start = partition_config.layer_start
        self.layer_end = partition_config.layer_end
        self.num_layers = self.layer_end - self.layer_start + 1
        
        # 判断是否是第一个或最后一个分层
        self.is_first_partition = (partition_idx == 0)
        self.is_last_partition = (partition_idx == total_partitions - 1)
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """初始化子模型组件"""
        # 如果是第一个分层，需要嵌入层
        if self.is_first_partition:
            self.embed_tokens = nn.Embedding(
                self.config.vocab_size, 
                self.config.hidden_size,
                self.config.pad_token_id
            )
        
        # 解码器层
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(self.config) 
            for _ in range(self.num_layers)
        ])
        
        # 如果是最后一个分层，需要归一化层和语言模型头
        ## TODO：后续需要在退出层直接添加一个lm_head，而不是在最后一个分层添加，从而直接输出中间态结果
        if self.is_last_partition:
            self.norm = LlamaRMSNorm(
                self.config.hidden_size, 
                eps=self.config.rms_norm_eps
            )
            self.lm_head = nn.Linear(
                self.config.hidden_size, 
                self.config.vocab_size, 
                bias=False
            )
    
    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        子模型前向传播
        
        Args:
            hidden_states: 隐藏状态（来自前一个分层）
            input_ids: 输入token ID（仅第一个分层使用）
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_values: KV缓存
            use_cache: 是否使用缓存
            output_attentions: 是否输出注意力权重
            return_dict: 是否返回字典格式
        
        Returns:
            Dict: 包含隐藏状态、KV缓存等的字典
        """
        # 如果是第一个分层，从input_ids开始
        if self.is_first_partition:
            if input_ids is None:
                raise ValueError("第一个分层必须提供input_ids")
            
            # 词嵌入
            hidden_states = self.embed_tokens(input_ids)
            batch_size, seq_length = input_ids.shape
        else:
            if hidden_states is None:
                raise ValueError("非第一个分层必须提供hidden_states")
            
            batch_size, seq_length, _ = hidden_states.shape
        
        # 计算past_length
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            # 检查第一个有效的past_key_value
            for pkv in past_key_values:
                if pkv is not None and len(pkv) > 0:
                    past_length = pkv[0].shape[2]
                    break
        
        # 准备注意力掩码 - 修复KV-cache时的掩码尺寸问题
        if attention_mask is None:
            # 创建默认的注意力掩码，覆盖完整的序列长度（包括过去的tokens）
            total_length = past_length + seq_length
            attention_mask = torch.ones(
                (batch_size, total_length), 
                dtype=torch.bool, 
                device=hidden_states.device
            )
        else:
            # 如果提供了attention_mask，确保它覆盖完整序列
            if past_length > 0 and attention_mask.shape[1] != (past_length + seq_length):
                # 如果attention_mask只覆盖当前序列，需要扩展到包含past_length
                if attention_mask.shape[1] == seq_length:
                    # 为过去的tokens创建掩码（全为1，表示可见）
                    past_mask = torch.ones(
                        (batch_size, past_length), 
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([past_mask, attention_mask], dim=1)
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length, 
                seq_length + past_length, 
                dtype=torch.long, 
                device=hidden_states.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # 准备因果掩码
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, 
            (batch_size, seq_length), 
            hidden_states, 
            past_length
        )
        
        # 通过解码器层
        all_hidden_states = () if output_attentions else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_attentions:
                all_hidden_states += (hidden_states,)
            
            # 获取对应的过去键值对
            past_key_value = None
            if past_key_values is not None:
                # 调整索引以匹配当前分层的层
                local_layer_idx = idx  # 使用本地层索引
                if local_layer_idx < len(past_key_values):
                    past_key_value = past_key_values[local_layer_idx]
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        # 如果是最后一个分层，应用归一化和语言模型头
        logits = None
        if self.is_last_partition:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
        
        # 组装输出
        result = {
            'hidden_states': hidden_states,
            'logits': logits,
            'past_key_values': next_decoder_cache if use_cache else None,
            'partition_idx': self.partition_idx,
            'is_last_partition': self.is_last_partition
        }
        
        if output_attentions:
            result['attentions'] = all_self_attns
            result['all_hidden_states'] = all_hidden_states
        
        return result
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """准备解码器注意力掩码"""
        # 创建因果掩码
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        
        if attention_mask is not None:
            # 扩展注意力掩码
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        
        return combined_attention_mask
    
    def get_memory_usage(self) -> int:
        """
        获取子模型的内存使用量
        
        Returns:
            int: 内存使用量（字节）
        """
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 2  # FP16
    
    def get_info(self) -> Dict:
        """
        获取子模型信息
        
        Returns:
            Dict: 子模型信息
        """
        return {
            'partition_idx': self.partition_idx,
            'total_partitions': self.total_partitions,
            'layer_start': self.layer_start,
            'layer_end': self.layer_end,
            'num_layers': self.num_layers,
            'is_first_partition': self.is_first_partition,
            'is_last_partition': self.is_last_partition,
            'memory_usage': self.get_memory_usage(),
            'device': str(next(self.parameters()).device),
            'target_device': self.partition_config.device
        }


class LlamaPartitioner:
    """
    Llama模型分层器
    
    负责根据分层策略将完整的Llama模型拆分成多个子模型。
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化分层器
        
        Args:
            model_path: 模型路径
            config: 模型配置字典
        """
        self.model_path = model_path
        self.config = config
        self.original_model = None
        self.analyzer = LlamaModelAnalyzer(model_path, config)
    
    def load_original_model(self, device: str = "cuda:0") -> LlamaForCausalLMSeq:
        """
        加载原始模型
        
        Args:
            device: 目标设备
            
        Returns:
            LlamaForCausalLMSeq: 原始模型
        """
        if self.original_model is not None:
            return self.original_model
        
        self.original_model = self.analyzer.load_model(device)
        return self.original_model
    
    def analyze_model(self, detailed: bool = True, device: str = None) -> ModelInfo:
        """
        分析模型结构
        
        Args:
            detailed: 是否进行详细分析
            
        Returns:
            ModelInfo: 模型分析结果
        """
        return self.analyzer.analyze_model(detailed=detailed, device=device)
    
    def partition(
        self, 
        strategy: PartitionStrategy,
        analyze_first: bool = True,
        copy_weights: bool = True
    ) -> List[LlamaSubModel]:
        """
        执行模型分层
        
        Args:
            strategy: 分层策略
            analyze_first: 是否先分析模型
            copy_weights: 是否复制权重到子模型
            
        Returns:
            List[LlamaSubModel]: 子模型列表
        """
        print("开始模型分层...")
        
        # 分析模型（如果需要）
        if analyze_first:
            model_info = self.analyze_model(detailed=True)
        else:
            # 获取基本信息并创建ModelInfo对象
            original_model = self.load_original_model()
            config = original_model.config
            
            # 创建简单的层信息
            layer_infos = []
            for i in range(config.num_hidden_layers):
                layer_info = LayerInfo(
                    layer_idx=i,
                    layer_type="LlamaDecoderLayer",
                    param_count=1,  # 简化值
                    memory_usage=1,  # 简化值
                    compute_cost=1.0  # 简化值
                )
                layer_infos.append(layer_info)
            
            model_info = ModelInfo(
                model_name="llama",
                num_layers=config.num_hidden_layers,
                total_params=1,  # 简化值
                total_memory=1,  # 简化值
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                layer_infos=layer_infos,
                layer_memory_costs=[1] * config.num_hidden_layers,
                layer_compute_costs=[1.0] * config.num_hidden_layers,
                layer_params=[1] * config.num_hidden_layers
            )
        
        # 创建分层配置
        partitions = strategy.create_partitions(model_info)
        
        # 验证分层策略
        if not strategy.validate_strategy(partitions, model_info):
            raise ValueError("分层策略验证失败")
        
        print(f"创建 {len(partitions)} 个子模型:")
        for i, partition in enumerate(partitions):
            print(f"  分层 {i}: 层 {partition.layer_start}-{partition.layer_end} -> {partition.device}")
        
        # 创建子模型
        submodels = []
        original_model = self.load_original_model()
        
        for i, partition_config in enumerate(partitions):
            # 创建子模型
            submodel = LlamaSubModel(
                config=original_model.config,
                partition_config=partition_config,
                partition_idx=i,
                total_partitions=len(partitions)
            )
            
            # 复制权重（如果需要）
            if copy_weights:
                self._copy_weights_to_submodel(original_model, submodel, partition_config)
            
            # 移动到目标设备
            if partition_config.device:
                submodel.to(partition_config.device)
            
            submodels.append(submodel)
        
        print(f"模型分层完成，创建了 {len(submodels)} 个子模型")
        return submodels
    
    def _copy_weights_to_submodel(
        self,
        original_model: LlamaForCausalLMSeq,
        submodel: LlamaSubModel,
        partition_config: PartitionConfig
    ):
        """
        将原始模型的权重复制到子模型
        
        Args:
            original_model: 原始模型
            submodel: 目标子模型
            partition_config: 分层配置
        """
        print(f"  复制权重到分层 {submodel.partition_idx}...")
        
        # 复制嵌入层权重（如果是第一个分层）
        if submodel.is_first_partition and hasattr(original_model.model, 'embed_tokens'):
            submodel.embed_tokens.load_state_dict(
                original_model.model.embed_tokens.state_dict()
            )
        
        # 复制解码器层权重
        original_layers = original_model.model.layers
        for i, target_layer in enumerate(submodel.layers):
            source_layer_idx = partition_config.layer_start + i
            if source_layer_idx < len(original_layers):
                source_layer = original_layers[source_layer_idx]
                target_layer.load_state_dict(source_layer.state_dict())
        
        # 复制归一化层和语言模型头权重（如果是最后一个分层）
        if submodel.is_last_partition:
            if hasattr(original_model.model, 'norm'):
                submodel.norm.load_state_dict(
                    original_model.model.norm.state_dict()
                )
            
            if hasattr(original_model, 'lm_head'):
                submodel.lm_head.load_state_dict(
                    original_model.lm_head.state_dict()
                )
    
    def save_partitioned_models(
        self, 
        submodels: List[LlamaSubModel], 
        output_dir: str,
        save_config: bool = True
    ):
        """
        保存分层后的子模型
        
        Args:
            submodels: 子模型列表
            output_dir: 输出目录
            save_config: 是否保存配置文件
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"保存 {len(submodels)} 个子模型到 {output_dir}...")
        
        # 保存每个子模型
        for submodel in submodels:
            submodel_dir = output_path / f"submodel_{submodel.partition_idx}"
            submodel_dir.mkdir(exist_ok=True)
            
            # 保存模型权重
            torch.save(
                submodel.state_dict(), 
                submodel_dir / "pytorch_model.bin"
            )
            
            # 保存子模型信息
            submodel_info = submodel.get_info()
            with open(submodel_dir / "submodel_info.json", 'w') as f:
                import json
                json.dump(submodel_info, f, indent=2)
            
            print(f"  已保存子模型 {submodel.partition_idx} 到 {submodel_dir}")
        
        # 保存整体配置（如果需要）
        if save_config and self.original_model:
            config_dict = self.original_model.config.to_dict()
            
            # 添加分层信息
            partition_info = {
                'total_partitions': len(submodels),
                'partitions': [
                    {
                        'partition_idx': sm.partition_idx,
                        'layer_start': sm.layer_start,
                        'layer_end': sm.layer_end,
                        'device': sm.partition_config.device
                    }
                    for sm in submodels
                ]
            }
            
            config_dict['partition_info'] = partition_info
            
            with open(output_path / "partition_config.json", 'w') as f:
                import json
                json.dump(config_dict, f, indent=2)
        
        print(f"所有子模型已保存到 {output_dir}")
    
    def load_partitioned_models(
        self, 
        input_dir: str,
        devices: Optional[List[str]] = None
    ) -> List[LlamaSubModel]:
        """
        加载分层后的子模型
        
        Args:
            input_dir: 输入目录
            devices: 目标设备列表
            
        Returns:
            List[LlamaSubModel]: 加载的子模型列表
        """
        input_path = Path(input_dir)
        
        # 加载配置
        config_file = input_path / "partition_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"未找到配置文件: {config_file}")
        
        with open(config_file, 'r') as f:
            import json
            config_data = json.load(f)
        
        # 提取分层信息
        partition_info = config_data.pop('partition_info')
        llama_config = LlamaConfig(**config_data)
        
        # 创建子模型列表
        submodels = []
        total_partitions = partition_info['total_partitions']
        
        print(f"加载 {total_partitions} 个子模型从 {input_dir}...")
        
        for partition_data in partition_info['partitions']:
            partition_idx = partition_data['partition_idx']
            
            # 创建分层配置
            partition_config = PartitionConfig(
                layer_start=partition_data['layer_start'],
                layer_end=partition_data['layer_end'],
                device=devices[partition_idx] if devices else partition_data['device']
            )
            
            # 创建子模型
            submodel = LlamaSubModel(
                config=llama_config,
                partition_config=partition_config,
                partition_idx=partition_idx,
                total_partitions=total_partitions
            )
            
            # 加载权重
            submodel_dir = input_path / f"submodel_{partition_idx}"
            weight_file = submodel_dir / "pytorch_model.bin"
            
            if weight_file.exists():
                state_dict = torch.load(weight_file, map_location='cpu')
                submodel.load_state_dict(state_dict)
                print(f"  已加载子模型 {partition_idx}")
            else:
                print(f"  警告：未找到子模型 {partition_idx} 的权重文件")
            
            # 移动到目标设备
            if partition_config.device:
                submodel.to(partition_config.device)
            
            submodels.append(submodel)
        
        print(f"所有子模型已加载")
        return submodels
    
    def validate_partitioned_models(
        self, 
        submodels: List[LlamaSubModel],
        sample_input: Optional[torch.Tensor] = None
    ) -> bool:
        """
        验证分层模型的完整性
        
        Args:
            submodels: 子模型列表
            sample_input: 测试输入
            
        Returns:
            bool: 验证是否通过
        """
        print("验证分层模型完整性...")
        
        # 检查分层连续性
        expected_layers = set()
        for submodel in submodels:
            for layer_idx in range(submodel.layer_start, submodel.layer_end + 1):
                if layer_idx in expected_layers:
                    print(f"错误：层 {layer_idx} 被多个分层覆盖")
                    return False
                expected_layers.add(layer_idx)
        
        # 检查第一个和最后一个分层
        first_partitions = [sm for sm in submodels if sm.is_first_partition]
        last_partitions = [sm for sm in submodels if sm.is_last_partition]
        
        if len(first_partitions) != 1:
            print(f"错误：应该有且仅有1个第一分层，实际有 {len(first_partitions)} 个")
            return False
        
        if len(last_partitions) != 1:
            print(f"错误：应该有且仅有1个最后分层，实际有 {len(last_partitions)} 个")
            return False
        
        # 功能性测试（如果提供了样本输入）
        if sample_input is not None:
            try:
                # 创建测试输入
                if sample_input.dim() == 1:
                    sample_input = sample_input.unsqueeze(0)  # 添加batch维度
                
                # 测试推理流程
                hidden_states = None
                past_key_values = None
                
                for submodel in submodels:
                    if submodel.is_first_partition:
                        # 第一个分层使用input_ids
                        result = submodel(
                            input_ids=sample_input.to(submodel.get_info()['device']),
                            use_cache=True
                        )
                    else:
                        # 后续分层使用hidden_states
                        result = submodel(
                            hidden_states=hidden_states.to(submodel.get_info()['device']),
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    
                    hidden_states = result['hidden_states']
                    past_key_values = result['past_key_values']
                
                print("功能性测试通过")
                
            except Exception as e:
                print(f"功能性测试失败: {e}")
                return False
        
        print("分层模型验证通过")
        return True 