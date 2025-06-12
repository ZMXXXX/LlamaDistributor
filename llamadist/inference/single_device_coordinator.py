"""
单设备分层推理协调器

专门处理在同一设备上的分层推理，不涉及跨设备通信。
适用于测试分层效果、内存优化和调试分层逻辑。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import time
import copy
from dataclasses import dataclass

# 使用内置的分层器模块
from ..partitioner.splitter import LlamaSubModel
from .coordinator import GenerationConfig, InferenceState


class SingleDeviceInference:
    """
    单设备分层推理引擎
    
    专门处理在同一设备上的多个子模型推理，特点：
    1. 所有子模型都在同一设备上
    2. 不需要跨设备状态传递
    3. 简化的缓存管理
    4. 更高效的内存利用
    5. 便于调试和分析
    """
    
    def __init__(
        self,
        submodels: List[LlamaSubModel],
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化单设备分层推理引擎
        
        Args:
            submodels: 子模型列表（按分层顺序）
            generation_config: 生成配置
            device: 目标设备（如果为None，使用第一个子模型的设备）
        """
        self.submodels = submodels
        self.generation_config = generation_config or GenerationConfig()
        
        # 验证子模型
        self._validate_submodels()
        
        # 确定设备
        self.device = device or self.submodels[0].get_info()['device']
        
        # 确保所有子模型在同一设备上
        self._ensure_same_device()
        
        # Early-exit submodel缓存
        self._early_exit_cache = {}  # 格式: {submodel_idx: prepared_submodel}
        self._original_lm_head_weights = None  # 保存原始lm_head权重用于复制
        
        # 性能统计
        self.stats = {
            'total_time_to_first_token': 0.0,
            'token_decode_time': 0.0,
            'total_generation_time': 0.0,
            'layer_processing_time': {},  # 每层的处理时间
            'memory_usage': {},           # 每层的内存使用
            'total_tokens_generated': 0,
            'inference_count': 0
        }
        
        print(f"单设备分层推理引擎初始化完成，{len(self.submodels)}个子模型 @ {self.device}")
        for sm in self.submodels:
            info = sm.get_info()
            print(f"  分层 {info['partition_idx']}: 层{info['layer_start']}-{info['layer_end']}")
    
    def set_original_lm_head_weights(self, lm_head_weights: torch.Tensor):
        """
        设置原始模型的lm_head权重，用于early-exit submodel
        
        Args:
            lm_head_weights: 原始模型的lm_head权重
        """
        self._original_lm_head_weights = lm_head_weights.clone().to(self.device)
    
    def _validate_submodels(self):
        """验证子模型的有效性"""
        if not self.submodels:
            raise ValueError("子模型列表不能为空")
        
        # 检查是否有第一个和最后一个分层
        first_partitions = [sm for sm in self.submodels if sm.is_first_partition]
        last_partitions = [sm for sm in self.submodels if sm.is_last_partition]
        
        if len(first_partitions) != 1:
            raise ValueError("必须有且仅有一个第一分层")
        
        if len(last_partitions) != 1:
            raise ValueError("必须有且仅有一个最后分层")
        
        # 按分层索引排序
        self.submodels.sort(key=lambda x: x.partition_idx)
    
    def _ensure_same_device(self):
        """确保所有子模型在同一设备上"""
        target_device = self.device
        
        for submodel in self.submodels:
            current_device = submodel.get_info()['device']
            if current_device != target_device:
                print(f"将子模型从 {current_device} 移动到 {target_device}")
                submodel.to(target_device)
    
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = True,
        exit_position: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行单设备分层submodels整体前向传播流程
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            past_key_values: 过去的键值对
            use_cache: 是否使用缓存
            exit_position: Early-exit位置（指定在哪个submodel结束时退出）
        
        Returns:
            Dict: 包含logits、hidden_states和past_key_values的字典
        """
        # 将输入移动到目标设备
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 移动KV缓存到目标设备
        if past_key_values is not None:
            past_key_values = self._transfer_kv_cache_to_device(past_key_values)
        
        # 初始化推理状态
        current_hidden_states = None
        current_past_key_values = past_key_values
        new_cache_list = []
        
        # 记录每层的处理时间和内存使用
        layer_stats = {}
        early_exit_submodel = None  # 保存early-exit时的最后一个submodel
        
        # 通过每个子模型进行推理
        for i, submodel in enumerate(self.submodels):
            # 检查是否应该停止（early-exit）
            # exit_position = n 表示执行前n个submodel（索引0到n-1）
            if exit_position is not None and i >= exit_position:
                break
            
            # 检查这是否是early-exit的最后一个submodel
            is_early_exit_last = (exit_position is not None and i == exit_position - 1)
            
            # 如果是early-exit的最后一个submodel且当前不是最后一个分层，则获取缓存的early-exit submodel
            if is_early_exit_last and not submodel.is_last_partition:
                submodel = self._get_early_exit_submodel(i)
                early_exit_submodel = submodel  # 保存准备好的submodel
            layer_start_time = time.time()
            
            # 记录处理前的内存使用
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated(self.device)
            else:
                memory_before = 0
            
            if submodel.is_first_partition:
                # 第一个分层使用input_ids
                current_input_ids = input_ids
                current_attention_mask = attention_mask
                
                if current_past_key_values is not None:
                    # 在有缓存时，只使用最后一个token
                    current_input_ids = input_ids[:, -1:]
                    # attention_mask需要覆盖完整序列（包括过去的tokens）
                    if attention_mask is not None:
                        batch_size = input_ids.shape[0]
                        seq_length = current_input_ids.shape[1]
                        
                        # 计算过去的序列长度
                        past_length = 0
                        if current_past_key_values is not None:
                            for pkv in current_past_key_values:
                                if pkv is not None and len(pkv) > 0 and pkv[0] is not None:
                                    past_length = pkv[0].shape[2]
                                    break
                        
                        # 创建完整的attention_mask
                        total_length = past_length + seq_length
                        current_attention_mask = torch.ones(
                            (batch_size, total_length),
                            dtype=attention_mask.dtype,
                            device=self.device
                        )
                
                model_input = {
                    'input_ids': current_input_ids,
                    'attention_mask': current_attention_mask,
                    'past_key_values': self._extract_relevant_kv_cache(current_past_key_values, submodel.layer_start, submodel.layer_end),
                    'use_cache': use_cache
                }
            else:
                # 后续分层使用hidden_states
                if current_hidden_states is None:
                    raise RuntimeError(f"子模型 {i} 需要hidden_states，但前一个子模型没有提供")
                
                model_input = {
                    'hidden_states': current_hidden_states,
                    'attention_mask': attention_mask,
                    'past_key_values': self._extract_relevant_kv_cache(current_past_key_values, submodel.layer_start, submodel.layer_end),
                    'use_cache': use_cache
                }
            
            # 执行子模型推理
            with torch.no_grad():
                output = submodel(**model_input)
            
            # 更新隐藏状态
            current_hidden_states = output['hidden_states']
            
            # 更新KV缓存
            if use_cache and 'past_key_values' in output and output['past_key_values'] is not None:
                # 将子模型的缓存合并到全局缓存
                current_past_key_values = self._merge_kv_cache(
                    current_past_key_values,
                    output['past_key_values'],
                    submodel.layer_start,
                    submodel.layer_end
                )
            
            # 记录层处理时间和内存使用
            layer_end_time = time.time()
            layer_time = layer_end_time - layer_start_time
            
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated(self.device)
                memory_delta = memory_after - memory_before
            else:
                memory_delta = 0
            
            layer_info = submodel.get_info()
            layer_key = f"partition_{i}_layers_{layer_info['layer_start']}-{layer_info['layer_end']}"
            layer_stats[layer_key] = {
                'processing_time': layer_time,
                'memory_delta': memory_delta,
                'memory_after': memory_after if torch.cuda.is_available() and self.device.startswith('cuda') else 0
            }
            
            # 如果这是early-exit的最后一个submodel，记录日志
            if is_early_exit_last:
                # 只在统计信息中记录early-exit，避免重复打印
                if not hasattr(self, '_early_exit_logged'):
                    print(f"Early-exit activated: executed {exit_position} submodels (0-{exit_position-1}), last submodel layers {submodel.layer_start}-{submodel.layer_end}")
                    self._early_exit_logged = True
        
        # 更新统计信息
   
        self.stats['inference_count'] += 1
        self.stats['layer_processing_time'].update(layer_stats)
        
        # 构建返回结果
        result = {
            'hidden_states': current_hidden_states,
            'past_key_values': current_past_key_values if use_cache else None
        }
        
        # 处理logits输出
        # 如果进行了early-exit，需要确保有logits输出
        # 对于early-exit的子模型，我们在_prepare_early_exit_submodel中已经添加了lm_head和norm
        if exit_position is not None:
            # Early-exit情况：使用保存的early_exit_submodel（如果有的话）
            if early_exit_submodel is not None:
                # 使用准备好的early-exit submodel
                last_submodel = early_exit_submodel
            else:
                # 如果没有准备early-exit submodel，使用原始的最后一个submodel
                last_processed_idx = exit_position - 1
                last_submodel = self.submodels[last_processed_idx]
            
            if hasattr(last_submodel, 'lm_head') and last_submodel.lm_head is not None:
                # 应用归一化（如果存在）
                normalized_hidden_states = current_hidden_states
                if hasattr(last_submodel, 'norm') and last_submodel.norm is not None:
                    normalized_hidden_states = last_submodel.norm(current_hidden_states)
                
                logits = last_submodel.lm_head(normalized_hidden_states)
                result['logits'] = logits
            else:
                # 如果最后一个submodel没有lm_head，需要创建临时的
                print(f"Warning: Early-exit submodel {exit_position-1} has no lm_head, creating temporary one...")
                temp_submodel = self._prepare_early_exit_submodel(self.submodels[exit_position - 1])
                normalized_hidden_states = current_hidden_states
                if hasattr(temp_submodel, 'norm') and temp_submodel.norm is not None:
                    normalized_hidden_states = temp_submodel.norm(current_hidden_states)
                
                logits = temp_submodel.lm_head(normalized_hidden_states)
                result['logits'] = logits
        else:
            # 正常情况：如果是最后一个分层，应该有logits
            if hasattr(self.submodels[-1], 'lm_head'):
                logits = self.submodels[-1].lm_head(current_hidden_states)
                result['logits'] = logits
        
        return result
    
    def _get_early_exit_submodel(self, submodel_idx: int) -> LlamaSubModel:
        """
        获取early-exit子模型（使用缓存优化）
        
        Args:
            submodel_idx: 子模型索引
        
        Returns:
            LlamaSubModel: 带有归一化层和语言模型头的子模型
        """
        # 检查缓存中是否已有准备好的early-exit submodel
        if submodel_idx in self._early_exit_cache:
            return self._early_exit_cache[submodel_idx]
        
        # 缓存中没有，创建新的early-exit submodel
        original_submodel = self.submodels[submodel_idx]
        early_exit_submodel = self._prepare_early_exit_submodel(original_submodel)
        
        # 缓存结果
        self._early_exit_cache[submodel_idx] = early_exit_submodel
        
        return early_exit_submodel
    
    def _prepare_early_exit_submodel(self, original_submodel: LlamaSubModel) -> LlamaSubModel:
        """
        为early-exit子模型，添加归一化层和语言模型头
        
        Args:
            original_submodel: 原始子模型
        
        Returns:
            LlamaSubModel: 带有归一化层和语言模型头的子模型副本
        """
        early_exit_submodel = copy.deepcopy(original_submodel)
        
        # 添加归一化层和语言模型头（如果还没有）
        if not hasattr(early_exit_submodel, 'norm') or early_exit_submodel.norm is None:
            from ..models.llama_seq import LlamaRMSNorm
            early_exit_submodel.norm = LlamaRMSNorm(
                early_exit_submodel.config.hidden_size,
                eps=early_exit_submodel.config.rms_norm_eps
            ).to(self.device)
            
            # 🔧 修复：如果有原始norm权重，复制过来
            if hasattr(self, '_original_norm_weights') and self._original_norm_weights is not None:
                early_exit_submodel.norm.weight.data.copy_(self._original_norm_weights.to(self.device))
                print(f"✅ 已为early-exit submodel复制原始norm权重")
        
        if not hasattr(early_exit_submodel, 'lm_head') or early_exit_submodel.lm_head is None:
            import torch.nn as nn
            early_exit_submodel.lm_head = nn.Linear(
                early_exit_submodel.config.hidden_size,
                early_exit_submodel.config.vocab_size,
                bias=False
            ).to(self.device)
            
            # 如果有原始模型可用，尝试复制权重
            if self._original_lm_head_weights is not None:
                early_exit_submodel.lm_head.weight.data.copy_(self._original_lm_head_weights)
                print(f"✅ 已为early-exit submodel复制原始lm_head权重")
        
        # 标记为最后分层，这样forward方法会输出logits
        early_exit_submodel.is_last_partition = True
        
        return early_exit_submodel
    
    def _transfer_kv_cache_to_device(self, past_key_values: List[torch.FloatTensor]) -> List[torch.FloatTensor]:
        """将KV缓存传输到目标设备"""
        if past_key_values is None:
            return None
        
        transferred_cache = []
        for layer_cache in past_key_values:
            if layer_cache is None:
                transferred_cache.append(None)
            else:
                transferred_layer_cache = tuple(
                    cache_tensor.to(self.device) if cache_tensor is not None else None
                    for cache_tensor in layer_cache
                )
                transferred_cache.append(transferred_layer_cache)
        
        return transferred_cache
    
    def _extract_relevant_kv_cache(
        self,
        past_key_values: Optional[List[torch.FloatTensor]],
        layer_start: int,
        layer_end: int
    ) -> Optional[List[torch.FloatTensor]]:
        """提取与当前子模型相关的KV缓存"""
        if past_key_values is None:
            return None
        
        relevant_cache = []
        for layer_idx in range(layer_start, layer_end + 1):
            if layer_idx < len(past_key_values):
                relevant_cache.append(past_key_values[layer_idx])
            else:
                relevant_cache.append(None)
        
        return relevant_cache
    
    def _merge_kv_cache(
        self,
        global_cache: Optional[List[torch.FloatTensor]],
        local_cache: Optional[List[torch.FloatTensor]],
        layer_start: int,
        layer_end: int
    ) -> Optional[List[torch.FloatTensor]]:
        """将子模型的KV缓存合并到全局缓存"""
        if local_cache is None:
            return global_cache
        
        if global_cache is None:
            # 初始化全局缓存
            total_layers = max(submodel.layer_end for submodel in self.submodels) + 1
            global_cache = [None] * total_layers
        
        # 合并缓存
        for i, layer_cache in enumerate(local_cache):
            global_layer_idx = layer_start + i
            if global_layer_idx < len(global_cache):
                global_cache[global_layer_idx] = layer_cache
        
        return global_cache
    
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        使用单设备分层推理生成文本
        
        Args:
            input_ids: 输入token ID
            generation_config: 生成配置
            tokenizer: 分词器
            **kwargs: 其他生成参数
        
        Returns:
            torch.Tensor: 生成的token序列
        """
        config = generation_config or self.generation_config
        
        # 将输入移动到目标设备
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]
        
        # 确定生成长度
        max_length = config.max_length
        max_new_tokens = config.max_new_tokens
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        
        # 初始化生成状态
        generated_tokens = input_ids.clone()
        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # 记录生成开始时间
        is_first_token = True
        decode_start_time = time.time()
        
        for step in range(input_ids.shape[1], max_length):
            # 准备当前步的输入
            if step == input_ids.shape[1]:
                # 第一步，使用完整输入
                current_input_ids = generated_tokens
            else:
                # 后续步骤，只使用最后一个token
                current_input_ids = generated_tokens[:, -1:]
            
            # 执行前向传播
            with torch.no_grad():
                outputs = self.forward_pass(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=config.use_cache,
                    exit_position=config.exit_position
                )
            
            # 获取下一个token的logits
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 应用温度
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # 采样下一个token
            if config.do_sample:
                # Top-k采样
                if config.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, config.top_k)
                    next_token_logits[next_token_logits < top_k_logits[:, -1, None]] = -float('inf')
                
                # Top-p采样
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx][indices_to_remove] = -float('inf')
                
                # 多项式采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪解码
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 只为未完成的序列更新token
            next_tokens = next_tokens * unfinished_sequences + (config.pad_token_id or 0) * (1 - unfinished_sequences)
            
            # 添加新token
            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新KV缓存
            if config.use_cache:
                past_key_values = outputs['past_key_values']
            
            if is_first_token:
                # 使用prompt开始时间计算TTFT，如果没有设置则使用decode开始时间（向后兼容）
                if hasattr(self, '_current_prompt_start_time') and hasattr(self, '_is_new_prompt') and self._is_new_prompt:
                    first_token_time = time.time() - self._current_prompt_start_time
                    self._is_new_prompt = False  # 重置标记
                else:
                    first_token_time = time.time() - decode_start_time
                
                self.stats['total_time_to_first_token'] += first_token_time
                is_first_token = False
                
                # 立即解码并打印第一个token，用于验证TTFT计算
                if tokenizer is not None:
                    first_token_text = tokenizer.decode(next_tokens[0], skip_special_tokens=True)
                    print(f"\n--- [分层模型] 第一个token '{first_token_text}' 生成完成，TTFT: {first_token_time*1000:.3f}ms")

            # 检查是否有序列完成
            if config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != config.eos_token_id).long())
            
            # 如果所有序列都完成，提前停止
            if unfinished_sequences.max() == 0:
                break
        
        # 更新统计信息
        decode_time = time.time() - decode_start_time   
        tokens_generated = generated_tokens.shape[1] - input_ids.shape[1]
        self.stats['token_decode_time'] += decode_time
        self.stats['total_tokens_generated'] += tokens_generated * batch_size
        
        return generated_tokens
    
    def generate_text(
        self,
        prompt: str,
        tokenizer: Any,
        generation_config: Optional[GenerationConfig] = None,
        return_full_text: bool = False,
        **kwargs
    ) -> str:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示
            tokenizer: 分词器
            generation_config: 生成配置
            return_full_text: 是否返回完整文本（包括输入）
            **kwargs: 其他参数
        
        Returns:
            str: 生成的文本
        """
        # 开始计时 - 从tokenization开始，与完整模型保持一致
        prompt_start_time = time.time()
        
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        input_length = input_ids.shape[1]

        generation_start_time = time.time()
        
        # 标记这是一个新的prompt，用于TTFT计算
        self._current_prompt_start_time = prompt_start_time
        self._is_new_prompt = True
        
        # 生成
        with torch.no_grad():
            generated_ids = self.generate(input_ids, generation_config, tokenizer, **kwargs)
        
        # 解码结果
        if return_full_text:
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            new_tokens = generated_ids[0][input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - generation_start_time
        self.stats['total_generation_time'] += generation_time

        return generated_text
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        
        avg_token_decode_time = (
            self.stats['token_decode_time'] / self.stats['total_tokens_generated']
            if self.stats['total_tokens_generated'] > 0 else 0
        )
        
        return {
            **self.stats,
            'avg_token_decode_time': avg_token_decode_time,
            'tokens_per_second': self.stats['total_tokens_generated'] / self.stats['total_generation_time'] if self.stats['total_generation_time'] > 0 else 0,
            'device': self.device,
            'num_submodels': len(self.submodels)
        }
    
    def get_layer_analysis(self) -> Dict[str, Any]:
        """获取分层分析信息"""
        layer_analysis = {}
        
        for layer_key, layer_stats in self.stats['layer_processing_time'].items():
            layer_analysis[layer_key] = {
                'avg_processing_time': layer_stats['processing_time'],
                'memory_usage': layer_stats.get('memory_delta', 0),
                'relative_time_percentage': 0  # 将在后面计算
            }
        
        # 计算相对时间百分比
        total_layer_time = sum(
            stats['processing_time'] 
            for stats in self.stats['layer_processing_time'].values()
        )
        
        if total_layer_time > 0:
            for layer_key in layer_analysis:
                processing_time = layer_analysis[layer_key]['avg_processing_time']
                layer_analysis[layer_key]['relative_time_percentage'] = (
                    processing_time / total_layer_time * 100
                )
        
        return layer_analysis
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'time_to_first_token': 0.0,
            'token_decode_time': 0.0,
            'total_generation_time': 0.0,
            'layer_processing_time': {},
            'memory_usage': {},
            'total_tokens_generated': 0,
            'inference_count': 0
        }
        # 重置early-exit日志标志
        if hasattr(self, '_early_exit_logged'):
            delattr(self, '_early_exit_logged')
    
    def clear_early_exit_cache(self):
        """清理early-exit submodel缓存，释放内存"""
        cached_count = len(self._early_exit_cache)
        
        # 删除缓存的submodel以释放内存
        for submodel_idx, cached_submodel in self._early_exit_cache.items():
            # 将缓存的submodel移动到CPU并删除以释放GPU内存
            if hasattr(cached_submodel, 'cpu'):
                cached_submodel.cpu()
            del cached_submodel
        
        self._early_exit_cache.clear()
        
        # 如果有CUDA，清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if cached_count > 0:
            print(f"Early-exit submodel缓存已清理（{cached_count}个缓存项）")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cached_submodels': list(self._early_exit_cache.keys()),
            'cache_size': len(self._early_exit_cache),
            'has_original_lm_head': self._original_lm_head_weights is not None
        }
    
    def print_layer_analysis(self):
        """打印分层分析结果"""
        print("\n" + "="*60)
        print("单设备分层推理分析")
        print("="*60)
        
        analysis = self.get_layer_analysis()
        stats = self.get_stats()
        
        print(f"设备: {self.device}")
        print(f"子模型数量: {len(self.submodels)}")
        print(f"总推理次数: {stats['inference_count']}")
        print(f"生成速度: {stats['tokens_per_second']:.2f} tokens/秒")
        
        print("\n分层性能分析:")
        print("-" * 60)
        for layer_key, layer_info in analysis.items():
            print(f"{layer_key}:")
            print(f"  处理时间: {layer_info['avg_processing_time']:.4f}秒 ({layer_info['relative_time_percentage']:.1f}%)")
            if layer_info['memory_usage'] > 0:
                print(f"  内存增量: {layer_info['memory_usage'] / (1024**2):.1f} MB") 