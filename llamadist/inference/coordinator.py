"""
分布式推理协调器

负责协调多个子模型的推理过程，实现高效的状态传递和KV缓存管理。
基于QLLM的隐藏状态传递机制和KV-Cache方法。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import queue
from dataclasses import dataclass

# 使用内置的分层器模块
from ..partitioner.splitter import LlamaSubModel


@dataclass
class InferenceState:
    """推理状态"""
    hidden_states: Optional[torch.Tensor] = None             # 隐藏状态
    past_key_values: Optional[List[torch.FloatTensor]] = None  # KV缓存
    attention_mask: Optional[torch.Tensor] = None             # 注意力掩码
    position_ids: Optional[torch.Tensor] = None               # 位置ID
    input_ids: Optional[torch.Tensor] = None                  # 输入token ID
    sequence_length: int = 0                                  # 序列长度
    batch_size: int = 1                                       # 批次大小


@dataclass
class GenerationConfig:
    """生成配置"""
    max_length: int = 512                     # 最大生成长度
    max_new_tokens: Optional[int] = None      # 最大新token数量
    temperature: float = 1.0                  # 温度
    top_p: float = 1.0                        # Top-p采样
    top_k: int = 50                          # Top-k采样
    do_sample: bool = True                   # 是否采样
    num_beams: int = 1                       # Beam search数量
    pad_token_id: Optional[int] = None       # Padding token ID
    eos_token_id: Optional[int] = None       # EOS token ID
    use_cache: bool = True                   # 是否使用KV缓存


class DistributedInference:
    """
    分布式推理引擎
    
    协调多个子模型的推理过程，实现：
    1. 高效的状态传递
    2. KV缓存管理和同步
    3. 异步推理优化
    4. 错误处理和恢复
    """
    
    def __init__(
        self,
        submodels: List[LlamaSubModel],
        generation_config: Optional[GenerationConfig] = None,
        enable_async: bool = False,
        max_workers: int = 4
    ):
        """
        初始化分布式推理引擎
        
        Args:
            submodels: 子模型列表（按分层顺序）
            generation_config: 生成配置
            enable_async: 是否启用异步推理
            max_workers: 异步工作线程数
        """
        self.submodels = submodels
        self.generation_config = generation_config or GenerationConfig()
        self.enable_async = enable_async
        self.max_workers = max_workers
        
        # 验证子模型
        self._validate_submodels()
        
        # 初始化异步组件
        if enable_async:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.device_queues = {
                sm.get_info()['device']: queue.Queue() 
                for sm in submodels
            }
        
        # 性能统计
        self.stats = {
            'total_inference_time': 0.0,
            'token_generation_time': 0.0,
            'state_transfer_time': 0.0,
            'cache_management_time': 0.0,
            'total_tokens_generated': 0,
            'inference_count': 0
        }
    
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
        
        print(f"分布式推理引擎初始化完成，{len(self.submodels)}个子模型")
        for sm in self.submodels:
            info = sm.get_info()
            print(f"  分层 {info['partition_idx']}: 层{info['layer_start']}-{info['layer_end']} @ {info['device']}")
    
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        执行分布式前向传播
        
        Args:
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            past_key_values: 过去的键值对
            use_cache: 是否使用缓存
        
        Returns:
            Dict: 包含logits、hidden_states和past_key_values的字典
        """
        start_time = time.time()
        
        # 初始化推理状态
        inference_state = InferenceState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            hidden_states=None,
            position_ids=None
        )
        
        # 通过每个子模型进行推理
        result = None
        for i, submodel in enumerate(self.submodels):
            transfer_start = time.time()
            
            if submodel.is_first_partition:
                # 第一个分层使用input_ids
                current_input_ids = input_ids
                current_attention_mask = attention_mask
                
                if past_key_values is not None:
                    # 在有缓存时，只使用最后一个token
                    current_input_ids = input_ids[:, -1:]
                    # attention_mask需要覆盖完整序列（包括过去的tokens）
                    if attention_mask is not None:
                        # 确保attention_mask维度正确
                        batch_size = input_ids.shape[0]
                        seq_length = current_input_ids.shape[1]  # 当前输入长度（通常为1）
                        
                        # 计算过去的序列长度
                        past_length = 0
                        if past_key_values is not None:
                            for pkv in past_key_values:
                                if pkv is not None and len(pkv) > 0 and pkv[0] is not None:
                                    past_length = pkv[0].shape[2]
                                    break
                        
                        # 创建完整的attention_mask
                        total_length = past_length + seq_length
                        current_attention_mask = torch.ones(
                            (batch_size, total_length),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                
                model_input = {
                    'input_ids': current_input_ids.to(submodel.get_info()['device']),
                    'attention_mask': current_attention_mask.to(submodel.get_info()['device']) if current_attention_mask is not None else None,
                    'position_ids': inference_state.position_ids.to(submodel.get_info()['device']) if inference_state.position_ids is not None else None,
                    'past_key_values': self._extract_relevant_kv_cache(past_key_values, submodel.layer_start, submodel.layer_end, submodel.get_info()['device']),
                    'use_cache': use_cache
                }
            else:
                # 后续分层使用hidden_states
                if inference_state.hidden_states is None:
                    raise RuntimeError(f"子模型 {i} 需要hidden_states，但前一个子模型没有提供")
                
                # 对于非第一分层，attention_mask处理更简单
                current_attention_mask = inference_state.attention_mask
                
                model_input = {
                    'hidden_states': inference_state.hidden_states.to(submodel.get_info()['device']),
                    'attention_mask': current_attention_mask.to(submodel.get_info()['device']) if current_attention_mask is not None else None,
                    'position_ids': inference_state.position_ids.to(submodel.get_info()['device']) if inference_state.position_ids is not None else None,
                    'past_key_values': self._extract_relevant_kv_cache(inference_state.past_key_values, submodel.layer_start, submodel.layer_end, submodel.get_info()['device']),
                    'use_cache': use_cache
                }
            
            self.stats['state_transfer_time'] += time.time() - transfer_start
            
            # 执行推理
            with torch.no_grad():
                result = submodel(**model_input)
            
            # 更新状态
            inference_state.hidden_states = result['hidden_states']
            if use_cache and result['past_key_values'] is not None:
                inference_state.past_key_values = self._merge_kv_cache(
                    inference_state.past_key_values, 
                    result['past_key_values'],
                    submodel.layer_start,
                    submodel.layer_end
                )
            
            # 确保attention_mask在推理状态中正确传递
            if submodel.is_first_partition:
                # 第一个分层处理后，更新推理状态中的attention_mask
                inference_state.attention_mask = current_attention_mask
        
        # 获取最终结果
        final_result = {
            'logits': result['logits'],
            'hidden_states': inference_state.hidden_states,
            'past_key_values': inference_state.past_key_values if use_cache else None
        }
        
        # 更新统计
        self.stats['total_inference_time'] += time.time() - start_time
        self.stats['inference_count'] += 1
        
        return final_result
    
    def _extract_relevant_kv_cache(
        self,
        past_key_values: Optional[List[torch.FloatTensor]],
        layer_start: int,
        layer_end: int,
        target_device: str
    ) -> Optional[List[torch.FloatTensor]]:
        """
        提取与当前子模型相关的KV缓存
        
        Args:
            past_key_values: 全局KV缓存
            layer_start: 子模型起始层
            layer_end: 子模型结束层
            target_device: 目标设备
        
        Returns:
            Optional[List[torch.FloatTensor]]: 相关的KV缓存
        """
        if past_key_values is None:
            return None
        
        # 提取当前子模型的缓存 - 修复索引计算
        relevant_cache = []
        num_layers_in_submodel = layer_end - layer_start + 1
        
        for local_idx in range(num_layers_in_submodel):
            global_layer_idx = layer_start + local_idx
            if global_layer_idx < len(past_key_values) and past_key_values[global_layer_idx] is not None:
                key_cache, value_cache = past_key_values[global_layer_idx]
                relevant_cache.append((
                    key_cache.to(target_device),
                    value_cache.to(target_device)
                ))
            else:
                relevant_cache.append(None)
        
        return relevant_cache
    
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        tokenizer: Optional[Any] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: 输入token ID
            generation_config: 生成配置
            tokenizer: 分词器
            **kwargs: 其他参数
        
        Returns:
            torch.Tensor: 生成的token ID
        """
        config = generation_config or self.generation_config
        
        # 合并kwargs到config中
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 确定生成长度
        max_length = config.max_length
        if config.max_new_tokens is not None:
            max_length = input_ids.shape[1] + config.max_new_tokens
        
        # 初始化
        generated_ids = input_ids.clone()
        past_key_values = None
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        
        start_time = time.time()
        
        # 生成循环
        for step in range(max_length - input_ids.shape[1]):
            # 准备当前输入
            if step == 0:
                current_input = generated_ids
            else:
                current_input = generated_ids[:, -1:]
            
            # 前向传播
            outputs = self.forward_pass(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=config.use_cache
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
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, config.top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p采样
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 多项式采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪心解码
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 更新生成的序列
            generated_ids = torch.cat([generated_ids, next_tokens[:, None]], dim=-1)
            
            # 更新KV缓存
            if config.use_cache:
                past_key_values = outputs['past_key_values']
            
            # 检查EOS token
            if config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(config.eos_token_id.shape[0], 1).ne(config.eos_token_id.unsqueeze(1)).prod(dim=0)
                )
                
                if unfinished_sequences.max() == 0:
                    break
        
        # 更新统计
        generation_time = time.time() - start_time
        tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
        
        self.stats['token_generation_time'] += generation_time
        self.stats['total_tokens_generated'] += tokens_generated * generated_ids.shape[0]
        
        return generated_ids
    
    def generate_text(
        self,
        prompt: str,
        tokenizer: Any,
        generation_config: Optional[GenerationConfig] = None,
        return_full_text: bool = False,
        **kwargs
    ) -> str:
        """
        生成文本（字符串接口）
        
        Args:
            prompt: 输入提示
            tokenizer: 分词器
            generation_config: 生成配置
            return_full_text: 是否返回完整文本
            **kwargs: 其他参数
        
        Returns:
            str: 生成的文本
        """
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # 获取第一个子模型的设备
        device = self.submodels[0].get_info()['device']
        input_ids = input_ids.to(device)
        
        # 生成
        generated_ids = self.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # 解码输出
        if return_full_text:
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        else:
            # 只返回新生成的部分
            new_tokens = generated_ids[0, input_ids.shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def _transfer_kv_cache(
        self, 
        past_key_values: Optional[List[torch.FloatTensor]], 
        target_device: str
    ) -> Optional[List[torch.FloatTensor]]:
        """
        转移KV缓存到目标设备
        
        Args:
            past_key_values: KV缓存
            target_device: 目标设备
        
        Returns:
            Optional[List[torch.FloatTensor]]: 转移后的KV缓存
        """
        if past_key_values is None:
            return None
        
        cache_start = time.time()
        
        transferred_cache = []
        for layer_cache in past_key_values:
            if layer_cache is not None:
                # 每层的缓存是(key, value)的元组
                key_cache, value_cache = layer_cache
                transferred_cache.append((
                    key_cache.to(target_device),
                    value_cache.to(target_device)
                ))
            else:
                transferred_cache.append(None)
        
        self.stats['cache_management_time'] += time.time() - cache_start
        return transferred_cache
    
    def _merge_kv_cache(
        self,
        global_cache: Optional[List[torch.FloatTensor]],
        local_cache: Optional[List[torch.FloatTensor]],
        layer_start: int,
        layer_end: int
    ) -> Optional[List[torch.FloatTensor]]:
        """
        合并局部KV缓存到全局缓存
        
        Args:
            global_cache: 全局KV缓存
            local_cache: 局部KV缓存
            layer_start: 开始层索引
            layer_end: 结束层索引
        
        Returns:
            Optional[List[torch.FloatTensor]]: 合并后的KV缓存
        """
        if local_cache is None:
            return global_cache
        
        cache_start = time.time()
        
        # 如果全局缓存为空，初始化
        if global_cache is None:
            # 假设总层数（这里需要从模型配置获取）
            total_layers = max(sm.layer_end for sm in self.submodels) + 1
            global_cache = [None] * total_layers
        
        # 更新对应层的缓存 - 修复索引映射
        num_layers_in_submodel = layer_end - layer_start + 1
        
        for local_idx, layer_cache in enumerate(local_cache):
            if local_idx < num_layers_in_submodel:
                global_layer_idx = layer_start + local_idx
                if global_layer_idx < len(global_cache):
                    global_cache[global_layer_idx] = layer_cache
        
        self.stats['cache_management_time'] += time.time() - cache_start
        return global_cache
    
    def async_generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
        callback: Optional[callable] = None
    ) -> asyncio.Future:
        """
        异步生成文本
        
        Args:
            input_ids: 输入token ID
            generation_config: 生成配置
            callback: 回调函数
        
        Returns:
            asyncio.Future: 异步结果
        """
        if not self.enable_async:
            raise RuntimeError("异步推理未启用")
        
        def generate_task():
            result = self.generate(input_ids, generation_config)
            if callback:
                callback(result)
            return result
        
        return self.executor.submit(generate_task)
    
    def batch_generate(
        self,
        batch_input_ids: List[torch.Tensor],
        generation_config: Optional[GenerationConfig] = None,
        max_batch_size: int = 4
    ) -> List[torch.Tensor]:
        """
        批量生成文本
        
        Args:
            batch_input_ids: 批量输入
            generation_config: 生成配置
            max_batch_size: 最大批次大小
        
        Returns:
            List[torch.Tensor]: 批量生成结果
        """
        results = []
        
        # 分批处理
        for i in range(0, len(batch_input_ids), max_batch_size):
            batch = batch_input_ids[i:i + max_batch_size]
            
            # 填充到相同长度
            max_len = max(ids.shape[1] for ids in batch)
            padded_batch = []
            
            for ids in batch:
                if ids.shape[1] < max_len:
                    pad_length = max_len - ids.shape[1]
                    pad_token_id = getattr(generation_config, 'pad_token_id', 0) if generation_config else 0
                    padded_ids = F.pad(ids, (0, pad_length), value=pad_token_id)
                else:
                    padded_ids = ids
                padded_batch.append(padded_ids)
            
            # 合并为批次
            batch_tensor = torch.cat(padded_batch, dim=0)
            
            # 生成
            batch_results = self.generate(batch_tensor, generation_config)
            
            # 分离结果
            for j, result in enumerate(batch_results):
                results.append(result.unsqueeze(0))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        
        # 计算平均值
        if stats['inference_count'] > 0:
            stats['avg_inference_time'] = stats['total_inference_time'] / stats['inference_count']
            stats['tokens_per_second'] = stats['total_tokens_generated'] / stats['token_generation_time'] if stats['token_generation_time'] > 0 else 0
        
        # 添加子模型信息
        stats['submodel_info'] = [sm.get_info() for sm in self.submodels]
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_inference_time': 0.0,
            'token_generation_time': 0.0,
            'state_transfer_time': 0.0,
            'cache_management_time': 0.0,
            'total_tokens_generated': 0,
            'inference_count': 0
        }
    
    def cleanup(self):
        """清理资源"""
        if self.enable_async and hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def __del__(self):
        """析构函数"""
        self.cleanup() 