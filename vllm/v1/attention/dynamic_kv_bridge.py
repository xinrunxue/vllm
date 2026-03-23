# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DynamicKV Attention Integration

Integrates DynamicKV compression into the attention computation pipeline.
"""

from typing import Optional
import torch

from vllm.logger import init_logger
from vllm.v1.core.dynamic_kv_config import DynamicKVConfig
from vllm.v1.core.dynamic_kv_manager import DynamicKVManager
from vllm.v1.core.compressed_kv_cache import CompressedKVCacheManager

logger = init_logger(__name__)


class DynamicKVAttentionBridge:
    """DynamicKV 与 Attention 层的桥接器
    
    职责：
    1. 在 KV cache 写入前执行压缩
    2. 在注意力计算时提供压缩后的 cache
    3. 管理压缩元数据
    
    使用方式：
    - 在 FlashAttentionImpl 中创建实例
    - 在 reshape_and_cache_flash 前调用 compress_kv_cache
    - 在 flash_attn_varlen_func 前调用 get_compressed_cache
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_seq_len: int,
        config: Optional[DynamicKVConfig] = None,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        
        self.enabled = config is not None and config.enabled
        
        if self.enabled:
            self.manager = DynamicKVManager(
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                config=config,
            )
            self.cache_manager = CompressedKVCacheManager(
                num_layers=num_layers,
            )
        else:
            self.manager = None
            self.cache_manager = None
        
        self._current_layer = 0
        self._compression_enabled_for_request: dict[str, bool] = {}
    
    def set_layer(self, layer_idx: int):
        """设置当前处理的层"""
        self._current_layer = layer_idx
    
    def should_compress_for_request(self, request_id: str, num_tokens: int) -> bool:
        """判断是否应该为请求执行压缩
        
        Args:
            request_id: 请求 ID
            num_tokens: 当前 token 数
            
        Returns:
            是否应该压缩
        """
        if not self.enabled or self.manager is None:
            return False
        
        if request_id not in self._compression_enabled_for_request:
            should_compress = self.manager.should_compress(num_tokens)
            self._compression_enabled_for_request[request_id] = should_compress
        
        return self._compression_enabled_for_request[request_id]
    
    def compress_and_cache(
        self,
        request_id: str,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """压缩 KV cache 并存储
        
        Args:
            request_id: 请求 ID
            layer_idx: 层索引
            key: [num_tokens, num_kv_heads, head_dim]
            value: [num_tokens, num_kv_heads, head_dim]
            key_cache: 原始 key cache
            value_cache: 原始 value cache
            slot_mapping: slot 映射
            attention_weights: 注意力权重 (可选)
            
        Returns:
            (compressed_key_cache, compressed_value_cache, retain_indices)
        """
        if not self.enabled or self.manager is None:
            return key_cache, value_cache, None
        
        num_tokens = key.shape[0]
        
        if not self.should_compress_for_request(request_id, num_tokens):
            return key_cache, value_cache, None
        
        result = self.manager.compress_kv_cache(
            layer_idx=layer_idx,
            key_cache=key,
            value_cache=value,
            attention_weights=attention_weights,
        )
        
        retain_indices = result.retain_indices
        
        compressed_key = key[retain_indices]
        compressed_value = value[retain_indices]
        
        block_ids = self._get_block_ids_from_slot_mapping(slot_mapping)
        
        for i, block_id in enumerate(block_ids):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, len(retain_indices))
            
            if start_idx >= len(retain_indices):
                break
            
            block_retain = retain_indices[start_idx:end_idx] - start_idx
            block_retain = block_retain[block_retain >= 0]
            
            if len(block_retain) > 0:
                block_key = compressed_key[start_idx:end_idx]
                block_value = compressed_value[start_idx:end_idx]
                
                self.cache_manager.store_compressed_cache(
                    request_id=request_id,
                    layer_idx=layer_idx,
                    block_id=block_id,
                    key_cache=block_key,
                    value_cache=block_value,
                    retain_indices=retain_indices[start_idx:end_idx],
                    importance_scores=result.importance_scores[start_idx:end_idx] if result.importance_scores is not None else None,
                    original_num_tokens=min(self.block_size, num_tokens - start_idx),
                )
        
        return compressed_key, compressed_value, retain_indices
    
    def get_compressed_cache_for_attention(
        self,
        layer_idx: int,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """获取压缩后的 KV cache 用于注意力计算
        
        Args:
            layer_idx: 层索引
            block_table: [num_seqs, max_num_blocks] block 表
            seq_lens: [num_seqs] 序列长度
            
        Returns:
            (key_cache, value_cache, token_mapping)
            - key_cache: 压缩后的 key cache
            - value_cache: 压缩后的 value cache
            - token_mapping: 原始 token 到压缩 token 的映射
        """
        if not self.enabled or self.cache_manager is None:
            return None, None, None
        
        all_keys = []
        all_values = []
        all_mappings = []
        
        num_seqs = block_table.shape[0]
        
        for seq_idx in range(num_seqs):
            seq_block_ids = block_table[seq_idx].tolist()
            seq_len = seq_lens[seq_idx].item()
            
            compressed_key, compressed_value, retain_indices = \
                self.cache_manager.get_compressed_cache(layer_idx, seq_block_ids)
            
            if compressed_key is not None:
                all_keys.append(compressed_key)
                all_values.append(compressed_value)
                
                mapping = torch.zeros(seq_len, dtype=torch.long, device=compressed_key.device)
                for new_idx, orig_idx in enumerate(retain_indices.tolist()):
                    if orig_idx < seq_len:
                        mapping[orig_idx] = new_idx
                all_mappings.append(mapping)
        
        if not all_keys:
            return None, None, None
        
        return (
            torch.cat(all_keys, dim=0),
            torch.cat(all_values, dim=0),
            torch.cat(all_mappings, dim=0) if all_mappings else None,
        )
    
    def _get_block_ids_from_slot_mapping(self, slot_mapping: torch.Tensor) -> list[int]:
        """从 slot_mapping 提取 block ID"""
        block_ids = set()
        for slot in slot_mapping.tolist():
            if slot >= 0:
                block_id = slot // self.block_size
                block_ids.add(block_id)
        return sorted(block_ids)
    
    def clear_request_cache(self, request_id: str):
        """清除请求的缓存"""
        if self.cache_manager:
            self.cache_manager.remove_request_blocks(request_id)
        if request_id in self._compression_enabled_for_request:
            del self._compression_enabled_for_request[request_id]
    
    def get_compression_stats(self) -> dict:
        """获取压缩统计"""
        if self.manager:
            stats = self.manager.get_statistics()
            if self.cache_manager:
                stats.update(self.cache_manager.get_compression_stats())
            return stats
        return {}
    
    def set_task_type(self, task_type: str):
        """设置任务类型"""
        if self.manager:
            self.manager.set_task_type(task_type)


def create_dynamic_kv_bridge(
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_seq_len: int,
    config: Optional[DynamicKVConfig] = None,
) -> DynamicKVAttentionBridge:
    """创建 DynamicKV 桥接器实例"""
    return DynamicKVAttentionBridge(
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_seq_len=max_seq_len,
        config=config,
    )
