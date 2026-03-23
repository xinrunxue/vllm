# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compressed KV Cache for DynamicKV

Provides data structures for storing compressed KV cache with importance metadata.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class CompressedKVBlock:
    """压缩后的 KV cache 块
    
    Attributes:
        block_id: 原始 block ID
        key_cache: 压缩后的 key cache [num_retained_tokens, num_heads, head_dim]
        value_cache: 压缩后的 value cache [num_retained_tokens, num_heads, head_dim]
        retain_indices: 保留的 token 在原始序列中的索引
        importance_scores: 每个 token 的重要性分数
        original_num_tokens: 压缩前的 token 数
        compression_ratio: 压缩比例
    """
    block_id: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    retain_indices: torch.Tensor
    importance_scores: Optional[torch.Tensor] = None
    original_num_tokens: int = 0
    compression_ratio: float = 1.0
    
    @property
    def num_retained_tokens(self) -> int:
        return self.key_cache.shape[0]
    
    def to(self, device: torch.device) -> "CompressedKVBlock":
        return CompressedKVBlock(
            block_id=self.block_id,
            key_cache=self.key_cache.to(device),
            value_cache=self.value_cache.to(device),
            retain_indices=self.retain_indices.to(device),
            importance_scores=self.importance_scores.to(device) if self.importance_scores is not None else None,
            original_num_tokens=self.original_num_tokens,
            compression_ratio=self.compression_ratio,
        )


@dataclass
class LayerCompressedKVCache:
    """单层的压缩 KV cache
    
    管理该层所有请求的压缩 KV cache
    """
    layer_idx: int
    blocks: dict[int, CompressedKVBlock] = field(default_factory=dict)
    
    def add_block(self, block: CompressedKVBlock):
        self.blocks[block.block_id] = block
    
    def get_block(self, block_id: int) -> Optional[CompressedKVBlock]:
        return self.blocks.get(block_id)
    
    def remove_block(self, block_id: int):
        if block_id in self.blocks:
            del self.blocks[block_id]
    
    def get_retain_indices(self, block_ids: list[int]) -> torch.Tensor:
        """获取多个 block 的保留索引映射"""
        all_indices = []
        offset = 0
        for block_id in block_ids:
            block = self.get_block(block_id)
            if block is not None:
                indices = block.retain_indices + offset
                all_indices.append(indices)
            offset += block.original_num_tokens if block else 0
        
        if all_indices:
            return torch.cat(all_indices)
        return torch.tensor([], dtype=torch.long)
    
    def get_compression_stats(self) -> dict:
        """获取压缩统计信息"""
        if not self.blocks:
            return {
                "layer_idx": self.layer_idx,
                "num_blocks": 0,
                "total_original_tokens": 0,
                "total_retained_tokens": 0,
                "avg_compression_ratio": 1.0,
            }
        
        total_original = sum(b.original_num_tokens for b in self.blocks.values())
        total_retained = sum(b.num_retained_tokens for b in self.blocks.values())
        
        return {
            "layer_idx": self.layer_idx,
            "num_blocks": len(self.blocks),
            "total_original_tokens": total_original,
            "total_retained_tokens": total_retained,
            "avg_compression_ratio": total_retained / total_original if total_original > 0 else 1.0,
        }


class CompressedKVCacheManager:
    """压缩 KV cache 管理器
    
    管理所有层的压缩 KV cache
    """
    
    def __init__(self, num_layers: int, device: torch.device = torch.device("cuda")):
        self.num_layers = num_layers
        self.device = device
        self.layer_caches: dict[int, LayerCompressedKVCache] = {
            i: LayerCompressedKVCache(layer_idx=i) for i in range(num_layers)
        }
        
        self._request_layer_blocks: dict[str, dict[int, list[int]]] = {}
    
    def store_compressed_cache(
        self,
        request_id: str,
        layer_idx: int,
        block_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        retain_indices: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None,
        original_num_tokens: int = 0,
    ):
        """存储压缩后的 KV cache
        
        Args:
            request_id: 请求 ID
            layer_idx: 层索引
            block_id: Block ID
            key_cache: 压缩后的 key cache
            value_cache: 压缩后的 value cache
            retain_indices: 保留的 token 索引
            importance_scores: 重要性分数
            original_num_tokens: 原始 token 数
        """
        compression_ratio = (
            key_cache.shape[0] / original_num_tokens 
            if original_num_tokens > 0 else 1.0
        )
        
        block = CompressedKVBlock(
            block_id=block_id,
            key_cache=key_cache.to(self.device),
            value_cache=value_cache.to(self.device),
            retain_indices=retain_indices.to(self.device),
            importance_scores=importance_scores.to(self.device) if importance_scores is not None else None,
            original_num_tokens=original_num_tokens,
            compression_ratio=compression_ratio,
        )
        
        self.layer_caches[layer_idx].add_block(block)
        
        if request_id not in self._request_layer_blocks:
            self._request_layer_blocks[request_id] = {}
        if layer_idx not in self._request_layer_blocks[request_id]:
            self._request_layer_blocks[request_id][layer_idx] = []
        self._request_layer_blocks[request_id][layer_idx].append(block_id)
    
    def get_compressed_cache(
        self,
        layer_idx: int,
        block_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取压缩后的 KV cache
        
        Args:
            layer_idx: 层索引
            block_ids: Block ID 列表
            
        Returns:
            (key_cache, value_cache, retain_indices)
            - key_cache: 拼接后的 key cache
            - value_cache: 拼接后的 value cache
            - retain_indices: 保留的 token 索引映射
        """
        layer_cache = self.layer_caches.get(layer_idx)
        if layer_cache is None:
            return None, None, None
        
        keys = []
        values = []
        indices = []
        
        for block_id in block_ids:
            block = layer_cache.get_block(block_id)
            if block is not None:
                keys.append(block.key_cache)
                values.append(block.value_cache)
                indices.append(block.retain_indices)
        
        if not keys:
            return None, None, None
        
        return (
            torch.cat(keys, dim=0),
            torch.cat(values, dim=0),
            torch.cat(indices, dim=0),
        )
    
    def get_block_retain_indices(
        self,
        layer_idx: int,
        block_id: int,
    ) -> Optional[torch.Tensor]:
        """获取单个 block 的保留索引"""
        layer_cache = self.layer_caches.get(layer_idx)
        if layer_cache is None:
            return None
        
        block = layer_cache.get_block(block_id)
        if block is None:
            return None
        
        return block.retain_indices
    
    def remove_request_blocks(self, request_id: str):
        """移除请求的所有 block"""
        if request_id not in self._request_layer_blocks:
            return
        
        for layer_idx, block_ids in self._request_layer_blocks[request_id].items():
            layer_cache = self.layer_caches.get(layer_idx)
            if layer_cache:
                for block_id in block_ids:
                    layer_cache.remove_block(block_id)
        
        del self._request_layer_blocks[request_id]
    
    def get_compression_stats(self) -> dict:
        """获取所有层的压缩统计"""
        stats = {
            "num_layers": self.num_layers,
            "layers": [],
            "total_original_tokens": 0,
            "total_retained_tokens": 0,
        }
        
        for layer_idx in range(self.num_layers):
            layer_stats = self.layer_caches[layer_idx].get_compression_stats()
            stats["layers"].append(layer_stats)
            stats["total_original_tokens"] += layer_stats["total_original_tokens"]
            stats["total_retained_tokens"] += layer_stats["total_retained_tokens"]
        
        if stats["total_original_tokens"] > 0:
            stats["overall_compression_ratio"] = (
                stats["total_retained_tokens"] / stats["total_original_tokens"]
            )
        else:
            stats["overall_compression_ratio"] = 1.0
        
        return stats
    
    def clear(self):
        """清空所有缓存"""
        for layer_idx in range(self.num_layers):
            self.layer_caches[layer_idx].blocks.clear()
        self._request_layer_blocks.clear()
