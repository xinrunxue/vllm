# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration module for DynamicKV with KVCacheManager

Provides the bridge between DynamicKV compression and vLLM's KV cache management.
"""

from typing import Optional
import torch

from vllm.logger import init_logger
from vllm.v1.core.dynamic_kv_config import DynamicKVConfig
from vllm.v1.core.dynamic_kv_manager import DynamicKVManager, CompressionResult
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request

logger = init_logger(__name__)


class DynamicKVIntegration:
    """DynamicKV 与 KVCacheManager 的集成层
    
    职责：
    1. 在 KV cache 分配时应用 DynamicKV 压缩
    2. 管理 token 到 block 的映射
    3. 与调度器协作处理压缩请求
    
    使用方式：
    - 在 KVCacheManager 初始化时创建
    - 在 allocate_slots 和 save_kv_cache 时调用
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        block_size: int,
        config: Optional[DynamicKVConfig] = None,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        
        if config and config.enabled:
            self.enabled = True
            self.manager = DynamicKVManager(
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                config=config,
            )
        else:
            self.enabled = False
            self.manager = None
        
        self._compression_cache: dict[str, dict[int, CompressionResult]] = {}
    
    def should_compress(self, request: Request, num_new_tokens: int = 1) -> bool:
        """判断是否应该对请求执行压缩
        
        Args:
            request: 当前请求
            num_new_tokens: 新增 token 数
            
        Returns:
            是否应该压缩
        """
        if not self.enabled or self.manager is None:
            return False
        
        return self.manager.should_compress(num_new_tokens)
    
    def detect_and_set_task(self, request: Request) -> str:
        """检测并设置请求的任务类型
        
        Args:
            request: 当前请求
            
        Returns:
            检测到的任务类型
        """
        if not self.enabled or self.manager is None:
            return "general"
        
        task_type = "general"
        
        if self.manager.task_analyzer:
            task_type = self.manager.task_analyzer.detect_task_type(
                prompt_text=getattr(request, 'prompt', None),
                metadata=getattr(request, 'metadata', None),
            )
            self.manager.set_task_type(task_type)
        
        return task_type
    
    def compute_compression_plan(
        self,
        request: Request,
        layer_idx: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Optional[CompressionResult]:
        """计算压缩计划
        
        Args:
            request: 当前请求
            layer_idx: 层索引
            key_cache: Key cache 张量
            value_cache: Value cache 张量
            attention_weights: 注意力权重 (可选)
            
        Returns:
            压缩结果，如果不需要压缩则返回 None
        """
        if not self.enabled or self.manager is None:
            return None
        
        result = self.manager.compress_kv_cache(
            layer_idx=layer_idx,
            key_cache=key_cache,
            value_cache=value_cache,
            attention_weights=attention_weights,
        )
        
        if request.request_id not in self._compression_cache:
            self._compression_cache[request.request_id] = {}
        self._compression_cache[request.request_id][layer_idx] = result
        
        return result
    
    def get_retain_indices(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """获取请求指定层的保留 token 索引
        
        Args:
            request_id: 请求 ID
            layer_idx: 层索引
            
        Returns:
            保留的 token 索引，如果不存在则返回 None
        """
        if request_id in self._compression_cache:
            layer_cache = self._compression_cache[request_id]
            if layer_idx in layer_cache:
                return layer_cache[layer_idx].retain_indices
        return None
    
    def map_indices_to_blocks(
        self,
        retain_indices: torch.Tensor,
        block_ids: list[int],
        block_size: int,
    ) -> tuple[list[int], list[int]]:
        """将保留的 token 索引映射到 block
        
        Args:
            retain_indices: 保留的 token 索引
            block_ids: 当前 block ID 列表
            block_size: 每个 block 的 token 数
            
        Returns:
            (new_block_ids, block_token_counts)
            - new_block_ids: 新的 block ID 列表
            - block_token_counts: 每个 block 中的 token 数
        """
        retain_indices_cpu = retain_indices.cpu().numpy()
        
        block_token_map: dict[int, list[int]] = {}
        
        for token_idx in retain_indices_cpu:
            block_idx = token_idx // block_size
            if block_idx < len(block_ids):
                block_id = block_ids[block_idx]
                if block_id not in block_token_map:
                    block_token_map[block_id] = []
                block_token_map[block_id].append(token_idx % block_size)
        
        new_block_ids = list(block_token_map.keys())
        block_token_counts = [len(tokens) for tokens in block_token_map.values()]
        
        return new_block_ids, block_token_counts
    
    def clear_request_cache(self, request_id: str):
        """清除请求的压缩缓存"""
        if request_id in self._compression_cache:
            del self._compression_cache[request_id]
    
    def get_statistics(self) -> dict:
        """获取压缩统计信息"""
        if self.enabled and self.manager:
            return self.manager.get_statistics()
        return {}
    
    def get_layer_budget(self, layer_idx: int) -> int:
        """获取指定层的预算"""
        if self.enabled and self.manager:
            budget = self.manager.get_layer_budget(layer_idx)
            return budget.max_tokens
        return self.max_seq_len
    
    def get_total_budget(self) -> int:
        """获取总预算"""
        if self.enabled and self.manager:
            return self.manager.get_total_budget()
        return self.max_seq_len * self.num_layers


def create_dynamic_kv_integration(
    num_layers: int,
    max_seq_len: int,
    block_size: int,
    config: Optional[DynamicKVConfig] = None,
) -> DynamicKVIntegration:
    """创建 DynamicKV 集成实例
    
    Args:
        num_layers: 模型层数
        max_seq_len: 最大序列长度
        block_size: KV cache block 大小
        config: DynamicKV 配置
        
    Returns:
        DynamicKVIntegration 实例
    """
    return DynamicKVIntegration(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        block_size=block_size,
        config=config,
    )
