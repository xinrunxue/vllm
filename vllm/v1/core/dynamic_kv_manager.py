# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DynamicKV Manager: Core component for task-aware adaptive KV cache compression

Based on paper: "DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs"
https://aclanthology.org/2025.findings-emnlp.426/
"""

from dataclasses import dataclass
from typing import Optional
import time

import torch

from vllm.logger import init_logger
from vllm.v1.core.dynamic_kv_config import (
    DynamicKVConfig,
    LayerKVBudget,
    CompressionStats,
)
from vllm.v1.core.task_pattern_analyzer import TaskPatternAnalyzer

logger = init_logger(__name__)

# Try to import CUDA kernels
try:
    from vllm.v1.core.dynamic_kv_cuda import DynamicKVCUDAKernels
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False
    logger.warning("DynamicKV CUDA kernels not available, using PyTorch fallback")


@dataclass
class CompressionResult:
    """KV cache 压缩结果
    
    Attributes:
        retain_indices: 保留的 token 索引
        num_tokens_before: 压缩前 token 数
        num_tokens_after: 压缩后 token 数
        compression_ratio: 压缩比例
        importance_scores: Token 重要性分数
    """
    retain_indices: torch.Tensor
    num_tokens_before: int
    num_tokens_after: int
    compression_ratio: float
    importance_scores: Optional[torch.Tensor] = None


class DynamicKVManager:
    """DynamicKV 核心管理器
    
    职责：
    1. 管理每层的 KV cache 预算
    2. 计算 token 重要性分数
    3. 执行 KV cache 压缩/驱逐
    4. 任务模式分析与适配
    
    核心算法：
    - 基于注意力权重或激活值计算 token 重要性
    - 自适应层级预算分配
    - 定期更新前面层的 cache 大小
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        config: Optional[DynamicKVConfig] = None,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.config = config or DynamicKVConfig()
        
        self.layer_budgets: list[LayerKVBudget] = self._init_layer_budgets()
        
        self._importance_cache: dict[int, torch.Tensor] = {}
        
        if self.config.enable_task_aware:
            self.task_analyzer = TaskPatternAnalyzer(num_layers=num_layers)
        else:
            self.task_analyzer = None
        
        self.stats = CompressionStats()
        
        self._token_count_since_update = 0
        self._current_task: Optional[str] = None
    
    def _init_layer_budgets(self) -> list[LayerKVBudget]:
        """初始化每层的 KV cache 预算"""
        budgets = []
        
        for layer_idx in range(self.num_layers):
            if self.config.layer_budget_strategy == "adaptive":
                ratio = self._compute_adaptive_ratio(layer_idx)
            elif self.config.layer_budget_strategy == "custom":
                if self.config.custom_layer_budgets:
                    ratio = self.config.custom_layer_budgets[
                        min(layer_idx, len(self.config.custom_layer_budgets) - 1)
                    ]
                else:
                    ratio = self.config.budget_ratio
            else:
                ratio = self.config.budget_ratio
            
            max_tokens = max(
                int(self.max_seq_len * ratio),
                self.config.min_tokens_per_layer
            )
            
            budgets.append(LayerKVBudget(
                layer_idx=layer_idx,
                max_tokens=max_tokens,
                current_tokens=0,
                importance_threshold=0.0,
                budget_ratio=ratio,
            ))
        
        return budgets
    
    def _compute_adaptive_ratio(self, layer_idx: int) -> float:
        """计算自适应预算比例
        
        论文观察：
        - 浅层：更多关注局部信息，保留更多 token
        - 深层：更多关注全局信息，可以保留较少 token
        
        使用指数衰减模型
        """
        decay_factor = 0.97
        base_ratio = self.config.budget_ratio * 1.5
        
        if self.task_analyzer and self._current_task:
            adjustments = self.task_analyzer.get_budget_adjustments(self._current_task)
            if layer_idx < len(adjustments):
                task_factor = adjustments[layer_idx]
            else:
                task_factor = 1.0
        else:
            task_factor = 1.0
        
        adaptive_ratio = base_ratio * (decay_factor ** layer_idx) * task_factor
        
        return min(adaptive_ratio, 1.0)
    
    def set_task_type(self, task_type: str):
        """设置当前任务类型"""
        self._current_task = task_type
        if self.task_analyzer:
            self.task_analyzer.set_current_task(task_type)
            self._update_budgets_for_task(task_type)
    
    def _update_budgets_for_task(self, task_type: str):
        """根据任务类型更新预算分配"""
        if self.task_analyzer is None:
            return
        
        adjustments = self.task_analyzer.get_budget_adjustments(task_type)
        
        for layer_idx, adjustment in enumerate(adjustments):
            if layer_idx < len(self.layer_budgets):
                budget = self.layer_budgets[layer_idx]
                base_max = int(self.max_seq_len * self.config.budget_ratio)
                budget.max_tokens = max(
                    int(base_max * adjustment),
                    self.config.min_tokens_per_layer
                )
    
    def compute_token_importance(
        self,
        layer_idx: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算 token 重要性分数
        
        Args:
            layer_idx: 层索引
            key_cache: [num_tokens, num_heads, head_dim] 或 [num_tokens, num_kv_heads, head_dim]
            value_cache: [num_tokens, num_heads, head_dim] 或 [num_tokens, num_kv_heads, head_dim]
            attention_weights: [num_heads, num_tokens] 注意力权重 (可选)
            
        Returns:
            importance_scores: [num_tokens] 每个 token 的重要性分数
        """
        method = self.config.importance_method
        
        # Use CUDA kernel if available and on GPU
        if (CUDA_KERNELS_AVAILABLE and 
            key_cache.is_cuda and 
            method in ["activation", "combined"]):
            try:
                num_heads = key_cache.shape[1]
                head_dim = key_cache.shape[2]
                importance = DynamicKVCUDAKernels.compute_importance(
                    key_cache, value_cache, num_heads, head_dim
                )
                if attention_weights is not None and method == "combined":
                    attn_importance = attention_weights.mean(dim=0)
                    importance = (importance + attn_importance) / 2
                return importance
            except Exception as e:
                logger.debug(f"CUDA kernel failed, falling back to PyTorch: {e}")
        
        # PyTorch fallback
        if method == "attention_weight" and attention_weights is not None:
            importance = attention_weights.mean(dim=0)
        elif method == "activation":
            key_norm = key_cache.norm(dim=-1).mean(dim=-1)
            value_norm = value_cache.norm(dim=-1).mean(dim=-1)
            importance = (key_norm + value_norm) / 2
        else:
            key_norm = key_cache.norm(dim=-1).mean(dim=-1)
            value_norm = value_cache.norm(dim=-1).mean(dim=-1)
            
            if attention_weights is not None:
                attn_importance = attention_weights.mean(dim=0)
                importance = (key_norm + value_norm + attn_importance) / 3
            else:
                importance = (key_norm + value_norm) / 2
        
        return importance
    
    def select_tokens_to_retain(
        self,
        layer_idx: int,
        importance_scores: torch.Tensor,
        num_tokens: int,
        preserve_recent: int = 0,
    ) -> torch.Tensor:
        """选择需要保留的 token 索引
        
        Args:
            layer_idx: 层索引
            importance_scores: [num_tokens] 重要性分数
            num_tokens: 当前 token 数量
            preserve_recent: 保留最近的 token 数 (不参与压缩)
            
        Returns:
            retain_indices: 需要保留的 token 索引 (已排序)
        """
        budget = self.layer_budgets[layer_idx]
        num_to_retain = min(budget.max_tokens, num_tokens)
        
        if preserve_recent > 0:
            num_to_retain = max(num_to_retain, preserve_recent)
        
        if num_tokens <= num_to_retain:
            return torch.arange(num_tokens, device=importance_scores.device)
        
        recent_indices = torch.arange(
            max(0, num_tokens - preserve_recent),
            num_tokens,
            device=importance_scores.device
        )
        
        num_from_importance = num_to_retain - len(recent_indices)
        
        if num_from_importance > 0:
            candidate_scores = importance_scores[:num_tokens - preserve_recent].clone()
            
            _, top_indices = torch.topk(candidate_scores, num_from_importance)
            
            retain_indices = torch.cat([top_indices, recent_indices])
        else:
            retain_indices = recent_indices
        
        retain_indices, _ = torch.sort(retain_indices)
        
        return retain_indices
    
    def compress_kv_cache(
        self,
        layer_idx: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> CompressionResult:
        """压缩 KV cache
        
        Args:
            layer_idx: 层索引
            key_cache: [num_tokens, ...] Key cache
            value_cache: [num_tokens, ...] Value cache
            attention_weights: [num_heads, num_tokens] 注意力权重
            
        Returns:
            CompressionResult: 压缩结果
        """
        start_time = time.perf_counter()
        
        num_tokens = key_cache.shape[0]
        
        importance = self.compute_token_importance(
            layer_idx, key_cache, value_cache, attention_weights
        )
        
        retain_indices = self.select_tokens_to_retain(
            layer_idx,
            importance,
            num_tokens,
            preserve_recent=self.config.preserve_recent_tokens,
        )
        
        num_after = len(retain_indices)
        compression_ratio = num_after / num_tokens if num_tokens > 0 else 1.0
        
        self.layer_budgets[layer_idx].current_tokens = num_after
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.update(num_tokens, num_after)
        self.stats.compression_time_ms += elapsed_ms
        
        return CompressionResult(
            retain_indices=retain_indices,
            num_tokens_before=num_tokens,
            num_tokens_after=num_after,
            compression_ratio=compression_ratio,
            importance_scores=importance,
        )
    
    def should_compress(self, num_new_tokens: int = 1) -> bool:
        """判断是否应该执行压缩
        
        Args:
            num_new_tokens: 新增的 token 数
            
        Returns:
            是否应该压缩
        """
        if not self.config.enabled:
            return False
        
        trigger = self.config.compression_trigger
        
        if trigger == "always":
            return True
        elif trigger == "periodic":
            self._token_count_since_update += num_new_tokens
            if self._token_count_since_update >= self.config.update_interval:
                self._token_count_since_update = 0
                return True
            return False
        elif trigger == "on_budget":
            for budget in self.layer_budgets:
                if budget.is_over_budget():
                    return True
            return False
        
        return False
    
    def get_layer_budget(self, layer_idx: int) -> LayerKVBudget:
        """获取指定层的预算配置"""
        if 0 <= layer_idx < len(self.layer_budgets):
            return self.layer_budgets[layer_idx]
        raise IndexError(f"Layer index {layer_idx} out of range")
    
    def get_total_budget(self) -> int:
        """获取所有层的总预算"""
        return sum(b.max_tokens for b in self.layer_budgets)
    
    def get_average_compression_ratio(self) -> float:
        """获取平均压缩比例"""
        return self.stats.compression_ratio
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_tokens_before": self.stats.total_tokens_before,
            "total_tokens_after": self.stats.total_tokens_after,
            "compression_ratio": self.stats.compression_ratio,
            "tokens_evicted": self.stats.tokens_evicted,
            "layers_processed": self.stats.layers_processed,
            "compression_time_ms": self.stats.compression_time_ms,
            "current_task": self._current_task,
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = CompressionStats()
        self._token_count_since_update = 0
    
    def update_budgets_from_pattern(self, task_type: str):
        """根据任务模式更新预算"""
        self.set_task_type(task_type)
