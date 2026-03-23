# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DynamicKV: Task-Aware Adaptive KV Cache Compression

Based on paper: "DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs"
https://aclanthology.org/2025.findings-emnlp.426/
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DynamicKVConfig:
    """DynamicKV 压缩配置
    
    Attributes:
        enabled: 是否启用 DynamicKV 压缩
        budget_ratio: 全局 KV cache 预算比例 (相对于原始大小)
        layer_budget_strategy: 层预算分配策略
            - "adaptive": 自适应分配 (浅层保留更多)
            - "uniform": 均匀分配
            - "custom": 使用自定义配置
        importance_method: Token 重要性评分方法
            - "attention_weight": 基于注意力权重
            - "activation": 基于激活值范数
            - "combined": 综合评分
        update_interval: KV cache 更新间隔 (token 数)
        min_tokens_per_layer: 每层最小保留 token 数
        enable_task_aware: 是否启用任务感知
        custom_layer_budgets: 自定义每层预算比例 (当 strategy="custom" 时使用)
        compression_trigger: 触发压缩的条件
            - "always": 总是压缩
            - "on_budget": 仅在达到预算上限时压缩
            - "periodic": 周期性压缩
        preserve_recent_tokens: 保留最近的 token 数 (不压缩)
    """
    
    enabled: bool = False
    
    budget_ratio: float = 0.05
    
    layer_budget_strategy: Literal["adaptive", "uniform", "custom"] = "adaptive"
    
    importance_method: Literal["attention_weight", "activation", "combined"] = "combined"
    
    update_interval: int = 128
    
    min_tokens_per_layer: int = 16
    
    enable_task_aware: bool = True
    
    custom_layer_budgets: Optional[list[float]] = None
    
    compression_trigger: Literal["always", "on_budget", "periodic"] = "periodic"
    
    preserve_recent_tokens: int = 32
    
    def __post_init__(self):
        if self.budget_ratio <= 0 or self.budget_ratio > 1:
            raise ValueError(f"budget_ratio must be in (0, 1], got {self.budget_ratio}")
        
        if self.custom_layer_budgets is not None:
            if self.layer_budget_strategy != "custom":
                self.layer_budget_strategy = "custom"
            for ratio in self.custom_layer_budgets:
                if ratio <= 0 or ratio > 1:
                    raise ValueError(
                        f"custom_layer_budgets must be in (0, 1], got {ratio}"
                    )


@dataclass
class LayerKVBudget:
    """每层的 KV cache 预算配置
    
    Attributes:
        layer_idx: 层索引
        max_tokens: 该层最大保留 token 数
        current_tokens: 当前保留 token 数
        importance_threshold: 重要性阈值 (低于此值的 token 可能被驱逐)
        budget_ratio: 该层的预算比例
    """
    layer_idx: int
    max_tokens: int
    current_tokens: int = 0
    importance_threshold: float = 0.0
    budget_ratio: float = 1.0
    
    def get_retention_count(self, total_tokens: int) -> int:
        """获取应该保留的 token 数量"""
        return min(self.max_tokens, total_tokens)
    
    def is_over_budget(self) -> bool:
        """检查是否超出预算"""
        return self.current_tokens > self.max_tokens


@dataclass
class CompressionStats:
    """压缩统计信息
    
    Attributes:
        total_tokens_before: 压缩前总 token 数
        total_tokens_after: 压缩后总 token 数
        compression_ratio: 压缩比例
        tokens_evicted: 被驱逐的 token 数
        layers_processed: 处理的层数
        compression_time_ms: 压缩耗时 (毫秒)
    """
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    compression_ratio: float = 1.0
    tokens_evicted: int = 0
    layers_processed: int = 0
    compression_time_ms: float = 0.0
    
    def update(self, tokens_before: int, tokens_after: int):
        """更新统计信息"""
        self.total_tokens_before += tokens_before
        self.total_tokens_after += tokens_after
        self.tokens_evicted += tokens_before - tokens_after
        self.layers_processed += 1
        if self.total_tokens_before > 0:
            self.compression_ratio = (
                self.total_tokens_after / self.total_tokens_before
            )
