# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DynamicKV: Task-Aware Adaptive KV Cache Compression

This module provides DynamicKV implementation for vLLM, enabling
adaptive KV cache compression based on task patterns and token importance.

Based on paper: "DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs"
https://aclanthology.org/2025.findings-emnlp.426/

Usage:
    from vllm.v1.core.dynamic_kv import DynamicKVConfig, DynamicKVManager
    
    # Create configuration
    config = DynamicKVConfig(
        enabled=True,
        budget_ratio=0.05,  # Retain 5% of KV cache
        layer_budget_strategy="adaptive",
        enable_task_aware=True,
    )
    
    # Create manager
    manager = DynamicKVManager(
        num_layers=32,
        max_seq_len=8192,
        config=config,
    )
    
    # Set task type for adaptive compression
    manager.set_task_type("rag")
    
    # Compress KV cache
    result = manager.compress_kv_cache(
        layer_idx=0,
        key_cache=key_tensor,
        value_cache=value_tensor,
    )
"""

from vllm.v1.core.dynamic_kv_config import (
    DynamicKVConfig,
    LayerKVBudget,
    CompressionStats,
)
from vllm.v1.core.task_pattern_analyzer import (
    TaskPattern,
    TaskPatternAnalyzer,
)
from vllm.v1.core.dynamic_kv_manager import (
    DynamicKVManager,
    CompressionResult,
    CUDA_KERNELS_AVAILABLE,
)
from vllm.v1.core.compressed_kv_cache import (
    CompressedKVBlock,
    LayerCompressedKVCache,
    CompressedKVCacheManager,
)
from vllm.v1.core.dynamic_kv_integration import (
    DynamicKVIntegration,
    create_dynamic_kv_integration,
)
from vllm.v1.attention.dynamic_kv_bridge import (
    DynamicKVAttentionBridge,
    create_dynamic_kv_bridge,
)

# CUDA kernels (optional)
try:
    from vllm.v1.core.dynamic_kv_cuda import (
        DynamicKVCUDAKernels,
        compute_token_importance_cuda,
        compress_kv_cache_cuda,
    )
except ImportError:
    DynamicKVCUDAKernels = None
    compute_token_importance_cuda = None
    compress_kv_cache_cuda = None

__all__ = [
    # Configuration
    "DynamicKVConfig",
    "LayerKVBudget",
    "CompressionStats",
    # Task Analysis
    "TaskPattern",
    "TaskPatternAnalyzer",
    # Core Manager
    "DynamicKVManager",
    "CompressionResult",
    # Compressed Cache
    "CompressedKVBlock",
    "LayerCompressedKVCache",
    "CompressedKVCacheManager",
    # Integration
    "DynamicKVIntegration",
    "create_dynamic_kv_integration",
    # Attention Bridge
    "DynamicKVAttentionBridge",
    "create_dynamic_kv_bridge",
]
