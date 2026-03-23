# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DynamicKV CUDA Kernel Interface

Python interface for DynamicKV CUDA kernels.
"""

from typing import Optional
import torch


def compute_token_importance_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    importance: torch.Tensor,
    num_heads: int,
    head_size: int,
) -> None:
    """Compute token importance scores using CUDA kernel.
    
    Args:
        key: [num_tokens, num_heads, head_size] Key tensor
        value: [num_tokens, num_heads, head_size] Value tensor
        importance: [num_tokens] Output importance scores
        num_heads: Number of attention heads
        head_size: Size of each attention head
    """
    torch.ops._C_cache_ops.compute_token_importance(
        key, value, importance, num_heads, head_size
    )


def compress_kv_cache_cuda(
    src_key: torch.Tensor,
    src_value: torch.Tensor,
    dst_key: torch.Tensor,
    dst_value: torch.Tensor,
    retain_indices: torch.Tensor,
    num_retained: int,
    num_heads: int,
    head_size: int,
) -> None:
    """Compress KV cache using CUDA kernel.
    
    Args:
        src_key: [num_tokens, num_heads, head_size] Source key tensor
        src_value: [num_tokens, num_heads, head_size] Source value tensor
        dst_key: [num_retained, num_heads, head_size] Destination key tensor
        dst_value: [num_retained, num_heads, head_size] Destination value tensor
        retain_indices: [num_retained] Indices of tokens to retain
        num_retained: Number of tokens to retain
        num_heads: Number of attention heads
        head_size: Size of each attention head
    """
    torch.ops._C_cache_ops.compress_kv_cache(
        src_key, src_value, dst_key, dst_value,
        retain_indices, num_retained, num_heads, head_size
    )


class DynamicKVCUDAKernels:
    """CUDA kernel wrapper for DynamicKV operations.
    
    This class provides a high-level interface for DynamicKV CUDA operations.
    """
    
    @staticmethod
    def compute_importance(
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        """Compute token importance scores.
        
        Args:
            key: [num_tokens, num_heads, head_size] Key tensor
            value: [num_tokens, num_heads, head_size] Value tensor
            num_heads: Number of attention heads
            head_size: Size of each attention head
            
        Returns:
            importance: [num_tokens] Importance scores for each token
        """
        num_tokens = key.shape[0]
        importance = torch.empty(num_tokens, dtype=torch.float32, device=key.device)
        
        compute_token_importance_cuda(key, value, importance, num_heads, head_size)
        
        return importance
    
    @staticmethod
    def compress_cache(
        src_key: torch.Tensor,
        src_value: torch.Tensor,
        retain_indices: torch.Tensor,
        num_heads: int,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress KV cache by retaining only important tokens.
        
        Args:
            src_key: [num_tokens, num_heads, head_size] Source key tensor
            src_value: [num_tokens, num_heads, head_size] Source value tensor
            retain_indices: [num_retained] Indices of tokens to retain
            num_heads: Number of attention heads
            head_size: Size of each attention head
            
        Returns:
            compressed_key: [num_retained, num_heads, head_size]
            compressed_value: [num_retained, num_heads, head_size]
        """
        num_retained = retain_indices.shape[0]
        
        compressed_key = torch.empty(
            num_retained, num_heads, head_size,
            dtype=src_key.dtype, device=src_key.device
        )
        compressed_value = torch.empty(
            num_retained, num_heads, head_size,
            dtype=src_value.dtype, device=src_value.device
        )
        
        compress_kv_cache_cuda(
            src_key, src_value,
            compressed_key, compressed_value,
            retain_indices, num_retained,
            num_heads, head_size
        )
        
        return compressed_key, compressed_value
    
    @staticmethod
    def select_top_k_tokens(
        importance: torch.Tensor,
        k: int,
        preserve_recent: int = 0,
    ) -> torch.Tensor:
        """Select top-k most important tokens.
        
        Args:
            importance: [num_tokens] Importance scores
            k: Number of tokens to select
            preserve_recent: Number of recent tokens to always preserve
            
        Returns:
            indices: [k] Indices of selected tokens (sorted)
        """
        num_tokens = importance.shape[0]
        k = min(k, num_tokens)
        
        if preserve_recent > 0:
            recent_indices = torch.arange(
                num_tokens - preserve_recent, num_tokens,
                device=importance.device
            )
            k_from_importance = k - preserve_recent
            
            if k_from_importance > 0:
                candidate_importance = importance[:num_tokens - preserve_recent]
                _, top_indices = torch.topk(candidate_importance, k_from_importance)
                indices = torch.cat([top_indices, recent_indices])
            else:
                indices = recent_indices[-k:]
        else:
            _, indices = torch.topk(importance, k)
        
        indices, _ = torch.sort(indices)
        
        return indices
