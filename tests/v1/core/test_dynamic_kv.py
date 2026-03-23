# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for DynamicKV implementation
"""

import pytest
import torch

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
)
from vllm.v1.core.compressed_kv_cache import (
    CompressedKVBlock,
    LayerCompressedKVCache,
    CompressedKVCacheManager,
)


class TestDynamicKVConfig:
    """Tests for DynamicKVConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DynamicKVConfig()
        
        assert config.enabled is False
        assert config.budget_ratio == 0.05
        assert config.layer_budget_strategy == "adaptive"
        assert config.importance_method == "combined"
        assert config.update_interval == 128
        assert config.min_tokens_per_layer == 16
        assert config.enable_task_aware is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = DynamicKVConfig(
            enabled=True,
            budget_ratio=0.1,
            layer_budget_strategy="uniform",
            importance_method="attention_weight",
            update_interval=256,
        )
        
        assert config.enabled is True
        assert config.budget_ratio == 0.1
        assert config.layer_budget_strategy == "uniform"
        assert config.importance_method == "attention_weight"
        assert config.update_interval == 256
    
    def test_invalid_budget_ratio(self):
        """Test that invalid budget ratio raises error"""
        with pytest.raises(ValueError):
            DynamicKVConfig(budget_ratio=0)
        
        with pytest.raises(ValueError):
            DynamicKVConfig(budget_ratio=1.5)
    
    def test_custom_layer_budgets(self):
        """Test custom layer budgets configuration"""
        config = DynamicKVConfig(
            layer_budget_strategy="custom",
            custom_layer_budgets=[0.1, 0.08, 0.06, 0.04],
        )
        
        assert config.layer_budget_strategy == "custom"
        assert len(config.custom_layer_budgets) == 4


class TestLayerKVBudget:
    """Tests for LayerKVBudget"""
    
    def test_budget_creation(self):
        """Test layer budget creation"""
        budget = LayerKVBudget(
            layer_idx=0,
            max_tokens=1000,
            current_tokens=500,
            importance_threshold=0.5,
        )
        
        assert budget.layer_idx == 0
        assert budget.max_tokens == 1000
        assert budget.current_tokens == 500
        assert budget.importance_threshold == 0.5
    
    def test_get_retention_count(self):
        """Test retention count calculation"""
        budget = LayerKVBudget(layer_idx=0, max_tokens=100)
        
        assert budget.get_retention_count(200) == 100
        assert budget.get_retention_count(50) == 50
    
    def test_is_over_budget(self):
        """Test budget overflow check"""
        budget = LayerKVBudget(layer_idx=0, max_tokens=100, current_tokens=50)
        assert not budget.is_over_budget()
        
        budget.current_tokens = 150
        assert budget.is_over_budget()


class TestCompressionStats:
    """Tests for CompressionStats"""
    
    def test_initial_stats(self):
        """Test initial statistics"""
        stats = CompressionStats()
        
        assert stats.total_tokens_before == 0
        assert stats.total_tokens_after == 0
        assert stats.compression_ratio == 1.0
        assert stats.tokens_evicted == 0
    
    def test_update_stats(self):
        """Test statistics update"""
        stats = CompressionStats()
        
        stats.update(100, 50)
        assert stats.total_tokens_before == 100
        assert stats.total_tokens_after == 50
        assert stats.tokens_evicted == 50
        assert stats.compression_ratio == 0.5
        
        stats.update(100, 30)
        assert stats.total_tokens_before == 200
        assert stats.total_tokens_after == 80
        assert stats.tokens_evicted == 120
        assert stats.compression_ratio == 0.4


class TestTaskPatternAnalyzer:
    """Tests for TaskPatternAnalyzer"""
    
    def test_pattern_creation(self):
        """Test task pattern creation"""
        analyzer = TaskPatternAnalyzer(num_layers=32)
        
        assert analyzer.num_layers == 32
        assert "rag" in TaskPatternAnalyzer.TASK_PATTERNS
        assert "summarization" in TaskPatternAnalyzer.TASK_PATTERNS
        assert "code" in TaskPatternAnalyzer.TASK_PATTERNS
    
    def test_budget_adjustments(self):
        """Test budget adjustment retrieval"""
        analyzer = TaskPatternAnalyzer(num_layers=20)
        
        rag_adjustments = analyzer.get_budget_adjustments("rag")
        assert len(rag_adjustments) == 20
        
        general_adjustments = analyzer.get_budget_adjustments("general")
        assert all(a == 1.0 for a in general_adjustments)
    
    def test_task_detection_from_text(self):
        """Test task detection from text"""
        analyzer = TaskPatternAnalyzer(num_layers=20)
        
        code_text = "def hello_world():\n    print('Hello')"
        assert analyzer._detect_from_text(code_text) == "code"
        
        rag_text = "Based on the following context: [document]"
        assert analyzer._detect_from_text(rag_text) == "rag"
        
        summary_text = "Please summarize the following article"
        assert analyzer._detect_from_text(summary_text) == "summarization"
    
    def test_layer_interpolation(self):
        """Test layer budget interpolation"""
        analyzer = TaskPatternAnalyzer(num_layers=10)
        
        for task_type, pattern in TaskPatternAnalyzer.TASK_PATTERNS.items():
            assert len(pattern.layer_budget_ratios) == 10


class TestDynamicKVManager:
    """Tests for DynamicKVManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a DynamicKVManager instance"""
        config = DynamicKVConfig(
            enabled=True,
            budget_ratio=0.1,
            min_tokens_per_layer=8,
        )
        return DynamicKVManager(
            num_layers=8,
            max_seq_len=1024,
            config=config,
        )
    
    def test_manager_creation(self, manager):
        """Test manager creation"""
        assert manager.num_layers == 8
        assert manager.max_seq_len == 1024
        assert len(manager.layer_budgets) == 8
    
    def test_adaptive_budget(self, manager):
        """Test adaptive budget allocation"""
        budgets = manager.layer_budgets
        
        assert budgets[0].max_tokens >= budgets[-1].max_tokens
    
    def test_token_importance(self, manager):
        """Test token importance calculation"""
        key_cache = torch.randn(100, 8, 64)
        value_cache = torch.randn(100, 8, 64)
        
        importance = manager.compute_token_importance(
            layer_idx=0,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        
        assert importance.shape == (100,)
        assert importance.min() >= 0
    
    def test_select_tokens_to_retain(self, manager):
        """Test token selection for retention"""
        importance = torch.randn(100)
        
        retain_indices = manager.select_tokens_to_retain(
            layer_idx=0,
            importance_scores=importance,
            num_tokens=100,
            preserve_recent=16,
        )
        
        assert len(retain_indices) <= manager.layer_budgets[0].max_tokens
        assert retain_indices[-1] == 99  # Last token should be preserved
    
    def test_compress_kv_cache(self, manager):
        """Test KV cache compression"""
        key_cache = torch.randn(100, 8, 64)
        value_cache = torch.randn(100, 8, 64)
        
        result = manager.compress_kv_cache(
            layer_idx=0,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        
        assert isinstance(result, CompressionResult)
        assert result.num_tokens_before == 100
        assert result.num_tokens_after <= 100
        assert result.compression_ratio <= 1.0
    
    def test_should_compress(self, manager):
        """Test compression trigger logic"""
        config = DynamicKVConfig(
            enabled=True,
            compression_trigger="periodic",
            update_interval=10,
        )
        manager.config = config
        
        for _ in range(9):
            assert not manager.should_compress(1)
        
        assert manager.should_compress(1)
    
    def test_task_type_setting(self, manager):
        """Test task type setting"""
        manager.set_task_type("rag")
        
        assert manager._current_task == "rag"
        assert manager.task_analyzer is not None


class TestCompressedKVCache:
    """Tests for CompressedKVCache"""
    
    @pytest.fixture
    def cache_manager(self):
        """Create a CompressedKVCacheManager instance"""
        return CompressedKVCacheManager(num_layers=8)
    
    def test_cache_manager_creation(self, cache_manager):
        """Test cache manager creation"""
        assert cache_manager.num_layers == 8
        assert len(cache_manager.layer_caches) == 8
    
    def test_store_and_retrieve(self, cache_manager):
        """Test storing and retrieving compressed cache"""
        key = torch.randn(50, 8, 64)
        value = torch.randn(50, 8, 64)
        retain_indices = torch.arange(50)
        
        cache_manager.store_compressed_cache(
            request_id="test_req",
            layer_idx=0,
            block_id=0,
            key_cache=key,
            value_cache=value,
            retain_indices=retain_indices,
            original_num_tokens=100,
        )
        
        retrieved_key, retrieved_value, retrieved_indices = \
            cache_manager.get_compressed_cache(layer_idx=0, block_ids=[0])
        
        assert retrieved_key is not None
        assert retrieved_key.shape == key.shape
    
    def test_compression_stats(self, cache_manager):
        """Test compression statistics"""
        key = torch.randn(50, 8, 64)
        value = torch.randn(50, 8, 64)
        retain_indices = torch.arange(50)
        
        cache_manager.store_compressed_cache(
            request_id="test_req",
            layer_idx=0,
            block_id=0,
            key_cache=key,
            value_cache=value,
            retain_indices=retain_indices,
            original_num_tokens=100,
        )
        
        stats = cache_manager.get_compression_stats()
        
        assert stats["num_layers"] == 8
        assert stats["total_original_tokens"] == 100
        assert stats["total_retained_tokens"] == 50
        assert stats["overall_compression_ratio"] == 0.5
    
    def test_remove_request_blocks(self, cache_manager):
        """Test removing request blocks"""
        key = torch.randn(50, 8, 64)
        value = torch.randn(50, 8, 64)
        retain_indices = torch.arange(50)
        
        cache_manager.store_compressed_cache(
            request_id="test_req",
            layer_idx=0,
            block_id=0,
            key_cache=key,
            value_cache=value,
            retain_indices=retain_indices,
            original_num_tokens=100,
        )
        
        cache_manager.remove_request_blocks("test_req")
        
        retrieved_key, _, _ = cache_manager.get_compressed_cache(
            layer_idx=0, block_ids=[0]
        )
        
        assert retrieved_key is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
