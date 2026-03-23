# DynamicKV: Task-Aware Adaptive KV Cache Compression

## 概述

DynamicKV 是一种任务感知的自适应 KV cache 压缩方法，可以根据不同任务的特性动态调整 KV cache 大小，在保持模型性能的同时大幅减少内存占用。

基于论文：[DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs](https://aclanthology.org/2025.findings-emnlp.426/)

## 核心特性

- **任务感知**：自动检测任务类型（RAG、摘要、代码等）并调整压缩策略
- **层级自适应**：不同层使用不同的压缩预算
- **Token 重要性评分**：基于注意力权重或激活值计算 token 重要性
- **高压缩率**：保留 1.7% KV cache 即可保持 90% 准确率

## 使用方法

### 1. 基本配置

```python
from vllm.v1.core.dynamic_kv import DynamicKVConfig, DynamicKVManager

# 创建配置
config = DynamicKVConfig(
    enabled=True,
    budget_ratio=0.05,              # 保留 5% 的 KV cache
    layer_budget_strategy="adaptive",  # 自适应层级预算
    importance_method="combined",   # 综合评分方法
    update_interval=128,            # 每 128 个 token 更新一次
    min_tokens_per_layer=16,        # 每层最少保留 16 个 token
    enable_task_aware=True,         # 启用任务感知
)
```

### 2. 创建管理器

```python
# 创建 DynamicKV 管理器
manager = DynamicKVManager(
    num_layers=32,      # 模型层数
    max_seq_len=8192,   # 最大序列长度
    config=config,
)
```

### 3. 设置任务类型

```python
# 自动检测任务类型
task_type = manager.task_analyzer.detect_task_type(
    prompt_text="Based on the following documents...",
)
manager.set_task_type(task_type)

# 或手动设置
manager.set_task_type("rag")  # 支持: rag, summarization, code, needle_haystack, general
```

### 4. 压缩 KV Cache

```python
import torch

# 假设 key_cache 和 value_cache 是 [num_tokens, num_heads, head_dim]
key_cache = torch.randn(1000, 32, 128)
value_cache = torch.randn(1000, 32, 128)

# 执行压缩
result = manager.compress_kv_cache(
    layer_idx=0,
    key_cache=key_cache,
    value_cache=value_cache,
)

# 获取压缩后的数据
compressed_key = key_cache[result.retain_indices]
compressed_value = value_cache[result.retain_indices]

print(f"压缩前: {result.num_tokens_before} tokens")
print(f"压缩后: {result.num_tokens_after} tokens")
print(f"压缩率: {result.compression_ratio:.2%}")
```

### 5. 与 vLLM 集成

```python
from vllm import LLM, SamplingParams
from vllm.v1.core.dynamic_kv import DynamicKVConfig

# 创建 DynamicKV 配置
dynamic_kv_config = DynamicKVConfig(
    enabled=True,
    budget_ratio=0.05,
    layer_budget_strategy="adaptive",
)

# 创建 LLM 实例（需要传递配置）
# 注意：具体集成方式取决于 vLLM 版本
llm = LLM(
    model="meta-llama/Llama-3-8B",
    # DynamicKV 配置将通过引擎配置传递
)
```

## 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `enabled` | bool | False | 是否启用 DynamicKV |
| `budget_ratio` | float | 0.05 | 全局 KV cache 预算比例 |
| `layer_budget_strategy` | str | "adaptive" | 层预算策略：adaptive/uniform/custom |
| `importance_method` | str | "combined" | 重要性评分方法：attention_weight/activation/combined |
| `update_interval` | int | 128 | KV cache 更新间隔 |
| `min_tokens_per_layer` | int | 16 | 每层最小保留 token 数 |
| `enable_task_aware` | bool | True | 是否启用任务感知 |
| `compression_trigger` | str | "periodic" | 压缩触发条件：always/on_budget/periodic |
| `preserve_recent_tokens` | int | 32 | 保留最近的 token 数 |

## 任务类型与压缩策略

| 任务类型 | 特点 | 压缩策略 |
|---------|------|---------|
| `rag` | 检索增强生成 | 深层保留更多 token 用于检索 |
| `summarization` | 长文本摘要 | 均匀压缩 |
| `code` | 代码生成 | 浅层保留更多（局部依赖） |
| `needle_haystack` | 大海捞针 | 深层保留更多以找到关键信息 |
| `general` | 通用任务 | 默认策略 |

## 性能指标

根据论文，DynamicKV 在 LongBench 数据集上的表现：

| 模型 | KV Cache 比例 | 准确率保持 |
|-----|--------------|-----------|
| LLaMA-3-8B-Instruct | 1.7% | 90% |
| Mistral-7B-Instruct-v0.2 | 1.7% | 87% |
| Qwen2-7B-Instruct | 1.7% | 78% |
| InternLM-2.5-7B-Chat-1M | 1.7% | 83% |

## 文件结构

```
vllm/v1/core/
├── dynamic_kv.py              # 导出模块
├── dynamic_kv_config.py       # 配置类
├── dynamic_kv_manager.py      # 核心管理器
├── task_pattern_analyzer.py   # 任务模式分析
├── compressed_kv_cache.py     # 压缩缓存管理
└── dynamic_kv_integration.py  # KVCacheManager 集成

vllm/v1/attention/
└── dynamic_kv_bridge.py       # Attention 层桥接

tests/v1/core/
└── test_dynamic_kv.py         # 单元测试
```

## 注意事项

1. **内存权衡**：压缩会带来轻微的计算开销，但大幅减少内存占用
2. **准确率**：极端压缩（<1%）可能导致准确率下降
3. **任务检测**：自动检测基于启发式规则，可能需要手动指定
4. **兼容性**：目前主要支持 FlashAttention 后端
