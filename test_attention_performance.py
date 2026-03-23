#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本：比较NEON、SVE和SME三种Attention实现的性能
"""

import torch
import time
import numpy as np
from typing import Dict, List

# 确保使用CPU
device = torch.device("cpu")

# 测试配置
TEST_CONFIGS = [
    # {"num_tokens": 32, "num_heads": 4, "head_dim": 32, "block_size": 16, "num_blocks": 2},
    # {"num_tokens": 64, "num_heads": 8, "head_dim": 64, "block_size": 32, "num_blocks": 2},
    {"num_tokens": 128, "num_heads": 16, "head_dim": 64, "block_size": 64, "num_blocks": 4},
    {"num_tokens": 256, "num_heads": 16, "head_dim": 128, "block_size": 64, "num_blocks": 8},
    # {"num_tokens": 512, "num_heads": 32, "head_dim": 128, "block_size": 64, "num_blocks": 16},
]

ISAS_TO_TEST = ["neon", "sve", "sme"]
REPEAT_TIMES = 3
WARMUP_TIMES = 2

def generate_test_data(num_tokens: int, num_heads: int, head_dim: int, 
                       num_kv_heads: int, block_size: int, num_blocks: int):
    """生成测试数据"""
    # Query: [num_tokens, num_heads, head_dim]
    query = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Key/Value Cache: [num_blocks, num_kv_heads, block_size, head_dim]
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=torch.float16)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=torch.float16)
    
    # Query start locations and sequence lengths
    query_start_loc = torch.tensor([0, num_tokens//2], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([num_tokens//2, num_tokens//2], dtype=torch.int32, device=device)
    
    # Block table: [num_requests, max_num_blocks]
    max_num_blocks = num_blocks // 2
    block_table = torch.randint(0, num_blocks, (2, max_num_blocks), dtype=torch.int32, device=device)
    
    return query, key_cache, value_cache, query_start_loc, seq_lens, block_table

def test_attention_performance(isa: str, config: Dict):
    """测试特定ISA的Attention性能"""
    num_tokens = config["num_tokens"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    block_size = config["block_size"]
    num_blocks = config["num_blocks"]
    num_kv_heads = num_heads // 2  # 使用多头注意力的常见配置
    
    print(f"\n测试配置: {config}")
    print(f"使用ISA: {isa}")
    
    # 生成测试数据
    query, key_cache, value_cache, query_start_loc, seq_lens, block_table = \
        generate_test_data(num_tokens, num_heads, head_dim, num_kv_heads, block_size, num_blocks)
    
    # 创建输出张量
    output = torch.empty_like(query)
    
    # 导入vllm的CPU注意力函数
    try:
        from vllm._C import cpu_attention_with_kv_cache
    except ImportError:
        print("错误：无法导入vllm._C.cpu_attention_with_kv_cache")
        return None
    
    # 预热运行
    print(f"预热运行 {WARMUP_TIMES} 次...")
    for _ in range(WARMUP_TIMES):
        cpu_attention_with_kv_cache(
            query, key_cache, value_cache, output, query_start_loc, seq_lens, 
            block_table, None, 1.0 / (head_dim ** 0.5), True, 0, 0, 0.0, None, None,
            isa
        )
    
    # 实际测试运行
    print(f"实际测试运行 {REPEAT_TIMES} 次...")
    run_times = []
    for i in range(REPEAT_TIMES):
        start_time = time.time()
        cpu_attention_with_kv_cache(
            query, key_cache, value_cache, output, query_start_loc, seq_lens, 
            block_table, None, 1.0 / (head_dim ** 0.5), True, 0, 0, 0.0, None, None,
            isa
        )
        end_time = time.time()
        run_time = (end_time - start_time) * 1000  # 转换为毫秒
        run_times.append(run_time)
        print(f"  运行 {i+1}: {run_time:.2f} ms")
    
    # 计算统计信息
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    min_time = np.min(run_times)
    max_time = np.max(run_times)
    
    print(f"统计结果 (毫秒):")
    print(f"  平均时间: {avg_time:.2f} ± {std_time:.2f}")
    print(f"  最小时间: {min_time:.2f}")
    print(f"  最大时间: {max_time:.2f}")
    
    return {
        "isa": isa,
        "config": config,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "run_times": run_times
    }

def main():
    """主函数"""
    print("开始测试Attention实现性能...")
    print(f"测试的ISA: {ISAS_TO_TEST}")
    print(f"测试配置: {TEST_CONFIGS}")
    print(f"每个测试重复: {REPEAT_TIMES}次 (预热{WARMUP_TIMES}次)")
    
    results = {}
    
    # 对每个ISA进行测试
    for isa in ISAS_TO_TEST:
        results[isa] = []
        print(f"\n===== 测试 {isa} =====")
        
        # 对每个配置进行测试
        for config in TEST_CONFIGS:
            try:
                result = test_attention_performance(isa, config)
                if result:
                    results[isa].append(result)
            except Exception as e:
                print(f"测试 {isa} 失败: {e}")
                continue
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("性能测试汇总结果")
    print("="*60)
    
    for config_idx, config in enumerate(TEST_CONFIGS):
        print(f"\n配置 {config_idx+1}: {config}")
        print("-" * 40)
        print(f"{'ISA':<8} {'平均时间 (ms)':<15} {'速度提升':<15}")
        print("-" * 40)
        
        # 找到最慢的实现作为基准
        base_time = float('inf')
        for isa in ISAS_TO_TEST:
            if config_idx < len(results[isa]):
                base_time = min(base_time, results[isa][config_idx]["avg_time"])
        
        # 打印每个实现的性能和相对提升
        for isa in ISAS_TO_TEST:
            if config_idx < len(results[isa]):
                avg_time = results[isa][config_idx]["avg_time"]
                speedup = base_time / avg_time if avg_time > 0 else 0
                print(f"{isa:<8} {avg_time:<15.2f} {speedup:<15.2fx}")
            else:
                print(f"{isa:<8} {'失败':<15} {'-':<15}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()
