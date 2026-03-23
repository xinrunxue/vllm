#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
注意力权重分析测试脚本

使用方法:
    python scripts/analyze_attention.py --model Qwen/Qwen3-32B --prompt "你的问题"

环境变量:
    VLLM_ATTENTION_ANALYSIS=1     启用注意力分析
    VLLM_ATTENTION_THRESHOLD=1e-4 设置接近零的阈值
"""

import argparse
import os

os.environ["VLLM_ATTENTION_ANALYSIS"] = "1"

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.flash_attn import (
    enable_attention_analysis,
    print_attention_analysis_summary,
    get_attention_analysis_summary,
)


def main():
    parser = argparse.ArgumentParser(description="分析模型注意力权重分布")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="模型名称")
    parser.add_argument("--prompt", type=str, default="请解释什么是机器学习，并举例说明其应用场景。", help="输入提示")
    parser.add_argument("--threshold", type=float, default=1e-4, help="接近零阈值")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行大小")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("注意力权重分析工具")
    print(f"{'='*80}")
    print(f"模型: {args.model}")
    print(f"阈值: {args.threshold}")
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"{'='*80}\n")
    
    enable_attention_analysis(threshold=args.threshold)
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    
    outputs = llm.generate([args.prompt], sampling_params)
    
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n生成的文本:\n{generated_text}")
    
    print_attention_analysis_summary()
    
    summary = get_attention_analysis_summary()
    print(f"\n压缩潜力分析:")
    cp = summary.get("compression_potential", {})
    print(f"  高稀疏层 (>50%): {cp.get('layers_with_50p_sparsity', 0)}")
    print(f"  中等稀疏层 (>30%): {cp.get('layers_with_30p_sparsity', 0)}")
    print(f"  预计平均压缩率: {cp.get('estimated_avg_compression', 0):.2%}")


if __name__ == "__main__":
    main()
