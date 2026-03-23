// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * DynamicKV Cache Kernel Declarations
 */

#pragma once

#include <torch/all.h>

namespace vllm {

void compute_token_importance(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& importance,
    int num_heads,
    int head_size);

void compress_kv_cache(
    torch::Tensor& src_key,
    torch::Tensor& src_value,
    torch::Tensor& dst_key,
    torch::Tensor& dst_value,
    torch::Tensor& retain_indices,
    int num_retained,
    int num_heads,
    int head_size);

}  // namespace vllm

using vllm::compute_token_importance;
using vllm::compress_kv_cache;
