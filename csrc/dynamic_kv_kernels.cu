// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * DynamicKV Cache Kernels
 * 
 * CUDA kernels for compressing and managing KV cache based on token importance.
 * Based on paper: "DynamicKV: Task-Aware Adaptive KV Cache Compression"
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "dynamic_kv.h"
#include "dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <cfloat>

namespace vllm {

constexpr int kMaxThreadsPerBlock = 256;

template <typename scalar_t>
__global__ void compute_token_importance_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    float* __restrict__ importance,
    const int num_heads,
    const int head_size,
    const int64_t key_stride,
    const int64_t value_stride) {
    
    const int64_t token_idx = blockIdx.x;
    const scalar_t* key_ptr = key + token_idx * key_stride;
    const scalar_t* value_ptr = value + token_idx * value_stride;
    
    float key_norm = 0.0f;
    float value_norm = 0.0f;
    
    const int total_elems = num_heads * head_size;
    for (int h = threadIdx.x; h < total_elems; h += blockDim.x) {
        float k_val = static_cast<float>(key_ptr[h]);
        float v_val = static_cast<float>(value_ptr[h]);
        key_norm += k_val * k_val;
        value_norm += v_val * v_val;
    }
    
    __shared__ float shared_key_norm[kMaxThreadsPerBlock];
    __shared__ float shared_value_norm[kMaxThreadsPerBlock];
    
    shared_key_norm[threadIdx.x] = key_norm;
    shared_value_norm[threadIdx.x] = value_norm;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_key_norm[threadIdx.x] += shared_key_norm[threadIdx.x + s];
            shared_value_norm[threadIdx.x] += shared_value_norm[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        importance[token_idx] = sqrtf(shared_key_norm[0] + shared_value_norm[0]) / 2.0f;
    }
}

template <typename scalar_t>
__global__ void compress_kv_cache_kernel(
    const scalar_t* __restrict__ src_key,
    const scalar_t* __restrict__ src_value,
    scalar_t* __restrict__ dst_key,
    scalar_t* __restrict__ dst_value,
    const int* __restrict__ retain_indices,
    const int num_retained,
    const int num_heads,
    const int head_size,
    const int64_t src_stride,
    const int64_t dst_stride) {
    
    const int retained_idx = blockIdx.x;
    const int elem_idx = threadIdx.x;
    
    if (retained_idx >= num_retained) return;
    
    const int orig_idx = retain_indices[retained_idx];
    const int total_elems = num_heads * head_size;
    
    for (int i = elem_idx; i < total_elems; i += blockDim.x) {
        dst_key[retained_idx * dst_stride + i] = 
            src_key[orig_idx * src_stride + i];
        dst_value[retained_idx * dst_stride + i] = 
            src_value[orig_idx * src_stride + i];
    }
}

__global__ void select_retained_tokens_kernel(
    const float* __restrict__ importance,
    int* __restrict__ retain_mask,
    int* __restrict__ retain_indices,
    int* __restrict__ num_retained,
    const float threshold,
    const int num_tokens,
    const int max_retained) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_tokens) {
        retain_mask[tid] = (importance[tid] >= threshold) ? 1 : 0;
    }
}

__global__ void compact_retained_indices_kernel(
    const int* __restrict__ retain_mask,
    int* __restrict__ retain_indices,
    int* __restrict__ num_retained,
    const int num_tokens,
    const int max_retained) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int shared_offset;
    if (threadIdx.x == 0) {
        shared_offset = 0;
    }
    __syncthreads();
    
    if (tid < num_tokens && retain_mask[tid]) {
        int pos = atomicAdd(&shared_offset, 1);
        if (pos < max_retained) {
            retain_indices[pos] = tid;
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        *num_retained = min(shared_offset, max_retained);
    }
}

}  // namespace vllm

void compute_token_importance(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& importance,
    int num_heads,
    int head_size) {
    
    TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
    TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
    TORCH_CHECK(importance.is_cuda(), "importance must be a CUDA tensor");
    
    TORCH_CHECK(key.dim() == 3, "key must be 3D [num_tokens, num_heads, head_size]");
    TORCH_CHECK(value.dim() == 3, "value must be 3D [num_tokens, num_heads, head_size]");
    TORCH_CHECK(importance.dim() == 1, "importance must be 1D [num_tokens]");
    
    const int64_t num_tokens = key.size(0);
    const int64_t key_stride = key.stride(0);
    const int64_t value_stride = value.stride(0);
    
    const int total_elems = num_heads * head_size;
    const int block_size = std::min(kMaxThreadsPerBlock, total_elems);
    const int grid_size = static_cast<int>(num_tokens);
    
    at::cuda::OptionalCUDAGuard device_guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    VLLM_DISPATCH_HALF_TYPES(key.scalar_type(), "compute_token_importance", [&] {
        vllm::compute_token_importance_kernel<scalar_t>
            <<<grid_size, block_size, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                importance.data_ptr<float>(),
                num_heads,
                head_size,
                key_stride,
                value_stride
            );
    });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compress_kv_cache(
    torch::Tensor& src_key,
    torch::Tensor& src_value,
    torch::Tensor& dst_key,
    torch::Tensor& dst_value,
    torch::Tensor& retain_indices,
    int num_retained,
    int num_heads,
    int head_size) {
    
    TORCH_CHECK(src_key.is_cuda(), "src_key must be a CUDA tensor");
    TORCH_CHECK(src_value.is_cuda(), "src_value must be a CUDA tensor");
    TORCH_CHECK(dst_key.is_cuda(), "dst_key must be a CUDA tensor");
    TORCH_CHECK(dst_value.is_cuda(), "dst_value must be a CUDA tensor");
    TORCH_CHECK(retain_indices.is_cuda(), "retain_indices must be a CUDA tensor");
    
    TORCH_CHECK(src_key.scalar_type() == dst_key.scalar_type(), 
                "src_key and dst_key must have the same dtype");
    TORCH_CHECK(src_value.scalar_type() == dst_value.scalar_type(),
                "src_value and dst_value must have the same dtype");
    
    const int64_t src_stride = src_key.stride(0);
    const int64_t dst_stride = dst_key.stride(0);
    const int total_elems = num_heads * head_size;
    const int block_size = std::min(kMaxThreadsPerBlock, total_elems);
    const int grid_size = num_retained;
    
    if (grid_size <= 0) return;
    
    at::cuda::OptionalCUDAGuard device_guard(src_key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    VLLM_DISPATCH_HALF_TYPES(src_key.scalar_type(), "compress_kv_cache", [&] {
        vllm::compress_kv_cache_kernel<scalar_t>
            <<<grid_size, block_size, 0, stream>>>(
                src_key.data_ptr<scalar_t>(),
                src_value.data_ptr<scalar_t>(),
                dst_key.data_ptr<scalar_t>(),
                dst_value.data_ptr<scalar_t>(),
                retain_indices.data_ptr<int>(),
                num_retained,
                num_heads,
                head_size,
                src_stride,
                dst_stride
            );
    });
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
