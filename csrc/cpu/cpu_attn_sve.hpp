#ifndef CPU_ATTN_SVE_HPP
#define CPU_ATTN_SVE_HPP

#include "cpu_attn_impl.hpp"
#include <arm_sve.h>
#include <type_traits>

namespace cpu_attention {

namespace {

#define BLOCK_SIZE_ALIGNMENT_SVE 32
#define HEAD_SIZE_ALIGNMENT_SVE 32
#define MAX_Q_HEAD_NUM_PER_ITER_SVE 16

// Load 8 elements from B as float32 using SVE
// Supports float, half, and bfloat16 data types
template <typename kv_cache_t>
FORCE_INLINE void load_row8_B_as_f32_sve(const kv_cache_t* p, svfloat32_t& b0, svfloat32_t& b1);

template <>
FORCE_INLINE void load_row8_B_as_f32_sve<float>(const float* p, svfloat32_t& b0, svfloat32_t& b1) {
  // For float, load directly
  b0 = svld1(svptrue_b32(), p + 0);
  b1 = svld1(svptrue_b32(), p + 4);
}

template <>
FORCE_INLINE void load_row8_B_as_f32_sve<c10::Half>(const c10::Half* p, svfloat32_t& b0, svfloat32_t& b1) {
  // For half, convert to float32 using SVE
  const float16_t* h = reinterpret_cast<const float16_t*>(p);
  svfloat16_t v0 = svld1(svptrue_b16(), h + 0);
  svfloat16_t v1 = svld1(svptrue_b16(), h + 4);
  b0 = svcvt_f32_f16_m(svptrue_b32(), b0, v0);
  b1 = svcvt_f32_f16_m(svptrue_b32(), b1, v1);
}

template <>
FORCE_INLINE void load_row8_B_as_f32_sve<c10::BFloat16>(const c10::BFloat16* p, svfloat32_t& b0, svfloat32_t& b1) {
  // For bfloat16, convert to float32 using SVE
  const uint16_t* u = reinterpret_cast<const uint16_t*>(p);
  svuint16_t u0 = svld1(svptrue_b16(), u + 0);
  svuint16_t u1 = svld1(svptrue_b16(), u + 4);
  svbfloat16_t bf0 = svreinterpret_bf16_u16(u0);
  svbfloat16_t bf1 = svreinterpret_bf16_u16(u1);
  b0 = svcvt_f32_bf16_m(svptrue_b32(), b0, bf0);
  b1 = svcvt_f32_bf16_m(svptrue_b32(), b1, bf1);
}

// SVE-based GEMM microkernel for Attention
// Mx8, with 1 <= M <= 16, K streamed, unroll-by-4 with SVE FMLAs
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_sve_fmla_Mx8_Ku4(
    const float* __restrict A,       // [M x K],
    const kv_cache_t* __restrict B,  // [K x 8],
    float* __restrict C,             // [M x 8],
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  // kernel supports max M of 16 with SVE (more than NEON's 8)
  static_assert(1 <= M && M <= 16, "M must be in [1,16]");

  // helpers for per-M codegen
#define ROWS_APPLY_SVE(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7) OP(8) OP(9) OP(10) OP(11) OP(12) OP(13) OP(14) OP(15)
#define IF_M_SVE(i) if constexpr (M > (i))

  // A row base pointers
#define DECL_A_SVE(i) const float* a##i = A + (i) * lda;
  ROWS_APPLY_SVE(DECL_A_SVE)
#undef DECL_A_SVE

  // declare 2 accumulators per row of M (using SVE vectors)
#define DECL_ACC_SVE(i) svfloat32_t acc##i##_0, acc##i##_1;
  ROWS_APPLY_SVE(DECL_ACC_SVE)
#undef DECL_ACC_SVE

  // Get SVE predicate for 4 elements (since we're processing 4 elements at a time)
  svbool_t p4 = svptrue_b32();

  // initialize accumulators
#define INIT_ACC_SVE(i)                              \
  IF_M_SVE(i) {                                      \
    if (accumulate) {                                \
      acc##i##_0 = svld1(p4, C + (i) * ldc + 0);     \
      acc##i##_1 = svld1(p4, C + (i) * ldc + 4);     \
    } else {                                         \
      acc##i##_0 = svdup_f32(0.f);                   \
      acc##i##_1 = svdup_f32(0.f);                   \
    }                                                \
  }
  ROWS_APPLY_SVE(INIT_ACC_SVE)
#undef INIT_ACC_SVE

  int32_t k = 0;

  // K unrolled by 4 - SVE can process 4 elements per iteration efficiently
  for (; k + 3 < K; k += 4) {
    // load A[k..k+3] for each active row (M)
    // Using SVE, we can load 4 elements at once for each row
#define LOAD_A4_SVE(i)     \
  svfloat32_t a##i##v;     \
  IF_M_SVE(i) a##i##v = svld1(p4, a##i + k);
    ROWS_APPLY_SVE(LOAD_A4_SVE)
#undef LOAD_A4_SVE

    // helper: FMA lane L from aiv using SVE
#define FMAS_LANE_SVE(i, aiv, L)                              \
  IF_M_SVE(i) {                                               \
    // For SVE, we can use svmla_lane to get the L-th lane from the vector  \
    acc##i##_0 = svmla_f32_m(p4, acc##i##_0, b0, svdup_lane_f32(aiv, L)); \
    acc##i##_1 = svmla_f32_m(p4, acc##i##_1, b1, svdup_lane_f32(aiv, L)); \
  }

    // Process k + 0
    {
      svfloat32_t b0, b1;
      load_row8_B_as_f32_sve<kv_cache_t>(B + (int64_t)(k + 0) * ldb, b0, b1);
#define STEP_K0_SVE(i) FMAS_LANE_SVE(i, a##i##v, 0)
      ROWS_APPLY_SVE(STEP_K0_SVE)
#undef STEP_K0_SVE
    }

    // Process k + 1
    {
      svfloat32_t b0, b1;
      load_row8_B_as_f32_sve<kv_cache_t>(B + (int64_t)(k + 1) * ldb, b0, b1);
#define STEP_K1_SVE(i) FMAS_LANE_SVE(i, a##i##v, 1)
      ROWS_APPLY_SVE(STEP_K1_SVE)
#undef STEP_K1_SVE
    }

    // Process k + 2
    {
      svfloat32_t b0, b1;
      load_row8_B_as_f32_sve<kv_cache_t>(B + (int64_t)(k + 2) * ldb, b0, b1);
#define STEP_K2_SVE(i) FMAS_LANE_SVE(i, a##i##v, 2)
      ROWS_APPLY_SVE(STEP_K2_SVE)
#undef STEP_K2_SVE
    }

    // Process k + 3
    {
      svfloat32_t b0, b1;
      load_row8_B_as_f32_sve<kv_cache_t>(B + (int64_t)(k + 3) * ldb, b0, b1);
#define STEP_K3_SVE(i) FMAS_LANE_SVE(i, a##i##v, 3)
      ROWS_APPLY_SVE(STEP_K3_SVE)
#undef STEP_K3_SVE
    }

#undef FMAS_LANE_SVE
  }

  // K tail - process remaining elements
  for (; k < K; ++k) {
    svfloat32_t b0, b1;
    load_row8_B_as_f32_sve<kv_cache_t>(B + (int64_t)k * ldb, b0, b1);

#define TAIL_ROW_SVE(i)                             \
  IF_M_SVE(i) {                                     \
    // Broadcast the scalar value to a vector      \
    svfloat32_t ai = svdup_f32(*(a##i + k));        \
    acc##i##_0 = svmla_f32_m(p4, acc##i##_0, b0, ai); \
    acc##i##_1 = svmla_f32_m(p4, acc##i##_1, b1, ai); \
  }
    ROWS_APPLY_SVE(TAIL_ROW_SVE)
#undef TAIL_ROW_SVE
  }

  // store accumulators to C using SVE
#define STORE_ROW_SVE(i)                          \
  IF_M_SVE(i) {                                   \
    svst1_m(p4, C + (i) * ldc + 0, acc##i##_0);   \
    svst1_m(p4, C + (i) * ldc + 4, acc##i##_1);   \
  }
  ROWS_APPLY_SVE(STORE_ROW_SVE)
#undef STORE_ROW_SVE

#undef ROWS_APPLY_SVE
#undef IF_M_SVE
}

// SVE-based macro GEMM that handles arbitrary M sizes
template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_sve_fmla_Mx8_Ku4(const float* __restrict A,
                                             const kv_cache_t* __restrict B,
                                             float* __restrict C, int32_t M,
                                             int32_t K, int64_t lda,
                                             int64_t ldb, int64_t ldc,
                                             bool accumulate) {
  // micro kernel is Mx8
  static_assert(N % 8 == 0, "N must be a multiple of 8");
  
  // With SVE, we can handle larger M sizes (up to 16) for better utilization
  for (int32_t m = 0; m < M;) {
    // Choose the largest block size that fits (up to 16 for SVE)
    int32_t mb = (M - m >= 16) ? 16 : 
                 (M - m >= 8) ? 8 : 
                 (M - m >= 4) ? 4 : 
                 (M - m >= 2) ? 2 : 1;
    
    const float* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    for (int32_t n = 0; n < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      
      // Dispatch to the appropriate microkernel based on mb
      switch (mb) {
        case 16:
          gemm_micro_sve_fmla_Mx8_Ku4<16, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
        case 8:
          gemm_micro_sve_fmla_Mx8_Ku4<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
        case 4:
          gemm_micro_sve_fmla_Mx8_Ku4<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
        case 2:
          gemm_micro_sve_fmla_Mx8_Ku4<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
        default:
          gemm_micro_sve_fmla_Mx8_Ku4<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
      }
    }
    // no tail loop for N as it's guaranteed to be a multiple of 8
    m += mb;
  }
}

// SVE-based TileGEMM implementation
template <typename kv_cache_t>
class TileGemmSVEFMLA {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::QK) {
      gemm_macro_sve_fmla_Mx8_Ku4<BLOCK_SIZE_ALIGNMENT_SVE, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc, accum_c);
    } else {
      gemm_macro_sve_fmla_Mx8_Ku4<HEAD_SIZE_ALIGNMENT_SVE, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb, ldc, accum_c);
    }
  }
};

}  // namespace

// Add SVE to the ISA enum
enum class ISA { AMX, VEC, VEC16, NEON, SVE };

// SVE implementation of AttentionImpl
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::SVE, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

  constexpr static int64_t BlockSizeAlignment = 
      BLOCK_SIZE_ALIGNMENT_SVE;  // KV token num unit of QK and PV phases
  constexpr static int64_t HeadDimAlignment = 
      HEAD_SIZE_ALIGNMENT_SVE;  // headdim num unit of PV phase
  constexpr static int64_t MaxQHeadNumPerIteration = MAX_Q_HEAD_NUM_PER_ITER_SVE;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::SVE;
  constexpr static bool scale_on_logits = false;  // apply scale on q_buffer

  // the gemm micro kernel is Mx8 with SVE
  static_assert(HeadDimAlignment % 8 == 0);
  static_assert(BlockSizeAlignment % 8 == 0);

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmSVEFMLA<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // k_cache_token_group_stride: stride of K cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment;  // layout of k_cache block is [head_dim,
                                // block_size], row-major
  }

  // v_cache_token_group_stride: stride of V cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t v_cache_token_group_stride(
      const int32_t block_size) {
    return head_dim * BlockSizeAlignment;  // layout of v_cache is [block_size,
                                           // head_dim], row-major
  }

  // v_cache_head_group_stride: stride of V cache when move to next
  // HeadDimAlignment head dims in a block
  constexpr static int64_t v_cache_head_group_stride(const int32_t block_size) {
    return HeadDimAlignment;  // layout of v_cache is [block_size, head_dim],
                              // row-major
  }

  // Copy q to q_buffer and cast it to fp32 using SVE
  static void copy_q_heads_tile(
      scalar_t* __restrict__ src,  // [q_num, q_heads_per_kv, head_size]
      float* __restrict__ q_buffer, const int32_t q_num,
      const int32_t q_heads_per_kv, const int64_t q_num_stride,
      const int64_t q_head_stride, float scale) {
    static_assert(head_dim % 16 == 0);
    constexpr int32_t unroll_size = head_dim / 16;
    
    // Use SVE vector for scale
    svfloat32_t scale_vec = svdup_f32(scale);
    svbool_t p16 = svptrue_b32();
    
    for (int32_t q_num_idx = 0; q_num_idx < q_num; ++q_num_idx) {
      for (int32_t q_head_idx = 0; q_head_idx < q_heads_per_kv; ++q_head_idx) {
        scalar_t* __restrict__ curr_q = 
            src + q_num_idx * q_num_stride + q_head_idx * q_head_stride;
        float* __restrict__ curr_q_buffer = 
            q_buffer + q_num_idx * q_heads_per_kv * head_dim + 
            q_head_idx * head_dim;

        // Process 16 elements at a time using SVE
        for (int32_t i = 0; i < unroll_size; ++i) {
          if constexpr (std::is_same<scalar_t, float>::value) {
            // Load float directly
            svfloat32_t vec = svld1(p16, curr_q);
            // Multiply by scale
            vec = svmul_f32_m(p16, vec, vec, scale_vec);
            // Store result
            svst1_m(p16, curr_q_buffer, vec);
          } else if constexpr (std::is_same<scalar_t, c10::Half>::value) {
            // Load half and convert to float32
            const float16_t* h_ptr = reinterpret_cast<const float16_t*>(curr_q);
            svfloat16_t vec_h = svld1(svptrue_b16(), h_ptr);
            svfloat32_t vec_f = svcvt_f32_f16_m(p16, svundef_f32(), vec_h);
            // Multiply by scale
            vec_f = svmul_f32_m(p16, vec_f, vec_f, scale_vec);
            // Store result
            svst1_m(p16, curr_q_buffer, vec_f);
          } else if constexpr (std::is_same<scalar_t, c10::BFloat16>::value) {
            // Load bfloat16 and convert to float32
            const uint16_t* u_ptr = reinterpret_cast<const uint16_t*>(curr_q);
            svuint16_t vec_u = svld1(svptrue_b16(), u_ptr);
            svbfloat16_t vec_bf = svreinterpret_bf16_u16(vec_u);
            svfloat32_t vec_f = svcvt_f32_bf16_m(p16, svundef_f32(), vec_bf);
            // Multiply by scale
            vec_f = svmul_f32_m(p16, vec_f, vec_f, scale_vec);
            // Store result
            svst1_m(p16, curr_q_buffer, vec_f);
          }

          curr_q += 16;
          curr_q_buffer += 16;
        }
      }
    }
  }

  // reshape K as column-major and V as row-major using SVE optimized loops
  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride) {
    // Use OpenMP parallelization similar to NEON implementation
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) {
          // skip
          continue;
        }

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;
        {
          // Write Key
          const scalar_t* key_start_ptr = key + 
                                          token_idx * key_token_num_stride +
                                          head_idx * key_head_num_stride;
          scalar_t* key_cache_start_ptr = 
              key_cache + block_idx * num_blocks_stride + 
              head_idx * cache_head_num_stride + block_offset;

          // Use SVE-friendly loop unrolling for better performance
#pragma GCC unroll 8
          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size) {
            key_cache_start_ptr[j] = key_start_ptr[i];
          }
        }
        {
          // Write Value - use memcpy for contiguous data
          const scalar_t* value_start_ptr = value + 
                                            token_idx * value_token_num_stride +
                                            head_idx * value_head_num_stride;
          scalar_t* value_cache_start_ptr = 
              value_cache + block_idx * num_blocks_stride + 
              head_idx * cache_head_num_stride + block_offset * head_dim;
          std::memcpy(value_cache_start_ptr, value_start_ptr,
                      sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};

}  // namespace cpu_attention

#endif  // #ifndef CPU_ATTN_SVE_HPP
