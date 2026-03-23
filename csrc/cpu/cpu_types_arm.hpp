#include <arm_neon.h>
#include <torch/all.h>
#include <cmath>
#include <limits>

#if defined(__APPLE__)
  #include "omp.h"
#endif

namespace vec_op {

#ifdef ARM_BF16_SUPPORT
  #define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#else
  #define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)
#endif

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
  #define CPU_KERNEL_GUARD_IN(NAME)
  #define CPU_KERNEL_GUARD_OUT(NAME)
#else
  #define CPU_KERNEL_GUARD_IN(NAME) \
    std::cout << #NAME << " invoked." << std::endl;
  #define CPU_KERNEL_GUARD_OUT(NAME) \
    std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline
// Number of elements in single ASIMD vector of given Datatype
#define NUM_ELEMENTS_REG(vec) (sizeof(vec) / sizeof(vec[0]))

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
};
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; };
};

struct FP32Vec8;
struct FP32Vec16;

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  float16x8_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(vld1q_f16(static_cast<const __fp16*>(ptr))) {};

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const { vst1q_f16(static_cast<__fp16*>(ptr), reg); }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  float16x8x2_t reg;

  explicit FP16Vec16(const void* ptr) {
    reg.val[0] = vld1q_f16(reinterpret_cast<const __fp16*>(ptr));
    reg.val[1] = vld1q_f16(reinterpret_cast<const __fp16*>(ptr) + 8);
  }

  explicit FP16Vec16(const FP32Vec16& vec);

  void save(void* ptr) const {
    vst1q_f16(reinterpret_cast<__fp16*>(ptr), reg.val[0]);
    vst1q_f16(reinterpret_cast<__fp16*>(ptr) + 8, reg.val[1]);
  }

  void save(void* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);

    if (full_blocks > 0) {
      vst1q_f16(reinterpret_cast<__fp16*>(ptr), reg.val[0]);
      if (full_blocks > 1) {
        vst1q_f16(reinterpret_cast<__fp16*>(ptr) + 8, reg.val[1]);
      }
    }

    // Note: below is the unrolled version of the following code:
    //
    // for (int i = 0; i < remainder; ++i) {
    //     reinterpret_cast<__fp16*>(ptr)[full_blocks * 8 + i] =
    //          vgetq_lane_f16(temp, i);
    // }
    //
    // For macOS build (Clang), the arm/neon intrinsics function
    // `vgetq_lane_f16` needs the parameter `i` to be constant at compile
    // time.

    if (remainder > 0) {
      float16x8_t temp = reg.val[full_blocks];
      __fp16* fp16_ptr = reinterpret_cast<__fp16*>(ptr);
      switch (remainder) {
        case 1:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          break;
        case 2:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          break;
        case 3:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          break;
        case 4:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          break;
        case 5:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          fp16_ptr[full_blocks * 8 + 4] = vgetq_lane_f16(temp, 4);
          break;
        case 6:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          fp16_ptr[full_blocks * 8 + 4] = vgetq_lane_f16(temp, 4);
          fp16_ptr[full_blocks * 8 + 5] = vgetq_lane_f16(temp, 5);
          break;
        case 7:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          fp16_ptr[full_blocks * 8 + 4] = vgetq_lane_f16(temp, 4);
          fp16_ptr[full_blocks * 8 + 5] = vgetq_lane_f16(temp, 5);
          fp16_ptr[full_blocks * 8 + 6] = vgetq_lane_f16(temp, 6);
          break;

        default:
          break;
      }
    }
  }
};

#ifdef ARM_BF16_SUPPORT
struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  bfloat16x8_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(*reinterpret_cast<const bfloat16x8_t*>(ptr)) {};

  explicit BF16Vec8(bfloat16x8_t data) : reg(data) {};

  explicit BF16Vec8(const FP32Vec8&);

  explicit BF16Vec8(float32x4x2_t v)
      : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1])) {};

  void save(void* ptr) const { *reinterpret_cast<bfloat16x8_t*>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  bfloat16x8x2_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(*reinterpret_cast<const bfloat16x8x2_t*>(ptr)) {};

  explicit BF16Vec16(bfloat16x8x2_t data) : reg(data) {};

  explicit BF16Vec16(const FP32Vec16&);

  explicit BF16Vec16(float32x4x4_t v)
      : reg({vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1]),
             vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[2]), v.val[3])}) {};

  void save(void* ptr) const { *reinterpret_cast<bfloat16x8x2_t*>(ptr) = reg; };
  void save(void* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    for (int i = 0; i < full_blocks; i++)
      vst1q_bf16(
          reinterpret_cast<__bf16*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);
    if (remainder > 0) {
      bfloat16x8_t temp = reg.val[full_blocks];
      bfloat16_t* base = reinterpret_cast<bfloat16_t*>(ptr) + full_blocks * 8;
      if (remainder > 0) base[0] = vgetq_lane_bf16(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_bf16(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_bf16(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_bf16(temp, 3);
      if (remainder > 4) base[4] = vgetq_lane_bf16(temp, 4);
      if (remainder > 5) base[5] = vgetq_lane_bf16(temp, 5);
      if (remainder > 6) base[6] = vgetq_lane_bf16(temp, 6);
    }
  };
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  bfloat16x8x4_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(*reinterpret_cast<const bfloat16x8x4_t*>(ptr)) {};

  explicit BF16Vec32(bfloat16x8x4_t data) : reg(data) {};

  explicit BF16Vec32(const BF16Vec8& vec8_data)
      : reg({vec8_data.reg, vec8_data.reg, vec8_data.reg, vec8_data.reg}) {};

  void save(void* ptr) const { *reinterpret_cast<bfloat16x8x4_t*>(ptr) = reg; };
  void save(void* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    for (int i = 0; i < full_blocks; i++)
      vst1q_bf16(
          reinterpret_cast<__bf16*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);
    if (remainder > 0) {
      bfloat16x8_t temp = reg.val[full_blocks];
      bfloat16_t* base = reinterpret_cast<bfloat16_t*>(ptr) + full_blocks * 8;
      base[0] = vgetq_lane_bf16(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_bf16(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_bf16(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_bf16(temp, 3);
      if (remainder > 4) base[4] = vgetq_lane_bf16(temp, 4);
      if (remainder > 5) base[5] = vgetq_lane_bf16(temp, 5);
      if (remainder > 6) base[6] = vgetq_lane_bf16(temp, 6);
    }
  };
};
#endif

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  union AliasReg {
    float32x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4_t reg;

  explicit FP32Vec4(float v) : reg(vdupq_n_f32(v)) {};

  explicit FP32Vec4() : reg(vdupq_n_f32(0.0f)) {};

  explicit FP32Vec4(const float* ptr) : reg(vld1q_f32(ptr)) {};

  explicit FP32Vec4(float32x4_t data) : reg(data) {};

  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {};
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    float32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x2_t reg;

  explicit FP32Vec8(float v) : reg({vmovq_n_f32(v), vmovq_n_f32(v)}) {};

  explicit FP32Vec8() : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0)}) {};

  explicit FP32Vec8(const float* ptr)
      : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4)}) {};

  explicit FP32Vec8(float32x4x2_t data) : reg(data) {};

  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {};

  explicit FP32Vec8(const FP16Vec8& v) {
    reg.val[0] = vcvt_f32_f16(vget_low_f16(v.reg));
    reg.val[1] = vcvt_f32_f16(vget_high_f16(v.reg));
  };

  explicit FP32Vec8(float16x8_t v)
      : reg({vcvt_f32_f16(vget_low_f16(v)), vcvt_f32_f16(vget_high_f16(v))}) {};

#ifdef ARM_BF16_SUPPORT

  explicit FP32Vec8(bfloat16x8_t v)
      : reg({vcvtq_low_f32_bf16(v), vcvtq_high_f32_bf16(v)}) {};

  explicit FP32Vec8(const BF16Vec8& v)
      : reg({vcvtq_low_f32_bf16(v.reg), vcvtq_high_f32_bf16(v.reg)}) {};

#endif

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&answer, &ar](int i) { answer += ar.values[i]; });

    return answer;
  }

  FP32Vec8 exp() const {
    float32x4_t res0, res1;
    
    // Check if we have ARMv8.4-A+ exponential instructions
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FMA)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t ln2 = vdupq_n_f32(0x1.62e42fefa39efp-1f);
    const float32x4_t inv_ln2 = vdupq_n_f32(0x1.71547652b82fep+0f);
    const float32x4_t c0 = vdupq_n_f32(0x1.0p+0f);
    const float32x4_t c1 = vdupq_n_f32(0x1.0p+0f);
    const float32x4_t c2 = vdupq_n_f32(0x1.5555555555555p-3f);
    const float32x4_t c3 = vdupq_n_f32(0x1.1111111111111p-7f);
    const float32x4_t c4 = vdupq_n_f32(0x1.6c16c16c16c16p-13f);
    const float32x4_t c5 = vdupq_n_f32(0x1.a01a01a01a01ap-19f);
    const float32x4_t c6 = vdupq_n_f32(0x1.a01a01a01a01ap-25f);
    const float32x4_t pos_special_bound = vdupq_n_f32(0x1.5d5e2ap+6f);
    const float32x4_t neg_special_bound = vnegq_f32(pos_special_bound);
    const float32x4_t inf = vdupq_n_f32(std::numeric_limits<float>::infinity());
    const float32x4_t zero = vdupq_n_f32(0.0f);
    
    // Process first 4 elements
    float32x4_t x = reg.val[0];
    float32x4_t abs_x = vabsq_f32(x);
    
    // Check for special cases
    uint32x4_t hi_mask = vcgeq_f32(x, pos_special_bound);
    uint32x4_t lo_mask = vcleq_f32(x, neg_special_bound);
    
    // Normal case: compute exp(x)
    float32x4_t n = vrndaq_f32(vmulq_f32(x, inv_ln2));
    float32x4_t r = vfmsq_f32(x, n, ln2);
    
    // Polynomial approximation: exp(r) = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*c6)))))
    float32x4_t p = vfmaq_f32(c5, c6, r);
    p = vfmaq_f32(c4, p, r);
    p = vfmaq_f32(c3, p, r);
    p = vfmaq_f32(c2, p, r);
    p = vfmaq_f32(c1, p, r);
    p = vfmaq_f32(c0, p, r);
    
    // Scale the result: exp(x) = 2^n * exp(r)
    int32x4_t n_int = vcvtq_s32_f32(n);
    float32x4_t scale = vreinterpretq_f32_s32(vaddq_s32(vshlq_n_s32(n_int, 23), vdupq_n_s32(0x3f800000)));
    res0 = vmulq_f32(p, scale);
    
    // Handle special cases
    res0 = vbslq_f32(hi_mask, inf, res0);
    res0 = vbslq_f32(lo_mask, zero, res0);
    
    // Process second 4 elements
    x = reg.val[1];
    abs_x = vabsq_f32(x);
    hi_mask = vcgeq_f32(x, pos_special_bound);
    lo_mask = vcleq_f32(x, neg_special_bound);
    
    n = vrndaq_f32(vmulq_f32(x, inv_ln2));
    r = vfmsq_f32(x, n, ln2);
    
    p = vfmaq_f32(c5, c6, r);
    p = vfmaq_f32(c4, p, r);
    p = vfmaq_f32(c3, p, r);
    p = vfmaq_f32(c2, p, r);
    p = vfmaq_f32(c1, p, r);
    p = vfmaq_f32(c0, p, r);
    
    n_int = vcvtq_s32_f32(n);
    scale = vreinterpretq_f32_s32(vaddq_s32(vshlq_n_s32(n_int, 23), vdupq_n_s32(0x3f800000)));
    res1 = vmulq_f32(p, scale);
    
    res1 = vbslq_f32(hi_mask, inf, res1);
    res1 = vbslq_f32(lo_mask, zero, res1);
#else
    // Fallback to original implementation for older ARM architectures
    // Implementation copied from Arm Optimized Routines (expf AdvSIMD)
    const float32x4_t inv_ln2 = vdupq_n_f32(0x1.715476p+0f);
    const float ln2_hi = 0x1.62e4p-1f;
    const float ln2_lo = 0x1.7f7d1cp-20f;
    const float c0 = 0x1.0e4020p-7f;
    const float c2 = 0x1.555e66p-3f;
    const float32x4_t ln2_c02 = {ln2_hi, ln2_lo, c0, c2};
    const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);
    const float32x4_t c1 = vdupq_n_f32(0x1.573e2ep-5f);
    const float32x4_t c3 = vdupq_n_f32(0x1.fffdb6p-2f);
    const float32x4_t c4 = vdupq_n_f32(0x1.ffffecp-1f);
    const float32x4_t pos_special_bound = vdupq_n_f32(0x1.5d5e2ap+6f);
    const float32x4_t neg_special_bound = vnegq_f32(pos_special_bound);
    const float32x4_t inf = vdupq_n_f32(std::numeric_limits<float>::infinity());
    const float32x4_t zero = vdupq_n_f32(0.0f);
    
    // Process first 4 elements
    float32x4_t values = reg.val[0];
    float32x4_t n = vrndaq_f32(vmulq_f32(values, inv_ln2));
    float32x4_t r = vfmsq_laneq_f32(values, n, ln2_c02, 0);
    r = vfmsq_laneq_f32(r, n, ln2_c02, 1);
    uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23);
    float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t p = vfmaq_laneq_f32(c1, r, ln2_c02, 2);
    float32x4_t q = vfmaq_laneq_f32(c3, r, ln2_c02, 3);
    q = vfmaq_f32(q, p, r2);
    p = vmulq_f32(c4, r);
    res0 = vfmaq_f32(p, q, r2);
    res0 = vfmaq_f32(scale, res0, scale);
    const uint32x4_t hi_mask = vcgeq_f32(values, pos_special_bound);
    const uint32x4_t lo_mask = vcleq_f32(values, neg_special_bound);
    res0 = vbslq_f32(hi_mask, inf, res0);
    res0 = vbslq_f32(lo_mask, zero, res0);
    
    // Process second 4 elements
    values = reg.val[1];
    n = vrndaq_f32(vmulq_f32(values, inv_ln2));
    r = vfmsq_laneq_f32(values, n, ln2_c02, 0);
    r = vfmsq_laneq_f32(r, n, ln2_c02, 1);
    e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23);
    scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));
    r2 = vmulq_f32(r, r);
    p = vfmaq_laneq_f32(c1, r, ln2_c02, 2);
    q = vfmaq_laneq_f32(c3, r, ln2_c02, 3);
    q = vfmaq_f32(q, p, r2);
    p = vmulq_f32(c4, r);
    res1 = vfmaq_f32(p, q, r2);
    res1 = vfmaq_f32(scale, res1, scale);
    const uint32x4_t hi_mask2 = vcgeq_f32(values, pos_special_bound);
    const uint32x4_t lo_mask2 = vcleq_f32(values, neg_special_bound);
    res1 = vbslq_f32(hi_mask2, inf, res1);
    res1 = vbslq_f32(lo_mask2, zero, res1);
#endif
    
    return FP32Vec8(float32x4x2_t({res0, res1}));
  }

  FP32Vec8 tanh() const {
    float32x4_t res0, res1;
    
    // Check if we have ARMv8.4-A+ vector instructions with FMA support
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FMA)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t three = vdupq_n_f32(3.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t sign_mask = vdupq_n_f32(-0.0f);
    const float32x4_t large_val = vdupq_n_f32(0x1.0a2b20p+3f); // ~10.0
    
    // Process first 4 elements
    float32x4_t x = reg.val[0];
    float32x4_t sign = vandq_f32(x, sign_mask);
    float32x4_t ax = vabsq_f32(x);
    
    // Check for large inputs where tanh(x) ≈ ±1
    uint32x4_t large_mask = vcgtq_f32(ax, large_val);
    
    // For small inputs, use polynomial approximation
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t x6 = vmulq_f32(x4, x2);
    float32x4_t x8 = vmulq_f32(x6, x2);
    
    // Coefficients for tanh(x) = x * (1 - x²/3 + 2x⁴/15 - 17x⁶/315 + 62x⁸/2835)
    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(-1.0f/3.0f);
    const float32x4_t c2 = vdupq_n_f32(2.0f/15.0f);
    const float32x4_t c3 = vdupq_n_f32(-17.0f/315.0f);
    const float32x4_t c4 = vdupq_n_f32(62.0f/2835.0f);
    
    // Polynomial evaluation: x * (c0 + x²*(c1 + x²*(c2 + x²*(c3 + x²*c4))))
    float32x4_t p = vfmaq_f32(c3, c4, x2);
    p = vfmaq_f32(c2, p, x2);
    p = vfmaq_f32(c1, p, x2);
    p = vfmaq_f32(c0, p, x2);
    p = vmulq_f32(p, x);
    
    // Handle large inputs: tanh(x) ≈ sign(x)
    float32x4_t tanhx = vbslq_f32(sign, vnegq_f32(one), one);
    tanhx = vbslq_f32(large_mask, tanhx, p);
    res0 = tanhx;
    
    // Process second 4 elements
    x = reg.val[1];
    sign = vandq_f32(x, sign_mask);
    ax = vabsq_f32(x);
    
    large_mask = vcgtq_f32(ax, large_val);
    
    x2 = vmulq_f32(x, x);
    x4 = vmulq_f32(x2, x2);
    x6 = vmulq_f32(x4, x2);
    x8 = vmulq_f32(x6, x2);
    
    p = vfmaq_f32(c3, c4, x2);
    p = vfmaq_f32(c2, p, x2);
    p = vfmaq_f32(c1, p, x2);
    p = vfmaq_f32(c0, p, x2);
    p = vmulq_f32(p, x);
    
    tanhx = vbslq_f32(sign, vnegq_f32(one), one);
    tanhx = vbslq_f32(large_mask, tanhx, p);
    res1 = tanhx;
#else
    // Fallback to original implementation for older ARM architectures
    // Implementation based on Arm Optimized Routines (tanhf AdvSIMD)
    const float32x4_t coeff_a1 = vdupq_n_f32(0x1.62e400p-1f);
    const float32x4_t coeff_a3 = vdupq_n_f32(0x1.172b84p-3f);
    const float32x4_t coeff_a5 = vdupq_n_f32(0x1.55c428p-5f);
    const float32x4_t coeff_a7 = vdupq_n_f32(0x1.573516p-7f);
    const float32x4_t coeff_a9 = vdupq_n_f32(0x1.05c610p-9f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t sign_mask = vdupq_n_f32(-0.0f);
    
    // Process first 4 elements
    float32x4_t x = reg.val[0];
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t sign = vandq_f32(x, sign_mask);
    float32x4_t ax = vabsq_f32(x);
    
    float32x4_t t = ax;
    float32x4_t p = coeff_a9;
    p = vfmaq_f32(p, coeff_a7, t);
    p = vfmaq_f32(p, coeff_a5, t);
    p = vfmaq_f32(p, coeff_a3, t);
    p = vfmaq_f32(p, coeff_a1, t);
    p = vmulq_f32(p, t);
    p = vmulq_f32(p, x2);
    
    float32x4_t e2x = vaddq_f32(two, vfmaq_f32(two, p, p));
    float32x4_t tanhx = vdivq_f32(p, e2x);
    tanhx = vaddq_f32(tanhx, one);
    tanhx = vbslq_f32(vcgtq_f32(ax, vdupq_n_f32(0x1.0a2b20p+3f)), one, tanhx);
    tanhx = vbslq_f32(sign, vnegq_f32(tanhx), tanhx);
    res0 = tanhx;
    
    // Process second 4 elements
    x = reg.val[1];
    x2 = vmulq_f32(x, x);
    sign = vandq_f32(x, sign_mask);
    ax = vabsq_f32(x);
    
    t = ax;
    p = coeff_a9;
    p = vfmaq_f32(p, coeff_a7, t);
    p = vfmaq_f32(p, coeff_a5, t);
    p = vfmaq_f32(p, coeff_a3, t);
    p = vfmaq_f32(p, coeff_a1, t);
    p = vmulq_f32(p, t);
    p = vmulq_f32(p, x2);
    
    e2x = vaddq_f32(two, vfmaq_f32(two, p, p));
    tanhx = vdivq_f32(p, e2x);
    tanhx = vaddq_f32(tanhx, one);
    tanhx = vbslq_f32(vcgtq_f32(ax, vdupq_n_f32(0x1.0a2b20p+3f)), one, tanhx);
    tanhx = vbslq_f32(sign, vnegq_f32(tanhx), tanhx);
    res1 = tanhx;
#endif
    
    return FP32Vec8(float32x4x2_t({res0, res1}));
  }

  FP32Vec8 er() const {
    float32x4_t res0, res1;
    
    // Check if we have ARMv8.4-A+ vector instructions with FMA support
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__ARM_FEATURE_FMA)
    // Implementation based on polynomial approximation of erf(x)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two_over_sqrt_pi = vdupq_n_f32(1.1283791670955125739f);
    const float32x4_t sign_mask = vdupq_n_f32(-0.0f);
    
    // Process first 4 elements
    float32x4_t x = reg.val[0];
    float32x4_t sign = vandq_f32(x, sign_mask);
    float32x4_t ax = vabsq_f32(x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t exp_neg_x2 = vexpq_f32(vnegq_f32(x2));
    
    // Polynomial coefficients for erf(x) approximation
    const float32x4_t c0 = vdupq_n_f32(0.0f);
    const float32x4_t c1 = vdupq_n_f32(0.99999999999980993227684700473478f);
    const float32x4_t c2 = vdupq_n_f32(-0.33333333333331391525489567742371f);
    const float32x4_t c3 = vdupq_n_f32(0.16666666666666701904553832439885f);
    const float32x4_t c4 = vdupq_n_f32(-0.074999999999999991118215802998746f);
    const float32x4_t c5 = vdupq_n_f32(0.031250000000000005551115123125783f);
    
    // Polynomial evaluation: t = x * (c1 + x2*(c2 + x2*(c3 + x2*(c4 + x2*c5))))
    float32x4_t t = vfmaq_f32(c4, c5, x2);
    t = vfmaq_f32(c3, t, x2);
    t = vfmaq_f32(c2, t, x2);
    t = vfmaq_f32(c1, t, x2);
    t = vmulq_f32(t, x);
    
    // erf(x) ≈ 2/√π * exp(-x²) * t
    float32x4_t erf_x = vmulq_f32(two_over_sqrt_pi, exp_neg_x2);
    erf_x = vmulq_f32(erf_x, t);
    
    // Apply sign
    erf_x = vbslq_f32(sign, vnegq_f32(erf_x), erf_x);
    res0 = erf_x;
    
    // Process second 4 elements
    x = reg.val[1];
    sign = vandq_f32(x, sign_mask);
    ax = vabsq_f32(x);
    x2 = vmulq_f32(x, x);
    exp_neg_x2 = vexpq_f32(vnegq_f32(x2));
    
    t = vfmaq_f32(c4, c5, x2);
    t = vfmaq_f32(c3, t, x2);
    t = vfmaq_f32(c2, t, x2);
    t = vfmaq_f32(c1, t, x2);
    t = vmulq_f32(t, x);
    
    erf_x = vmulq_f32(two_over_sqrt_pi, exp_neg_x2);
    erf_x = vmulq_f32(erf_x, t);
    erf_x = vbslq_f32(sign, vnegq_f32(erf_x), erf_x);
    res1 = erf_x;
#else
    // Fallback to original implementation for older ARM architectures
    AliasReg ar;
    ar.reg = reg;

    float32x2_t er_vec0 = {static_cast<float32_t>(erf(ar.values[0])),
                           static_cast<float32_t>(erf(ar.values[1]))};
    float32x2_t er_vec1 = {static_cast<float32_t>(erf(ar.values[2])),
                           static_cast<float32_t>(erf(ar.values[3]))};
    float32x2_t er_vec2 = {static_cast<float32_t>(erf(ar.values[4])),
                           static_cast<float32_t>(erf(ar.values[5]))};
    float32x2_t er_vec3 = {static_cast<float32_t>(erf(ar.values[6])),
                           static_cast<float32_t>(erf(ar.values[7]))};

    float32x4_t result0 = vcombine_f32(er_vec0, er_vec1);
    float32x4_t result1 = vcombine_f32(er_vec2, er_vec3);

    res0 = result0;
    res1 = result1;
#endif
    
    return FP32Vec8(float32x4x2_t({res0, res1}));
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vmulq_f32(reg.val[0], b.reg.val[0]),
                                   vmulq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vaddq_f32(reg.val[0], b.reg.val[0]),
                                   vaddq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vsubq_f32(reg.val[0], b.reg.val[0]),
                                   vsubq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vdivq_f32(reg.val[0], b.reg.val[0]),
                                   vdivq_f32(reg.val[1], b.reg.val[1])}));
  }

  void save(float* ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
  }
};

struct INT32Vec16 : public Vec<INT32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    int32x4x4_t reg;
    int32_t values[VEC_ELEM_NUM];
  };
  int32x4x4_t reg;

  explicit INT32Vec16(const void* ptr) {
    reg.val[0] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr));
    reg.val[1] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr) + 4);
    reg.val[2] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr) + 8);
    reg.val[3] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr) + 12);
  }

  void save(int32_t* ptr) const {
    vst1q_s32(ptr, reg.val[0]);
    vst1q_s32(ptr + 4, reg.val[1]);
    vst1q_s32(ptr + 8, reg.val[2]);
    vst1q_s32(ptr + 12, reg.val[3]);
  };

  void save(int32_t* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);

    for (int i = 0; i < full_blocks; i++)
      vst1q_s32(
          reinterpret_cast<__int32_t*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);

    if (remainder > 0) {
      int32x4_t temp = reg.val[full_blocks];
      int32_t* base = reinterpret_cast<int32_t*>(ptr) + full_blocks * 4;
      if (remainder > 0) base[0] = vgetq_lane_s32(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_s32(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_s32(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_s32(temp, 3);
    }
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    float32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x4_t reg;

  explicit FP32Vec16(float v)
      : reg({vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v)}) {}

  explicit FP32Vec16()
      : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0), vmovq_n_f32(0.0),
             vmovq_n_f32(0.0)}) {}

  explicit FP32Vec16(const float* ptr)
      : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4), vld1q_f32(ptr + 8),
             vld1q_f32(ptr + 12)}) {}

  explicit FP32Vec16(float32x4x4_t data) : reg(data) {}

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[1];
  }

  explicit FP32Vec16(const FP32Vec16& data) : reg(data.reg) {}

  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v.reg)) {}

#ifdef ARM_BF16_SUPPORT
  explicit FP32Vec16(bfloat16x8x2_t v)
      : reg({vcvtq_low_f32_bf16(v.val[0]), vcvtq_high_f32_bf16(v.val[0]),
             vcvtq_low_f32_bf16(v.val[1]), vcvtq_high_f32_bf16(v.val[1])}) {};
#endif

  explicit FP32Vec16(const FP32Vec4& data) {
    reg.val[0] = data.reg;
    reg.val[1] = data.reg;
    reg.val[2] = data.reg;
    reg.val[3] = data.reg;
  };

#ifdef ARM_BF16_SUPPORT
  explicit FP32Vec16(const BF16Vec16& v)
      : reg({vcvtq_low_f32_bf16(v.reg.val[0]),
             vcvtq_high_f32_bf16(v.reg.val[0]),
             vcvtq_low_f32_bf16(v.reg.val[1]),
             vcvtq_high_f32_bf16(v.reg.val[1])}) {};

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};
#endif

  explicit FP32Vec16(const FP16Vec16& v) {
    reg.val[0] = vcvt_f32_f16(vget_low_f16(v.reg.val[0]));
    reg.val[1] = vcvt_f32_f16(vget_high_f16(v.reg.val[0]));
    reg.val[2] = vcvt_f32_f16(vget_low_f16(v.reg.val[1]));
    reg.val[3] = vcvt_f32_f16(vget_high_f16(v.reg.val[1]));
  };
  explicit FP32Vec16(const INT32Vec16& v) {
    reg.val[0] = vcvtq_f32_s32(v.reg.val[0]);
    reg.val[1] = vcvtq_f32_s32(v.reg.val[1]);
    reg.val[2] = vcvtq_f32_s32(v.reg.val[2]);
    reg.val[3] = vcvtq_f32_s32(v.reg.val[3]);
  };
  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vaddq_f32(reg.val[0], b.reg.val[0]),
                                    vaddq_f32(reg.val[1], b.reg.val[1]),
                                    vaddq_f32(reg.val[2], b.reg.val[2]),
                                    vaddq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vmulq_f32(reg.val[0], b.reg.val[0]),
                                    vmulq_f32(reg.val[1], b.reg.val[1]),
                                    vmulq_f32(reg.val[2], b.reg.val[2]),
                                    vmulq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vsubq_f32(reg.val[0], b.reg.val[0]),
                                    vsubq_f32(reg.val[1], b.reg.val[1]),
                                    vsubq_f32(reg.val[2], b.reg.val[2]),
                                    vsubq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vdivq_f32(reg.val[0], b.reg.val[0]),
                                    vdivq_f32(reg.val[1], b.reg.val[1]),
                                    vdivq_f32(reg.val[2], b.reg.val[2]),
                                    vdivq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(float32x4x4_t(
        {vminq_f32(max.reg.val[0], vmaxq_f32(min.reg.val[0], reg.val[0])),
         vminq_f32(max.reg.val[1], vmaxq_f32(min.reg.val[1], reg.val[1])),
         vminq_f32(max.reg.val[2], vmaxq_f32(min.reg.val[2], reg.val[2])),
         vminq_f32(max.reg.val[3], vmaxq_f32(min.reg.val[3], reg.val[3]))}));
  };

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vmaxq_f32(b.reg.val[0], reg.val[0]),
                                    vmaxq_f32(b.reg.val[1], reg.val[1]),
                                    vmaxq_f32(b.reg.val[2], reg.val[2]),
                                    vmaxq_f32(b.reg.val[3], reg.val[3])}));
  };

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    float32x4x4_t temp;

    for (int i = 0; i < full_blocks; i++)
      temp.val[i] = vmaxq_f32(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 0),
                             vgetq_lane_f32(b.reg.val[full_blocks], 0));
      temp.val[full_blocks] = vsetq_lane_f32(max_v, temp.val[full_blocks], 0);
    }
    if (remainder > 1) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 1),
                             vgetq_lane_f32(b.reg.val[full_blocks], 1));
      temp.val[full_blocks] = vsetq_lane_f32(max_v, temp.val[full_blocks], 1);
    }
    if (remainder > 2) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 2),
                             vgetq_lane_f32(b.reg.val[full_blocks], 2));
      temp.val[full_blocks] = vsetq_lane_f32(max_v, temp.val[full_blocks], 2);
    }
    return FP32Vec16(temp);
  };

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({
        vminq_f32(b.reg.val[0], reg.val[0]),
        vminq_f32(b.reg.val[1], reg.val[1]),
        vminq_f32(b.reg.val[2], reg.val[2]),
        vminq_f32(b.reg.val[3], reg.val[3]),
    }));
  };
  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    const int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    float32x4x4_t temp;
    for (int i = 0; i < full_blocks; i++)
      temp.val[i] = vminq_f32(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 0),
                             vgetq_lane_f32(b.reg.val[full_blocks], 0));
      temp.val[full_blocks] = vsetq_lane_f32(min_v, temp.val[full_blocks], 0);
    }
    if (remainder > 1) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 1),
                             vgetq_lane_f32(b.reg.val[full_blocks], 1));
      temp.val[full_blocks] = vsetq_lane_f32(min_v, temp.val[full_blocks], 1);
    }
    if (remainder > 2) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 2),
                             vgetq_lane_f32(b.reg.val[full_blocks], 2));
      temp.val[full_blocks] = vsetq_lane_f32(min_v, temp.val[full_blocks], 2);
    }

    return FP32Vec16(temp);
  };
  FP32Vec16 abs() const {
    return FP32Vec16(
        float32x4x4_t({vabsq_f32(reg.val[0]), vabsq_f32(reg.val[1]),
                       vabsq_f32(reg.val[2]), vabsq_f32(reg.val[3])}));
  }
  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&answer, &ar](int i) { answer += ar.values[i]; });

    return answer;
  };

  float reduce_max() const {
    AliasReg ar;
    ar.reg = reg;
    float max_v = std::numeric_limits<float>::lowest();
    unroll_loop<int, VEC_ELEM_NUM>(
        [&max_v, &ar](int i) { max_v = std::max(max_v, ar.values[i]); });
    return max_v;
  }

  float reduce_min() const {
    AliasReg ar;
    ar.reg = reg;
    float min_v = std::numeric_limits<float>::max();
    unroll_loop<int, VEC_ELEM_NUM>(
        [&min_v, &ar](int i) { min_v = std::min(min_v, ar.values[i]); });
    return min_v;
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&answer, &start, ar](int i) { answer += ar.values[start + i]; });

    return answer;
  };

  void save(float* ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
    vst1q_f32(ptr + 8, reg.val[2]);
    vst1q_f32(ptr + 12, reg.val[3]);
  };

  void save(float* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);

    for (int i = 0; i < full_blocks; i++)
      vst1q_f32(
          reinterpret_cast<float32_t*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);

    if (remainder > 0) {
      float32x4_t temp = reg.val[full_blocks];
      float* base = reinterpret_cast<float32_t*>(ptr) +
                    full_blocks * NUM_ELEMENTS_REG(reg.val[0]);
      if (remainder > 0) base[0] = vgetq_lane_f32(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_f32(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_f32(temp, 2);
    }
  }
};

struct INT8Vec16 : public Vec<INT8Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    int8x16_t reg;
    int8_t values[VEC_ELEM_NUM];
  };
  int8x16_t reg;

  explicit INT8Vec16(const FP32Vec16& vec) {
    // Convert each 128-bit float32 vector to int32
    int32x4_t part0 =
        vcvtq_s32_f32(vec.reg.val[0]);  // Convert first 128-bit block
    int32x4_t part1 =
        vcvtq_s32_f32(vec.reg.val[1]);  // Convert second 128-bit block
    int32x4_t part2 =
        vcvtq_s32_f32(vec.reg.val[2]);  // Convert third 128-bit block
    int32x4_t part3 =
        vcvtq_s32_f32(vec.reg.val[3]);  // Convert fourth 128-bit block

    // Narrow each 32-bit vector to 8 bits and combine
    int8x8_t lower =
        vqmovn_s16(vcombine_s16(vqmovn_s32(part0), vqmovn_s32(part1)));
    int8x8_t upper =
        vqmovn_s16(vcombine_s16(vqmovn_s32(part2), vqmovn_s32(part3)));
    reg = vcombine_s8(lower, upper);  // Combine to form a single 128-bit vector
  }

  void save(int8_t* ptr) const { vst1q_s8(ptr, reg); };

  void save(int8_t* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg);

    for (int i = 0; i < full_blocks; i++)
      vst1q_s8(reinterpret_cast<int8_t*>(ptr) + NUM_ELEMENTS_REG(reg) * i, reg);
    if (remainder > 0) {
      int8x16_t temp = reg;
      int8_t* base =
          reinterpret_cast<int8_t*>(ptr) + full_blocks * NUM_ELEMENTS_REG(reg);
      if (remainder > 0) base[0] = vgetq_lane_s8(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_s8(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_s8(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_s8(temp, 3);
      if (remainder > 4) base[4] = vgetq_lane_s8(temp, 4);
      if (remainder > 5) base[5] = vgetq_lane_s8(temp, 5);
      if (remainder > 6) base[6] = vgetq_lane_s8(temp, 6);
      if (remainder > 7) base[7] = vgetq_lane_s8(temp, 7);
      if (remainder > 8) base[8] = vgetq_lane_s8(temp, 8);
      if (remainder > 9) base[9] = vgetq_lane_s8(temp, 9);
      if (remainder > 10) base[10] = vgetq_lane_s8(temp, 10);
      if (remainder > 11) base[11] = vgetq_lane_s8(temp, 11);
      if (remainder > 12) base[12] = vgetq_lane_s8(temp, 12);
      if (remainder > 13) base[13] = vgetq_lane_s8(temp, 13);
      if (remainder > 14) base[14] = vgetq_lane_s8(temp, 14);
    }
  };
};

template <typename T>
struct VecType {
  using vec_type = void;
};

template <typename T>
using vec_t = typename VecType<T>::vec_type;

template <>
struct VecType<float> {
  using vec_type = FP32Vec8;
};

template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec8;
};

#ifdef ARM_BF16_SUPPORT
template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
};
#endif

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}

template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  *reinterpret_cast<__fp16*>(ptr) = v;
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  float16x4_t low_0 = vcvt_f16_f32(v.reg.val[0]);
  float16x4_t high_0 = vcvt_f16_f32(v.reg.val[1]);
  float16x4_t low_1 = vcvt_f16_f32(v.reg.val[2]);
  float16x4_t high_1 = vcvt_f16_f32(v.reg.val[3]);

  reg.val[0] = vcombine_f16(low_0, high_0);
  reg.val[1] = vcombine_f16(low_1, high_1);
};

inline FP16Vec8 ::FP16Vec8(const FP32Vec8& v) {
  float16x4_t lower_half = vcvt_f16_f32(v.reg.val[0]);
  float16x4_t upper_half = vcvt_f16_f32(v.reg.val[1]);

  reg = vcombine_f16(lower_half, upper_half);
};

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a.reg.val[0], b.reg.val[0]);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a.reg.val[1], b.reg.val[1]);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a.reg.val[2], b.reg.val[2]);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a.reg.val[3], b.reg.val[3]);
};

#ifdef ARM_BF16_SUPPORT
inline void fma(FP32Vec16& acc, BF16Vec32& a, BF16Vec32& b) {
  float32x4_t a0_low = vcvt_f32_bf16(vget_low_bf16(a.reg.val[0]));
  float32x4_t a0_high = vcvt_f32_bf16(vget_high_bf16(a.reg.val[0]));
  float32x4_t a1_low = vcvt_f32_bf16(vget_low_bf16(a.reg.val[1]));
  float32x4_t a1_high = vcvt_f32_bf16(vget_high_bf16(a.reg.val[1]));

  float32x4_t b0_low = vcvt_f32_bf16(vget_low_bf16(b.reg.val[0]));
  float32x4_t b0_high = vcvt_f32_bf16(vget_high_bf16(b.reg.val[0]));
  float32x4_t b1_low = vcvt_f32_bf16(vget_low_bf16(b.reg.val[1]));
  float32x4_t b1_high = vcvt_f32_bf16(vget_high_bf16(b.reg.val[1]));

  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a0_low, b0_low);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a0_high, b0_high);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a1_low, b1_low);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a1_high, b1_high);
};
#endif

#ifdef ARM_BF16_SUPPORT
inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[0]), v.reg.val[1])) {
      };

inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : reg({vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[0]), v.reg.val[1]),
           vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[2]),
                               v.reg.val[3])}) {};
#endif

inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 1); };

#ifdef ARM_BF16_SUPPORT
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *reinterpret_cast<__bf16*>(ptr) = vcvth_bf16_f32(v);
};
#endif
};  // namespace vec_op