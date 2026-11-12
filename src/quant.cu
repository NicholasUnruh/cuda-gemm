#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "quant.cuh"

static int grid_for(size_t n4) {
  return int(std::min<size_t>((n4 + 255) / 256, 1024));
}

__device__ __forceinline__ uint8_t e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ uint8_t e2m1(float x) {
  return __nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest);
}
// Max of v over aligned groups of `width` lanes (width a power of two <= 32).
__device__ __forceinline__ float group_max(float v, int width) {
  for (int o = width / 2; o > 0; o /= 2)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float abs_max4(float4 v) {
  return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
}

// Grid-stride loops below run whole warps together: n4 = n / 4 is a multiple
// of 32, so the shuffles always have all 32 lanes.

__global__ void amax_kernel(const float4 *x, size_t n4, unsigned *amax) {
  float m = 0.f;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x)
    m = fmaxf(m, abs_max4(x[i]));
  m = group_max(m, 32);
  // Non-negative floats order like their bit patterns.
  if (threadIdx.x % 32 == 0) atomicMax(amax, __float_as_uint(m));
}

__global__ void e4m3_tensor_kernel(const float4 *x, size_t n4,
                                   const unsigned *amax, uchar4 *codes) {
  const float m = __uint_as_float(*amax);
  const double scale = 448.0 / (m > 0.f ? m : 1.0);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 v = x[i];
    auto q = [&](float e) {
      return __nv_cvt_double_to_fp8(e * scale, __NV_SATFINITE, __NV_E4M3);
    };
    codes[i] = make_uchar4(q(v.x), q(v.y), q(v.z), q(v.w));
  }
}

// Each thread converts 4 elements; 8 lanes share one 32-element block.
template <bool kFp4>
__global__ void mx_kernel(const float4 *x, size_t n4, uint8_t *codes,
                          uint8_t *scales) {
  constexpr int kEmax = kFp4 ? 2 : 8;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 v = x[i];
    float m = group_max(abs_max4(v), 8);
    // floor(log2 m) is the exponent field; ilogbf is exact, also for
    // subnormals.
    int shared = m > 0.f ? ilogbf(m) - kEmax : -127;
    shared = max(-127, min(127, shared));
    if (i % 8 == 0) scales[i / 8] = uint8_t(shared + 127);
    // Scaling by a power of two is exact for every value that does not round
    // to zero in the element format.
    float a = scalbnf(v.x, -shared), b = scalbnf(v.y, -shared),
          c = scalbnf(v.z, -shared), d = scalbnf(v.w, -shared);
    if (kFp4)
      reinterpret_cast<uint16_t *>(codes)[i] = uint16_t(
          e2m1(a) | e2m1(b) << 4 | e2m1(c) << 8 | e2m1(d) << 12);
    else
      reinterpret_cast<uchar4 *>(codes)[i] =
          make_uchar4(e4m3(a), e4m3(b), e4m3(c), e4m3(d));
  }
}

// Each thread converts 4 elements; 4 lanes share one 16-element block.
__global__ void nvfp4_kernel(const float4 *x, size_t n4, const unsigned *amax,
                             uint8_t *codes, uint8_t *scales) {
  const float m = __uint_as_float(*amax);
  const double encode = 448.0 * 6.0 / (m > 0.f ? m : 1.0);
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 v = x[i];
    double bm = group_max(abs_max4(v), 4);
    uint8_t s = __nv_cvt_double_to_fp8(bm / 6.0 * encode, __NV_SATFINITE,
                                       __NV_E4M3);
    if (i % 4 == 0) scales[i / 4] = s;
    double divide = __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
    auto q = [&](float e) -> unsigned {
      return divide == 0.0 ? 0u
                           : __nv_cvt_double_to_fp4(e * encode / divide,
                                                    __NV_E2M1, cudaRoundNearest);
    };
    reinterpret_cast<uint16_t *>(codes)[i] =
        uint16_t(q(v.x) | q(v.y) << 4 | q(v.z) << 8 | q(v.w) << 12);
  }
}

__global__ void bf16_kernel(const float4 *x, size_t n4, __nv_bfloat162 *out) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 v = x[i];
    out[2 * i] = __floats2bfloat162_rn(v.x, v.y);
    out[2 * i + 1] = __floats2bfloat162_rn(v.z, v.w);
  }
}

void quant_bf16(const float *x, size_t n, __nv_bfloat16 *out) {
  bf16_kernel<<<grid_for(n / 4), 256>>>(reinterpret_cast<const float4 *>(x),
                                        n / 4,
                                        reinterpret_cast<__nv_bfloat162 *>(out));
}

// Each E2M1 value is also an E4M3 value, so this re-encoding is exact.
__global__ void e2m1_to_e4m3_kernel(const uint16_t *packed, size_t n4,
                                    uchar4 *codes) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned p = packed[i];
    auto c = [&](int k) {
      __half h = __nv_cvt_fp4_to_halfraw(p >> 4 * k & 0xf, __NV_E2M1);
      return e4m3(__half2float(h));
    };
    codes[i] = make_uchar4(c(0), c(1), c(2), c(3));
  }
}

// An E2M1 element times its E4M3 block scale has at most 6 significant bits
// and lies within 2^-10 .. 2688, so the FP16 product is exact.
__global__ void nvfp4_to_half_kernel(const uint16_t *packed,
                                     const uint8_t *scales, size_t n4,
                                     __half2 *out) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n4;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned p = packed[i];
    float s = __half2float(__nv_cvt_fp8_to_halfraw(scales[i / 4], __NV_E4M3));
    auto v = [&](int k) {
      return s * __half2float(__nv_cvt_fp4_to_halfraw(p >> 4 * k & 0xf,
                                                      __NV_E2M1));
    };
    out[2 * i] = __floats2half2_rn(v(0), v(1));
    out[2 * i + 1] = __floats2half2_rn(v(2), v(3));
  }
}

void e2m1_to_e4m3(const uint8_t *packed, size_t n, uint8_t *codes) {
  e2m1_to_e4m3_kernel<<<grid_for(n / 4), 256>>>(
      reinterpret_cast<const uint16_t *>(packed), n / 4,
      reinterpret_cast<uchar4 *>(codes));
}

void nvfp4_to_half(const uint8_t *codes, const uint8_t *scales, size_t n,
                   __half *out) {
  nvfp4_to_half_kernel<<<grid_for(n / 4), 256>>>(
      reinterpret_cast<const uint16_t *>(codes), scales, n / 4,
      reinterpret_cast<__half2 *>(out));
}

void quant_amax(const float *x, size_t n, unsigned *amax) {
  cudaMemsetAsync(amax, 0, sizeof(unsigned));
  amax_kernel<<<grid_for(n / 4), 256>>>(reinterpret_cast<const float4 *>(x),
                                        n / 4, amax);
}

void quant_e4m3_tensor(const float *x, size_t n, const unsigned *amax,
                       uint8_t *codes) {
  e4m3_tensor_kernel<<<grid_for(n / 4), 256>>>(
      reinterpret_cast<const float4 *>(x), n / 4, amax,
      reinterpret_cast<uchar4 *>(codes));
}

void quant_mx(const float *x, size_t n, bool fp4, uint8_t *codes,
              uint8_t *scales) {
  auto *x4 = reinterpret_cast<const float4 *>(x);
  if (fp4)
    mx_kernel<true><<<grid_for(n / 4), 256>>>(x4, n / 4, codes, scales);
  else
    mx_kernel<false><<<grid_for(n / 4), 256>>>(x4, n / 4, codes, scales);
}

void quant_nvfp4(const float *x, size_t n, const unsigned *amax,
                 uint8_t *codes, uint8_t *scales) {
  nvfp4_kernel<<<grid_for(n / 4), 256>>>(reinterpret_cast<const float4 *>(x),
                                         n / 4, amax, codes, scales);
}
