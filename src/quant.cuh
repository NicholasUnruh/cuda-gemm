#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// FP32 -> narrow-format quantizers on the GPU. Every one rounds to nearest
// even and saturates, and matches the C++ reference models of the chip-lab
// numerics study bit for bit (check/quant_crosscheck.cu). n must be a multiple
// of 128, and blocks run along consecutive elements, so a row-major matrix
// with K a multiple of 128 gets blocks along K.

// BF16, round to nearest even: the operand format of the BF16 baseline.
void quant_bf16(const float *x, size_t n, __nv_bfloat16 *out);

// *amax = max |x| over n floats, as float bits.
void quant_amax(const float *x, size_t n, unsigned *amax);

// Per-tensor FP8: code = E4M3(x * 448 / amax), the product taken in double as
// the reference does. Dequantize with amax / 448.
void quant_e4m3_tensor(const float *x, size_t n, const unsigned *amax,
                       uint8_t *codes);

// OCP MX: each block of 32 shares an E8M0 scale 2^(floor(log2 amax_block) -
// emax), stored as the biased byte; elements are E4M3 (MXFP8, emax 8, one
// byte each) or E2M1 (MXFP4, emax 2, two per byte, low nibble first).
void quant_mx(const float *x, size_t n, bool fp4, uint8_t *codes,
              uint8_t *scales);

// NVFP4: global scale S = 448 * 6 / amax; each block of 16 stores the E4M3
// scale s = E4M3(amax_block / 6 * S); elements are E2M1(x * S / s) (0 when s
// is 0), two per byte, low nibble first. All in double, as the mx-training
// reference does. Dequantize as code * s / S.
void quant_nvfp4(const float *x, size_t n, const unsigned *amax,
                 uint8_t *codes, uint8_t *scales);

// Decoders for emulating FP4 formats on Ada, which has no FP4 tensor cores.
// MXFP4 codes (packed E2M1) -> the same values as E4M3 bytes, for the MXFP8
// kernel with the MXFP4 scales.
void e2m1_to_e4m3(const uint8_t *packed, size_t n, uint8_t *codes);
// NVFP4 codes and scales -> FP16 values code * s (exact); the GEMM output
// then needs the factor (amax_a / 2688) * (amax_b / 2688).
void nvfp4_to_half(const uint8_t *codes, const uint8_t *scales, size_t n,
                   __half *out);
