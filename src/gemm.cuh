#pragma once

#include <cuda_fp16.h>

#include <cstdint>

// Row-major GEMM: C = alpha * A * B + beta * C, with A M x K, B K x N, C M x N.
// Every kernel launcher has one of these signatures; pointers are device
// memory. Tiled kernels need M, N and K to be multiples of 128.
using GemmFn = void (*)(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C);
// FP16 inputs, FP32 accumulation and output (tensor-core kernels).
using GemmHalfFn = void (*)(int M, int N, int K, float alpha, const __half *A,
                            const __half *B, float beta, float *C);
// FP8 (E4M3 bytes) inputs, FP32 accumulation and output. B is passed
// transposed, as Bt (N x K), so both operands are K-contiguous.
using GemmFp8Fn = void (*)(int M, int N, int K, float alpha, const uint8_t *A,
                           const uint8_t *Bt, float beta, float *C);

void gemm_naive(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C);
void gemm_coalesced(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C);
void gemm_smem(int M, int N, int K, float alpha, const float *A, const float *B,
               float beta, float *C);
void gemm_reg2d(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C);
void gemm_vec(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C);
void gemm_wmma(int M, int N, int K, float alpha, const __half *A,
               const __half *B, float beta, float *C);
void gemm_cpasync(int M, int N, int K, float alpha, const __half *A,
                  const __half *B, float beta, float *C);
void gemm_fp8(int M, int N, int K, float alpha, const uint8_t *A,
              const uint8_t *Bt, float beta, float *C);
