#pragma once

// Row-major SGEMM: C = alpha * A * B + beta * C, with A M x K, B K x N, C M x N.
// Every kernel launcher has this signature; pointers are device memory.
// Tiled kernels need M, N and K to be multiples of 128.
using GemmFn = void (*)(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C);

void gemm_naive(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C);
void gemm_coalesced(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C);
void gemm_smem(int M, int N, int K, float alpha, const float *A,
               const float *B, float beta, float *C);
void gemm_reg2d(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C);
