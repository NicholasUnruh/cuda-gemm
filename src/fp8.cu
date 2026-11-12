#include <cstdint>

#include "async.cuh"
#include "gemm.cuh"

constexpr int BM = 128, BN = 128, BK = 64;     // block tile of C; K step in bytes
constexpr int WM = 64, WN = 32;                // warp tile of C
constexpr int kWarps = (BM / WM) * (BN / WN);  // 8 warps, 2 x 4
constexpr int kThreads = kWarps * 32;          // 256
constexpr int MI = WM / 16, NI = WN / 8;       // 4 x 4 m16n8 tiles per warp
// Shared rows are 64 data bytes plus 16 of padding: the 8 rows one ldmatrix
// phase reads are then 80 bytes apart and land in 8 different 4-bank groups.
constexpr int LDS = BK + 16;
constexpr int kSteps = BK / 32;  // k32 MMAs per K tile
constexpr int kStages = 2;
constexpr int kGroup = 8;  // block rows per launch group, see grouped_tile

// Four 8x8 matrices of 16-bit elements (8 rows of 16 bytes) from shared
// memory. Lane l supplies the address of row l % 8 of matrix l / 8 and gets
// back, in register i, bytes 4 * (l % 4) .. +3 of row l / 4 of matrix i:
// exactly the 8-bit mma fragment layout, so FP8 tiles load with no shuffling.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *p) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(s));
}

// d += a * b on a 16x8x32 tile: E4M3 inputs, FP32 accumulators (sm_89+).
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  __trap();  // FP8 tensor cores need Ada or later; bench checks first
#endif
}

// FP8 (E4M3) GEMM on tensor cores with per-tensor scales: C = alpha * A Bt^T
// + beta * C, where A is M x K and Bt is N x K, both K-contiguous bytes, and
// alpha carries the two dequantization factors. Step 7's pipeline and block
// order, with WMMA replaced by mma.sync PTX, because WMMA has no FP8
// fragments. 8-bit ldmatrix cannot transpose, so B arrives K-major.
__global__ void __launch_bounds__(kThreads)
    fp8_kernel(int M, int N, int K, float alpha, const uint8_t *A,
               const uint8_t *Bt, float beta, float *C) {
  __shared__ __align__(128) uint8_t As[kStages][BM * LDS];
  __shared__ __align__(128) uint8_t Bs[kStages][BN * LDS];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int warpRow = warp / (BN / WN), warpCol = warp % (BN / WN);
  int blockRow, blockCol;
  grouped_tile(M / BM, N / BN, kGroup, blockRow, blockCol);
  A += size_t(blockRow) * BM * K;
  Bt += size_t(blockCol) * BN * K;
  C += size_t(blockRow) * BM * N + blockCol * BN;

  // K tile t into buffer s: 128 rows x 64 bytes per operand, 512 16-byte
  // chunks each, two per thread per operand.
  auto load_tile = [&](int t, int s) {
    for (int c = tid; c < BM * BK / 16; c += kThreads) {
      int r = c / (BK / 16), col = c % (BK / 16) * 16;
      cp_async16(&As[s][r * LDS + col], &A[size_t(r) * K + t * BK + col]);
      cp_async16(&Bs[s][r * LDS + col], &Bt[size_t(r) * K + t * BK + col]);
    }
    cp_async_commit();
  };

  float acc[MI][NI][4] = {};
  const int tiles = K / BK;
  load_tile(0, 0);
  for (int t = 0; t < tiles; ++t) {
    const int s = t % kStages;
    cp_async_wait<0>();
    __syncthreads();  // tile t visible; every warp is done with tile t-1
    if (t + 1 < tiles) load_tile(t + 1, (t + 1) % kStages);
    uint32_t a[kSteps][MI][4], b[kSteps][NI][2];
    for (int k = 0; k < kSteps; ++k) {
      // A: one x4 load per 16x32 tile. Lanes 0-15 address rows 0-15 at byte
      // 32k, lanes 16-31 the same rows at 32k + 16: registers come back as
      // (rows 0-7, k lo), (rows 8-15, k lo), (rows 0-7, k hi), (8-15, k hi).
      for (int i = 0; i < MI; ++i)
        ldmatrix_x4(a[k][i], &As[s][(warpRow * WM + i * 16 + lane % 16) * LDS +
                                    k * 32 + lane / 16 * 16]);
      // B: one x4 load covers two n8 tiles: (n 0-7, k lo), (n 0-7, k hi),
      // (n 8-15, k lo), (n 8-15, k hi).
      for (int j = 0; j < NI; j += 2) {
        uint32_t r[4];
        ldmatrix_x4(r, &Bs[s][(warpCol * WN + j * 8 + lane % 8 + lane / 16 * 8) *
                                  LDS +
                              k * 32 + lane / 8 % 2 * 16]);
        b[k][j][0] = r[0], b[k][j][1] = r[1];
        b[k][j + 1][0] = r[2], b[k][j + 1][1] = r[3];
      }
    }
    // Promotion: the tensor core keeps only about 13 bits below the largest
    // addend and truncates the rest (check/mma_probe.cu), so a long K chain
    // inside it loses accuracy as the sum grows. Each K tile is summed in a
    // fresh tensor-core accumulator and then added into FP32 registers.
    for (int i = 0; i < MI; ++i)
      for (int j = 0; j < NI; ++j) {
        float p[4] = {};
        for (int k = 0; k < kSteps; ++k) mma_e4m3(p, a[k][i], b[k][j]);
        for (int e = 0; e < 4; ++e) acc[i][j][e] += p[e];
      }
  }

  // Accumulator layout: lane (g = lane / 4, q = lane % 4) holds columns 2q,
  // 2q + 1 of rows g and g + 8 of each 16x8 tile.
  const int g = lane / 4, q = lane % 4;
  for (int i = 0; i < MI; ++i)
    for (int j = 0; j < NI; ++j)
      for (int h = 0; h < 2; ++h) {
        float2 *c = reinterpret_cast<float2 *>(
            &C[size_t(warpRow * WM + i * 16 + g + h * 8) * N + warpCol * WN +
               j * 8 + 2 * q]);
        float2 v = {alpha * acc[i][j][2 * h], alpha * acc[i][j][2 * h + 1]};
        if (beta != 0.f) {
          float2 old = *c;
          v.x += beta * old.x, v.y += beta * old.y;
        }
        *c = v;
      }
}

void gemm_fp8(int M, int N, int K, float alpha, const uint8_t *A,
              const uint8_t *Bt, float beta, float *C) {
  fp8_kernel<<<(M / BM) * (N / BN), kThreads>>>(M, N, K, alpha, A, Bt, beta, C);
}
