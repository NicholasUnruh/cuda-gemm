// Probes how an FP8 tensor-core MMA (mma.sync m16n8k32 E4M3, FP32
// accumulator) adds: one instruction on hand-built operands per experiment,
// D[0][0] printed exactly next to what an FP32 add would give.
// Usage: mma_probe (needs sm_89+). Build: make probe ARCH=sm_89
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// One m16n8k32 MMA. A is 16 x 32 bytes row-major, Bt is 8 x 32 bytes (B
// stored K-major), C and D are 16 x 8 floats. Fragments are gathered with the
// PTX ISA layout formulas directly, no ldmatrix.
__global__ void probe_kernel(const uint8_t *A, const uint8_t *Bt,
                             const float *C, float *D) {
#if __CUDA_ARCH__ >= 890
  const int lane = threadIdx.x, g = lane / 4, q = lane % 4;
  uint32_t a[4], b[2];
  for (int r = 0; r < 4; ++r) {
    int row = g + (r % 2) * 8, col = q * 4 + (r / 2) * 16;
    memcpy(&a[r], &A[row * 32 + col], 4);
  }
  for (int r = 0; r < 2; ++r) memcpy(&b[r], &Bt[g * 32 + q * 4 + r * 16], 4);
  float d[4];
  for (int h = 0; h < 4; ++h) d[h] = C[(g + h / 2 * 8) * 8 + 2 * q + h % 2];
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  for (int h = 0; h < 4; ++h) D[(g + h / 2 * 8) * 8 + 2 * q + h % 2] = d[h];
#endif
}

static uint8_t e4m3(double x) {
  return __nv_cvt_double_to_fp8(x, __NV_NOSAT, __NV_E4M3);
}

// D[0][0] = c + sum_k a[k] * b[k]: row 0 of A and column 0 of B hold the
// factors. Each factor must be exactly representable in E4M3.
static float run(float c, const std::vector<double> &a,
                 const std::vector<double> &b) {
  std::vector<uint8_t> A(16 * 32, 0), Bt(8 * 32, 0);
  std::vector<float> C(16 * 8, 0.f), D(16 * 8);
  for (size_t k = 0; k < a.size(); ++k) {
    A[k] = e4m3(a[k]);
    Bt[k] = e4m3(b[k]);
  }
  C[0] = c;
  uint8_t *dA, *dB;
  float *dC, *dD;
  cudaMalloc(&dA, A.size());
  cudaMalloc(&dB, Bt.size());
  cudaMalloc(&dC, C.size() * 4);
  cudaMalloc(&dD, D.size() * 4);
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bt.data(), Bt.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
  probe_kernel<<<1, 32>>>(dA, dB, dC, dD);
  cudaError_t e = cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(2);
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dD);
  return D[0];
}

int main() {
  printf("experiment,c,products,mma,fp32\n");
  // 1. Accumulator width: 2^e + 1 * 1. FP32 keeps the 1 up to e = 23.
  for (int e = 8; e <= 25; ++e) {
    float c = std::ldexp(1.f, e);
    printf("width,2^%d,1*1,%.9g,%.9g\n", e, run(c, {1}, {1}), c + 1.f);
  }
  // 2. Rounding of dropped bits: 2^e + t for t = 0.25, 0.5, 0.75, 1.5.
  //    Truncation drops all of t below the cut; nearest rounds 0.75 and 1.5
  //    up.
  for (int e = 10; e <= 16; ++e) {
    float c = std::ldexp(1.f, e);
    for (double t : {0.25, 0.5, 0.75, 1.5})
      printf("round,2^%d,%g*1,%.9g,%.9g\n", e, t, run(c, {t}, {1}),
             c + float(t));
  }
  // 3. Negative accumulator: -2^e + 1 (truncation toward zero or down?).
  for (int e = 10; e <= 16; ++e) {
    float c = -std::ldexp(1.f, e);
    for (double t : {0.75, 1.0})
      printf("negative,-2^%d,%g*1,%.9g,%.9g\n", e, t, run(c, {t}, {1}),
             c + float(t));
  }
  // 4. Products among themselves (c = 0): 2^16 + 2^-k, with 2^16 = 256 * 256.
  for (int k = 0; k <= 18; k += 2) {
    double s1 = std::ldexp(1.0, -k / 2), s2 = std::ldexp(1.0, -(k - k / 2));
    printf("products,0,256*256+2^-%d,%.9g,%.9g\n", k,
           run(0.f, {256, s1}, {256, s2}), float(65536.0 + s1 * s2));
  }
  // 5. Many small products: 2^e + 32 x (1 * 1).
  for (int e = 12; e <= 24; e += 2) {
    float c = std::ldexp(1.f, e);
    printf("many,2^%d,32x1*1,%.9g,%.9g\n", e,
           run(c, std::vector<double>(32, 1), std::vector<double>(32, 1)),
           c + 32.f);
  }
  return 0;
}
