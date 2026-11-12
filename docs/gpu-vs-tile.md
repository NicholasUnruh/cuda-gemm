# GPU vs custom tile

The [openroad-chip-lab](https://github.com/NicholasUnruh/openroad-chip-lab) numerics study designs one dot-product unit per number format and ranks them by throughput per mm². This page puts the same formats on a real GPU.

**What kind of number each is.** GPU figures are **measured** on an RTX 4060 ([rtx4060.csv](results/rtx4060.csv), medians of three at 4096³). chip-lab figures are **predictive**: routed layouts in ASAP7, a predictive 7 nm PDK, never foundry numbers. Each side is compared only as ratios to its own 16-bit baseline.

## Throughput relative to 16-bit

| Format | GPU TFLOP/s | GPU ratio | chip-lab unit | chip-lab GMAC/s/mm² ratio |
|---|---:|---:|---|---:|
| 16-bit | 30.2 (cuBLAS FP16) | 1.00 | `dot8_bf16` | 1.00 |
| FP8 E4M3, per tensor | 57.8 (step 8) | 1.91 | `dot8_e4m3` | 1.53 |
| MXFP8 | 37.9 (step 9) | 1.25 | `dot32_mxfp8` | 1.90 |
| MXFP4 | 39.1 (emulated) | 1.29 | `dot32_mxfp4` | 5.70 |

The GPU baseline is cuBLAS FP16 with FP32 accumulation, the reference column of the `cpasync` rows. Ada's whitepaper gives BF16 and FP16 the same tensor rate. MXFP4 runs the MXFP8 kernel, so its 3% lead is run-to-run spread. chip-lab source: its `docs/numerics-results.md`.

## What the comparison shows

1. **Per-tensor FP8 pays off on both.** cuBLASLt FP8 runs at about 2× cuBLAS FP16 (61 against 30 TFLOP/s). chip-lab's FP8 and BF16 dot8 units both do 8 MACs per clock, so its 1.53× comes from less area per MAC (117 against 166 µm²) and a higher fmax.
2. **MX block scaling is cheap in silicon and expensive in software.** chip-lab's MXFP8 unit adds one exponent per 32 MACs. Ada has no block-scaled MMA, so every 32-deep partial sum is rescaled in FP32 registers, and the best format per mm² becomes slower than per-tensor FP8. Blackwell's block-scaled MMA (`.block_scale` in PTX) is the hardware answer; it is not tested here.
3. **FP4 needs FP4 hardware.** chip-lab predicts 5.70× for MXFP4. On Ada, E2M1 values still run on FP8 tensor cores, so FP4 costs the MXFP8 rate and saves only storage. Its error matches chip-lab's model: 2.60e-3 median at K = 4096 (Gaussian), against 2.6e-3. In chip-lab's training study, MXFP4 fails the 1% loss gate.
4. **Accumulators are narrower than the study assumed.** chip-lab sweeps BF16 alignment frames from 24 to 40 bits and finds 24 enough. The GPU's FP8 adder keeps about 14 bits ([step 8](steps/8-fp8.md)); FP32 promotion every 64-deep K tile makes it accurate. A custom FP8 unit can do the same.

## Accuracy is the same on both

On chip-lab's 2,000 test vectors, GPU error matches chip-lab's bit-exact C++ models to two digits for BF16, per-tensor FP8, MXFP8 and MXFP4 ([step 9](steps/9-mx.md)).
## Not compared

- **Energy.** Board power reads N/A through `nvidia-smi` on this RTX 4060 (Windows and WSL2), so GPU energy per MAC is not measured. chip-lab's energies come from zero-delay simulation and are low by an unknown factor.
- **Absolute density.** GPU area per tensor core is not public.
