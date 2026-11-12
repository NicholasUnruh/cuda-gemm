# GPU vs custom tile

The [openroad-chip-lab](https://github.com/NicholasUnruh/openroad-chip-lab) numerics study designs one dot-product unit per number format and ranks them by throughput per mm² and energy per MAC. This page puts the same formats on a real GPU.

**What kind of number each is.** GPU figures are **measured** on an RTX 4060 (this repository, `docs/results/`). chip-lab figures are **predictive**: routed ASAP7 layouts, a predictive 7 nm PDK, so they compare design points with each other and are never foundry numbers. The two are compared only as ratios to each side's own BF16 baseline.

## Throughput relative to BF16

| Format | GPU TFLOP/s, 4096³ | GPU × BF16 | chip-lab unit | chip-lab GMAC/s/mm² × BF16 |
|---|---:|---:|---|---:|
| BF16 | 28.2 (cuBLAS) | 1.00 | `dot8_bf16` | 1.00 |
| FP8 E4M3, per tensor | 57.8 (step 8) | 2.05 | `dot8_e4m3` | 1.64 |
| MXFP8 | MXFP8_TF (step 9) | MXFP8_X | `dot32_mxfp8` | 2.17 |
| MXFP4 | MXFP4_TF (emulated) | MXFP4_X | `dot32_mxfp4` | 5.86 |

GPU sources: [numerics](results/numerics-rtx4060.csv) (BF16), [rtx4060.csv](results/rtx4060.csv) (FP8, MXFP8, MXFP4, medians of three). chip-lab source: its `docs/numerics-results.md`.

## What the comparison shows

1. **Per-tensor FP8 pays off on both.** On the GPU, cuBLASLt FP8 runs at about 2× cuBLAS FP16 (61 against 30 TFLOP/s), and the step-8 kernel gets 2.05× BF16. chip-lab's FP8 and BF16 dot8 units both do 8 MACs per clock, so its 1.64× comes from smaller area per MAC (117 against 165 µm²) and a higher fmax.
2. **MX block scaling is cheap in silicon and expensive in software.** chip-lab's best point is the 32-term MXFP8 unit: one exponent add per 32 MACs, folded into the alignment. Ada has no block-scaled MMA, so each 32-deep partial sum leaves the tensor core and is rescaled in FP32. That turns MXFP8 from the best format per mm² into one slower than per-tensor FP8. Blackwell's PTX adds block-scaled MMA (`.block_scale`), the hardware answer to this cost; it is not tested here.
3. **FP4 needs FP4 hardware.** chip-lab predicts 5.86× for an MXFP4 unit. On Ada, FP4 data still runs on FP8 tensor cores (E2M1 values re-encoded as E4M3), so it costs the MXFP8 rate and buys only storage. Its accuracy matches chip-lab's model: MXFP4_ERR median error at K = 4096 (Gaussian), against 2.6e-3. In chip-lab's training study, MXFP4 fails the 1% loss gate.
4. **Accumulators are narrower than the study assumed.** chip-lab sweeps BF16 alignment frames from 24 to 40 bits and finds 24 enough. The GPU's FP8 adder keeps about 14 bits ([step 8](steps/8-fp8.md)), and FP32 promotion every 64-deep K tile is what makes it accurate. A custom FP8 unit can use a narrow frame if the design budgets for promotion.

## Accuracy is the same on both

On chip-lab's 2,000 test vectors, the GPU's error matches chip-lab's bit-exact C++ models to two digits for BF16, per-tensor FP8, MXFP8 and MXFP4 ([step 9](steps/9-mx.md)). The formats' error comes from input rounding, which both sides model identically. The GPU's accumulator matters only without promotion.

## Not compared

- **Energy.** Board power reads N/A through `nvidia-smi` on this RTX 4060 (Windows and WSL2), so GPU energy per MAC is not measured. chip-lab's energies come from zero-delay simulation and are themselves low by an unknown factor.
- **Absolute density.** GPU area per tensor core is not public. Only ratios are compared.
