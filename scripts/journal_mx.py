# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Key figure of the MX journal entry: per format, GEMM error on chip-lab's
vectors (GPU against chip-lab's bit-exact C++ model) and time per 4096³ GEMM
split into quantizing and multiplying.

Usage: uv run scripts/journal_mx.py <out.png> [light|dark]
Reads docs/results/numerics-rtx4060.csv, docs/results/chiplab-gemm-error.csv and
docs/results/rtx4060.csv (median-of-three GEMM TFLOP/s for the labels).
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import THEMES

# (numerics format, numerics impl, chip-lab format or None, label, rtx4060.csv kernel or None)
FORMATS = [
    ("bf16", "cuBLAS", "bf16", "BF16", None),
    ("e4m3_tensor", "fp8 (step 8)", "e4m3_tensor", "FP8 per tensor", "fp8"),
    ("mxfp8", "mxfp8 (step 9)", "mxfp8", "MXFP8", "mxfp8"),
    ("mxfp4", "mxfp8 kernel on E2M1 values", "mxfp4", "MXFP4*", "mxfp4"),
    ("nvfp4", "FP16 cuBLAS on decoded values", None, "NVFP4*", None),
]


def read(path):
    with open(path) as f:
        return [r for r in csv.reader(ln for ln in f if not ln.startswith("#"))]


def main(dst, theme="light"):
    rows = read("docs/results/numerics-rtx4060.csv")
    err = {(r[0], r[1], r[2], r[3]): float(r[4]) for r in rows if len(r) == 6
           and r[0] != "distribution" and r[1] == "4096"}
    cost = {(r[0], r[1]): r for r in rows if len(r) == 6 and r[0] != "distribution"
            and not r[1].isdigit()}
    chip = {(r[0], r[1]): float(r[2]) for r in read("docs/results/chiplab-gemm-error.csv")[1:]}
    runs = {}
    for r in read("docs/results/rtx4060.csv"):
        if len(r) == 7 and r[1] == "4096":
            runs.setdefault(r[0], []).append(float(r[3]) / 1000)
    median = {k: sorted(v)[len(v) // 2] for k, v in runs.items() if len(v) == 3}

    c = THEMES[theme]
    labels = [f[3] for f in FORMATS]
    y = range(len(FORMATS))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200)
    fig.patch.set_facecolor(c["surface"])

    ax = axes[0]
    gpu = [err[("gaussian", "4096", f[0], f[1])] for f in FORMATS]
    ax.barh(y, gpu, color=c["tc"], height=0.6, label="GPU, measured")
    for i, f in enumerate(FORMATS):
        if f[2]:
            ax.plot(chip[(f[2], "gaussian")], i, "D", color=c["ink"], ms=5,
                    label="chip-lab C++ model" if i == 0 else None)
        right = max(gpu[i], chip[(f[2], "gaussian")] if f[2] else 0)
        ax.text(right * 1.25, i, f"{gpu[i]:.1e}", va="center", fontsize=7.5,
                color=c["ink"])
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 3e-2)
    ax.set_title("Median GEMM error, Gaussian, K = 4096", fontsize=9,
                 color=c["ink"], loc="left")
    leg = ax.legend(frameon=False, fontsize=7, loc="upper right")
    for t in leg.get_texts():
        t.set_color(c["ink"])

    ax = axes[1]
    q = [float(cost[(f[0], f[1])][2]) for f in FORMATS]
    g = [float(cost[(f[0], f[1])][3]) for f in FORMATS]
    ax.barh(y, g, color=c["tc"], height=0.6, label="GEMM")
    ax.barh(y, q, left=g, color=c["muted"], height=0.6, label="quantize both operands")
    for i in y:
        kernel = FORMATS[i][4]
        if kernel:
            text = f"{median[kernel]:.1f} TFLOP/s GEMM, median of 3"
        else:
            text = f"{float(cost[(FORMATS[i][0], FORMATS[i][1])][4]):.1f} TFLOP/s GEMM, one run"
        ax.text(g[i] + q[i] + 0.1, i, text, va="center",
                fontsize=7.5, color=c["ink"])
    ax.set_xlim(0, max(a + b for a, b in zip(g, q)) * 1.6)
    ax.set_title("Milliseconds at 4096³, one run", fontsize=9, color=c["ink"], loc="left")
    leg = ax.legend(frameon=False, fontsize=7, loc="upper right")
    for t in leg.get_texts():
        t.set_color(c["ink"])

    for ax in axes:
        ax.set_facecolor(c["surface"])
        ax.set_yticks(list(y), labels if ax is axes[0] else [])
        ax.set_ylim(len(FORMATS) - 0.4, -1.4)  # top to bottom, legend room on top
        ax.tick_params(colors=c["muted"], labelsize=7.5)
        for s in ax.spines.values():
            s.set_color(c["grid"])
    fig.suptitle(
        "RTX 4060: number formats on one GPU (* emulated: Ada has no FP4 tensor cores)",
        fontsize=10, color=c["ink"], x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(dst, facecolor=c["surface"])


if __name__ == "__main__":
    main(*sys.argv[1:])
