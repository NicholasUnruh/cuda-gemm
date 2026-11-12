# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Key figure of the FP8 journal entry: where the FP8 kernel adds its tensor-core
partial sums into FP32 ("promotion"), against error and throughput.

Usage: uv run scripts/journal_fp8.py <out.png> [light|dark]
Reads docs/results/fp8-promotion.csv.
"""

import csv
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import THEMES

VARIANTS = [  # (csv key, label)
    ("none", "never (one long\ntensor-core chain)"),
    ("64", "every 64-deep K tile\n(committed)"),
    ("32", "after every MMA"),
]


def main(dst, theme="light"):
    with open("docs/results/fp8-promotion.csv") as f:
        rows = list(csv.DictReader(ln for ln in f if not ln.startswith("#")))
    sizes = sorted({int(r["n"]) for r in rows})

    def med(key, n, col):
        return statistics.median(
            float(r[col]) for r in rows if r["promotion"] == key and int(r["n"]) == n
        )

    c = THEMES[theme]
    shades = [c["muted"], c["tc"], c["fp32"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax = axes[0]
    for (key, label), color in zip(VARIANTS, shades):
        ax.plot(sizes, [med(key, n, "exact_err") for n in sizes], "o-", color=color,
                label=label.replace("\n", " "), lw=1.8, ms=4)
    ax.plot(sizes, [med("64", n, "cublaslt_exact_err") for n in sizes], "s--",
            color=c["ink"], label="cuBLASLt FP8", lw=1.2, ms=3.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlabel("n = K", color=c["ink"], fontsize=8)
    ax.set_title("Max error vs exact FP64 dot products", fontsize=9, color=c["ink"],
                 loc="left")
    leg = ax.legend(frameon=False, fontsize=7)
    for t in leg.get_texts():
        t.set_color(c["ink"])

    ax = axes[1]
    labels = [label for _, label in VARIANTS] + ["cuBLASLt FP8"]
    vals = [med(key, 4096, "gflops") / 1e3 for key, _ in VARIANTS]
    vals.append(med("64", 4096, "cublaslt_gflops") / 1e3)  # same runs as "64"
    bars = ax.barh(labels[::-1], vals[::-1], color=(shades + [c["ink"]])[::-1],
                   height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {v:.1f}",
                va="center", fontsize=8, color=c["ink"])
    ax.set_xlim(0, max(vals) * 1.25)
    ax.set_xticks([])
    ax.set_title("TFLOP/s at 4096³ (median of three)", fontsize=9, color=c["ink"],
                 loc="left")
    for a in axes:
        a.set_facecolor(c["surface"])
        a.tick_params(colors=c["muted"], labelsize=7.5)
        for s in a.spines.values():
            s.set_color(c["grid"])
    for s in axes[1].spines.values():
        s.set_visible(False)
    fig.suptitle(
        "RTX 4060, FP8 (E4M3) GEMM: where partial sums leave the tensor core",
        fontsize=10, color=c["ink"], x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(dst, facecolor=c["surface"])


if __name__ == "__main__":
    main(*sys.argv[1:])
