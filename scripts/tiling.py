# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Tiling of every step: parameters cited to the source, checked against ncu.

Usage: uv run scripts/tiling.py <out.png|.webp> [light|dark] [tiling.json]
Run from the repo root. Each parameter carries the file:line it comes from,
and the script fails if that line no longer contains the cited text. The
global-load requests and sectors each tiling implies are recomputed and
compared with docs/results/ncu-rtx4060.csv; GFLOPS and % cuBLAS are the
medians of the three 4096 runs in docs/results/rtx4060.csv.
"""

import csv
import json
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plotstyle import THEMES
from roofline import load

N = 4096  # M = N = K for every profiled and README run

# block: threads (x, y); tile: block's C tile (BM, BN) and K step BK;
# thread_tile: C elements per thread (rows, cols); warp0: C region of warp 0
# (row0, col0, rows, cols); cite: parameter -> (file, line, text on that line).
# loads_per_kstep: per thread per K step; "global_ld" are global load
# instructions per warp per K step with sectors each (from the address
# pattern); "epilogue" the same for the read of C before it is scaled.
STEPS = [
    dict(
        step=1,
        kernel="naive",
        doc="README.md",
        block=(32, 32),
        tile=(32, 32, 1),
        thread_tile=(1, 1),
        warp0=(0, 0, 32, 1),
        vector_bytes=4,
        mapping="row = blockIdx.x*32 + threadIdx.x, col = blockIdx.y*32 + threadIdx.y: "
        "a warp computes 32 rows of one column",
        loads_per_kstep="1 float of A and 1 of B, global to registers; no shared memory",
        global_ld=[
            ("A[row*K+k]: 32 rows, stride K", 32),
            ("B[k*N+col]: one address", 1),
        ],
        epilogue=[("C[row*N+col]", 1, 32)],
        cite=dict(
            block=("src/naive.cu", 18, "dim3 block(32, 32)"),
            mapping=(
                "src/naive.cu",
                8,
                "int row = blockIdx.x * blockDim.x + threadIdx.x",
            ),
            mapping_col=(
                "src/naive.cu",
                9,
                "int col = blockIdx.y * blockDim.y + threadIdx.y",
            ),
            loads=("src/naive.cu", 12, "A[row * K + k] * B[k * N + col]"),
            epilogue=("src/naive.cu", 13, "beta * C[row * N + col]"),
        ),
    ),
    dict(
        step=2,
        kernel="coalesced",
        doc="docs/steps/2-coalesced.md",
        block=(32, 32),
        tile=(32, 32, 1),
        thread_tile=(1, 1),
        warp0=(0, 0, 1, 32),
        vector_bytes=4,
        mapping="col = blockIdx.x*32 + threadIdx.x, row = blockIdx.y*32 + threadIdx.y: "
        "a warp computes 32 neighbouring columns of one row",
        loads_per_kstep="1 float of A and 1 of B, global to registers; no shared memory",
        global_ld=[
            ("A[row*K+k]: one address, broadcast", 1),
            ("B[k*N+col]: 32 floats in a row", 4),
        ],
        epilogue=[("C[row*N+col]", 1, 4)],
        cite=dict(
            block=("src/coalesced.cu", 21, "dim3 block(32, 32)"),
            mapping=(
                "src/coalesced.cu",
                11,
                "int col = blockIdx.x * blockDim.x + threadIdx.x",
            ),
            mapping_row=(
                "src/coalesced.cu",
                12,
                "int row = blockIdx.y * blockDim.y + threadIdx.y",
            ),
            loads=("src/coalesced.cu", 15, "A[row * K + k] * B[k * N + col]"),
            doc_table=("docs/steps/2-coalesced.md", 15, "32 rows, 32 sectors"),
        ),
    ),
    dict(
        step=3,
        kernel="smem",
        doc="docs/steps/3-smem.md",
        block=(32, 32),
        tile=(32, 32, 32),
        thread_tile=(1, 1),
        warp0=(0, 0, 1, 32),
        vector_bytes=4,
        mapping="row = blockIdx.y*32 + ty, col = blockIdx.x*32 + tx (as coalesced)",
        loads_per_kstep="global to shared: 1 float of A (As[ty][tx]) and 1 of B per thread; "
        "shared to registers: 32 k steps x (As[ty][k] broadcast + Bs[k][tx]), 32 FMAs",
        global_ld=[("A tile row: 32 floats", 4), ("B tile row: 32 floats", 4)],
        epilogue=[("C[row*N+col]", 1, 4)],
        cite=dict(
            tile=("src/smem.cu", 3, "constexpr int kTile = 32"),
            smem=("src/smem.cu", 12, "__shared__ float As[kTile][kTile]"),
            mapping=("src/smem.cu", 15, "int row = blockIdx.y * kTile + ty"),
            gmem_to_smem=("src/smem.cu", 19, "As[ty][tx] = A[row * K + k0 + tx]"),
            gmem_to_smem_b=("src/smem.cu", 20, "Bs[ty][tx] = B[(k0 + ty) * N + col]"),
            smem_to_reg=("src/smem.cu", 24, "acc += As[ty][k] * Bs[k][tx]"),
            block=("src/smem.cu", 32, "dim3 block(kTile, kTile)"),
        ),
    ),
    dict(
        step=4,
        kernel="reg2d",
        doc="docs/steps/4-reg2d.md",
        block=(256, 1),
        tile=(128, 128, 8),
        thread_tile=(8, 8),
        warp0=(0, 0, 16, 128),
        vector_bytes=4,
        mapping="threadRow = tid/16, threadCol = tid%16; thread owns C rows "
        "threadRow*8..+7, cols threadCol*8..+7 of the block tile",
        loads_per_kstep="global to shared: 4 floats of A (4 passes of 32 rows) and 4 of B "
        "(4 passes of 2 rows) per thread; shared to registers per k: 8 A + 8 B values "
        "(regM, regN), then 64 FMAs",
        global_ld=[("A tile, 4 passes: 4 rows x 8 floats per warp", 4)] * 4
        + [("B tile, 4 passes: 32 floats of one row per warp", 4)] * 4,
        epilogue=[
            ("C, 64 scalar loads per thread: 16 threads x 32 B stride, 2 rows", 64, 32)
        ],
        cite=dict(
            tile=("src/reg2d.cu", 3, "constexpr int BM = 128, BN = 128, BK = 8"),
            thread_tile=("src/reg2d.cu", 4, "constexpr int TM = 8, TN = 8"),
            block=("src/reg2d.cu", 5, "constexpr int kThreads = (BM / TM) * (BN / TN)"),
            mapping=("src/reg2d.cu", 17, "const int threadRow = tid / (BN / TN)"),
            load_map=("src/reg2d.cu", 25, "const int aRow = tid / BK, aCol = tid % BK"),
            gmem_to_smem=(
                "src/reg2d.cu",
                33,
                "As[(aRow + r) * BK + aCol] = A[(aRow + r) * K + aCol]",
            ),
            gmem_to_smem_b=(
                "src/reg2d.cu",
                35,
                "Bs[(bRow + r) * BN + bCol] = B[(bRow + r) * N + bCol]",
            ),
            smem_to_reg=(
                "src/reg2d.cu",
                40,
                "regM[i] = As[(threadRow * TM + i) * BK + k]",
            ),
            fma=("src/reg2d.cu", 43, "acc[i][j] += regM[i] * regN[j]"),
            epilogue=(
                "src/reg2d.cu",
                49,
                "float &c = C[(threadRow * TM + i) * N + threadCol * TN + j]",
            ),
            sass=("docs/steps/4-reg2d.md", 13, "512 `FFMA` and 32 `LDS.128`"),
        ),
    ),
    dict(
        step=5,
        kernel="vec",
        doc="docs/steps/5-vec.md",
        block=(256, 1),
        tile=(128, 128, 8),
        thread_tile=(8, 8),
        warp0=(0, 0, 16, 128),
        vector_bytes=16,
        mapping="as reg2d: threadRow = tid/16, threadCol = tid%16, 8x8 outputs per thread",
        loads_per_kstep="global to shared: 1 float4 of A (stored transposed, 4 scalar "
        "stores) and 1 float4 of B per thread; shared to registers per k: 2 float4 of A + "
        "2 float4 of B, then 64 FMAs",
        global_ld=[
            ("A float4: 16 rows x 32 B per warp", 16),
            ("B float4: one 512 B row per warp", 16),
        ],
        epilogue=[("C, 16 float4 loads per thread: 16 B at 32 B stride", 16, 32)],
        cite=dict(
            tile=("src/vec.cu", 3, "constexpr int BM = 128, BN = 128, BK = 8"),
            thread_tile=("src/vec.cu", 4, "constexpr int TM = 8, TN = 8"),
            block=("src/vec.cu", 5, "constexpr int kThreads = (BM / TM) * (BN / TN)"),
            mapping=("src/vec.cu", 20, "const int threadRow = tid / (BN / TN)"),
            load_map=(
                "src/vec.cu",
                28,
                "const int aRow = tid / (BK / 4), aCol = tid % (BK / 4)",
            ),
            gmem_a=(
                "src/vec.cu",
                34,
                "float4 a = reinterpret_cast<const float4 *>(&A[aRow * K])[aCol]",
            ),
            transpose=("src/vec.cu", 35, "As[(aCol * 4 + 0) * BM + aRow] = a.x"),
            gmem_b=("src/vec.cu", 39, "Bs4[bRow * (BN / 4) + bCol] ="),
            smem_to_reg=(
                "src/vec.cu",
                46,
                "float4 m = As4[(k * BM + threadRow * TM + i) / 4]",
            ),
            epilogue=("src/vec.cu", 62, "float4 c = *p"),
        ),
    ),
    dict(
        step=6,
        kernel="wmma",
        doc="docs/steps/6-wmma.md",
        block=(256, 1),
        tile=(128, 128, 32),
        thread_tile=None,
        warp0=(0, 0, 64, 32),
        vector_bytes=16,
        warp_tile=(64, 32),
        fragments=(4, 2),
        fragment_shape=(16, 16, 16),
        smem_pad_halves=8,
        mapping="warp = tid/32; warpRow = warp/4, warpCol = warp%4; warp owns C rows "
        "warpRow*64..+63, cols warpCol*32..+31 as 4x2 16x16 accumulator fragments "
        "(element-to-thread layout is opaque in WMMA)",
        loads_per_kstep="global to shared: 2 uint4 (8 halves) of A and 2 of B per thread; "
        "shared to fragments per 16 of K: 4 A + 2 B fragments per warp (ldmatrix), "
        "8 mma_sync",
        global_ld=[("A uint4: 8 rows x 64 B per warp", 16)] * 2
        + [("B uint4: 2 rows x 256 B per warp", 16)] * 2,
        epilogue=None,  # load_matrix_sync from global: instruction count is the compiler's
        cite=dict(
            tile=("src/wmma.cu", 7, "constexpr int BM = 128, BN = 128, BK = 32"),
            warp_tile=("src/wmma.cu", 8, "constexpr int WM = 64, WN = 32"),
            warps=("src/wmma.cu", 9, "constexpr int kWarps = (BM / WM) * (BN / WN)"),
            block=("src/wmma.cu", 10, "constexpr int kThreads = kWarps * 32"),
            fragments=("src/wmma.cu", 11, "constexpr int FM = WM / 16, FN = WN / 16"),
            pad=("src/wmma.cu", 15, "constexpr int kPad = 8"),
            mapping=(
                "src/wmma.cu",
                30,
                "const int warpRow = warp / (BN / WN), warpCol = warp % (BN / WN)",
            ),
            gmem_a=(
                "src/wmma.cu",
                41,
                "for (int c = tid; c < BM * BK / 8; c += kThreads)",
            ),
            gmem_b=(
                "src/wmma.cu",
                46,
                "for (int c = tid; c < BK * BN / 8; c += kThreads)",
            ),
            frag_a=(
                "src/wmma.cu",
                58,
                "wmma::load_matrix_sync(a[i], &As[(warpRow * WM + i * 16) * LDA + kk]",
            ),
            mma=("src/wmma.cu", 65, "wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j])"),
            sass=("docs/steps/6-wmma.md", 13, "32 `HMMA` per K step"),
        ),
    ),
]


def check_cites(step):
    for key, (path, line, text) in step["cite"].items():
        with open(path) as f:
            src = f.read().splitlines()[line - 1]
        if text not in src:
            sys.exit(f"{step['kernel']}.{key}: {path}:{line} lacks {text!r}: {src!r}")


def predict(step):
    """Global load requests and sectors implied by the tiling at N = 4096."""
    bm, bn, bk = step["tile"]
    warps = N * N // (bm * bn) * (step["block"][0] * step["block"][1] // 32)
    ksteps = N // bk
    req = warps * ksteps * len(step["global_ld"])
    sect = warps * ksteps * sum(s for _, s in step["global_ld"])
    if step["epilogue"]:
        req += warps * sum(n for _, n, _ in step["epilogue"])
        sect += warps * sum(n * s for _, n, s in step["epilogue"])
    return warps, ksteps, req, sect


def medians(path):
    runs = {}
    with open(path) as f:
        for r in csv.DictReader(ln for ln in f if not ln.startswith("#")):
            if int(r["n"]) == N:
                runs.setdefault(r["kernel"], []).append(r)
    return {
        k: {
            c: statistics.median(float(r[c]) for r in rs)
            for c in ("gflops", "pct_cublas")
        }
        for k, rs in runs.items()
    }


def draw(ax, s, c):
    """One step: B tile on top, A tile on the left, block C tile, warp 0 and
    thread 0 outputs highlighted. One unit = one element, same scale for all."""
    bm, bn, bk = s["tile"]
    gap = 6
    ax.add_patch(Rectangle((0, 0), bn, bm, fc=c["grid"], ec=c["muted"], lw=0.8))
    ax.add_patch(Rectangle((-gap - bk, 0), bk, bm, fc=c["fp32"], ec="none", alpha=0.55))
    ax.add_patch(Rectangle((0, -gap - bk), bn, bk, fc=c["fp32"], ec="none", alpha=0.55))
    r0, c0, wr, wc = s["warp0"]
    ax.add_patch(Rectangle((c0, r0), wc, wr, fc=c["tc"], ec="none", alpha=0.45))
    tr, tcol = s["thread_tile"] or (16, 16)  # wmma: one 16x16 fragment
    ax.add_patch(Rectangle((c0, r0), tcol, tr, fc=c["tc"], ec="none"))
    fs = 7.5
    ax.text(
        -gap - bk - 2,
        bm / 2,
        f"A {bm}x{bk}",
        rotation=90,
        ha="right",
        va="center",
        fontsize=fs,
        color=c["muted"],
    )
    ax.text(
        bn / 2,
        -gap - bk - 2,
        f"B {bk}x{bn}",
        ha="center",
        va="bottom",
        fontsize=fs,
        color=c["muted"],
    )
    ax.text(bn + 3, 0, f"C tile {bm}x{bn}", va="top", fontsize=fs, color=c["muted"])
    unit = "fragment 16x16" if s["thread_tile"] is None else f"thread {tr}x{tcol}"
    ax.text(
        bn + 3, 12, f"warp 0: {wr}x{wc}\n{unit}", va="top", fontsize=fs, color=c["tc"]
    )
    bx, by = s["block"]
    ax.set_title(
        f"{s['step']}. {s['kernel']}   {s['gflops']:,.1f} GFLOPS\n"
        f"block {bx}x{by} threads, K step {bk}, {s['vector_bytes'] * 8}-bit loads",
        loc="left",
        fontsize=8.5,
        color=c["ink"],
    )
    ax.set_xlim(-50, 185)
    ax.set_ylim(bm + 7, -50)
    ax.set_aspect("equal")
    ax.axis("off")


def main(dst, theme="light", json_out=None):
    ncu = {r["name"]: r for r in load("docs/results/ncu-rtx4060.csv")}
    med = medians("docs/results/rtx4060.csv")
    for s in STEPS:
        check_cites(s)
        warps, ksteps, req, sect = predict(s)
        r = ncu[s["kernel"]]
        mreq = r["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"]
        msect = r["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]
        s.update(med[s["kernel"]], warps=warps, ksteps=ksteps)
        s["check"] = dict(
            predicted_requests=req,
            measured_requests=int(mreq),
            predicted_sectors=sect,
            measured_sectors=int(msect),
        )
        if s["epilogue"] is None:
            s["check"]["note"] = (
                "prediction covers the K loop only; the rest is the C read"
            )
        print(
            f"{s['kernel']}: requests {req} vs {mreq:.0f}, sectors {sect} vs {msect:.0f}"
        )

    c = THEMES[theme]
    # Row heights follow each row's y extent (32- vs 128-row tiles), so every
    # panel keeps the same units-per-inch.
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), dpi=133.34, gridspec_kw={"height_ratios": [89, 185]}
    )
    fig.patch.set_facecolor(c["surface"])
    for ax, s in zip(axes.flat, STEPS):
        draw(ax, s, c)
    fig.suptitle(
        "Block tiling per step, drawn to one scale (1 unit = 1 element of C, A or B)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=c["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    webp = {"quality": 82} if dst.endswith(".webp") else None
    fig.savefig(dst, facecolor=c["surface"], pil_kwargs=webp)

    if json_out:
        for s in STEPS:
            s["cite"] = {k: f"{p}:{n}" for k, (p, n, _) in s["cite"].items()}
            s["global_ld"] = [dict(what=w, sectors=n) for w, n in s["global_ld"]]
            if s["epilogue"]:
                s["epilogue"] = [
                    dict(what=w, per_warp=n, sectors=x) for w, n, x in s["epilogue"]
                ]
        with open(json_out, "w") as f:
            fields = dict(
                tile="[BM, BN, BK]: block's C tile and K step",
                warp0="[row, col, rows, cols] of C computed by warp 0 in the block tile",
                global_ld="global load instructions per warp per K step, sectors each",
                epilogue="global loads of C per warp before scaling, sectors each",
                check="global load requests/sectors implied by the tiling vs ncu at 4096",
                gflops="median of three 4096 runs, docs/results/rtx4060.csv",
            )
            json.dump(
                dict(size=N, gpu="RTX 4060 (sm_89)", fields=fields, steps=STEPS),
                f,
                separators=(",", ":"),
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
