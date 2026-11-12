"""Colours shared by the plotting scripts: `light` for the repo docs, `dark` for
near-black web pages. Blue marks FP32 kernels, orange the tensor-core kernel."""

THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        muted="#52514e",
        grid="#e4e3df",
        fp32="#2a78d6",
        tc="#eb6834",
    ),
    "dark": dict(
        surface="#0a0b12",
        ink="#f2f2f0",
        muted="#9a9ca6",
        grid="#262833",
        fp32="#4a8ddc",
        tc="#e0652f",
    ),
}
