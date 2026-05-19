"""
Assignment 3 – Attention Kernel Optimisation
CPU Template  (Intel / Apple Silicon / any x86 machine)

A sample code snippet is provided below to get you started.
Feel free to change the code in any way you see fit.

Parallelism tool : OpenMP  (CPU threads + SIMD vectorisation)
Profiling tool   : Intel Advisor or something similar (generates a roofline chart)
Reference        : torch.nn.functional.scaled_dot_product_attention
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# C++ / OpenMP source
#
# For CPU extensions the entire implementation goes in cpp_sources.
# There is no cuda_sources here — that is only used for .cu (CUDA) files.


_CPP_SRC = r"""
#include <torch/extension.h>
#include <cmath>
#include <cfloat>

#ifdef _OPENMP
  #include <omp.h>
#endif


### YOUR C++ / OpenMP CODE implementing Attention HERE
"""

# ─────────────────────────────────────────────────────────────────────────────
# Compile the extension
#
# -O3           : full compiler optimisations
# -fopenmp      : enable OpenMP parallel regions and SIMD hints
# -march=native : use AVX2 / AVX-512 instructions available on this CPU
# -g            : keep debug symbols so Intel Advisor maps counters to lines
#
# macOS note: Apple clang does not ship OpenMP.
#   Install libomp first:  brew install libomp
#   Then replace the flags list below with:
#     ["-O3", "-Xpreprocessor", "-fopenmp", "-lomp", "-march=native", "-g"]
# Some Linux may also require -fopenmp=libomp and -lomp if using the LLVM toolchain instead of GNU.
# ─────────────────────────────────────────────────────────────────────────────

_attn_ext = load_inline(
    name="attn_cpu_ext",
    cpp_sources=_CPP_SRC,  # full C++ source — no cuda_sources for CPU
    functions=["attention_forward"],
    extra_compile_args={"cxx": ["-O3", "-fopenmp", "-march=native", "-g"]},
    # macOS: Apple clang does not bundle OpenMP.
    #   Install it first:  brew install libomp
    #   Then replace the flags above with:
    #   ["-O3", "-Xpreprocessor", "-fopenmp", "-lomp", "-march=native", "-g"]
    verbose=False,
)


def attention_cpu(Q, K, V):
    """Your C++ / OpenMP attention implementation."""
    return _attn_ext.attention_forward(Q.contiguous(), K.contiguous(), V.contiguous())


def attention_pytorch(Q, K, V):
    """PyTorch reference — optimised BLAS + fused softmax on CPU."""
    return F.scaled_dot_product_attention(Q, K, V)


def check_correctness():
    print("=" * 55)
    print("  Step 1: Correctness check")
    print("=" * 55)

    torch.manual_seed(42)
    configs = [
        (1, 1, 64, 64),  # minimal — easiest to debug
        (2, 4, 128, 64),  # small
        (2, 8, 512, 64),  # medium — closer to profiling size
    ]

    all_passed = True
    for B, H, S, D in configs:
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.randn(B, H, S, D)

        out_cpu = attention_cpu(Q, K, V)
        out_pytorch = attention_pytorch(Q, K, V)

        max_err = (out_cpu - out_pytorch).abs().max().item()
        passed = max_err < 1e-3
        status = "✓  PASS" if passed else "✗  FAIL"
        all_passed = all_passed and passed

        print(f"  B={B} H={H} S={S:<4} D={D}  " f"max_err={max_err:.2e}  {status}")

    print()
    if all_passed:
        print("  All configurations passed. Proceed to Step 2.")
    else:
        print("  One or more configurations FAILED.")
        print("  Fix your kernel before running the benchmark or profiler.")
    print()
    return all_passed


# Step 2 — Performance benchmark
# Run after correctness passes.


def run_benchmark():
    """Benchmark the CPU implementation against PyTorch's reference."""

    " TODO: Implement a benchmark that compares the runtime of attention_cpu vs attention_pytorch. "


# Main

if __name__ == "__main__":
    passed = check_correctness()
    if passed:
        run_benchmark()

#  Step 3 - Intel Advisor profiling
#
#  Run these commands from your terminal after Step 1 and Step 2 pass.
#  Intel Advisor requires two separate passes to build the roofline chart.
#
#  Prerequisites:
#    Download Intel oneAPI Base Toolkit (free for students):
#      https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
#
#    Activate the environment before running the commands below:
#      source /opt/intel/oneapi/setvars.sh          # Linux
#      source ~/intel/oneapi/setvars.sh             # Linux, user install
#      C:\Intel\oneAPI\setvars.bat                  # Windows
#
#
#  ===========3a. Collect the profile (two passes required) =============

#  ===========3b. Open and analyse the report =============
#  │
#  │  Option A — GUI (recommended):
#  │    1. Open Intel Advisor
#  │    2. File → Open Project → advisor_project/
#  │    3. Click the "Roofline" tab
#  │       Each loop in your C++ code appears as a labelled dot.
#  │
#  │  Option B — Standalone HTML (no GUI needed):
#  │    advisor --report=roofline \
#  │            --project-dir=./advisor_project \
#  │            --report-output=roofline.html \
#  │            --format=html
#  │    Then open roofline.html in any browser.
