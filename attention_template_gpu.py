"""
Assignment 3 – Attention Kernel Optimisation
GPU Template  (NVIDIA CUDA)

A sample code snippet is provided below to get you started.
Feel free to change the code in any way you see fit.
"""

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r""" YOUR INLINE CUDA CODE HERE


#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

\\ TODO: Implement the necessary CUDA kernels for the attention computation.

torch::Tensor attention_forward(
    torch::Tensor Q,   // [B, H, S, D]  float32, CUDA
    torch::Tensor K,
    torch::Tensor V
) { 

    TORCH_CHECK(Q.device().is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kFloat32);

    // Your CUDA code here should compute the attention output and return it
}


"""


_CPP_DECL = (
    "torch::Tensor attention_forward(torch::Tensor, torch::Tensor, torch::Tensor);"
)

_attn_ext = load_inline(
    name="attn_ext",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["attention_forward"],
    extra_cuda_cflags=["-O2"],
    verbose=False,
)


def attention_cuda(Q, K, V):
    """Your CUDA attention implementation."""
    return _attn_ext.attention_forward(Q.contiguous(), K.contiguous(), V.contiguous())


def attention_pytorch(Q, K, V):
    """PyTorch reference - uses Flash Attention backend."""
    return F.scaled_dot_product_attention(Q, K, V)


# Step 1 — Correctness check
# Run this first. Do not proceed to profiling until all configs pass.


def check_correctness():
    print("=" * 55)
    print("  Step 1: Correctness check")
    print("=" * 55)

    torch.manual_seed(42)
    configs = [
        (1, 1, 64, 64),  # minimal - easiest to debug
        (2, 4, 128, 64),  # small
        (2, 8, 512, 64),  # medium - closer to profiling size
    ]

    all_passed = True
    for B, H, S, D in configs:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        out_cuda = attention_cuda(Q, K, V)
        out_pytorch = attention_pytorch(Q, K, V)

        max_err = (out_cuda - out_pytorch).abs().max().item()
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
    """Benchmark the CUDA implementation against PyTorch's Flash Attention."""

    " TODO: Implement a benchmark that compares the runtime of attention_cuda vs attention_pytorch. "


# Main

if __name__ == "__main__":
    passed = check_correctness()
    if passed:
        run_benchmark()


#  Step 3 — Nsight Compute profiling
#
#  Run these commands from your terminal after Step 1 and Step 2 pass.

#  ===========3a. Collect the profile =============
#  │
#  │  ncu \
#  │    --target-processes all \
#  │    --kernel-name-base function \
#  │    --launch-skip 5 \
#  │    --launch-count 5 \
#  │    --section SpeedOfLight \
#  │    --section MemoryWorkloadAnalysis \
#  │    --section ComputeWorkloadAnalysis \
#  │    --section Occupancy \
#  │    --export attention_profile \
#  │    --force-overwrite \
#  │    python attention_template_gpu.py
#  │
#  │  This produces:  attention_profile.ncu-rep
#  │
#  │  Flag reference:
#  │    --launch-skip 5        skip 5 launches (avoids JIT / warmup noise)
#  │    --launch-count 5        capture the next 5 launches after the skip
#  │    --section SpeedOfLight  the critical section — shows DRAM% vs SM%
#  │    --section MemoryWorkloadAnalysis   L1/L2 hit rates, sector counts
#  │    --section ComputeWorkloadAnalysis  warp efficiency, IPC
#  │    --section Occupancy     active warps vs hardware maximum
#  │    --export <name>         write report to <name>.ncu-rep
#  │    --force-overwrite       overwrite if the file already exists

# Note: Feel free to modify the sections you capture based on what you want to analyze. The above sections are a good starting point for attention kernels.


#  ===========3b. Open and Analyze the report =============
#  │  ncu-ui attention_profile.ncu-rep

#  │  The Nsight Compute GUI works on Mac and Windows without a GPU.
#  │  Download: https://developer.nvidia.com/nsight-compute
