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

import os
# for gcc version issue in csc, might not apply in other environment
_extra_cuda_cflags = ["-O2"]
_gcc13 = "/appl/spack/v020/install-tree/gcc-8.5.0/gcc-13.1.0-how4ki/bin/g++"
_extra_cuda_cflags += ["-ccbin", _gcc13]
os.environ["CXX"] = _gcc13

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Kernel 1: compute P = Q * K^T / sqrt(D)
// each thread computes one element P[bh][i][j]
// by doing a dot product of row i of Q and row j of K
__global__ void qk_dot(
    const float* Q,   // [BH, S, D]
    const float* K,   // [BH, S, D]
    float* P,         // [BH, S, S]
    int S, int D
) {
    int bh = blockIdx.x;                            // which batch-head
    int i  = blockIdx.y * blockDim.y + threadIdx.y; // row in P (query row)
    int j  = blockIdx.z * blockDim.x + threadIdx.x; // col in P (key row)

    if (i >= S || j >= S) return;

    float scale = 1.0f / sqrtf((float)D);

    // dot product of Q[bh][i][:] and K[bh][j][:]
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += Q[bh * S * D + i * D + d] * K[bh * S * D + j * D + d];
    }

    P[bh * S * S + i * S + j] = sum * scale;
}

// Kernel 2: row-wise softmax on P, done in-place
// each thread handles one full row P[bh][i][:]
__global__ void softmax_rows(
    float* P,         // [BH, S, S]
    int S, int BH
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // flat index
    if (idx >= BH * S) return;

    int bh = idx / S;  // which batch-head
    int i  = idx % S;  // which row

    float* row = P + bh * S * S + i * S;

    // pass 1: find the max value in this row (for numerical stability)
    float max_val = -FLT_MAX;
    for (int j = 0; j < S; j++) {
        if (row[j] > max_val) max_val = row[j];
    }

    // pass 2: compute exp(val - max) and accumulate the sum
    float sum = 0.0f;
    for (int j = 0; j < S; j++) {
        row[j] = expf(row[j] - max_val);
        sum += row[j];
    }

    // pass 3: divide by the sum so the row adds up to 1
    for (int j = 0; j < S; j++) {
        row[j] /= sum;
    }
}


// Kernel 3: compute O = A * V
// each thread computes one element O[bh][i][d]
// by doing a dot product of row i of A and column d of V
__global__ void av_mul(
    const float* A,   // [BH, S, S]
    const float* V,   // [BH, S, D]
    float* O,         // [BH, S, D]
    int S, int D
) {
    int bh = blockIdx.x;                            // which batch-head
    int i  = blockIdx.y * blockDim.y + threadIdx.y; // row in O (sequence pos)
    int d  = blockIdx.z * blockDim.x + threadIdx.x; // col in O (head dim)

    if (i >= S || d >= D) return;

    // dot product of A[bh][i][:] and V[bh][:][d]
    float sum = 0.0f;
    for (int j = 0; j < S; j++) {
        sum += A[bh * S * S + i * S + j] * V[bh * S * D + j * D + d];
    }

    O[bh * S * D + i * D + d] = sum;
}

torch::Tensor attention_forward(
    torch::Tensor Q,   // [B, H, S, D]
    torch::Tensor K,
    torch::Tensor V
) {
    TORCH_CHECK(Q.device().is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dtype() == torch::kFloat32);

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);

    // P holds the score matrix, shape [B, H, S, S]
    auto P = torch::empty({B, H, S, S}, Q.options());

    // O is the final output, shape [B, H, S, D]
    auto O = torch::empty({B, H, S, D}, Q.options());

    // flatten batch and heads into one dimension
    const int BH = B * H;

    // --- Kernel 1: P = Q * K^T / sqrt(D) ---
    // one thread per element of P, so grid covers [BH, S, S]
    dim3 block1(16, 16);
    dim3 grid1(BH, (S + 15) / 16, (S + 15) / 16);
    qk_dot<<<grid1, block1>>>(Q.data_ptr<float>(), K.data_ptr<float>(),
                              P.data_ptr<float>(), S, D);

    // --- Kernel 2: softmax each row of P in-place ---
    // one thread per row, so we need BH * S threads total
    int threads2 = 256;
    int blocks2 = (BH * S + threads2 - 1) / threads2;
    softmax_rows<<<blocks2, threads2>>>(P.data_ptr<float>(), S, BH);

    // --- Kernel 3: O = A * V ---
    // one thread per element of O, so grid covers [BH, S, D]
    dim3 block3(16, 16);
    dim3 grid3(BH, (S + 15) / 16, (D + 15) / 16);
    av_mul<<<grid3, block3>>>(P.data_ptr<float>(), V.data_ptr<float>(),
                             O.data_ptr<float>(), S, D);

    return O;
}
"""



_CPP_DECL = (
    "torch::Tensor attention_forward(torch::Tensor, torch::Tensor, torch::Tensor);"
)


import shutil
_cache_dir = os.path.expanduser("~/.cache/torch_extensions/py311_cu124/attn_ext")
if os.path.exists(_cache_dir):
    shutil.rmtree(_cache_dir)
_attn_ext = load_inline(
    name="attn_ext",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["attention_forward"],
    extra_cuda_cflags=_extra_cuda_cflags,
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

    print("=" * 55)
    print("  Step 2: Performance Benchmark")
    print("=" * 55)

    B, H, D = 2, 8, 64
    configs = [128, 512, 1024]

    print(f"  {'S':<6} {'Your Kernel (ms)':<20} {'PyTorch (ms)':<20} {'Slowdown'}")
    print("  " + "-" * 50)

    for S in configs:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        t_cuda = Timer(
            stmt="attention_cuda(Q, K, V)",
            globals={"attention_cuda": attention_cuda, "Q": Q, "K": K, "V": V}
        ).blocked_autorange(min_run_time=1.0)

        t_pytorch = Timer(
            stmt="attention_pytorch(Q, K, V)",
            globals={"attention_pytorch": attention_pytorch, "Q": Q, "K": K, "V": V}
        ).blocked_autorange(min_run_time=1.0)

        your_ms = t_cuda.median * 1e3
        pytorch_ms = t_pytorch.median * 1e3
        slowdown = your_ms / pytorch_ms

        print(f"  {S:<6} {your_ms:<20.3f} {pytorch_ms:<20.3f} {slowdown:.1f}x")

    print()

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
