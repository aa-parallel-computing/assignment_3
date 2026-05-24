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
    dim3 block1(16, 16);
    dim3 grid1(BH, (S + 15) / 16, (S + 15) / 16);
    qk_dot<<<grid1, block1>>>(Q.data_ptr<float>(), K.data_ptr<float>(),
                              P.data_ptr<float>(), S, D);

    // --- Kernel 2: softmax each row of P in-place ---
    int threads2 = 256;
    int blocks2 = (BH * S + threads2 - 1) / threads2;
    softmax_rows<<<blocks2, threads2>>>(P.data_ptr<float>(), S, BH);

    // --- Kernel 3: O = A * V ---
    dim3 block3(16, 16);
    dim3 grid3(BH, (S + 15) / 16, (D + 15) / 16);
    av_mul<<<grid3, block3>>>(P.data_ptr<float>(), V.data_ptr<float>(),
                             O.data_ptr<float>(), S, D);

    return O;
}


#define FLASH_Bc  16
#define FLASH_Br  64
#define FLASH_D   64

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,   // [BH, S, D]
    const float* __restrict__ K,   // [BH, S, D]
    const float* __restrict__ V,   // [BH, S, D]
    float*       __restrict__ O,   // [BH, S, D]
    int S, int D
) {
    const int bh       = blockIdx.x;
    const int q_tile   = blockIdx.y;
    const int q_local  = threadIdx.x;
    const int q_global = q_tile * FLASH_Bc + q_local;

    if (q_global >= S) return;

    __shared__ float Ks[FLASH_Br][FLASH_D + 1];
    __shared__ float Vs[FLASH_Br][FLASH_D + 1];

    float q_reg[FLASH_D];
    {
        const float* Q_ptr = Q + bh * S * D + q_global * D;
        for (int d = 0; d < D; d++)
            q_reg[d] = Q_ptr[d];
    }

    float m_i = -FLT_MAX;
    float l_i = 0.0f;    
    float o_reg[FLASH_D];
    for (int d = 0; d < D; d++) o_reg[d] = 0.0f;

    const float inv_sqrt_D = rsqrtf((float)D);  // 1/sqrt(D) scaling factor

    for (int kv_start = 0; kv_start < S; kv_start += FLASH_Br) {

        const int kv_end = (kv_start + FLASH_Br < S) ? FLASH_Br : (S - kv_start);

        for (int row = q_local; row < kv_end; row += FLASH_Bc) {
            const int   kv_idx = kv_start + row;
            const float* K_ptr = K + bh * S * D + kv_idx * D;
            const float* V_ptr = V + bh * S * D + kv_idx * D;
            for (int d = 0; d < D; d++) {
                Ks[row][d] = K_ptr[d];
                Vs[row][d] = V_ptr[d];
            }
        }
        __syncthreads();

        for (int j = 0; j < kv_end; j++) {

            // Dot product: q_reg · Ks[j]
            float s_j = 0.0f;
            for (int d = 0; d < D; d++)
                s_j += q_reg[d] * Ks[j][d];
            s_j *= inv_sqrt_D;

            // Online softmax rescaling step
            float m_new    = fmaxf(m_i, s_j);
            float old_scale = expf(m_i - m_new);
            float new_exp   = expf(s_j - m_new);

            l_i = l_i * old_scale + new_exp;
            for (int d = 0; d < D; d++)
                o_reg[d] = o_reg[d] * old_scale + new_exp * Vs[j][d];

            m_i = m_new;
        }

        __syncthreads();
    }

    float* O_ptr = O + bh * S * D + q_global * D;
    for (int d = 0; d < D; d++)
        O_ptr[d] = o_reg[d] / l_i;
}

torch::Tensor attention_forward_fused(
    torch::Tensor Q,
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

    TORCH_CHECK(D <= FLASH_D,
        "flash_attention_kernel: D must be <= FLASH_D (got D=", D, ")");

    auto O = torch::empty({B, H, S, D}, Q.options());
    const int BH = B * H;

    dim3 grid(BH, (S + FLASH_Bc - 1) / FLASH_Bc);
    dim3 block(FLASH_Bc);

    flash_attention_kernel<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        S, D
    );

    return O;
}
"""


_CPP_DECL = (
    "torch::Tensor attention_forward(torch::Tensor, torch::Tensor, torch::Tensor);\n"
    "torch::Tensor attention_forward_fused(torch::Tensor, torch::Tensor, torch::Tensor);"
)


import shutil

_cache_dir = os.path.expanduser("~/.cache/torch_extensions/py311_cu124/attn_ext")
if os.path.exists(_cache_dir):
    shutil.rmtree(_cache_dir)
_attn_ext = load_inline(
    name="attn_ext",
    cpp_sources=_CPP_DECL,
    cuda_sources=_CUDA_SRC,
    functions=["attention_forward", "attention_forward_fused"],
    extra_cuda_cflags=_extra_cuda_cflags,
    verbose=False,
)


def attention_cuda(Q, K, V):
    return _attn_ext.attention_forward(Q.contiguous(), K.contiguous(), V.contiguous())


def attention_fused_cuda(Q, K, V):
    return _attn_ext.attention_forward_fused(
        Q.contiguous(), K.contiguous(), V.contiguous()
    )


def attention_pytorch(Q, K, V):
    """PyTorch reference - uses Flash Attention backend."""
    return F.scaled_dot_product_attention(Q, K, V)


def check_correctness():
    print("=" * 60)
    print("  Step 1: Correctness check")
    print("=" * 60)

    torch.manual_seed(42)
    configs = [
        (1, 1, 64, 64),  # minimal — easiest to debug
        (2, 4, 128, 64),  # small
        (2, 8, 512, 64),  # medium — closer to profiling size
    ]

    print("\n  [Naive kernel]")
    all_passed = True
    for B, H, S, D in configs:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        out_cuda = attention_cuda(Q, K, V)
        out_pytorch = attention_pytorch(Q, K, V)

        max_err = (out_cuda - out_pytorch).abs().max().item()
        ok = max_err < 1e-3
        status = "✓  PASS" if ok else "✗  FAIL"
        all_passed = all_passed and ok

        print(f"  B={B} H={H} S={S:<4} D={D}  max_err={max_err:.2e}  {status}")

    print("\n  Fused kernel")
    fused_passed = True
    for B, H, S, D in configs:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        out_fused = attention_fused_cuda(Q, K, V)
        out_pytorch = attention_pytorch(Q, K, V)

        max_err = (out_fused - out_pytorch).abs().max().item()
        ok = max_err < 1e-3
        status = "✓  PASS" if ok else "✗  FAIL"
        fused_passed = fused_passed and ok

        print(f"  B={B} H={H} S={S:<4} D={D}  max_err={max_err:.2e}  {status}")

    all_passed = all_passed and fused_passed

    print()
    if all_passed:
        print("  All configurations passed (naive + fused). Proceed to Step 2.")
    else:
        print("  One or more configurations FAILED.")
        print("  Fix the kernel before running the benchmark or profiler.")
    print()
    return all_passed


# ============================================================
# Step 2 — Performance benchmark
# Run after correctness passes.
# ============================================================


def run_benchmark():
    """Benchmark naive vs fused kernel vs PyTorch Flash Attention."""

    print("=" * 75)
    print("  Step 2: Performance Benchmark  (B=2, H=8, D=64)")
    print("=" * 75)

    B, H, D = 2, 8, 64
    configs = [128, 512, 1024]

    hdr = (
        f"  {'S':<6} {'Naive (ms)':<16} {'Fused (ms)':<16}"
        f" {'PyTorch (ms)':<16} {'Speedup (F/N)':<16} {'Slowdown (F/PT)'}"
    )
    print(hdr)
    print("  " + "-" * 80)

    for S in configs:
        Q = torch.randn(B, H, S, D, device="cuda")
        K = torch.randn(B, H, S, D, device="cuda")
        V = torch.randn(B, H, S, D, device="cuda")

        t_naive = Timer(
            stmt="attention_cuda(Q, K, V)",
            globals={"attention_cuda": attention_cuda, "Q": Q, "K": K, "V": V},
        ).blocked_autorange(min_run_time=1.0)

        t_fused = Timer(
            stmt="attention_fused_cuda(Q, K, V)",
            globals={
                "attention_fused_cuda": attention_fused_cuda,
                "Q": Q,
                "K": K,
                "V": V,
            },
        ).blocked_autorange(min_run_time=1.0)

        t_pytorch = Timer(
            stmt="attention_pytorch(Q, K, V)",
            globals={"attention_pytorch": attention_pytorch, "Q": Q, "K": K, "V": V},
        ).blocked_autorange(min_run_time=1.0)

        naive_ms = t_naive.median * 1e3
        fused_ms = t_fused.median * 1e3
        pytorch_ms = t_pytorch.median * 1e3

        speedup = naive_ms / fused_ms  # how much faster fused is than naive
        slowdown = fused_ms / pytorch_ms  # how much slower fused is vs PyTorch

        print(
            f"  {S:<6} {naive_ms:<16.3f} {fused_ms:<16.3f} {pytorch_ms:<16.3f} "
            f"{speedup:<16.1f}x {slowdown:.1f}x"
        )

    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    passed = check_correctness()

    print(f"Passed : {passed}")

    # if passed:
    #     run_benchmark()

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
#  │    --launch-count 5       capture the next 5 launches after the skip
#  │    --section SpeedOfLight the critical section — shows DRAM% vs SM%
#  │    --section MemoryWorkloadAnalysis   L1/L2 hit rates, sector counts
#  │    --section ComputeWorkloadAnalysis  warp efficiency, IPC
#  │    --section Occupancy    active warps vs hardware maximum
#  │    --export <name>        write report to <name>.ncu-rep
#  │    --force-overwrite      overwrite if the file already exists

#  ===========3b. Open and Analyze the report =============
#  │  ncu-ui attention_profile.ncu-rep
#  │
#  │  The Nsight Compute GUI works on Mac and Windows without a GPU.
#  │  Download: https://developer.nvidia.com/nsight-compute
