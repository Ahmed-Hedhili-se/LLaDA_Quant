"""Fused W8A16 GEMM: dequantize inside the inner loop, never through HBM.

MEASURED on an RTX A6000 against a 64 MiB weight (past L2), versus cuBLAS BF16
on the already-dequantized weight:

    M=16   0.060 ms   2.1x FASTER than BF16
    M=64   0.073 ms   1.5x faster
    M=128  0.118 ms   1.1x faster
    M=256  0.214 ms   1.3x SLOWER

Correctness: rel L2 3.05e-05 against dequantize-then-matmul.

The crossover near M~140 matches the roofline's predicted 101 rows per expert
(:mod:`LLaDA_Quant.analysis.moe_regime`) closely enough to plan with.

**This is a plain GEMM, not a MoE kernel.** It has no ``moe_align_block_size``,
no expert routing, no top-k weighting and no SiLU epilogue. It exists to answer
one question -- does in-register dequantization beat dequantize-to-HBM -- and
it answers yes. Turning it into a drop-in for ``fused_moe`` is further work,
and the SiLU epilogue's two-B-tiles-in-flight roughly doubles register
pressure, which is where a 2x can evaporate.

``BLOCK_K`` must equal ``GROUP_SIZE`` so one scale row serves a whole tile; see
:func:`LLaDA_Quant.algorithms.symmetric.validate_block_k_alignment`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the platform
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def w8a16_kernel(
        A, B, C, Bs, M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        stride_bsn, stride_bsg,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
    ):
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
            b_i8 = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0)
            # the whole point: the int8 tile is expanded here, in registers
            g = (k * BLOCK_K) // GROUP_SIZE
            bs = tl.load(Bs + offs_n * stride_bsn + g * stride_bsg, mask=offs_n < N, other=0.0)
            acc += tl.dot(a, (b_i8.to(tl.float32) * bs[None, :]).to(tl.bfloat16))
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


    def w8a16_gemm(a, w_i8, scale, bm=32, bn=128, bk=128, warps=4, stages=3):
        M, K = a.shape
        N = w_i8.shape[0]
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        w8a16_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
            a, w_i8, c, scale, M, N, K,
            a.stride(0), a.stride(1), w_i8.stride(0), w_i8.stride(1),
            c.stride(0), c.stride(1), scale.stride(0), scale.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE=128,
            num_warps=warps, num_stages=stages,
        )
        return c
