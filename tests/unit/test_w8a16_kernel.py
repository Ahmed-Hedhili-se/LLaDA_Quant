"""The fused W8A16 GEMM. Numerics need a GPU; the import contract does not."""

from __future__ import annotations

import pytest
import torch

from LLaDA_Quant.algorithms.symmetric import quantize_tensor, validate_block_k_alignment
from LLaDA_Quant.runtime.kernels import w8a16_gemm as kernel

needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and kernel.HAS_TRITON),
    reason="fused kernel needs CUDA and Triton",
)


def test_module_imports_without_triton():
    """The package must stay importable on a laptop with no Triton."""
    assert isinstance(kernel.HAS_TRITON, bool)
    if not kernel.HAS_TRITON:
        assert not hasattr(kernel, "w8a16_gemm")


def test_block_k_must_match_the_group_size():
    """One scale row serves a whole tile, so BLOCK_K == GROUP_SIZE."""
    validate_block_k_alignment(128, 128)
    with pytest.raises(ValueError, match="straddles a group boundary"):
        validate_block_k_alignment(96, 128)


@needs_gpu
@pytest.mark.parametrize("m", [16, 64, 129])
def test_fused_is_as_accurate_as_cublas_against_ground_truth(m):
    """Measure both paths against fp32, not against each other.

    Comparing the fused kernel to ``a @ dequantize(w).t()`` measures the gap
    between two approximations, and cuBLAS is the one that moves: its error
    against truth ranges 1.64e-03 to 2.37e-03 as it changes tiling by shape,
    while the fused kernel holds 1.65e-03 everywhere. A pairwise check
    therefore swings 100x for reasons that have nothing to do with this kernel.
    """
    torch.manual_seed(0)
    k, n = 2048, 2048
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q = quantize_tensor(w, bits=8, group_size=128)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    w_deq = q.dequantize(torch.bfloat16)
    truth = a.float() @ w_deq.float().t()
    cublas = a @ w_deq.t()
    fused = kernel.w8a16_gemm(a, q.q.contiguous(), q.scale.contiguous())

    assert fused.shape == cublas.shape and fused.dtype == cublas.dtype

    def rel(x):
        return ((x.double() - truth.double()).norm() / truth.double().norm()).item()

    err_fused, err_cublas = rel(fused), rel(cublas)
    # both sit at the bf16 output-rounding floor (~2^-9)
    assert err_fused < 3e-3, f"fused error {err_fused:.2e}"
    # and the fused path must not be materially worse than cuBLAS
    assert err_fused <= err_cublas * 1.2, (
        f"fused {err_fused:.2e} vs cuBLAS {err_cublas:.2e}"
    )
    # for scale: INT8 quantization itself costs ~6.5e-3
    assert err_fused < 6.5e-3 / 2


@needs_gpu
def test_handles_a_ragged_m():
    """Token counts per expert are never a nice multiple of BLOCK_M."""
    torch.manual_seed(0)
    w = (torch.randn(256, 512, device="cuda") * 0.02).to(torch.bfloat16)
    q = quantize_tensor(w, bits=8, group_size=128)
    for m in (1, 7, 33):
        a = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        out = kernel.w8a16_gemm(a, q.q.contiguous(), q.scale.contiguous())
        assert out.shape == (m, 256)
        assert torch.isfinite(out.float()).all()
        truth = a.float() @ q.dequantize(torch.float32).t()
        rel = ((out.double() - truth.double()).norm() / truth.double().norm()).item()
        assert rel < 3e-3, f"M={m} drifts at the ragged tail: {rel:.2e}"
