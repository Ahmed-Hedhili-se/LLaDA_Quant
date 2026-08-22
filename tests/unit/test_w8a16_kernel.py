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
def test_fused_matches_dequantize_then_matmul(m):
    torch.manual_seed(0)
    k, n = 512, 256
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    q = quantize_tensor(w, bits=8, group_size=128)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    reference = a @ q.dequantize(torch.bfloat16).t()
    fused = kernel.w8a16_gemm(a, q.q.contiguous(), q.scale.contiguous())

    assert fused.shape == reference.shape and fused.dtype == reference.dtype
    rel = (fused.float() - reference.float()).norm() / reference.float().norm()
    assert rel < 2e-2, f"relative L2 {rel:.2e}"


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
