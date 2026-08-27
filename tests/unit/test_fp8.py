import torch

from LLaDA_Quant.algorithms.fp8 import (
    FP8_DTYPE,
    FP8_E4M3_MAX,
    dequantize_tensor_fp8,
    quantize_tensor_fp8,
)
from LLaDA_Quant.algorithms.symmetric import QuantResult
from LLaDA_Quant.config import QuantConfig


def test_zero_tensor_quantizes_to_zero():
    w = torch.zeros(4, 8, 16)
    q = quantize_tensor_fp8(w, group_size=8)
    assert torch.all(q.q.float() == 0)
    assert torch.all(q.scale == 0.0)


def test_storage_dtype_is_float8_e4m3():
    torch.manual_seed(0)
    w = torch.randn(4, 16, 256) * 0.5
    q = quantize_tensor_fp8(w, group_size=32)
    assert q.q.dtype == FP8_DTYPE
    assert not q.packed
    assert q.qtype == "fp8_e4m3"


def test_roundtrip_error_within_e4m3_budget():
    """E4M3's worst-case relative step near its own amax is ~1/16 (3 mantissa
    bits); nowhere close to int8's 1/127, but that is the format, not a bug.
    A generous bound catches breakage without asserting int8-level precision
    fp8 was never going to have."""
    torch.manual_seed(0)
    w = torch.randn(4, 16, 256) * 0.5
    q = quantize_tensor_fp8(w, group_size=32)
    w_hat = q.dequantize()
    rel_err = (w - w_hat).abs().max() / w.abs().max()
    assert rel_err < 0.10


def test_dequantize_dispatches_through_quantresult():
    """QuantResult.dequantize() must route fp8-tagged results through
    dequantize_tensor_fp8, not the int path -- this is the actual wiring the
    PACKED-mode runtime depends on, so it is tested directly rather than only
    through quantize_tensor_fp8's own round trip."""
    torch.manual_seed(1)
    w = torch.randn(2, 8, 64)
    q = quantize_tensor_fp8(w, group_size=16)
    assert isinstance(q, QuantResult)
    direct = dequantize_tensor_fp8(q.q, q.scale, q.group_size, dtype=torch.float32)
    via_result = q.dequantize(dtype=torch.float32)
    assert torch.equal(direct, via_result)


def test_scale_maps_amax_to_format_max():
    torch.manual_seed(2)
    w = torch.randn(2, 8, 64)
    q = quantize_tensor_fp8(w, group_size=16)
    w_g = w.reshape(2, 8, 4, 16)
    expected = w_g.abs().amax(dim=-1) / FP8_E4M3_MAX
    assert torch.allclose(q.scale, expected, atol=1e-6)


def test_per_tensor_fallback_when_not_divisible():
    torch.manual_seed(3)
    w = torch.randn(3, 7, 50)
    q = quantize_tensor_fp8(w, group_size=33)
    assert q.group_size == -1
    w_hat = q.dequantize()
    # E4M3's step widens with magnitude; budget scales off amax the same way
    # symmetric.py's equivalent test does off qmax_for_bits(8).
    assert torch.allclose(w, w_hat, atol=w.abs().max() * 0.10 + 1e-4)


def test_storage_bytes_matches_int8_at_same_shape():
    """FP8 and INT8 are both 1 byte/value with the same scale dtype, so a
    fair memory comparison between the two quant modes should show identical
    storage_bytes() for identical shapes -- any difference here would mean
    the two paths are not actually comparable at the memory axis."""
    from LLaDA_Quant.algorithms.symmetric import quantize_tensor

    torch.manual_seed(4)
    w = torch.randn(4, 32, 128)
    q_int8 = quantize_tensor(w, bits=8, group_size=32)
    q_fp8 = quantize_tensor_fp8(w, group_size=32)
    assert q_fp8.storage_bytes() == q_int8.storage_bytes()


def test_bfloat16_input_handled():
    w = torch.randn(3, 8, 128, dtype=torch.bfloat16) * 0.1
    q = quantize_tensor_fp8(w, group_size=32)
    w_hat = q.dequantize(dtype=torch.bfloat16)
    assert w_hat.dtype == torch.bfloat16
    assert (w.float() - w_hat.float()).abs().max() < 0.05


def test_quantconfig_accepts_fp8_dtype():
    cfg = QuantConfig(bits=8, dtype="fp8_e4m3")
    assert cfg.dtype == "fp8_e4m3"


def test_quantconfig_rejects_fp8_with_wrong_bits():
    try:
        QuantConfig(bits=4, dtype="fp8_e4m3")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_quantconfig_rejects_unknown_dtype():
    try:
        QuantConfig(dtype="fp16")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_packed_residency_dequantizes_fp8_on_access():
    """End-to-end through runtime.moe's PACKED-mode plumbing: quantize a
    fake expert block to fp8, install packed access, and confirm block.w1/w2
    read back through _packed_expert_weight's fp8 branch rather than the int
    branch -- this is the exact path apply_quantization(dtype='fp8_e4m3')
    exercises against the real model."""
    import torch.nn as nn

    from LLaDA_Quant.runtime.moe import (
        QuantExpertWeights,
        attach_packed_buffers,
        install_packed_expert_access,
        is_packed_expert_block,
    )

    torch.manual_seed(5)

    class FakeExpertBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(4, 32, 16) * 0.3)
            self.w2 = nn.Parameter(torch.randn(4, 16, 16) * 0.3)

    block = FakeExpertBlock()
    w1_ref, w2_ref = block.w1.detach().clone(), block.w2.detach().clone()

    weights = QuantExpertWeights.quantize(
        block.w1.detach(), block.w2.detach(),
        group_size=16, dtype="fp8_e4m3",
    )
    attach_packed_buffers(block, weights, compute_dtype=torch.float32)
    install_packed_expert_access(block)

    assert is_packed_expert_block(block)
    assert block._qw1.dtype == FP8_DTYPE
    assert "w1" not in block._parameters  # Parameter removed, now a property

    rel_err_w1 = (block.w1 - w1_ref).abs().max() / w1_ref.abs().max()
    rel_err_w2 = (block.w2 - w2_ref).abs().max() / w2_ref.abs().max()
    assert rel_err_w1 < 0.10
    assert rel_err_w2 < 0.10
