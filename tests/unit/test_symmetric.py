import torch

from LLaDA_Quant.algorithms.symmetric import (
    dequantize_tensor,
    pack_int4,
    qmax_for_bits,
    quantize_tensor,
    unpack_int4,
)


def test_qmax_values():
    assert qmax_for_bits(8) == 127
    assert qmax_for_bits(4) == 7


def test_zero_tensor_quantizes_to_zero():
    w = torch.zeros(4, 8, 16)
    q = quantize_tensor(w, bits=8, group_size=8)
    assert torch.all(q.q == 0)
    assert torch.all(q.scale == 0.0)


def test_roundtrip_error_within_budget():
    torch.manual_seed(0)
    w = torch.randn(4, 16, 256) * 0.5
    q = quantize_tensor(w, bits=8, group_size=32)
    w_hat = q.dequantize()
    rel_err = (w - w_hat).abs().max() / w.abs().max()
    assert rel_err < 1 / 127 * 1.1


def test_scale_matches_amax_divided_by_qmax():
    torch.manual_seed(1)
    w = torch.randn(2, 8, 64)
    q = quantize_tensor(w, bits=8, group_size=16)
    w_g = w.reshape(2, 8, 4, 16)
    expected = w_g.abs().amax(dim=-1) / 127
    assert torch.allclose(q.scale, expected, atol=1e-6)


def test_per_tensor_fallback_when_not_divisible():
    torch.manual_seed(2)
    w = torch.randn(3, 7, 50)
    q = quantize_tensor(w, bits=8, group_size=33)
    assert q.group_size == -1
    w_hat = q.dequantize()
    assert torch.allclose(w, w_hat, atol=w.abs().max() / 127 + 1e-4)


def test_int4_roundtrip_and_packing():
    torch.manual_seed(3)
    w = torch.randn(2, 16, 64)
    q = quantize_tensor(w, bits=4, group_size=16)
    assert torch.all(q.q >= -8) and torch.all(q.q <= 7)
    packed = pack_int4(q.q)
    assert packed.shape[-1] == q.q.shape[-1] // 2
    assert torch.equal(unpack_int4(packed), q.q)
    w_hat = q.dequantize()
    assert (w - w_hat).abs().max() / w.abs().max() < 1 / 7 * 1.1


def test_bfloat16_input_handled():
    w = torch.randn(3, 8, 128, dtype=torch.bfloat16) * 0.1
    q = quantize_tensor(w, bits=8, group_size=32)
    w_hat = q.dequantize(dtype=torch.bfloat16)
    assert w_hat.dtype == torch.bfloat16
    assert (w.float() - w_hat.float()).abs().max() < 0.05