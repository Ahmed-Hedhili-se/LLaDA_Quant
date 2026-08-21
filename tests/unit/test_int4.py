"""Phase 2: INT4 must be genuinely packed, not a cosmetic config value.

The bug: ``pack_int4`` existed, was tested, and was never called by the
expert adapter, so ``bits=4`` produced the same byte count as ``bits=8`` —
strictly more error for zero benefit.
"""

from __future__ import annotations

import pytest
import torch

from LLaDA_Quant import ExecutionMode, QuantConfig, quantize_model
from LLaDA_Quant.algorithms.symmetric import (
    dequantize_tensor,
    pack_int4,
    qmax_for_bits,
    qmin_for_bits,
    quantize_tensor,
    storage_bytes,
    unpack_int4,
    validate_int4_layout,
)


# --------------------------------------------------------------------------
# Packing primitives
# --------------------------------------------------------------------------


def test_pack_unpack_roundtrip_over_the_whole_int4_range():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(4).reshape(4, 16)
    packed = pack_int4(values)
    assert packed.shape == (4, 8)
    assert torch.equal(unpack_int4(packed), values)


def test_sign_extension_is_correct_for_every_negative_nibble():
    for value in range(-8, 0):
        pair = torch.tensor([[value, 0]], dtype=torch.int8)
        recovered = unpack_int4(pack_int4(pair))
        assert int(recovered[0, 0]) == value, f"nibble for {value} did not sign-extend"


def test_nibble_order_is_low_then_high():
    packed = pack_int4(torch.tensor([[1, 2]], dtype=torch.int8))
    assert int(packed[0, 0]) & 0x0F == 1, "even index must occupy the low nibble"
    assert (int(packed[0, 0]) >> 4) & 0x0F == 2, "odd index must occupy the high nibble"


def test_pack_rejects_out_of_range_and_odd_shapes():
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        pack_int4(torch.tensor([[8, 0]], dtype=torch.int8))
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        pack_int4(torch.tensor([[-9, 0]], dtype=torch.int8))
    with pytest.raises(ValueError, match="even"):
        pack_int4(torch.zeros(1, 3, dtype=torch.int8))


def test_validate_int4_layout_guards_group_alignment():
    validate_int4_layout((4, 128), 64)
    validate_int4_layout((4, 128), -1)
    with pytest.raises(ValueError, match="even last dimension"):
        validate_int4_layout((4, 127), 64)
    with pytest.raises(ValueError, match="even group_size"):
        validate_int4_layout((4, 128), 33)


# --------------------------------------------------------------------------
# Quantization contract at 4 bits
# --------------------------------------------------------------------------


def test_int4_storage_is_half_of_int8():
    torch.manual_seed(0)
    w = torch.randn(4, 64, 128)
    q8 = quantize_tensor(w, bits=8, group_size=64)
    q4 = quantize_tensor(w, bits=4, group_size=64)
    assert q4.packed and not q8.packed
    assert q4.q.shape[-1] * 2 == q8.q.shape[-1]
    values8 = q8.q.numel() * q8.q.element_size()
    values4 = q4.q.numel() * q4.q.element_size()
    assert values4 * 2 == values8
    # scales are identical in both, so total is just above half
    assert 0.5 < q4.storage_bytes() / q8.storage_bytes() < 0.55


def test_int4_roundtrip_stays_inside_the_quantization_contract():
    torch.manual_seed(0)
    w = torch.randn(4, 32, 128) * 0.02
    result = quantize_tensor(w, bits=4, group_size=64)
    recovered = result.dequantize(torch.float32)
    assert recovered.shape == w.shape
    # symmetric quant: error is bounded by half a step of the group's scale
    scale = result.scale.repeat_interleave(64, dim=-1)
    assert torch.all((recovered - w).abs() <= scale / 2 + 1e-6)


def test_int4_values_occupy_the_full_representable_range():
    torch.manual_seed(0)
    w = torch.randn(2, 16, 64)
    result = quantize_tensor(w, bits=4, group_size=32)
    unpacked = unpack_int4(result.q)
    assert int(unpacked.min()) >= qmin_for_bits(4) == -8
    assert int(unpacked.max()) <= qmax_for_bits(4) == 7
    assert int(unpacked.max()) == 7, "amax element must map to qmax"


def test_int4_logical_shape_survives_packing():
    w = torch.randn(3, 8, 64)
    result = quantize_tensor(w, bits=4, group_size=32)
    assert result.logical_shape == (3, 8, 64)
    assert result.q.shape == (3, 8, 32)


def test_expert_dimension_stays_aligned_at_int4():
    """Packing runs along K only; the expert axis must be untouched."""
    torch.manual_seed(0)
    experts, n, k = 5, 8, 64
    w = torch.randn(experts, n, k)
    result = quantize_tensor(w, bits=4, group_size=32)
    assert result.q.shape[:2] == (experts, n)
    assert result.scale.shape == (experts, n, k // 32)
    recovered = result.dequantize(torch.float32)
    # every expert must reconstruct independently and correctly
    for e in range(experts):
        single = quantize_tensor(w[e], bits=4, group_size=32).dequantize(torch.float32)
        assert torch.allclose(recovered[e], single)


def test_dequantize_without_the_packed_flag_fails_loudly():
    """Forgetting ``packed`` must never yield a plausible-looking tensor."""
    w = torch.randn(2, 4, 32)
    result = quantize_tensor(w, bits=4, group_size=16)
    with pytest.raises(RuntimeError):
        dequantize_tensor(result.q, result.scale, 4, 16, packed=False)
    correct = dequantize_tensor(result.q, result.scale, 4, 16, packed=True)
    assert correct.shape == w.shape


def test_storage_bytes_formula_matches_measurement():
    w = torch.randn(4, 16, 128)
    for bits in (8, 4):
        result = quantize_tensor(w, bits=bits, group_size=64)
        assert storage_bytes(w.numel(), bits, 64) == result.storage_bytes()


# --------------------------------------------------------------------------
# End to end through the adapter
# --------------------------------------------------------------------------


def test_expert_adapter_stores_packed_int4(moe_model):
    config = QuantConfig(bits=4, group_size=64, targets=("expert",))
    result = quantize_model(moe_model, config)
    assert all(t.packed for t in result.targets)
    block = moe_model.layers[0].mlp
    # w1 is [E, 2I, H] = [4, 128, 128]; packed stores half the K columns
    assert block._qw1.shape == (4, 128, 64)
    assert block.w1.shape == (4, 128, 128)


def test_int4_config_rejects_odd_group_size():
    with pytest.raises(ValueError, match="even group_size"):
        QuantConfig(bits=4, group_size=63)


def test_int4_and_int8_both_reconstruct_after_a_forward(moe_model):
    import copy

    x = torch.randn(6, 128, dtype=torch.bfloat16)
    reference = moe_model.layers[0].mlp(x)
    for bits, tolerance in ((8, 0.05), (4, 0.6)):
        clone = copy.deepcopy(moe_model)
        quantize_model(clone, QuantConfig(bits=bits, group_size=64, targets=("expert",)))
        out = clone.layers[0].mlp(x)
        relative = (out - reference).abs().max() / reference.abs().max()
        assert relative < tolerance, f"INT{bits} relative error {relative:.3f}"


def test_unpack_sign_extension_is_exhaustively_correct():
    """Arithmetic-shift unpacking must match the reference for every byte.

    The int8 shift trick relies on `>>` being an arithmetic (sign-propagating)
    shift. If that ever changed, low-nibble values 8..15 would decode as
    positive instead of -8..-1, silently corrupting half of every INT4 weight.
    """
    def reference(packed):
        p = packed.to(torch.int16) & 0xFF
        lo, hi = p & 0x0F, (p >> 4) & 0x0F
        lo = torch.where(lo >= 8, lo - 16, lo)
        hi = torch.where(hi >= 8, hi - 16, hi)
        shape = packed.shape
        return torch.stack([lo, hi], -1).reshape(*shape[:-1], shape[-1] * 2).to(torch.int8)

    every_byte = torch.arange(-128, 128, dtype=torch.int8).reshape(1, 256)
    assert torch.equal(unpack_int4(every_byte), reference(every_byte))


def test_every_int4_pair_survives_a_roundtrip():
    values = torch.arange(-8, 8, dtype=torch.int8)
    pairs = torch.stack(torch.meshgrid(values, values, indexing="ij"), -1).reshape(1, -1)
    assert torch.equal(unpack_int4(pack_int4(pairs)), pairs)


def test_unpack_rejects_non_int8_storage():
    with pytest.raises(ValueError, match="must be int8"):
        unpack_int4(torch.zeros(4, dtype=torch.int16))


def test_unpack_preserves_leading_dimensions():
    packed = torch.randint(-128, 127, (3, 5, 8), dtype=torch.int8)
    assert unpack_int4(packed).shape == (3, 5, 16)
