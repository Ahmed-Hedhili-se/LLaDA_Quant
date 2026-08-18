"""The LLaDA expert adapter: detection, both residency modes, restore."""

from __future__ import annotations

import pytest
import torch

from LLaDA_Quant import ExecutionMode, QuantConfig
from LLaDA_Quant.adapters.llada_moe import (
    describe_fused_expert_block,
    quantize_llada_experts,
    restore_llada_experts_from_buffers,
)
from LLaDA_Quant.runtime.moe import (
    is_packed_expert_block,
    materialize_expert_params,
    quant_result_from_buffers,
    quantize_fused_experts,
)

from conftest import FusedExpertBlock


def _config(bits=8, group_size=64, mode=ExecutionMode.PACKED):
    return QuantConfig(
        bits=bits, group_size=group_size, targets=("expert",), execution_mode=mode.value
    )


def test_quantization_leaves_the_router_bit_identical(expert_block):
    original_gate = expert_block.gate.weight.detach().clone()
    quantize_llada_experts(expert_block, _config())
    assert torch.equal(expert_block.gate.weight, original_gate)


def test_packed_mode_shapes_and_error_budget(expert_block):
    original = expert_block.w1.detach().clone()
    records = quantize_llada_experts(expert_block, _config())

    assert len(records) == 1 and records[0].name == ""
    assert is_packed_expert_block(expert_block)
    assert expert_block.w1.shape == original.shape
    assert expert_block.w1.dtype == torch.bfloat16
    assert expert_block._sw1.shape == (4, 128, 2)  # K=128, group 64 -> 2 groups

    error = (expert_block.w1.float() - original.float()).abs()
    assert error.max() < original.abs().max() / 127 * 1.1


def test_reference_mode_keeps_parameters_and_matches_packed_values(expert_block):
    import copy

    packed = copy.deepcopy(expert_block)
    reference = copy.deepcopy(expert_block)
    quantize_llada_experts(packed, _config(mode=ExecutionMode.PACKED))
    quantize_llada_experts(reference, _config(mode=ExecutionMode.REFERENCE))

    assert is_packed_expert_block(packed)
    assert not is_packed_expert_block(reference)
    assert torch.equal(packed.w1, reference.w1), "both modes must dequantize identically"
    assert torch.equal(packed._qw1, reference._qw1)


def test_quantization_is_reproducible():
    config = _config()
    a, b = FusedExpertBlock(), FusedExpertBlock()
    quantize_llada_experts(a, config)
    quantize_llada_experts(b, config)
    for attr in ("_qw1", "_sw1", "_qw2", "_sw2"):
        assert torch.equal(getattr(a, attr), getattr(b, attr))
    assert torch.equal(a.w1, b.w1)


def test_dequantization_happens_per_access_not_once(expert_block):
    quantize_llada_experts(expert_block, _config())
    first, second = expert_block.w1, expert_block.w1
    assert torch.equal(first, second)
    assert first.data_ptr() != second.data_ptr(), (
        "a cached tensor would defeat the point of PACKED mode"
    )


def test_materialize_refuses_to_write_into_a_packed_block(expert_block):
    weights = quantize_fused_experts(
        expert_block.w1.detach(), expert_block.w2.detach(), bits=8, group_size=64
    )
    quantize_llada_experts(expert_block, _config())
    with pytest.raises(RuntimeError, match="cannot be assigned"):
        materialize_expert_params(expert_block, weights)


def test_restore_reinstalls_access_after_a_bare_buffer_load(expert_block):
    quantize_llada_experts(expert_block, _config())
    expected = expert_block.w1.clone()
    # simulate a load that dropped the runtime metadata
    delattr(expert_block, "_llada_quant_meta")
    with pytest.raises(RuntimeError, match="reload through|Reload through"):
        _ = expert_block.w1
    assert restore_llada_experts_from_buffers(expert_block, _config()) == 1
    assert torch.equal(expert_block.w1, expected)


def test_quant_result_from_buffers_derives_the_group_size():
    torch.manual_seed(0)
    w1 = torch.randn(2, 8, 128) * 0.02
    w2 = torch.randn(2, 128, 4) * 0.02
    weights = quantize_fused_experts(w1, w2, bits=8, group_size=64)
    rebuilt = quant_result_from_buffers(weights.w1.q, weights.w1.scale, 8)
    assert rebuilt.group_size == 64
    assert rebuilt.logical_shape == (2, 8, 128)
    assert torch.equal(rebuilt.dequantize(torch.float32), weights.w1.dequantize(torch.float32))


def test_quant_result_from_buffers_handles_per_tensor_fallback():
    w = torch.randn(2, 4, 100)  # 64 does not divide 100
    result = quantize_fused_experts(w, w, bits=8, group_size=64).w1
    assert result.group_size == -1
    rebuilt = quant_result_from_buffers(result.q, result.scale, 8)
    assert rebuilt.group_size == -1


def test_quantize_fused_experts_shapes_and_error():
    torch.manual_seed(0)
    w1 = torch.randn(4, 512, 128) * 0.01
    w2 = torch.randn(4, 128, 256) * 0.01
    weights = quantize_fused_experts(w1, w2, bits=8, group_size=64)
    assert weights.w1.q.shape == w1.shape
    assert weights.w1.scale.shape == (4, 512, 2)
    assert weights.w2.scale.shape == (4, 128, 4)
    w1_hat, w2_hat = weights.dequantize(dtype=torch.bfloat16)
    assert (w1_hat.float() - w1).abs().max() < w1.abs().max() / 127 * 1.1
    assert (w2_hat.float() - w2).abs().max() < w2.abs().max() / 127 * 1.1


def test_expert_block_shape_reports_element_count():
    shape = describe_fused_expert_block(FusedExpertBlock())
    assert shape.numel == 4 * (2 * 64 * 128 + 128 * 64)
    assert "E=4" in shape.describe()
