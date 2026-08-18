"""Public API surface: results, modes, and non-destructive variants."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from LLaDA_Quant import (
    ExecutionMode,
    QuantConfig,
    quantize_and_measure,
    quantize_model,
    quantized_model,
)
from LLaDA_Quant.runtime.linear import QuantLinear
from LLaDA_Quant.runtime.moe import is_packed_expert_block


def test_quantize_model_returns_an_audit_trail(moe_model):
    result = quantize_model(moe_model, QuantConfig(group_size=64, targets=("expert",)))
    assert result.names == ["layers.0.mlp", "layers.1.mlp"]
    assert len(result.expert_blocks) == 2 and result.linears == []
    assert result.quantized_bytes < result.source_bytes
    assert result.weight_ratio == pytest.approx(
        result.quantized_bytes / result.source_bytes
    )


def test_summary_states_the_mode_honestly(moe_model):
    import copy

    packed = quantize_model(copy.deepcopy(moe_model), QuantConfig(group_size=64))
    assert "PACKED mode" in packed.summary()
    assert "Resident memory drops" in packed.summary()

    reference = quantize_model(
        copy.deepcopy(moe_model), QuantConfig(group_size=64, execution_mode="reference")
    )
    assert "LARGER than unquantized" in reference.summary()
    assert "Validation only" in reference.summary()


def test_quantized_model_leaves_the_original_untouched(moe_model):
    clone = quantized_model(moe_model, QuantConfig(group_size=64, targets=("expert",)))
    assert is_packed_expert_block(clone.layers[0].mlp)
    assert not is_packed_expert_block(moe_model.layers[0].mlp)
    assert isinstance(moe_model.layers[0].mlp.w1, nn.Parameter)


def test_quantize_and_measure_returns_a_measured_comparison(moe_model):
    clone, result, comparison = quantize_and_measure(
        moe_model, QuantConfig(group_size=64, targets=("expert",))
    )
    assert comparison.baseline.total > comparison.quantized.total
    assert comparison.ratio < 1.0
    assert "0." in comparison.describe()
    assert len(result.targets) == 2
    assert not is_packed_expert_block(moe_model.layers[0].mlp), "original must be untouched"


def test_linear_quantization_preserves_call_semantics():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32, bias=False), nn.Linear(32, 16, bias=False))
    config = QuantConfig(bits=8, group_size=16, targets=("linear",), linear_include=("0", "1"))
    result = quantize_model(model, config)

    assert len(result.linears) == 2
    assert isinstance(model[0], QuantLinear) and isinstance(model[1], QuantLinear)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3, 16) and out.dtype == torch.bfloat16


def test_quantlinear_output_tracks_the_float_reference():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32))
    clone = quantized_model(
        model, QuantConfig(bits=8, group_size=16, targets=("linear",), linear_include=("0",))
    )
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        reference = model[0](x.to(torch.float32)).to(torch.bfloat16)
        out = clone(x)
    assert (out.float() - reference.float()).abs().mean() < 1e-2


def test_expert_and_linear_targets_compose(moe_model):
    config = QuantConfig(
        bits=8,
        group_size=64,
        targets=("expert", "linear"),
        linear_include=("q_proj",),
        expect_expert_blocks=2,
        expect_linears=2,
    )
    result = quantize_model(moe_model, config)
    assert len(result.expert_blocks) == 2
    assert len(result.linears) == 2
    assert is_packed_expert_block(moe_model.layers[0].mlp)
    assert isinstance(moe_model.layers[0].q_proj, QuantLinear)


def test_execution_mode_enum_and_reduces_memory_flag():
    assert QuantConfig().mode is ExecutionMode.PACKED
    assert QuantConfig().reduces_memory
    assert not QuantConfig(execution_mode="reference").reduces_memory
