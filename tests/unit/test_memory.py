"""Phase 1 regression tests: the resident-memory bug must not come back.

The bug: the expert adapter registered packed buffers while leaving the BF16
``w1``/``w2`` Parameters alive, so "quantization" grew the resident model by
52% and the benchmark still reported a 47% saving. Every test here fails if
that behaviour returns in any form.
"""

from __future__ import annotations

import copy

import pytest
import torch

from LLaDA_Quant import (
    ExecutionMode,
    QuantConfig,
    compare_resident_memory,
    quantize_and_measure,
    quantize_model,
    resident_memory,
)
from LLaDA_Quant.runtime.moe import is_packed_expert_block


def _expert_config(bits=8, mode=ExecutionMode.PACKED, **kw):
    return QuantConfig(
        bits=bits, group_size=64, targets=("expert",), execution_mode=mode.value, **kw
    )


def test_packed_mode_actually_reduces_resident_memory(moe_model):
    _, _, comparison = quantize_and_measure(moe_model, _expert_config())
    assert comparison.is_saving, comparison.describe()
    assert comparison.ratio < 1.0
    assert comparison.saved_bytes > 0


def test_packed_mode_removes_the_bf16_parameters(moe_model):
    quantize_model(moe_model, _expert_config())
    for layer in moe_model.layers:
        names = dict(layer.mlp.named_parameters())
        assert "w1" not in names, "BF16 w1 is still a resident Parameter"
        assert "w2" not in names, "BF16 w2 is still a resident Parameter"
        assert is_packed_expert_block(layer.mlp)
        # ...but attribute access still works, so the model code is untouched
        assert layer.mlp.w1.shape == (4, 128, 128)
        assert layer.mlp.w1.dtype == torch.bfloat16


def test_packed_state_dict_has_no_bf16_expert_copy(moe_model):
    quantize_model(moe_model, _expert_config())
    keys = set(moe_model.state_dict())
    assert not [k for k in keys if k.endswith(("mlp.w1", "mlp.w2"))]
    assert {"layers.0.mlp._qw1", "layers.0.mlp._sw1"} <= keys


def test_int4_saves_strictly_more_resident_memory_than_int8(moe_model):
    int8_model, int8_result, int8 = quantize_and_measure(moe_model, _expert_config(bits=8))
    int4_model, int4_result, int4 = quantize_and_measure(moe_model, _expert_config(bits=4))
    assert int4.quantized.total < int8.quantized.total
    assert int4.saved_bytes > int8.saved_bytes

    # The int8 storage buffers themselves must be ~2x the int4 ones. Scales are
    # identical in both, so the ratio is slightly above 0.5 rather than exactly.
    def q_bytes(model):
        return sum(
            layer.mlp._qw1.numel() * layer.mlp._qw1.element_size()
            + layer.mlp._qw2.numel() * layer.mlp._qw2.element_size()
            for layer in model.layers
        )

    assert q_bytes(int4_model) * 2 == q_bytes(int8_model)
    assert int4_result.weight_ratio < int8_result.weight_ratio


def test_reference_mode_is_larger_and_says_so(moe_model):
    _, result, comparison = quantize_and_measure(
        moe_model, _expert_config(mode=ExecutionMode.REFERENCE)
    )
    assert not comparison.is_saving
    assert comparison.ratio > 1.0
    assert "REGRESSION" in comparison.describe()
    assert "LARGER than unquantized" in result.summary()
    assert not result.config.reduces_memory


def test_reference_mode_keeps_writable_parameters(moe_model):
    quantize_model(moe_model, _expert_config(mode=ExecutionMode.REFERENCE))
    block = moe_model.layers[0].mlp
    assert not is_packed_expert_block(block)
    assert isinstance(block.w1, torch.nn.Parameter)
    assert hasattr(block, "_qw1")


def test_packed_weights_are_read_only(moe_model):
    quantize_model(moe_model, _expert_config())
    block = moe_model.layers[0].mlp
    with pytest.raises(AttributeError):
        block.w1 = torch.zeros(1)


def test_in_place_weight_loaders_are_blocked_in_packed_mode(moe_model):
    """LLaDA's real block writes ``self.w1[i].copy_(...)`` at build time.

    In PACKED mode that write lands in a dequantized temporary and vanishes.
    Shadowing the method turns a silent no-op into an error that says to load
    the weights before quantizing.
    """
    import LLaDA_Quant.runtime.moe as moe_module
    from conftest import FusedExpertBlock

    loaded = []
    FusedExpertBlock.load_state_dict_from_unfused = lambda self, src: loaded.append(src)
    # the shadow set is computed when the packed subclass is generated, and
    # earlier tests already cached one for this base class
    moe_module._PACKED_CLASS_CACHE.clear()
    try:
        moe_model.layers[0].mlp.load_state_dict_from_unfused("before")  # not yet quantized
        assert loaded == ["before"]

        quantize_model(moe_model, _expert_config())
        with pytest.raises(RuntimeError, match="silently discarded"):
            moe_model.layers[0].mlp.load_state_dict_from_unfused("after")
        assert loaded == ["before"], "the blocked call must not have run"
    finally:
        del FusedExpertBlock.load_state_dict_from_unfused
        moe_module._PACKED_CLASS_CACHE.clear()


def test_reference_mode_leaves_in_place_loaders_alone(moe_model):
    """REFERENCE mode keeps real Parameters, so the write would land."""
    quantize_model(moe_model, _expert_config(mode=ExecutionMode.REFERENCE))
    block = moe_model.layers[0].mlp
    with torch.no_grad():
        block.w1[0].copy_(torch.ones_like(block.w1[0]))
    assert torch.all(block.w1[0] == 1)


def test_resident_memory_counts_shared_storage_once():
    linear = torch.nn.Linear(8, 8, bias=False)
    module = torch.nn.Module()
    module.a = linear
    module.b = linear  # same object twice
    report = resident_memory(module)
    assert report.tensor_count == 1
    assert report.total == 8 * 8 * linear.weight.element_size()


def test_memory_comparison_reports_growth_as_negative_saving(moe_model):
    clone = copy.deepcopy(moe_model)
    quantize_model(clone, _expert_config(mode=ExecutionMode.REFERENCE))
    comparison = compare_resident_memory(moe_model, clone, label="ref mode")
    assert comparison.saved_bytes < 0
    assert comparison.to_dict()["saved_pct"] < 0
