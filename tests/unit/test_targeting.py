"""Phase 4: a quantizer must never silently convert the wrong module.

The old targeting matched ``"expert" in name or "mlp" in name`` and detected
expert blocks by "3-D w1/w2 with an even second dim". Both are guesses. These
tests pin the replacement: structural matching for experts, explicit naming
for linears, and a loud failure whenever the match set is not what was asked
for.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from LLaDA_Quant import QuantConfig, TargetingError, quantize_model
from LLaDA_Quant.adapters.llada_moe import (
    describe_fused_expert_block,
    find_expert_blocks,
    is_fused_expert_block,
)
from LLaDA_Quant.config import is_excluded, matches_linear
from LLaDA_Quant.runtime.linear import QuantLinear

from conftest import FusedExpertBlock


# --------------------------------------------------------------------------
# Structural expert detection
# --------------------------------------------------------------------------


def test_detects_the_real_fused_layout(expert_block):
    shape = describe_fused_expert_block(expert_block)
    assert shape is not None
    assert (shape.num_experts, shape.hidden, shape.intermediate) == (4, 128, 64)


@pytest.mark.parametrize(
    "w1_shape, w2_shape, why",
    [
        ((4, 128, 128), (5, 128, 64), "expert counts disagree"),
        ((4, 128, 128), (4, 99, 64), "hidden dims disagree"),
        ((4, 130, 128), (4, 128, 64), "w1 is not 2x the intermediate"),
        ((4, 128), (4, 128, 64), "w1 is not 3-D"),
    ],
)
def test_rejects_shapes_that_only_look_like_experts(w1_shape, w2_shape, why):
    block = nn.Module()
    block.w1 = nn.Parameter(torch.zeros(*w1_shape))
    block.w2 = nn.Parameter(torch.zeros(*w2_shape))
    assert not is_fused_expert_block(block), f"should have been rejected: {why}"


def test_a_module_named_mlp_is_not_enough():
    """The old rule matched on the name; this one matches on the layout."""
    model = nn.Module()
    model.mlp = nn.Linear(8, 8)
    assert find_expert_blocks(model, QuantConfig(targets=("expert",))) == []


def test_a_correctly_shaped_block_with_an_odd_name_is_still_found():
    model = nn.Module()
    model.something_unexpected = FusedExpertBlock()
    found = find_expert_blocks(model, QuantConfig(targets=("expert",)))
    assert [name for name, _, _ in found] == ["something_unexpected"]


# --------------------------------------------------------------------------
# Exclusion is component-wise, not substring
# --------------------------------------------------------------------------


def test_excludes_match_path_components():
    config = QuantConfig(targets=("linear",), exclude=("router", "*norm*", "lm_head"))
    assert is_excluded("model.router", config)
    assert is_excluded("layers.0.input_layernorm", config)
    assert is_excluded("lm_head", config)
    assert not is_excluded("layers.0.q_proj", config)


def test_exclusion_does_not_leak_across_components():
    """``gate`` must not knock out ``gate_proj`` unless asked to."""
    config = QuantConfig(targets=("linear",), exclude=("gate",))
    assert is_excluded("layers.0.mlp.gate", config)
    assert not is_excluded("layers.0.mlp.gate_proj", config)


# --------------------------------------------------------------------------
# Linears require explicit naming
# --------------------------------------------------------------------------


def test_linear_include_has_no_implicit_default(moe_model):
    config = QuantConfig(targets=("expert", "linear"), group_size=64)
    result = quantize_model(moe_model, config)
    assert result.linears == [], "an empty linear_include must quantize no linear"
    assert len(result.expert_blocks) == 2


def test_named_linears_are_quantized_and_others_are_not(moe_model):
    config = QuantConfig(
        targets=("linear",), group_size=64, linear_include=("q_proj",), expect_linears=2
    )
    result = quantize_model(moe_model, config)
    assert result.names == ["layers.0.q_proj", "layers.1.q_proj"]
    assert isinstance(moe_model.layers[0].q_proj, QuantLinear)
    assert isinstance(moe_model.lm_head, nn.Linear)
    assert not isinstance(moe_model.lm_head, QuantLinear)


def test_matches_linear_accepts_globs_and_full_paths():
    config = QuantConfig(targets=("linear",), linear_include=("*_proj", "layers.0.special"))
    assert matches_linear("layers.3.q_proj", config)
    assert matches_linear("layers.0.special", config)
    assert not matches_linear("layers.1.special", config)


def test_excludes_win_over_includes():
    config = QuantConfig(
        targets=("linear",), linear_include=("*",), exclude=("lm_head", "*norm*")
    )
    assert matches_linear("layers.0.q_proj", config)
    assert not matches_linear("lm_head", config)


# --------------------------------------------------------------------------
# Loud failure
# --------------------------------------------------------------------------


def test_matching_nothing_is_an_error_not_a_silent_no_op():
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(TargetingError, match="matched no modules"):
        quantize_model(model, QuantConfig(targets=("expert",)))


def test_allow_no_matches_is_an_explicit_opt_out():
    model = nn.Sequential(nn.Linear(4, 4))
    result = quantize_model(model, QuantConfig(targets=("expert",), allow_no_matches=True))
    assert result.targets == []


def test_wrong_expert_count_fails_loudly(moe_model):
    with pytest.raises(TargetingError, match="expected 5 expert block"):
        quantize_model(moe_model, QuantConfig(targets=("expert",), group_size=64,
                                              expect_expert_blocks=5))


def test_wrong_linear_count_fails_loudly(moe_model):
    with pytest.raises(TargetingError, match="expected 9 linear"):
        quantize_model(
            moe_model,
            QuantConfig(targets=("linear",), group_size=64, linear_include=("q_proj",),
                        expect_linears=9),
        )


def test_double_quantization_is_refused(moe_model):
    config = QuantConfig(targets=("expert",), group_size=64)
    quantize_model(moe_model, config)
    with pytest.raises(RuntimeError, match="already quantized"):
        quantize_model(moe_model, config)


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------


def test_result_records_what_was_converted(moe_model):
    config = QuantConfig(bits=8, group_size=64, targets=("expert", "linear"),
                         linear_include=("q_proj",))
    result = quantize_model(moe_model, config)
    assert len(result.targets) == 4
    expert = result.expert_blocks[0]
    assert expert.name == "layers.0.mlp"
    assert expert.kind == "expert"
    assert expert.shapes == {"w1": [4, 128, 128], "w2": [4, 128, 64]}
    assert expert.bits == 8 and expert.group_size == 64
    assert expert.execution_mode == "packed"
    assert 0 < expert.quantized_bytes < expert.source_bytes
    assert expert.compression_ratio < 1.0


def test_config_rejects_unknown_targets_and_modes():
    with pytest.raises(ValueError, match="unknown targets"):
        QuantConfig(targets=("attention",))
    with pytest.raises(ValueError, match="execution_mode must be"):
        QuantConfig(execution_mode="fast")


def test_config_roundtrips_through_json():
    config = QuantConfig(bits=4, group_size=64, targets=("expert", "linear"),
                         linear_include=("q_proj", "k_proj"), execution_mode="reference",
                         expect_expert_blocks=16)
    import json

    restored = QuantConfig.from_dict(json.loads(config.to_json()))
    assert restored == config
