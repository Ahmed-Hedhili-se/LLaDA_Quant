"""Phase 3: a quantized checkpoint must be self-contained and non-redundant.

The bug: the artifact stored the BF16 ``w1``/``w2`` *and* the packed buffers,
even though the load path recomputes the former from the latter. The
"quantized" checkpoint was 1.52x the size of the unquantized one.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from LLaDA_Quant import (
    ExecutionMode,
    QuantConfig,
    QuantizationManifest,
    load_quantized_weights,
    quantize_model,
    save_quantized_checkpoint,
)
from LLaDA_Quant.formats.manifest import (
    MANIFEST_FILENAME,
    SOURCE_FILENAME,
    QuantEntry,
    TargetedModule,
    tensor_hash,
)
from LLaDA_Quant.formats.safetensors import (
    checkpoint_size_bytes,
    derivable_tensor_names,
    find_weights_file,
    load_quantized_checkpoint,
)

from conftest import TinyMoEModel


def _save(model, config, directory):
    result = quantize_model(model, config)
    manifest = QuantizationManifest(
        source_checkpoint="hf://test/model", config=config, targets=result.targets
    )
    path = save_quantized_checkpoint(model, manifest, directory)
    return result, manifest, path


def _bf16_reference_size(model, directory):
    path = os.path.join(directory, "bf16.safetensors")
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, path)
    size = os.path.getsize(path)
    os.remove(path)
    return size


# --------------------------------------------------------------------------
# No redundancy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("mode", [ExecutionMode.PACKED, ExecutionMode.REFERENCE])
def test_checkpoint_contains_no_bf16_expert_copy(tmp_path, bits, mode):
    torch.manual_seed(0)
    model = TinyMoEModel()
    config = QuantConfig(bits=bits, group_size=64, targets=("expert",),
                         execution_mode=mode.value)
    _save(model, config, str(tmp_path))
    keys = set(load_file(find_weights_file(str(tmp_path))))
    assert not [k for k in keys if k.endswith(("mlp.w1", "mlp.w2"))], (
        f"BF16 expert weights leaked into an INT{bits} {mode.value} checkpoint"
    )
    assert "layers.0.mlp._qw1" in keys and "layers.0.mlp._sw1" in keys


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_checkpoint_is_smaller_than_the_bf16_one(tmp_path, bits):
    torch.manual_seed(0)
    model = TinyMoEModel()
    bf16_size = _bf16_reference_size(model, str(tmp_path))
    _, _, path = _save(model, QuantConfig(bits=bits, group_size=64, targets=("expert",)),
                       str(tmp_path))
    assert os.path.getsize(path) < bf16_size


def test_int4_checkpoint_is_smaller_than_int8(tmp_path):
    torch.manual_seed(0)
    sizes = {}
    for bits in (8, 4):
        directory = tmp_path / f"int{bits}"
        _, _, path = _save(TinyMoEModel(), QuantConfig(bits=bits, group_size=64,
                                                       targets=("expert",)), str(directory))
        sizes[bits] = os.path.getsize(path)
    assert sizes[4] < sizes[8]


def test_derivable_names_lists_exactly_the_expert_params():
    manifest = QuantizationManifest(
        targets=[
            TargetedModule("layers.0.mlp", "expert", "Block", {}, 8, 64, False, "packed", 10, 5),
            TargetedModule("layers.0.q_proj", "linear", "Linear", {}, 8, 64, False, "packed", 4, 2),
        ]
    )
    assert derivable_tensor_names(manifest) == {"layers.0.mlp.w1", "layers.0.mlp.w2"}


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("mode", [ExecutionMode.PACKED, ExecutionMode.REFERENCE])
def test_save_load_is_bit_exact(tmp_path, bits, mode):
    torch.manual_seed(0)
    model = TinyMoEModel()
    config = QuantConfig(bits=bits, group_size=64, targets=("expert",),
                         execution_mode=mode.value)
    _save(model, config, str(tmp_path))

    fresh = TinyMoEModel(seed=99)  # deliberately different weights
    manifest = load_quantized_weights(fresh, str(tmp_path))
    assert manifest.config == config

    for original, loaded in zip(model.layers, fresh.layers):
        assert torch.equal(original.mlp._qw1, loaded.mlp._qw1)
        assert torch.equal(original.mlp._sw1, loaded.mlp._sw1)
        assert torch.equal(original.mlp._qw2, loaded.mlp._qw2)
        assert torch.equal(original.mlp._sw2, loaded.mlp._sw2)
        assert torch.equal(original.mlp.w1, loaded.mlp.w1), "reconstructed w1 differs"
        assert torch.equal(original.mlp.w2, loaded.mlp.w2), "reconstructed w2 differs"


def test_loading_into_a_plain_model_registers_buffers_and_reconstructs(tmp_path):
    torch.manual_seed(0)
    model = TinyMoEModel()
    config = QuantConfig(bits=8, group_size=64, targets=("expert",))
    _save(model, config, str(tmp_path))

    plain = TinyMoEModel(seed=7)
    assert not hasattr(plain.layers[0].mlp, "_qw1")
    load_quantized_weights(plain, str(tmp_path))
    assert hasattr(plain.layers[0].mlp, "_qw1")
    assert "w1" not in dict(plain.layers[0].mlp.named_parameters())
    assert plain.layers[0].mlp.w1.shape == (4, 128, 128)


def test_forward_after_reload_matches_forward_before_save(tmp_path):
    torch.manual_seed(0)
    model = TinyMoEModel()
    config = QuantConfig(bits=8, group_size=64, targets=("expert",))
    _save(model, config, str(tmp_path))
    x = torch.randn(5, 128, dtype=torch.bfloat16)
    before = model.layers[0].mlp(x)

    fresh = TinyMoEModel(seed=3)
    load_quantized_weights(fresh, str(tmp_path))
    assert torch.equal(fresh.layers[0].mlp(x), before)


def test_group_size_is_recovered_from_the_scales_not_the_config(tmp_path):
    """A group size that does not divide K falls back to per-tensor at save
    time; reload must follow the stored scales, not the config's request."""
    torch.manual_seed(0)
    model = TinyMoEModel()
    config = QuantConfig(bits=8, group_size=48, targets=("expert",))  # 48 does not divide 128
    _save(model, config, str(tmp_path))
    assert model.layers[0].mlp._sw1.shape[-1] == 1, "expected per-tensor fallback"

    fresh = TinyMoEModel(seed=5)
    load_quantized_weights(fresh, str(tmp_path))
    assert torch.equal(fresh.layers[0].mlp.w1, model.layers[0].mlp.w1)


def test_strict_load_still_catches_a_real_mismatch(tmp_path):
    torch.manual_seed(0)
    model = TinyMoEModel()
    _save(model, QuantConfig(bits=8, group_size=64, targets=("expert",)), str(tmp_path))
    with pytest.raises(RuntimeError, match="state dict mismatch"):
        load_quantized_weights(TinyMoEModel(layers=1), str(tmp_path))


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def test_manifest_records_the_targeting_audit_and_totals(tmp_path):
    torch.manual_seed(0)
    model = TinyMoEModel()
    config = QuantConfig(bits=4, group_size=64, targets=("expert",))
    result, _, _ = _save(model, config, str(tmp_path))

    with open(tmp_path / MANIFEST_FILENAME) as f:
        data = json.load(f)
    assert data["config"]["bits"] == 4
    assert data["config"]["execution_mode"] == "packed"
    assert [t["name"] for t in data["targets"]] == ["layers.0.mlp", "layers.1.mlp"]
    assert all(t["packed"] for t in data["targets"])
    assert data["totals"]["module_count"] == 2
    assert data["totals"]["quantized_bytes"] < data["totals"]["source_bytes"]
    assert os.path.exists(tmp_path / SOURCE_FILENAME)


def test_manifest_roundtrips_through_json():
    config = QuantConfig(bits=4, group_size=64)
    manifest = QuantizationManifest(
        source_checkpoint="hf://x",
        config=config,
        entries=[QuantEntry("w1", [1, 2], 4, 64, "int8", "bfloat16")],
        targets=[TargetedModule("m", "expert", "B", {"w1": [1, 2]}, 4, 64, True, "packed", 8, 3)],
    )
    restored = QuantizationManifest.from_dict(json.loads(manifest.to_json()))
    assert restored.config == config
    assert restored.targets[0].name == "m"
    assert restored.targets[0].packed is True
    assert restored.entries[0].tensor_name == "w1"


def test_checkpoint_size_counts_every_file(tmp_path):
    torch.manual_seed(0)
    _save(TinyMoEModel(), QuantConfig(bits=8, group_size=64, targets=("expert",)), str(tmp_path))
    total = checkpoint_size_bytes(str(tmp_path))
    parts = sum(
        os.path.getsize(tmp_path / f) for f in os.listdir(tmp_path)
    )
    assert total == parts


def test_tensor_hash_is_value_deterministic():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_hash(a) == tensor_hash(a.clone())
    assert tensor_hash(a) != tensor_hash(a + 1)


def test_load_reports_a_missing_weights_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_weights_file(str(tmp_path))
