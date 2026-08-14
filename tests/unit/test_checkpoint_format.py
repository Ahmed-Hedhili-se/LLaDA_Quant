import torch

from LLaDA_Quant.config import QuantConfig
from LLaDA_Quant.formats.manifest import MANIFEST_FILENAME, SOURCE_FILENAME, QuantEntry, QuantizationManifest
from LLaDA_Quant.formats.safetensors import (
    WEIGHTS_FILENAME,
    load_quantized_checkpoint,
    save_quantized_checkpoint,
)
from LLaDA_Quant.runtime.linear import QuantLinear


def test_manifest_json_roundtrip(tmp_path):
    cfg = QuantConfig(bits=8, group_size=128, targets=("expert",))
    manifest = QuantizationManifest(
        source_checkpoint="hf://inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        config=cfg,
        entries=[QuantEntry(tensor_name="layers.0._qw1", shape=[8, 2048, 512], bits=8, group_size=128, storage_dtype="int8", compute_dtype="bfloat16")],
    )
    manifest.save(tmp_path)
    assert (tmp_path / MANIFEST_FILENAME).exists()
    loaded = QuantizationManifest.from_dict(
        __import__("json").loads((tmp_path / MANIFEST_FILENAME).read_text())
    )
    assert loaded.config == cfg
    assert loaded.entries[0].tensor_name == "layers.0._qw1"


def test_checkpoint_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=True)
    qlin = QuantLinear.from_linear(lin, bits=8, group_size=16)
    cfg = QuantConfig(bits=8, group_size=16, targets=("linear",))
    manifest = QuantizationManifest(source_checkpoint="dummy.pt", config=cfg)
    save_quantized_checkpoint(qlin, manifest, str(tmp_path))

    assert (tmp_path / WEIGHTS_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()
    assert (tmp_path / SOURCE_FILENAME).exists()

    state, loaded = load_quantized_checkpoint(str(tmp_path))
    assert "qweight" in state and "scale" in state
    assert loaded.config == cfg

    qlin2 = QuantLinear(64, 32, bits=8, group_size=16, bias=True)
    qlin2.load_state_dict(state)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(qlin(x), qlin2(x))


def make_fake_fused_block(num_experts=8, hidden=128, intermediate=256, seed=0):
    torch.manual_seed(seed)
    block = torch.nn.Module()
    block.gate = torch.nn.Linear(hidden, num_experts, bias=False)
    block.w1 = torch.nn.Parameter((torch.randn(num_experts, 2 * intermediate, hidden) * 0.02).to(torch.bfloat16))
    block.w2 = torch.nn.Parameter((torch.randn(num_experts, hidden, intermediate) * 0.02).to(torch.bfloat16))
    return block


def test_load_quantized_weights_restores_expert_buffers(tmp_path):
    torch.manual_seed(0)
    cfg = QuantConfig(bits=8, group_size=64, targets=("expert",))
    block = make_fake_fused_block()
    from LLaDA_Quant.adapters.llada_moe import quantize_llada_experts

    quantize_llada_experts(block, cfg)
    fresh = make_fake_fused_block(seed=99)
    manifest = QuantizationManifest(config=cfg, entries=[QuantEntry(tensor_name="x", shape=[1], bits=8, group_size=64, storage_dtype="int8", compute_dtype="bfloat16")])
    save_quantized_checkpoint(block, manifest, str(tmp_path))

    from LLaDA_Quant.api import load_quantized_weights

    load_quantized_weights(fresh, str(tmp_path))
    for attr in ("_qw1", "_sw1", "_qw2", "_sw2"):
        assert torch.equal(getattr(fresh, attr), getattr(block, attr))
    assert torch.allclose(fresh.w1, block.w1)
    assert fresh.w1 is not fresh.w2