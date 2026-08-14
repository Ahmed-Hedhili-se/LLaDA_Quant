import copy

import torch

from LLaDA_Quant.api import quantize_model, quantized_model
from LLaDA_Quant.config import QuantConfig, matches
from LLaDA_Quant.runtime.linear import QuantLinear


def test_matches_respects_excludes():
    cfg = QuantConfig(bits=8, targets=("linear",), exclude=("router", "norm"))
    assert matches("layers.0.self_attn.q_proj", cfg)
    assert not matches("layers.0.mlp.gate", cfg)
    assert not matches("layers.0.mlp.gate_proj", cfg)
    assert not matches("model.norm", cfg)


def test_quantize_model_linear_targets():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32, bias=False),
        torch.nn.Linear(32, 16, bias=False),
    )
    cfg = QuantConfig(bits=8, group_size=16, targets=("linear",))
    quantized = quantize_model(model, cfg)
    assert len(quantized) == 2
    assert isinstance(model[0], QuantLinear) and isinstance(model[1], QuantLinear)

    x = torch.randn(3, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3, 16)
    assert out.dtype == torch.bfloat16


def test_quantize_model_linear_excludes():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.Linear(32, 16),
    )
    cfg = QuantConfig(bits=8, group_size=16, targets=("linear",), exclude=("1",))
    quantized = quantize_model(model, cfg)
    assert quantized == ["0"]
    assert isinstance(model[0], QuantLinear)
    assert isinstance(model[1], torch.nn.Linear)


def test_quantized_model_preserves_original():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 32))
    cfg = QuantConfig(bits=8, group_size=16, targets=("linear",))
    clone = quantized_model(model, cfg)
    assert isinstance(clone[0], QuantLinear)
    assert isinstance(model[0], torch.nn.Linear)

    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = model[0](x.to(torch.float32)).to(torch.bfloat16)
        out = clone(x)
    assert (out.float() - ref.float()).abs().mean() < 1e-2


def test_quantize_model_moe_like_container():
    torch.manual_seed(0)
    container = torch.nn.Module()
    container.layers = torch.nn.ModuleList(
        [copy.deepcopy(make_block(i)) for i in range(3)]
    )
    cfg = QuantConfig(bits=8, group_size=32, targets=("expert",))
    names = quantize_model(container, cfg)
    assert len(names) == 3
    for layer in container.layers:
        assert hasattr(layer, "_qw1") and hasattr(layer, "_qw2")
        assert layer.w1.dtype == torch.bfloat16


def make_block(i):
    torch.manual_seed(i)
    block = torch.nn.Module()
    block.w1 = torch.nn.Parameter((torch.randn(4, 256, 64) * 0.01).to(torch.bfloat16))
    block.w2 = torch.nn.Parameter((torch.randn(4, 64, 128) * 0.01).to(torch.bfloat16))
    return block