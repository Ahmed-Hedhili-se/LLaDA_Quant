import copy

import torch

from llada_quant.adapters.llada_moe import is_fused_expert_block, quantize_llada_experts
from llada_quant.config import QuantConfig
from llada_quant.runtime.moe import quantize_fused_experts


def make_fake_fused_block(num_experts=8, hidden=128, intermediate=256, seed=0):
    torch.manual_seed(seed)
    block = torch.nn.Module()
    block.gate = torch.nn.Linear(hidden, num_experts, bias=False)
    block.w1 = torch.nn.Parameter((torch.randn(num_experts, 2 * intermediate, hidden) * 0.02).to(torch.bfloat16))
    block.w2 = torch.nn.Parameter((torch.randn(num_experts, hidden, intermediate) * 0.02).to(torch.bfloat16))
    return block


def test_is_fused_expert_block_detection():
    assert is_fused_expert_block(make_fake_fused_block())
    lin = torch.nn.Linear(4, 4)
    assert not is_fused_expert_block(lin)
    torch.manual_seed(0)
    plain = torch.nn.Parameter(torch.randn(4, 8, 16))
    assert not is_fused_expert_block(plain)


def test_expert_quantization_preserves_router_and_shapes():
    block = make_fake_fused_block()
    orig_w1 = block.w1.detach().clone()
    orig_w2 = block.w2.detach().clone()
    orig_gate = block.gate.weight.detach().clone()

    cfg = QuantConfig(bits=8, group_size=64, targets=("expert",))
    quantized = quantize_llada_experts(block, cfg)
    assert quantized == [""]

    assert torch.equal(block.gate.weight, orig_gate)
    assert block.w1.shape == orig_w1.shape
    assert block.w2.shape == orig_w2.shape
    assert block.w1.dtype == torch.bfloat16

    err1 = (block.w1.float() - orig_w1.float()).abs()
    err2 = (block.w2.float() - orig_w2.float()).abs()
    assert err1.mean() < 1e-2 and err2.mean() < 1e-2
    assert err1.max() < orig_w1.abs().max() / 127 * 1.1
    assert torch.all(block._qw1 >= -128) and torch.all(block._qw1 <= 127)
    assert block._sw1.shape == (8, 512, 2)


def test_expert_quantization_is_reproducible():
    cfg = QuantConfig(bits=8, group_size=64, targets=("expert",))
    b1 = make_fake_fused_block()
    b2 = make_fake_fused_block()
    quantize_llada_experts(b1, cfg)
    quantize_llada_experts(b2, cfg)
    for attr in ("_qw1", "_sw1", "_qw2", "_sw2", "w1", "w2"):
        assert torch.equal(getattr(b1, attr), getattr(b2, attr))


def test_quantize_fused_experts_shapes():
    torch.manual_seed(0)
    w1 = torch.randn(4, 512, 128) * 0.01
    w2 = torch.randn(4, 128, 256) * 0.01
    qe = quantize_fused_experts(w1, w2, bits=8, group_size=64)
    assert qe.w1.q.shape == w1.shape
    assert qe.w1.scale.shape == (4, 512, 2)
    assert qe.w2.scale.shape == (4, 128, 4)
    w1_hat, w2_hat = qe.dequantize(dtype=torch.bfloat16)
    assert (w1_hat.float() - w1).abs().max() < w1.abs().max() / 127 * 1.1
    assert (w2_hat.float() - w2).abs().max() < w2.abs().max() / 127 * 1.1