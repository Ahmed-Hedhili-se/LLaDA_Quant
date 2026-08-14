import torch
import pytest

from LLaDA_Quant.adapters.llada_moe import quantize_llada_experts
from LLaDA_Quant.config import QuantConfig
from LLaDA_Quant.validation.metrics import (
    cosine_similarity,
    max_abs_error,
    router_overlap,
    summarize_metrics,
)


def test_router_overlap():
    a = torch.tensor([[0, 1, 2], [3, 4, 5]])
    b = torch.tensor([[0, 1, 7], [3, 4, 5]])
    assert router_overlap(a, b) == pytest.approx(5 / 6)
    assert router_overlap(a, a) == 1.0


def test_router_overlap_shape_mismatch():
    with pytest.raises(ValueError):
        router_overlap(torch.zeros(2, 3), torch.zeros(2, 4))


def test_metrics_values():
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b = torch.tensor([1.05, 2.0, 2.9], dtype=torch.float64)
    m = summarize_metrics(a, b)
    assert m["max_abs_error"] == pytest.approx(0.1)
    assert m["cosine_similarity"] > 0.999
    assert max_abs_error(a, b).item() == pytest.approx(0.1)
    assert 0.0 < cosine_similarity(a, b).item() <= 1.0


def test_expert_quant_moe_output_error():
    torch.manual_seed(0)
    n_e, hidden, inter = 8, 128, 256
    w1 = torch.randn(n_e, 2 * inter, hidden) * 0.02
    w2 = torch.randn(n_e, hidden, inter) * 0.02
    block = torch.nn.Module()
    block.w1 = torch.nn.Parameter(w1.to(torch.bfloat16))
    block.w2 = torch.nn.Parameter(w2.to(torch.bfloat16))
    x = torch.randn(16, hidden, dtype=torch.bfloat16)

    def moe_forward(w1_t, w2_t):
        xf = x.float()
        gate = torch.einsum("th,enh->etn", xf, w1_t.float()[:, :inter])
        up = torch.einsum("th,enh->etn", xf, w1_t.float()[:, inter:])
        h = torch.nn.functional.silu(gate[0]) * up[0]
        out = torch.einsum("tk,enk->etn", h, w2_t.float())
        return out.sum(dim=0)

    with torch.no_grad():
        ref = moe_forward(block.w1, block.w2)

    quantize_llada_experts(block, QuantConfig(bits=8, group_size=64, targets=("expert",)))
    with torch.no_grad():
        out = moe_forward(block.w1, block.w2)

    m = summarize_metrics(ref, out)
    assert m["max_abs_error"] / ref.abs().max() < 1 / 127 * 3
    assert m["cosine_similarity"] > 0.999