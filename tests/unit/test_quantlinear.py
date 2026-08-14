import torch

from LLaDA_Quant.config import QuantConfig
from LLaDA_Quant.runtime.linear import QuantLinear


def test_quantlinear_matches_linear_within_tolerance():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 128, bias=True)
    qlin = QuantLinear.from_linear(lin, bits=8, group_size=64, compute_dtype=torch.bfloat16)

    x = torch.randn(4, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = lin(x.to(torch.float32)).to(torch.bfloat16)
        out = qlin(x)
    assert out.shape == ref.shape
    err = (out.float() - ref.float()).abs()
    assert err.mean() < 1e-2
    assert err.max() < 0.1


def test_quantlinear_weight_layout():
    torch.manual_seed(1)
    lin = torch.nn.Linear(128, 64, bias=False)
    qlin = QuantLinear.from_linear(lin, bits=8, group_size=32)
    assert qlin.qweight.shape == (64, 128)
    assert qlin.qweight.dtype == torch.int8
    assert qlin.scale.shape == (64, 4)
    assert qlin.bias is None


def test_quantlinear_per_tensor_fallback():
    torch.manual_seed(2)
    lin = torch.nn.Linear(100, 50, bias=False)
    qlin = QuantLinear.from_linear(lin, bits=8, group_size=64)
    assert qlin.scale.shape == (50, 1)
    x = torch.randn(3, 100, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = lin(x.to(torch.float32)).to(torch.bfloat16)
        out = qlin(x)
    assert (out.float() - ref.float()).abs().max() < 0.1


def test_quantlinear_state_roundtrip():
    torch.manual_seed(3)
    lin = torch.nn.Linear(64, 32, bias=True)
    qlin = QuantLinear.from_linear(lin, bits=8, group_size=16)
    qlin2 = QuantLinear(64, 32, bits=8, group_size=16, bias=True)
    qlin2.load_state_dict(qlin.state_dict())
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(qlin(x), qlin2(x))