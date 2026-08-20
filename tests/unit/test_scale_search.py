"""MSE-optimal scale search: better INT4, identical storage contract.

``s = amax / Qmax`` spends the whole grid accommodating the single largest
weight in a group. At 8 bits that is nearly free; at 4 bits, with 16 levels,
it is why INT4 lands around 12% relative error against INT8's 0.65%. Searching
a clipping ratio recovers part of that.

The invariant these tests protect is that it changes **only the value of the
scale**. The dequantize formula, the packed layout, the byte count and the
checkpoint format are all unchanged, so nothing downstream — including a
future Triton kernel — needs to know the search ran.
"""

from __future__ import annotations

import copy

import pytest
import torch

from LLaDA_Quant import (
    QuantConfig,
    QuantizationManifest,
    load_quantized_weights,
    quantize_model,
    save_quantized_checkpoint,
)
from LLaDA_Quant.algorithms.symmetric import (
    _as_groups,
    quantize_tensor,
    search_group_scale,
    unpack_int4,
)

from conftest import TinyMoEModel


def rel_error(w, r):
    return ((r - w).norm() / w.norm()).item()


def roundtrip(w, **kw):
    return quantize_tensor(w, **kw).dequantize(torch.float32)


@pytest.fixture
def heavy_tailed():
    torch.manual_seed(0)
    return torch.distributions.StudentT(3.0).sample((32, 512)) * 0.01


# --------------------------------------------------------------------------
# It does what it claims
# --------------------------------------------------------------------------


def test_mse_search_reduces_int4_error(heavy_tailed):
    w = heavy_tailed
    amax = rel_error(w, roundtrip(w, bits=4, group_size=128, scale_search="amax"))
    mse = rel_error(w, roundtrip(w, bits=4, group_size=128, scale_search="mse"))
    assert mse < amax
    assert (1 - mse / amax) > 0.05, f"expected a real reduction, got {1 - mse / amax:.3f}"


def test_mse_search_never_increases_the_error_it_optimises():
    """Per group, the searched scale must beat or tie amax on squared error."""
    torch.manual_seed(1)
    w = torch.randn(16, 256)
    w_g, _ = _as_groups(w, 64)

    def group_sse(scale):
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.round(w_g.float() / safe).clamp(-8, 7)
        return ((q * safe - w_g.float()) ** 2).sum(dim=-1, keepdim=True)

    amax_scale = w_g.abs().amax(dim=-1, keepdim=True).float() / 7
    searched = search_group_scale(w_g, bits=4)
    assert torch.all(group_sse(searched) <= group_sse(amax_scale) + 1e-12)


def test_int8_barely_benefits_int4_does(heavy_tailed):
    """256 levels absorb an outlier; 16 do not. The search is an INT4 tool."""
    w = heavy_tailed
    gain = {}
    for bits in (8, 4):
        a = rel_error(w, roundtrip(w, bits=bits, group_size=128, scale_search="amax"))
        m = rel_error(w, roundtrip(w, bits=bits, group_size=128, scale_search="mse"))
        gain[bits] = 1 - m / a
    assert gain[4] > gain[8]
    assert gain[8] < 0.02, "INT8 should see almost nothing"


def test_search_is_deterministic(heavy_tailed):
    a = quantize_tensor(heavy_tailed, bits=4, group_size=64, scale_search="mse")
    b = quantize_tensor(heavy_tailed, bits=4, group_size=64, scale_search="mse")
    assert torch.equal(a.q, b.q) and torch.equal(a.scale, b.scale)


# --------------------------------------------------------------------------
# The storage contract is untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [8, 4])
def test_storage_layout_is_identical_to_amax(bits, heavy_tailed):
    a = quantize_tensor(heavy_tailed, bits=bits, group_size=128, scale_search="amax")
    m = quantize_tensor(heavy_tailed, bits=bits, group_size=128, scale_search="mse")
    assert a.q.shape == m.q.shape and a.q.dtype == m.q.dtype
    assert a.scale.shape == m.scale.shape and a.scale.dtype == m.scale.dtype
    assert a.packed == m.packed and a.group_size == m.group_size
    assert a.storage_bytes() == m.storage_bytes(), "the search must not cost a byte"


def test_dequantize_formula_is_unchanged(heavy_tailed):
    """Still W ~= q * s, zero-point-free, one scale per group."""
    result = quantize_tensor(heavy_tailed, bits=4, group_size=128, scale_search="mse")
    q = unpack_int4(result.q).float().reshape(32, 4, 128)
    manual = (q * result.scale.reshape(32, 4, 1).float()).reshape(32, 512)
    assert torch.allclose(manual, result.dequantize(torch.float32))


def test_searched_scales_are_never_larger_than_amax(heavy_tailed):
    """The search only ever clips inward, so nothing overflows the range."""
    w_g, _ = _as_groups(heavy_tailed, 128)
    amax = w_g.abs().amax(dim=-1, keepdim=True).float() / 7
    searched = search_group_scale(w_g, bits=4)
    assert torch.all(searched <= amax + 1e-12)
    assert torch.all(searched > 0)


def test_quantized_values_stay_in_range(heavy_tailed):
    """Clipping is intentional, but the stored codes must still be legal."""
    result = quantize_tensor(heavy_tailed, bits=4, group_size=128, scale_search="mse")
    unpacked = unpack_int4(result.q)
    assert int(unpacked.min()) >= -8 and int(unpacked.max()) <= 7


def test_zero_groups_get_zero_scale():
    w = torch.zeros(4, 64)
    result = quantize_tensor(w, bits=4, group_size=32, scale_search="mse")
    assert torch.all(result.scale == 0)
    assert torch.all(result.dequantize(torch.float32) == 0)


# --------------------------------------------------------------------------
# Wiring and validation
# --------------------------------------------------------------------------


def test_search_hyperparameters_are_validated():
    w = torch.randn(4, 1, 64)
    with pytest.raises(ValueError, match="grid must be >= 1"):
        search_group_scale(w, bits=4, grid=0)
    with pytest.raises(ValueError, match="max_shrink must lie in"):
        search_group_scale(w, bits=4, max_shrink=1.0)
    with pytest.raises(ValueError, match="scale_search must be"):
        quantize_tensor(torch.randn(4, 64), bits=4, group_size=32, scale_search="magic")


def test_config_validates_and_records_the_search():
    with pytest.raises(ValueError, match="scale_search must be"):
        QuantConfig(scale_search="magic")
    with pytest.raises(ValueError, match="search_grid must be >= 1"):
        QuantConfig(search_grid=0)
    config = QuantConfig(bits=4, group_size=64, scale_search="mse", search_grid=8)
    assert QuantConfig.from_dict(config.to_dict()) == config, "must survive a manifest roundtrip"


def test_config_reaches_the_expert_adapter(moe_model):
    """End to end: the search must actually run when the config asks for it."""
    base = copy.deepcopy(moe_model)
    original = base.layers[0].mlp.w1.float()

    amax_model = copy.deepcopy(base)
    amax_result = quantize_model(
        amax_model, QuantConfig(bits=4, group_size=64, scale_search="amax")
    )
    mse_model = copy.deepcopy(base)
    mse_result = quantize_model(
        mse_model, QuantConfig(bits=4, group_size=64, scale_search="mse")
    )

    assert amax_result.quantized_bytes == mse_result.quantized_bytes, "same bytes"
    err_amax = (amax_model.layers[0].mlp.w1.float() - original).norm()
    err_mse = (mse_model.layers[0].mlp.w1.float() - original).norm()
    assert err_mse < err_amax


def test_search_survives_a_checkpoint_roundtrip(tmp_path, moe_model):
    config = QuantConfig(bits=4, group_size=64, targets=("expert",), scale_search="mse")
    result = quantize_model(moe_model, config)
    save_quantized_checkpoint(
        moe_model,
        QuantizationManifest(config=config, targets=result.targets),
        str(tmp_path),
    )
    fresh = TinyMoEModel(seed=42)
    manifest = load_quantized_weights(fresh, str(tmp_path))
    assert manifest.config.scale_search == "mse"
    assert torch.equal(fresh.layers[0].mlp.w1, moe_model.layers[0].mlp.w1)


def test_chunking_is_exact_and_bounds_memory(heavy_tailed):
    """A 268M-element expert tensor needs an fp32 copy plus a temporary; doing
    that unchunked is several GB of transient VRAM per block."""
    w_g, _ = _as_groups(heavy_tailed, 128)
    whole = search_group_scale(w_g, bits=4, grid=8, chunk_elements=0)
    chunked = search_group_scale(w_g, bits=4, grid=8, chunk_elements=1024)
    assert torch.equal(whole, chunked), "chunking must not change the result"
    assert chunked.shape == whole.shape


def test_chunking_handles_a_single_row():
    w_g = torch.randn(1, 4, 64)
    assert search_group_scale(w_g, bits=4, grid=4, chunk_elements=1).shape == (1, 4, 1)
