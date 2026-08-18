"""Router top-k capture: exact, per-model, and detachable.

``TritonFusedMoEBlock`` computes ``topk_ids`` inside ``forward`` and never
exposes it, so measuring router overlap needs a pre-hook that recomputes the
routing from the block's input. These tests pin the two things that make that
trustworthy: the recomputation is bit-identical to the block's own, and two
models' captures never contaminate each other.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from LLaDA_Quant import QuantConfig, quantize_model
from LLaDA_Quant.trajectory import (
    DiffusionState,
    attach_router_capture,
    gates_fn_for,
    router_fn_for,
)
from LLaDA_Quant.trajectory.llada import RouterCapture
from LLaDA_Quant.validation.metrics import router_overlap

from conftest import HIDDEN, NUM_EXPERTS, FusedExpertBlock


def _state(tokens: int = 6) -> DiffusionState:
    ids = torch.zeros(1, tokens, dtype=torch.long)
    return DiffusionState(step=0, input_ids=ids, mask_positions=torch.ones_like(ids).bool())


def _drive(model, x):
    """Feed the same ``x`` to every fused block.

    Note this does *not* chain layers, so each block's router sees an identical
    input in both models. That is deliberate: it isolates the router itself. In
    a real forward the hidden state is chained, so routing at layer *i* also
    reflects quantization error accumulated in layers 0..i-1 — which is the
    divergence the on-GPU experiment measures and these tests cannot.
    """
    with torch.no_grad():
        for layer in model.layers:
            layer.mlp(x)


# --------------------------------------------------------------------------
# Exactness
# --------------------------------------------------------------------------


def test_capture_populates_one_entry_per_block(moe_model):
    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16)
    with attach_router_capture(moe_model, top_k=2) as capture:
        _drive(moe_model, x)
        ids = capture.topk_ids
    assert sorted(ids) == ["layers.0.mlp", "layers.1.mlp"]
    for tensor in ids.values():
        assert tensor.shape == (6, 2)
        assert tensor.dtype == torch.int64


def test_recomputation_is_identical_to_the_blocks_own_routing(moe_model):
    """The block does softmax(gate(x), float32) then topk. So must the hook."""
    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16)
    block = moe_model.layers[0].mlp
    with torch.no_grad():
        expected = (
            F.softmax(block.gate(x.reshape(-1, HIDDEN)), dim=-1, dtype=torch.float32)
            .topk(2, dim=-1)
            .indices
        )
    with attach_router_capture(moe_model, top_k=2) as capture:
        _drive(moe_model, x)
        assert torch.equal(capture.topk_ids["layers.0.mlp"], expected.cpu())


def test_gates_are_the_raw_router_logits(moe_model):
    x = torch.randn(4, HIDDEN, dtype=torch.bfloat16)
    block = moe_model.layers[0].mlp
    with torch.no_grad():
        expected = block.gate(x.reshape(-1, HIDDEN)).float()
    with attach_router_capture(moe_model, top_k=2) as capture:
        _drive(moe_model, x)
        gates = capture.gates["layers.0.mlp"]
    assert gates.shape == (4, NUM_EXPERTS)
    assert torch.allclose(gates, expected.cpu())


def test_capture_reflects_only_the_most_recent_forward(moe_model):
    with attach_router_capture(moe_model, top_k=2) as capture:
        _drive(moe_model, torch.randn(4, HIDDEN, dtype=torch.bfloat16))
        first = capture.topk_ids["layers.0.mlp"].clone()
        _drive(moe_model, torch.randn(9, HIDDEN, dtype=torch.bfloat16) * 5)
        second = capture.topk_ids["layers.0.mlp"]
    assert second.shape[0] == 9, "a later forward must replace, not append"
    assert first.shape[0] == 4


def test_clear_empties_the_capture(moe_model):
    with attach_router_capture(moe_model, top_k=2) as capture:
        _drive(moe_model, torch.randn(4, HIDDEN, dtype=torch.bfloat16))
        assert capture.topk_ids
        capture.clear()
        assert capture.topk_ids == {} and capture.gates == {}


# --------------------------------------------------------------------------
# Per-model isolation
# --------------------------------------------------------------------------


def test_two_models_do_not_contaminate_each_other(moe_model):
    """capture_shared runs both models; one shared registry would overwrite."""
    quantized = copy.deepcopy(moe_model)
    quantize_model(quantized, QuantConfig(bits=4, group_size=64, scale_search="mse"))
    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16)

    ref_cap = attach_router_capture(moe_model, top_k=2)
    qnt_cap = attach_router_capture(quantized, top_k=2)
    try:
        _drive(moe_model, x)
        ref_ids = ref_cap.topk_ids["layers.0.mlp"].clone()
        _drive(quantized, x)
        assert torch.equal(ref_cap.topk_ids["layers.0.mlp"], ref_ids), (
            "the quantized model's forward overwrote the reference's capture"
        )
    finally:
        ref_cap.remove()
        qnt_cap.remove()


def test_router_fn_dispatches_by_model_identity(moe_model):
    other = copy.deepcopy(moe_model)
    with torch.no_grad():
        other.layers[0].mlp.gate.weight.mul_(-1.0)  # force different routing
    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16)

    a = attach_router_capture(moe_model, top_k=2)
    b = attach_router_capture(other, top_k=2)
    try:
        _drive(moe_model, x)
        _drive(other, x)
        router_fn = router_fn_for(a, b)
        state = _state(6)
        ids_a = router_fn(moe_model, state)["layers.0.mlp"]
        ids_b = router_fn(other, state)["layers.0.mlp"]
        assert not torch.equal(ids_a, ids_b), "flipping the gate must change routing"
        assert router_overlap(ids_a, ids_a) == 1.0
        assert router_fn(nn.Linear(2, 2), state) is None, "unknown model -> None"

        gates_fn = gates_fn_for(a, b)
        assert set(gates_fn(moe_model, state)) == {"layers.0.mlp", "layers.1.mlp"}
    finally:
        a.remove()
        b.remove()


def test_quantizing_experts_does_not_perturb_the_router_itself(moe_model):
    """Given the same input, INT4 experts must not change expert selection.

    The router is a BF16 ``nn.Linear`` excluded from quantization, and it runs
    *before* the experts. So on identical inputs the routing has to be
    bit-identical, and every overlap below 1.0 in the real experiment is
    attributable to the hidden state having drifted upstream — not to the
    router having been damaged. That attribution is the whole point of the
    measurement, so it is worth pinning here.
    """
    quantized = copy.deepcopy(moe_model)
    quantize_model(quantized, QuantConfig(bits=4, group_size=64, scale_search="mse"))
    x = torch.randn(32, HIDDEN, dtype=torch.bfloat16)

    a = attach_router_capture(moe_model, top_k=2)
    b = attach_router_capture(quantized, top_k=2)
    try:
        _drive(moe_model, x)
        _drive(quantized, x)
        for layer in ("layers.0.mlp", "layers.1.mlp"):
            assert torch.equal(
                moe_model.get_submodule(layer).gate.weight,
                quantized.get_submodule(layer).gate.weight,
            ), "the router weight was quantized; it must be excluded"
            assert router_overlap(a.topk_ids[layer], b.topk_ids[layer]) == 1.0
    finally:
        a.remove()
        b.remove()


# --------------------------------------------------------------------------
# Lifecycle and failure modes
# --------------------------------------------------------------------------


def test_remove_detaches_and_is_idempotent(moe_model):
    capture = attach_router_capture(moe_model, top_k=2)
    capture.remove()
    capture.remove()
    capture.clear()
    _drive(moe_model, torch.randn(4, HIDDEN, dtype=torch.bfloat16))
    assert capture.topk_ids == {}, "hooks were not detached"


def test_attaching_to_a_model_without_fused_blocks_fails_loudly():
    with pytest.raises(RuntimeError, match="no fused expert block"):
        attach_router_capture(nn.Sequential(nn.Linear(4, 4)))


def test_top_k_must_be_derivable_or_supplied(moe_model):
    """The toy block has no cfg.TOPK, so top_k must be passed explicitly."""
    with attach_router_capture(moe_model) as capture:
        with pytest.raises(RuntimeError, match="cannot determine top_k"):
            _drive(moe_model, torch.randn(4, HIDDEN, dtype=torch.bfloat16))
        assert capture.layer_names == []


def test_top_k_is_read_from_cfg_when_present(moe_model):
    """The real TritonFusedMoEBlock carries cfg.TOPK; honour it."""

    class Cfg:
        TOPK = 3

    for layer in moe_model.layers:
        layer.mlp.cfg = Cfg()
    with attach_router_capture(moe_model) as capture:
        _drive(moe_model, torch.randn(5, HIDDEN, dtype=torch.bfloat16))
        assert capture.topk_ids["layers.0.mlp"].shape == (5, 3)


def test_top_k_is_clamped_to_the_expert_count(moe_model):
    class Cfg:
        TOPK = 999

    for layer in moe_model.layers:
        layer.mlp.cfg = Cfg()
    with attach_router_capture(moe_model) as capture:
        _drive(moe_model, torch.randn(5, HIDDEN, dtype=torch.bfloat16))
        assert capture.topk_ids["layers.0.mlp"].shape == (5, NUM_EXPERTS)


def test_capture_works_on_a_bare_block():
    block = FusedExpertBlock()
    model = nn.Module()
    model.mlp = block
    with attach_router_capture(model, top_k=2) as capture:
        with torch.no_grad():
            block(torch.randn(4, HIDDEN, dtype=torch.bfloat16))
        assert capture.layer_names == ["mlp"]
