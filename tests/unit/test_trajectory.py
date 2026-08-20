"""Trajectory states, masked-token metrics, and the two capture modes."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from LLaDA_Quant import QuantConfig, quantized_model
from LLaDA_Quant.trajectory import (
    DiffusionState,
    capture_free_running,
    capture_shared,
    fully_masked_state,
    make_masked_states,
    mask_positions_from_ids,
)
from LLaDA_Quant.trajectory.metrics import (
    commit_order_agreement,
    kl_divergence,
    predictive_entropy,
    router_gate_entropy,
    router_margin,
    tie_fraction,
    top1_agreement,
    top2_margin,
    topk_kl_lower_bound,
    unmask_selection_agreement,
)

VOCAB, HIDDEN, MASK_ID = 32, 32, 31


class ToyDiffusionLM(nn.Module):
    """Bidirectional toy: every position sees the whole sequence, as in LLaDA."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        h = h + h.mean(dim=1, keepdim=True)
        return self.head(torch.tanh(self.proj(h)))


def logits_fn(model, state):
    return model(state.input_ids)


def router_fn(model, state):
    with torch.no_grad():
        return {"layers.0.mlp": model(state.input_ids).argmax(dim=-1) % 4}


def gates_fn(model, state):
    with torch.no_grad():
        return {"layers.0.mlp": model(state.input_ids)[..., :8]}


def make_advance_fn(k: int = 1):
    """Confidence-based remasking: unmask the ``k`` most confident positions."""

    def advance(state, logits):
        if state.num_masked == 0:
            return None
        conf, pred = torch.softmax(logits.float(), dim=-1).max(dim=-1)
        conf = conf.masked_fill(~state.mask_positions, -1.0)
        ids = state.input_ids.clone()
        mask = state.mask_positions.clone()
        for row in range(ids.shape[0]):
            take = min(k, int(state.mask_positions[row].sum().item()))
            if take == 0:
                continue
            idx = conf[row].topk(take).indices
            ids[row, idx] = pred[row, idx]
            mask[row, idx] = False
        return DiffusionState(step=state.step + 1, input_ids=ids, mask_positions=mask)

    return advance


def perturbed(model, scale: float = 0.5):
    other = copy.deepcopy(model)
    with torch.no_grad():
        other.head.weight.add_(torch.randn_like(other.head.weight) * scale)
    return other


def states():
    prompt = torch.arange(4).unsqueeze(0)
    completion = torch.arange(4, 20).unsqueeze(0)
    return make_masked_states(
        prompt, completion, MASK_ID, ratios=(1.0, 0.5, 0.25),
        generator=torch.Generator().manual_seed(1),
    )


# --------------------------------------------------------------------------
# State construction
# --------------------------------------------------------------------------


def test_fully_masked_state():
    state = fully_masked_state(torch.arange(4).unsqueeze(0), 6, MASK_ID)
    assert state.input_ids.shape == (1, 10)
    assert state.num_masked == 6
    assert not state.mask_positions[0, :4].any()
    assert (state.input_ids[0, 4:] == MASK_ID).all()


def test_make_masked_states_is_monotone_and_spares_the_prompt():
    prompt = torch.arange(4).unsqueeze(0)
    completion = torch.arange(4, 24).unsqueeze(0)
    built = make_masked_states(
        prompt, completion, MASK_ID, ratios=(1.0, 0.75, 0.5, 0.25, 0.0),
        generator=torch.Generator().manual_seed(0),
    )
    assert [s.num_masked for s in built] == [20, 15, 10, 5, 0]
    expected = torch.cat([prompt, completion], dim=1)
    for state in built:
        assert not state.mask_positions[:, :4].any(), "prompt must never be masked"
        keep = ~state.mask_positions
        assert (state.input_ids[keep] == expected[keep]).all()
    for earlier, later in zip(built, built[1:]):
        assert not (later.mask_positions & ~earlier.mask_positions).any()


def test_make_masked_states_rejects_bad_input():
    prompt, completion = torch.arange(4).unsqueeze(0), torch.arange(8).unsqueeze(0)
    with pytest.raises(ValueError):
        make_masked_states(prompt, completion, MASK_ID, ratios=())
    with pytest.raises(ValueError):
        make_masked_states(prompt, completion, MASK_ID, ratios=(1.5,))
    with pytest.raises(ValueError):
        make_masked_states(prompt, torch.arange(8).repeat(3, 1), MASK_ID)


def test_diffusion_state_validates_shapes():
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError):
        DiffusionState(step=0, input_ids=ids[0], mask_positions=ids[0].bool())
    with pytest.raises(ValueError):
        DiffusionState(step=0, input_ids=ids, mask_positions=torch.zeros(1, 5, dtype=torch.bool))


def test_diffusion_state_rejects_a_split_device_state():
    """A CPU/CUDA-mixed state used to construct fine and fail later, inside the
    caller's indexing. `meta` stands in for a second device without a GPU."""
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="must live on one device"):
        DiffusionState(
            step=0,
            input_ids=ids,
            mask_positions=torch.zeros(1, 4, dtype=torch.bool, device="meta"),
        )
    with pytest.raises(ValueError, match="must live on one device"):
        DiffusionState(
            step=0,
            input_ids=ids,
            mask_positions=ids.bool(),
            attention_mask=torch.ones(1, 4, device="meta"),
        )


def test_mask_positions_from_ids():
    ids = torch.tensor([[1, MASK_ID, 3, MASK_ID]])
    assert mask_positions_from_ids(ids, MASK_ID).tolist() == [[False, True, False, True]]


# --------------------------------------------------------------------------
# Masked-token and routing metrics
# --------------------------------------------------------------------------


def test_top1_agreement_respects_positions():
    a = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    b = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]])
    assert top1_agreement(a, b) == pytest.approx(0.5)
    assert top1_agreement(a, b, torch.tensor([[False, True]])) == pytest.approx(1.0)
    assert top1_agreement(a, b, torch.tensor([[False, False]])) == 1.0


def test_kl_divergence_is_zero_for_identical_and_shifted_logits():
    logits = torch.randn(2, 5, 7)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)
    assert kl_divergence(logits, logits + 3.0) == pytest.approx(0.0, abs=1e-5)
    assert kl_divergence(logits, torch.zeros_like(logits)) > 0.0


def test_unmask_selection_agreement_catches_reordering():
    a = torch.tensor([[[9.0, 0.0], [1.0, 0.0]]])
    b = torch.tensor([[[1.0, 0.0], [9.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    assert unmask_selection_agreement(a, b, mask, k=1) == pytest.approx(0.0)
    assert unmask_selection_agreement(a, b, mask, k=2) == pytest.approx(1.0)
    # top-1 token agreement alone would have called this perfect
    assert top1_agreement(a, b, mask) == pytest.approx(1.0)


def test_tie_fraction_separates_coin_tosses_from_damage():
    ref = torch.tensor([[[5.0, 3.0, 1.0], [2.0, 2.0 - 1e-4, 0.0]]])
    shifted = ref + torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1e-2, 0.0]]])
    assert tie_fraction(ref, shifted, torch.tensor([[True, False]])) == 0.0
    assert tie_fraction(ref, shifted, torch.tensor([[False, True]])) == 1.0
    assert top1_agreement(ref, shifted, torch.tensor([[False, True]])) == 0.0
    assert tie_fraction(ref, ref) == 0.0


def test_top2_margin():
    logits = torch.tensor([[[5.0, 3.0, 1.0], [2.0, 2.0, 0.0]]])
    assert top2_margin(logits) == pytest.approx(1.0)
    assert top2_margin(logits, torch.tensor([[True, False]])) == pytest.approx(2.0)


def test_predictive_entropy_is_zero_for_a_certain_distribution():
    certain = torch.tensor([[[100.0, 0.0, 0.0]]])
    uniform = torch.zeros(1, 1, 3)
    assert predictive_entropy(certain) == pytest.approx(0.0, abs=1e-5)
    assert predictive_entropy(uniform) == pytest.approx(torch.tensor(3.0).log().item(), abs=1e-5)


def test_router_margin_measures_the_top_k_boundary_gap():
    gates = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    assert router_margin(gates, top_k=2) == pytest.approx(3.0)  # 4.0 - 1.0
    assert router_margin(gates, top_k=1) == pytest.approx(1.0)  # 5.0 - 4.0
    assert router_margin(gates, top_k=4) == 0.0, "no boundary when k == num_experts"


def test_router_gate_entropy_matches_a_uniform_router():
    assert router_gate_entropy(torch.zeros(2, 4)) == pytest.approx(
        torch.tensor(4.0).log().item(), abs=1e-5
    )


def test_topk_kl_lower_bound_is_zero_for_identical_slices():
    logprobs = torch.log_softmax(torch.randn(4, 8), dim=-1)
    assert topk_kl_lower_bound(logprobs, logprobs) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        topk_kl_lower_bound(logprobs, logprobs[:, :4])


def test_commit_order_agreement():
    assert commit_order_agreement([[1], [2]], [[1], [2]]) == 1.0
    assert commit_order_agreement([[1], [2]], [[2], [1]]) == 0.0
    assert commit_order_agreement([[1], [2]], [[1], [3]]) == 0.5


# --------------------------------------------------------------------------
# Mode A: shared state
# --------------------------------------------------------------------------


def test_mode_a_identical_models_show_no_divergence():
    model = ToyDiffusionLM().eval()
    capture = capture_shared(model, model, states(), logits_fn, router_fn, gates_fn)

    assert len(capture.reference) == len(capture.quantized) == 3
    for step in capture.quantized.steps:
        assert step.scalars["pair.top1_agreement"].value == 1.0
        assert step.scalars["pair.kl_masked"].value == pytest.approx(0.0, abs=1e-6)
        assert step.scalars["pair.router_overlap.layers.0.mlp"].value == 1.0
        assert step.scalars["pair.tie_fraction"].value == 0.0
        assert step.scalars["pair.logit_cosine"].precision == "exact"


def test_mode_a_captures_layer_stats_and_self_scalars():
    model = ToyDiffusionLM().eval()
    capture = capture_shared(model, model, states(), logits_fn, router_fn, gates_fn)
    step = capture.reference.steps[0]
    assert "entropy_masked" in step.scalars and "top2_margin" in step.scalars
    layer = step.layers["layers.0.mlp"]
    assert layer.router_margin >= 0.0
    assert layer.router_topk_ids is None, "router ids must be opt-in, they dominate trace size"


def test_mode_a_records_int8_error(moe_model=None):
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    config = QuantConfig(bits=8, group_size=16, targets=("linear",),
                         linear_include=("proj", "head"), expect_linears=2)
    quantized = quantized_model(model, config).eval()
    capture = capture_shared(model, quantized, states(), logits_fn, router_fn, unmask_k=4)

    for step in capture.quantized.steps:
        assert step.scalars["pair.logit_cosine"].value > 0.99
        assert step.scalars["pair.max_abs_error"].value > 0.0
        assert 0.0 <= step.scalars["pair.tie_fraction"].value <= 1.0


def test_mode_a_rejects_mismatched_logits():
    model = ToyDiffusionLM().eval()
    other = ToyDiffusionLM(seed=1).eval()

    def mixed(m, state):
        if m is model:
            return logits_fn(m, state)
        return torch.randn(1, state.input_ids.shape[1], VOCAB - 1)

    with pytest.raises(ValueError, match="logits shape mismatch"):
        capture_shared(model, other, states(), mixed)


def test_router_ids_are_stored_only_when_requested():
    model = ToyDiffusionLM().eval()
    capture = capture_shared(
        model, model, states()[:1], logits_fn, router_fn, store_router_ids=True
    )
    assert capture.reference.steps[0].layers["layers.0.mlp"].router_topk_ids is not None


# --------------------------------------------------------------------------
# Mode B: free running
# --------------------------------------------------------------------------


def test_mode_b_identical_models_commit_identically():
    model = ToyDiffusionLM().eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 6, MASK_ID)
    capture = capture_free_running(model, model, start, logits_fn, make_advance_fn(1),
                                   max_steps=20)
    assert len(capture.reference) == len(capture.quantized) == 6
    for ref, qnt in zip(capture.reference.steps, capture.quantized.steps):
        assert ref.committed_positions == qnt.committed_positions
        assert ref.committed_tokens == qnt.committed_tokens


def test_mode_b_records_commits_and_stops_when_done():
    model = ToyDiffusionLM().eval()
    other = perturbed(model, scale=1.5).eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 8, MASK_ID)
    capture = capture_free_running(model, other, start, logits_fn, make_advance_fn(1),
                                   max_steps=20)
    assert capture.reference.gen_length == 8
    assert capture.reference.prompt_length == 4
    assert sum(len(s.committed_positions) for s in capture.reference.steps) == 8
    assert all(len(s.committed_positions) == 1 for s in capture.reference.steps)


def test_mode_b_honours_max_steps():
    model = ToyDiffusionLM().eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 10, MASK_ID)
    capture = capture_free_running(model, model, start, logits_fn, make_advance_fn(1),
                                   max_steps=3)
    assert len(capture.reference) == 3
    with pytest.raises(ValueError):
        capture_free_running(model, model, start, logits_fn, make_advance_fn(1), max_steps=0)


def test_mode_b_stores_no_pairwise_logit_scalars():
    """Once inputs drift apart a logit distance conflates two causes."""
    model = ToyDiffusionLM().eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 4, MASK_ID)
    capture = capture_free_running(model, model, start, logits_fn, make_advance_fn(1))
    for step in capture.quantized.steps:
        assert not [k for k in step.scalars if k.startswith("pair.")]
