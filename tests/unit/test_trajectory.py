import copy

import pytest
import torch
import torch.nn as nn

from LLaDA_Quant.api import quantized_model
from LLaDA_Quant.config import QuantConfig
from LLaDA_Quant.validation.diffusion import (
    DiffusionState,
    fully_masked_state,
    make_masked_states,
    mask_positions_from_ids,
)
from LLaDA_Quant.validation.metrics import (
    kl_divergence,
    tie_fraction,
    top1_agreement,
    top2_margin,
    unmask_selection_agreement,
)
from LLaDA_Quant.validation.trajectory import compare_free_running, compare_trajectory

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
        h = h + h.mean(dim=1, keepdim=True)  # crude global mixing
        return self.head(torch.tanh(self.proj(h)))


def logits_fn(model, state):
    return model(state.input_ids)


def router_fn(model, state):
    """Stand-in for top-k expert ids: stable buckets derived from the logits."""
    with torch.no_grad():
        ids = model(state.input_ids).argmax(dim=-1) % 4
    return {"layers.0.mlp": ids}


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


# --------------------------------------------------------------------------
# State construction
# --------------------------------------------------------------------------


def test_fully_masked_state():
    prompt = torch.arange(4).unsqueeze(0)
    state = fully_masked_state(prompt, gen_length=6, mask_token_id=MASK_ID)
    assert state.input_ids.shape == (1, 10)
    assert state.num_masked == 6
    assert not state.mask_positions[0, :4].any()
    assert state.mask_positions[0, 4:].all()
    assert (state.input_ids[0, 4:] == MASK_ID).all()


def test_fully_masked_state_accepts_1d_prompt():
    state = fully_masked_state(torch.arange(3), gen_length=2, mask_token_id=MASK_ID)
    assert state.input_ids.shape == (1, 5)


def test_make_masked_states_is_monotone_and_spares_the_prompt():
    prompt = torch.arange(4).unsqueeze(0)
    completion = torch.arange(4, 24).unsqueeze(0)
    ratios = (1.0, 0.75, 0.5, 0.25, 0.0)
    states = make_masked_states(
        prompt, completion, MASK_ID, ratios=ratios, generator=torch.Generator().manual_seed(0)
    )

    assert [s.num_masked for s in states] == [20, 15, 10, 5, 0]
    for i, state in enumerate(states):
        assert state.step == i
        assert not state.mask_positions[:, :4].any(), "prompt must never be masked"
        assert (state.input_ids[state.mask_positions] == MASK_ID).all()
        # revealed tokens keep their ground-truth value
        keep = ~state.mask_positions
        expected = torch.cat([prompt, completion], dim=1)
        assert (state.input_ids[keep] == expected[keep]).all()

    for earlier, later in zip(states, states[1:]):
        nested = later.mask_positions & ~earlier.mask_positions
        assert not nested.any(), "positions must only ever be revealed, never re-masked"


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
    with pytest.raises(ValueError):
        DiffusionState(
            step=0,
            input_ids=ids,
            mask_positions=ids.bool(),
            attention_mask=torch.ones(1, 5),
        )


def test_mask_positions_from_ids():
    ids = torch.tensor([[1, MASK_ID, 3, MASK_ID]])
    assert mask_positions_from_ids(ids, MASK_ID).tolist() == [[False, True, False, True]]


# --------------------------------------------------------------------------
# Masked-token metrics
# --------------------------------------------------------------------------


def test_top1_agreement_respects_positions():
    a = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])  # argmax: 0, 1
    b = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]])  # argmax: 1, 1
    assert top1_agreement(a, b) == pytest.approx(0.5)
    assert top1_agreement(a, b, torch.tensor([[False, True]])) == pytest.approx(1.0)
    assert top1_agreement(a, b, torch.tensor([[True, False]])) == pytest.approx(0.0)
    # nothing selected is vacuously perfect agreement
    assert top1_agreement(a, b, torch.tensor([[False, False]])) == 1.0


def test_top1_agreement_rejects_bad_positions():
    a = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        top1_agreement(a, a, torch.zeros(1, 5, dtype=torch.bool))


def test_kl_divergence_is_zero_for_identical_logits():
    logits = torch.randn(2, 5, 7)
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)
    assert kl_divergence(logits, logits + 3.0) == pytest.approx(0.0, abs=1e-5), "shift-invariant"
    assert kl_divergence(logits, torch.zeros_like(logits)) > 0.0


def test_unmask_selection_agreement_catches_reordering():
    # position 0 is the confident one for A, position 1 for B
    a = torch.tensor([[[9.0, 0.0], [1.0, 0.0]]])
    b = torch.tensor([[[1.0, 0.0], [9.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    assert unmask_selection_agreement(a, b, mask, k=1) == pytest.approx(0.0)
    assert unmask_selection_agreement(a, b, mask, k=2) == pytest.approx(1.0)
    assert unmask_selection_agreement(a, a, mask, k=1) == pytest.approx(1.0)
    # top-1 token agreement alone would have reported perfect agreement here
    assert top1_agreement(a, b, mask) == pytest.approx(1.0)


def test_top2_margin():
    logits = torch.tensor([[[5.0, 3.0, 1.0], [2.0, 2.0, 0.0]]])  # margins 2.0 and 0.0
    assert top2_margin(logits) == pytest.approx(1.0)
    assert top2_margin(logits, torch.tensor([[True, False]])) == pytest.approx(2.0)
    assert top2_margin(logits, torch.tensor([[False, False]])) == 0.0


def test_tie_fraction_separates_coin_tosses_from_damage():
    # position 0: reference is decisive (margin 2.0); position 1: a near-tie (1e-4)
    ref = torch.tensor([[[5.0, 3.0, 1.0], [2.0, 2.0 - 1e-4, 0.0]]])
    perturbed_logits = ref + torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1e-2, 0.0]]])

    assert tie_fraction(ref, perturbed_logits, torch.tensor([[True, False]])) == 0.0
    assert tie_fraction(ref, perturbed_logits, torch.tensor([[False, True]])) == 1.0
    assert tie_fraction(ref, perturbed_logits) == pytest.approx(0.5)
    # the argmax did flip at position 1 — without the tie fraction that reads as damage
    assert top1_agreement(ref, perturbed_logits, torch.tensor([[False, True]])) == 0.0
    assert tie_fraction(ref, ref) == 0.0


def test_tie_fraction_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        tie_fraction(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))


def test_unmask_selection_agreement_skips_unmasked_rows():
    logits = torch.randn(1, 3, 4)
    assert unmask_selection_agreement(logits, logits + 1, torch.zeros(1, 3, dtype=torch.bool)) == 1.0
    with pytest.raises(ValueError):
        unmask_selection_agreement(logits, logits, torch.ones(1, 3, dtype=torch.bool), k=0)


# --------------------------------------------------------------------------
# Teacher-forced trajectory
# --------------------------------------------------------------------------


def _states():
    prompt = torch.arange(4).unsqueeze(0)
    completion = torch.arange(4, 20).unsqueeze(0)
    return make_masked_states(
        prompt,
        completion,
        MASK_ID,
        ratios=(1.0, 0.5, 0.25),
        generator=torch.Generator().manual_seed(1),
    )


def test_compare_trajectory_identical_models_report_no_divergence():
    model = ToyDiffusionLM().eval()
    report = compare_trajectory(model, model, _states(), logits_fn, router_fn)

    assert len(report.states) == 3
    for state in report.states:
        assert state.top1_agreement == 1.0
        assert state.unmask_agreement == 1.0
        assert state.kl_masked == pytest.approx(0.0, abs=1e-6)
        assert state.router_overlap["layers.0.mlp"] == 1.0
        assert state.logit_metrics["max_abs_error"] == 0.0
        assert state.tie_fraction == 0.0, "no perturbation means nothing can be tied"
    assert report.min_router_overlap == 1.0
    assert report.series("top1_agreement") == [1.0, 1.0, 1.0]


def test_compare_trajectory_tracks_int8_quantization_error():
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    config = QuantConfig(bits=8, group_size=16, targets=("linear",), exclude=("embed_tokens",))
    qmodel = quantized_model(model, config).eval()

    report = compare_trajectory(model, qmodel, _states(), logits_fn, router_fn, unmask_k=4)

    assert len(report.states) == 3
    for state in report.states:
        assert state.logit_metrics["cosine_similarity"] > 0.99
        assert state.logit_metrics["max_abs_error"] > 0.0, "INT8 must actually perturb the logits"
        assert 0.0 <= state.top1_agreement <= 1.0
        assert 0.0 <= state.unmask_agreement <= 1.0
        assert 0.0 <= state.tie_fraction <= 1.0
        assert state.kl_masked >= 0.0
        assert state.reference_top2_margin >= 0.0
    assert report.worst_state is not None
    assert "step" in report.to_table()
    assert report.to_dict()["kind"] == "teacher_forced"


def test_tie_fraction_flags_the_degenerate_fully_masked_state():
    """Regression guard for a trap this harness must not hide.

    The toy gives every masked position identical logits, so its top-2 margin
    (~1e-5) sits far below INT8 noise (~5e-3). ``top1_agreement`` then reads
    0.0 while nothing was actually damaged. The tie fraction is what makes
    that readable instead of alarming.
    """
    torch.manual_seed(0)
    ref = ToyDiffusionLM().eval()
    config = QuantConfig(bits=8, group_size=16, targets=("linear",), exclude=("embed_tokens",))
    qmodel = quantized_model(ref, config).eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 16, MASK_ID)

    state = compare_trajectory(ref, qmodel, [start], logits_fn).states[0]

    assert state.tie_fraction == 1.0
    assert state.reference_top2_margin < state.logit_metrics["max_abs_error"]


def test_compare_trajectory_records_mask_ratio_and_labels():
    model = ToyDiffusionLM().eval()
    report = compare_trajectory(model, model, _states(), logits_fn)
    assert [s.num_masked for s in report.states] == [16, 8, 4]
    assert report.states[0].label == "100% masked"
    assert report.states[0].mask_ratio == pytest.approx(16 / 20)
    assert report.states[0].router_overlap == {}
    assert report.states[0].mean_router_overlap == 1.0


def test_compare_trajectory_rejects_mismatched_logits():
    model = ToyDiffusionLM().eval()

    def short_logits(_model, state):
        return torch.randn(1, state.input_ids.shape[1], VOCAB - 1)

    def mixed(m, state):
        return logits_fn(m, state) if m is model else short_logits(m, state)

    with pytest.raises(ValueError, match="logits shape mismatch"):
        compare_trajectory(model, ToyDiffusionLM(seed=1).eval(), _states(), mixed)


def test_compare_trajectory_rejects_router_key_mismatch():
    model = ToyDiffusionLM().eval()
    other = ToyDiffusionLM(seed=1).eval()

    def lopsided_router(m, state):
        return {"layers.0.mlp" if m is model else "layers.9.mlp": m(state.input_ids).argmax(-1)}

    with pytest.raises(ValueError, match="router_fn returned"):
        compare_trajectory(model, other, _states(), logits_fn, lopsided_router)


# --------------------------------------------------------------------------
# Free-running trajectory
# --------------------------------------------------------------------------


def test_free_running_identical_models_never_diverge():
    model = ToyDiffusionLM().eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 6, MASK_ID)

    report = compare_free_running(model, model, start, logits_fn, make_advance_fn(1), max_steps=20)

    assert report.first_divergence_step is None
    assert report.final_token_agreement == 1.0
    assert report.steps[-1].num_masked_reference == 0
    assert len(report.steps) == 6, "one position unmasked per step, then it stops"
    assert all(s.resolved_disagreements == 0 for s in report.steps)
    assert torch.equal(report.reference_ids, report.quantized_ids)


def test_free_running_surfaces_compounding_divergence():
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    other = perturbed(model, scale=1.5).eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 8, MASK_ID)

    report = compare_free_running(model, other, start, logits_fn, make_advance_fn(1), max_steps=20)

    assert report.first_divergence_step is not None
    assert report.final_token_agreement < 1.0
    # agreement is monotonically non-increasing: resolved tokens are never revisited
    agreements = [s.token_agreement for s in report.steps]
    assert agreements == sorted(agreements, reverse=True)
    assert report.to_dict()["first_divergence_step"] == report.first_divergence_step
    assert "step" in report.to_table()


def test_free_running_honours_max_steps():
    model = ToyDiffusionLM().eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 10, MASK_ID)

    report = compare_free_running(model, model, start, logits_fn, make_advance_fn(1), max_steps=3)

    assert len(report.steps) == 3
    assert report.steps[-1].num_masked_reference == 7
    with pytest.raises(ValueError):
        compare_free_running(model, model, start, logits_fn, make_advance_fn(1), max_steps=0)


def test_free_running_stops_when_advance_fn_returns_none():
    model = ToyDiffusionLM().eval()
    start = fully_masked_state(torch.arange(4).unsqueeze(0), 10, MASK_ID)
    calls = {"n": 0}

    def advance_twice(state, logits):
        calls["n"] += 1
        return None if state.step >= 2 else make_advance_fn(1)(state, logits)

    report = compare_free_running(model, model, start, logits_fn, advance_twice, max_steps=20)

    assert len(report.steps) == 3
    assert report.steps[-1].num_masked_reference == 8, "state freezes, it is not advanced further"
