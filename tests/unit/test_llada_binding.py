"""Phase 8: Mode B must delegate to the production decoder, not copy it.

These tests do not check LLaDA's decoding semantics — that is the inference
repository's job and its code is the source of truth. They check the thing
that can silently rot: that ``make_llada_advance_fn`` really routes every
decision through the injected functions, and that
``assert_matches_production_decoder`` notices when it stops doing so.
"""

from __future__ import annotations

import os

import pytest
import torch

from LLaDA_Quant.trajectory import DiffusionState, fully_masked_state
from LLaDA_Quant.trajectory.llada import (
    LLADA_MASK_ID,
    LLaDADecoder,
    assert_matches_production_decoder,
    load_llada_decoder,
    make_llada_advance_fn,
)

INFERENCE_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..",
    "test_llada",
)


def _spy_decoder(calls: list[str]) -> LLaDADecoder:
    """A decoder whose primitives record that they were consulted."""

    def add_gumbel_noise(logits, temperature):
        calls.append("add_gumbel_noise")
        return logits

    def get_num_transfer_tokens(mask_index, steps):
        calls.append("get_num_transfer_tokens")
        rows = mask_index.shape[0]
        return torch.ones(rows, steps, dtype=torch.long)

    def select_transfer_indices(confidence, num_transfer_step):
        calls.append("select_transfer_indices")
        out = torch.zeros_like(confidence, dtype=torch.bool)
        for row in range(confidence.shape[0]):
            take = int(num_transfer_step[row])
            if take:
                out[row, confidence[row].topk(take).indices] = True
        return out

    return LLaDADecoder(
        add_gumbel_noise=add_gumbel_noise,
        get_num_transfer_tokens=get_num_transfer_tokens,
        select_transfer_indices=select_transfer_indices,
        source="spy",
    )


def _state(gen: int = 4) -> DiffusionState:
    return fully_masked_state(torch.arange(3).unsqueeze(0), gen, LLADA_MASK_ID)


def test_every_decision_goes_through_the_injected_primitives():
    calls: list[str] = []
    advance = make_llada_advance_fn(_spy_decoder(calls), steps=4)
    logits = torch.randn(1, 7, 40)
    advance(_state(), logits)
    assert calls == ["get_num_transfer_tokens", "add_gumbel_noise", "select_transfer_indices"]


def test_advance_commits_and_shrinks_the_mask():
    advance = make_llada_advance_fn(_spy_decoder([]), steps=4)
    state = _state(4)
    nxt = advance(state, torch.randn(1, 7, 40))
    assert nxt is not None
    assert nxt.num_masked == state.num_masked - 1
    assert nxt.step == state.step + 1
    assert not nxt.mask_positions[0, :3].any(), "the prompt must stay untouched"


def test_advance_returns_none_when_nothing_is_masked():
    advance = make_llada_advance_fn(_spy_decoder([]), steps=4)
    ids = torch.arange(5).unsqueeze(0)
    done = DiffusionState(step=0, input_ids=ids, mask_positions=torch.zeros_like(ids).bool())
    assert advance(done, torch.randn(1, 5, 40)) is None


def test_equivalence_check_passes_for_the_real_adapter():
    decoder = _spy_decoder([])
    assert_matches_production_decoder(decoder, torch.randn(1, 7, 40), _state(4), steps=4)


def test_equivalence_check_catches_a_drifted_adapter():
    """If the adapter ever stops matching the primitives, this must fail."""
    decoder = _spy_decoder([])
    drifted = LLaDADecoder(
        add_gumbel_noise=lambda logits, t: logits.flip(-1),  # different from what we verify against
        get_num_transfer_tokens=decoder.get_num_transfer_tokens,
        select_transfer_indices=decoder.select_transfer_indices,
        source="drifted",
    )
    logits = torch.randn(1, 7, 40)
    state = _state(4)
    # build the advance_fn from the drifted decoder, verify against the honest one
    import LLaDA_Quant.trajectory.llada as llada

    original = llada.make_llada_advance_fn
    try:
        llada.make_llada_advance_fn = lambda d, steps, **kw: original(drifted, steps, **kw)
        with pytest.raises(AssertionError, match="diverge"):
            llada.assert_matches_production_decoder(decoder, logits, state, steps=4)
    finally:
        llada.make_llada_advance_fn = original


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="steps must be >= 1"):
        make_llada_advance_fn(_spy_decoder([]), steps=0)


def test_unknown_remasking_is_rejected():
    advance = make_llada_advance_fn(_spy_decoder([]), steps=4, remasking="telepathy")
    with pytest.raises(ValueError, match="unknown remasking"):
        advance(_state(), torch.randn(1, 7, 40))


def test_loading_a_missing_repo_explains_itself(tmp_path):
    with pytest.raises(ImportError, match="Mode B against the production decoder"):
        load_llada_decoder(str(tmp_path), module="definitely_not_a_module")


@pytest.mark.skipif(
    not os.path.isdir(INFERENCE_REPO), reason="inference repository not present"
)
def test_real_decoder_import_when_the_repo_is_available():
    """Opportunistic: binds to the real model_update.generate when importable.

    Skipped rather than failed when the inference repo cannot be imported
    here (it needs triton and a CUDA build), which is the normal case on a
    development laptop.
    """
    try:
        decoder = load_llada_decoder(os.path.abspath(INFERENCE_REPO))
    except ImportError as exc:
        pytest.skip(f"inference repo present but not importable: {exc}")
    assert callable(decoder.add_gumbel_noise)
    assert callable(decoder.get_num_transfer_tokens)
    assert callable(decoder.select_transfer_indices)
    assert_matches_production_decoder(
        decoder, torch.randn(1, 7, 40), _state(4), steps=4
    )
