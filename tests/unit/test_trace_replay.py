"""Phases 9 and 10: offline replay, trace compactness, and the noise floor.

Metrics must be recomputable from a stored trace without the model, traces
must stay small, approximations must never be labelled exact, and every
quantization result must be readable against a BF16-vs-BF16 baseline.
"""

from __future__ import annotations

import pytest
import torch

from LLaDA_Quant.trajectory import (
    MetricPrecision,
    ScalarMetric,
    Trace,
    TraceStep,
    TrajectoryReport,
    capture_free_running,
    capture_shared,
    fully_masked_state,
    replay_free_running,
    replay_shared,
    verify_replay,
)

from test_trajectory import (
    MASK_ID,
    ToyDiffusionLM,
    logits_fn,
    make_advance_fn,
    perturbed,
    router_fn,
    states,
)


def _start(gen=6):
    return fully_masked_state(torch.arange(4).unsqueeze(0), gen, MASK_ID)


# --------------------------------------------------------------------------
# Trace container
# --------------------------------------------------------------------------


def test_trace_roundtrips_through_json(tmp_path):
    trace = Trace(label="x", mode="shared", top_k_stored=4)
    trace.steps.append(
        TraceStep(
            step=0,
            num_masked=2,
            mask_ratio=0.5,
            topk_ids=[[1, 2]],
            topk_logprobs=[[-0.1, -2.0]],
            scalars={"pair.top1_agreement": ScalarMetric(0.75)},
        )
    )
    path = tmp_path / "trace.json"
    trace.save(str(path))
    restored = Trace.load(str(path))
    assert restored.label == "x" and restored.mode == "shared"
    assert restored.steps[0].topk_ids == [[1, 2]]
    assert restored.steps[0].scalars["pair.top1_agreement"].value == 0.75
    assert restored.steps[0].scalars["pair.top1_agreement"].is_exact


def test_trace_rejects_an_unknown_format_version():
    with pytest.raises(ValueError, match="format version"):
        Trace.from_dict({"format_version": 999})


def test_scalar_metric_precision_is_explicit():
    exact = ScalarMetric(1.0, MetricPrecision.EXACT.value)
    approx = ScalarMetric(1.0, MetricPrecision.TOPK.value, "top-8 truncation")
    assert exact.is_exact and not approx.is_exact
    assert approx.to_dict()["note"] == "top-8 truncation"


def test_trace_stays_compact():
    """A trace must not grow with the vocabulary."""
    model = ToyDiffusionLM().eval()
    capture = capture_shared(model, model, states(), logits_fn, router_fn, top_k=8)
    size = capture.quantized.size_estimate_bytes()
    # 3 steps over a 20-token sequence; full logits would be 3*20*32 floats
    assert size < 40_000, f"trace ballooned to {size} bytes"
    for step in capture.quantized.steps:
        for row in step.topk_ids:
            assert len(row) <= 8, "stored top-k must respect top_k"


def test_scalar_series_reads_across_steps():
    model = ToyDiffusionLM().eval()
    capture = capture_shared(model, model, states(), logits_fn)
    series = capture.quantized.scalar_series("pair.top1_agreement")
    assert series == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def test_replay_reproduces_what_capture_measured(tmp_path):
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    other = perturbed(model, 0.3).eval()
    capture = capture_shared(model, other, states(), logits_fn, router_fn)
    capture.save(str(tmp_path))

    reference = Trace.load(str(tmp_path / "modeA-reference.json"))
    quantized = Trace.load(str(tmp_path / "modeA-quantized.json"))
    report = replay_shared(reference, quantized)

    assert verify_replay(report) == [], "offline replay drifted from the on-device capture"
    assert report.summary["steps"] == 3
    assert 0.0 <= report.summary["min_top1_agreement"] <= 1.0


def test_replay_labels_its_own_numbers_as_truncated():
    model = ToyDiffusionLM().eval()
    capture = capture_shared(model, model, states(), logits_fn)
    report = replay_shared(capture.reference, capture.quantized)
    step = report.steps[0]
    assert step.replayed["topk_set_overlap"].precision == MetricPrecision.TOPK.value
    assert "Jaccard" in step.replayed["topk_set_overlap"].note
    assert step.stored_exact["pair.logit_cosine"].is_exact


def test_verify_replay_reports_a_planted_inconsistency():
    model = ToyDiffusionLM().eval()
    capture = capture_shared(model, model, states(), logits_fn)
    capture.quantized.steps[1].scalars["pair.top1_agreement"] = ScalarMetric(0.123)
    problems = verify_replay(replay_shared(capture.reference, capture.quantized))
    assert len(problems) == 1 and "step 1" in problems[0]


def test_replay_rejects_mismatched_modes():
    shared = Trace(mode="shared")
    free = Trace(mode="free_running")
    with pytest.raises(ValueError, match="mode='shared'"):
        replay_shared(shared, free)
    with pytest.raises(ValueError, match="mode='free_running'"):
        replay_free_running(free, shared)


def test_replay_free_running_finds_the_divergence_point():
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    other = perturbed(model, 1.5).eval()
    capture = capture_free_running(model, other, _start(8), logits_fn, make_advance_fn(1))
    report = replay_free_running(capture.reference, capture.quantized)
    assert report.summary["final_token_agreement"] <= 1.0
    assert report.summary["positions_committed_by_both"] > 0
    agreements = report.series("token_agreement_on_committed")
    assert agreements == sorted(agreements, reverse=True), "committed tokens are never revisited"


def test_replay_free_running_is_clean_for_identical_models():
    model = ToyDiffusionLM().eval()
    capture = capture_free_running(model, model, _start(6), logits_fn, make_advance_fn(1))
    report = replay_free_running(capture.reference, capture.quantized)
    assert report.summary["final_token_agreement"] == 1.0
    assert report.summary["first_divergence_step"] == -1.0
    assert report.summary["commit_order_agreement"] == 1.0


def test_replay_needs_no_model_or_gpu(tmp_path):
    """The whole point: capture once, then analyse from JSON forever."""
    model = ToyDiffusionLM().eval()
    capture = capture_free_running(model, model, _start(4), logits_fn, make_advance_fn(1))
    capture.save(str(tmp_path))
    del model, capture
    report = replay_free_running(
        Trace.load(str(tmp_path / "modeB-reference.json")),
        Trace.load(str(tmp_path / "modeB-quantized.json")),
    )
    assert report.summary["steps"] == 4


# --------------------------------------------------------------------------
# Noise floor (Phase 10)
# --------------------------------------------------------------------------


def test_bf16_vs_bf16_noise_floor_is_clean():
    """Step 1 of the protocol: the same model against itself must be exact."""
    model = ToyDiffusionLM().eval()
    floor_a = replay_shared(*_pair(capture_shared(model, model, states(), logits_fn, router_fn)))
    assert floor_a.summary["mean_top1_agreement"] == 1.0
    assert floor_a.summary["mean_tie_fraction"] == 0.0


def _pair(capture):
    return capture.reference, capture.quantized


def test_report_exposes_the_floor_next_to_the_result():
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    other = perturbed(model, 0.4).eval()

    mode_a = replay_shared(*_pair(capture_shared(model, other, states(), logits_fn, router_fn)))
    mode_b = replay_free_running(
        *_pair(capture_free_running(model, other, _start(6), logits_fn, make_advance_fn(1)))
    )
    floor_a = replay_shared(*_pair(capture_shared(model, model, states(), logits_fn, router_fn)))
    floor_b = replay_free_running(
        *_pair(capture_free_running(model, model, _start(6), logits_fn, make_advance_fn(1)))
    )

    report = TrajectoryReport(
        mode_a=mode_a, mode_b=mode_b, noise_floor_a=floor_a, noise_floor_b=floor_b,
        label="INT8",
    )
    table = report.to_table()
    assert "BF16 floor" in table
    assert "Mode A" in table and "Mode B" in table
    assert report.per_step_signal >= 0.0
    payload = report.to_dict()
    assert payload["noise_floor_a"]["mean_top1_agreement"] == 1.0
    assert "Mode A = error injected per step" in payload["interpretation"]


def test_report_amplification_is_undefined_without_both_modes():
    report = TrajectoryReport(mode_a=None, mode_b=None)
    assert report.amplification != report.amplification  # nan
    assert "amplification" in report.to_table()


def test_report_distinguishes_per_step_error_from_amplification():
    torch.manual_seed(0)
    model = ToyDiffusionLM().eval()
    other = perturbed(model, 1.5).eval()
    mode_a = replay_shared(*_pair(capture_shared(model, other, states(), logits_fn)))
    mode_b = replay_free_running(
        *_pair(capture_free_running(model, other, _start(8), logits_fn, make_advance_fn(1)))
    )
    report = TrajectoryReport(mode_a=mode_a, mode_b=mode_b)
    assert report.to_dict()["mode_a_shared_state"]["mode"] == "shared"
    assert report.to_dict()["mode_b_free_running"]["mode"] == "free_running"
