"""Phase 6: the roofline analysis that decides whether to build the kernel."""

from __future__ import annotations

import pytest
import torch

from LLaDA_Quant.analysis import (
    LLADA_MOE_7B_A1B,
    RTX_A6000,
    SCHEMES,
    MoEShape,
    Workload,
    crossover_m,
    expert_token_stats,
    gemm_regime,
    ideal_tokens_per_expert,
    regime_sweep,
    suffix_lengths_for_schedule,
)


def test_llada_shape_matches_the_real_config():
    """From test_llada/src/model.py: H=2048, NL=16, NE=64, TOPK=8, EI=1024."""
    shape = LLADA_MOE_7B_A1B
    assert (shape.num_experts, shape.top_k) == (64, 8)
    assert (shape.hidden, shape.intermediate, shape.num_layers) == (2048, 1024, 16)
    # w1 [64, 2048, 2048] + w2 [64, 2048, 1024] per layer, 16 layers
    assert shape.expert_elements_per_layer == 64 * (2 * 1024 * 2048 + 2048 * 1024)
    assert shape.expert_elements == pytest.approx(6.44e9, rel=0.01)


def test_suffix_lengths_shrink_as_blocks_complete():
    """LLaDA forwards x[:, block_start:], so M falls block by block."""
    assert suffix_lengths_for_schedule(128, 32) == [128, 96, 64, 32]
    with pytest.raises(ValueError, match="must divide"):
        suffix_lengths_for_schedule(100, 32)


def test_tokens_per_expert_is_tokens_times_topk_over_experts():
    shape = LLADA_MOE_7B_A1B
    assert ideal_tokens_per_expert(shape, Workload(1, 128)) == pytest.approx(16.0)
    assert ideal_tokens_per_expert(shape, Workload(1, 32)) == pytest.approx(4.0)
    assert ideal_tokens_per_expert(shape, Workload(57, 128)) == pytest.approx(912.0)


def test_crossover_scales_with_weight_bytes():
    """AI = 2M/bytes, so halving the weight bytes halves the crossover."""
    bf16 = crossover_m(SCHEMES["BF16"], RTX_A6000)
    w8 = crossover_m(SCHEMES["W8A16"], RTX_A6000)
    w4 = crossover_m(SCHEMES["W4A16"], RTX_A6000)
    assert bf16 == pytest.approx(RTX_A6000.balance("bf16"))
    assert w8 == pytest.approx(bf16 / 2)
    assert w4 == pytest.approx(bf16 / 4)
    # W8A8 recovers the BF16 crossover because int8 peak is ~2x bf16 peak
    # (A6000: 309.7 TOPS vs 154.8 TFLOPS, so the match is close, not exact)
    assert crossover_m(SCHEMES["W8A8"], RTX_A6000) == pytest.approx(bf16, rel=1e-3)


def test_batch_one_is_memory_bound_in_every_scheme():
    shape = LLADA_MOE_7B_A1B
    m = ideal_tokens_per_expert(shape, Workload(1, 128))
    for name, scheme in SCHEMES.items():
        regime = gemm_regime(shape, m, scheme, RTX_A6000)
        assert regime.is_memory_bound, f"{name} unexpectedly compute-bound at M={m}"


def test_large_batch_is_compute_bound_so_weight_only_buys_no_latency():
    shape = LLADA_MOE_7B_A1B
    m = ideal_tokens_per_expert(shape, Workload(57, 128))
    for name in ("BF16", "W8A16", "W4A16", "W8A8"):
        regime = gemm_regime(shape, m, SCHEMES[name], RTX_A6000)
        assert not regime.is_memory_bound, f"{name} unexpectedly memory-bound at M={m}"


def test_gemm_shapes_match_the_fused_layout():
    shape = LLADA_MOE_7B_A1B
    regime = gemm_regime(shape, 16, SCHEMES["W8A16"], RTX_A6000)
    assert regime.gemm_w1 == (16, 2 * shape.intermediate, shape.hidden)
    assert regime.gemm_w2 == (16, shape.hidden, shape.intermediate)


def test_regime_crossover_property_agrees_with_the_free_function():
    for name, scheme in SCHEMES.items():
        regime = gemm_regime(LLADA_MOE_7B_A1B, 32, scheme, RTX_A6000)
        assert regime.crossover_m == pytest.approx(crossover_m(scheme, RTX_A6000))


def test_expert_token_stats_counts_real_routing():
    topk = torch.tensor([[0, 1], [0, 2], [0, 1]])
    stats = expert_token_stats(topk, num_experts=4)
    assert stats.counts == [3, 2, 1, 0]
    assert stats.active_experts == 3
    assert stats.mean == pytest.approx(1.5)
    assert stats.imbalance == pytest.approx(2.0)
    assert stats.to_dict()["max"] == 3


def test_expert_token_stats_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[tokens, top_k\]"):
        expert_token_stats(torch.zeros(4), num_experts=4)


def test_perfectly_balanced_routing_matches_the_ideal_estimate():
    shape = MoEShape(num_experts=4, top_k=2, hidden=8, intermediate=4)
    topk = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])
    stats = expert_token_stats(topk, shape.num_experts)
    assert stats.imbalance == 1.0
    assert stats.mean == ideal_tokens_per_expert(shape, Workload(1, 4))


def test_sweep_report_is_serializable_and_flags_the_balance_assumption():
    """Default machine is A40-24Q — the GPU the inference repo actually
    measured on. This assertion previously pinned 50.4, the A6000 value."""
    report = regime_sweep()
    payload = report.to_dict()
    assert payload["model"] == "LLaDA-MoE-7B-A1B"
    assert payload["machine"]["name"] == "A40-24Q"
    assert payload["measured_routing"] is None
    assert "optimistic" in payload["routing_note"]
    assert payload["crossovers_m_per_expert"]["W4A16"] == pytest.approx(53.8, abs=0.2)
    assert len(payload["rows"]) == len(report.rows)
    table = report.to_table()
    assert "memory" in table and "compute" in table


def test_sweep_carries_measured_routing_when_supplied():
    stats = expert_token_stats(torch.tensor([[0, 1], [0, 1]]), num_experts=4)
    report = regime_sweep(measured=stats)
    assert report.to_dict()["measured_routing"]["imbalance_max_over_mean"] == 2.0


# --------------------------------------------------------------------------
# Kernel-planning helpers (from INFERENCE_REPO_CHANGES.md sections 2 and 4)
# --------------------------------------------------------------------------


def test_a40_is_the_machine_the_measurements_came_from():
    """The handoff quotes ~696 GB/s and a ~215 flops/byte balance."""
    from LLaDA_Quant.analysis import A40_24Q

    assert A40_24Q.bandwidth_gbps == 696.0
    assert A40_24Q.memory_gb == 24.0
    assert A40_24Q.balance("bf16") == pytest.approx(215, abs=1.0)


def test_a40_crossovers_reproduce_the_handoff_verdict():
    """Section 3.4: at the throughput operating point M/expert = 128,
    BF16 is memory-bound but W8A16 is already compute-bound."""
    from LLaDA_Quant.analysis import A40_24Q

    m = 128.0
    assert gemm_regime(LLADA_MOE_7B_A1B, m, SCHEMES["BF16"], A40_24Q).is_memory_bound
    assert not gemm_regime(LLADA_MOE_7B_A1B, m, SCHEMES["W8A16"], A40_24Q).is_memory_bound
    assert not gemm_regime(LLADA_MOE_7B_A1B, m, SCHEMES["W4A16"], A40_24Q).is_memory_bound


def test_traced_workload_is_predicted_memory_bound():
    """Section 3.2: batch 11 x 32 tokens -> M/expert = 44, measured at 81% of
    weight-streaming peak. The model must call that memory-bound."""
    from LLaDA_Quant.analysis import A40_24Q

    m = ideal_tokens_per_expert(LLADA_MOE_7B_A1B, Workload(11, 32))
    assert m == pytest.approx(44.0)
    assert gemm_regime(LLADA_MOE_7B_A1B, m, SCHEMES["BF16"], A40_24Q).is_memory_bound


def test_block_k_must_divide_or_be_divided_by_group_size():
    from LLaDA_Quant.algorithms.symmetric import (
        aligned_block_k_values,
        block_k_is_scale_aligned,
        validate_block_k_alignment,
    )

    for block_k in (32, 64, 128, 256):
        assert block_k_is_scale_aligned(block_k, 128)
    assert not block_k_is_scale_aligned(96, 128)
    assert not block_k_is_scale_aligned(48, 128)
    assert block_k_is_scale_aligned(96, -1), "per-tensor scaling has one scale"

    assert aligned_block_k_values(128, [32, 64, 96, 128, 256]) == [32, 64, 128, 256]
    validate_block_k_alignment(64, 128)
    with pytest.raises(ValueError, match="straddles a group boundary"):
        validate_block_k_alignment(96, 128)


def test_shared_memory_generalises_over_weight_precision():
    """The inference repo's _shmem_bytes hardcodes 2 bytes per element."""
    from LLaDA_Quant.analysis import kernel_shared_memory_bytes, shared_memory_headroom

    args = dict(block_m=16, block_n=128, block_k=64, num_stages=2)
    # matches the repo's formula exactly at bf16: (bm*bk + 2*bk*bn) * ns * 2
    assert kernel_shared_memory_bytes(**args, weight_bytes=2.0) == (
        16 * 64 + 2 * 64 * 128
    ) * 2 * 2
    int8 = kernel_shared_memory_bytes(**args, weight_bytes=1.0)
    int4 = kernel_shared_memory_bytes(**args, weight_bytes=0.5)
    assert int4 < int8 < kernel_shared_memory_bytes(**args, weight_bytes=2.0)

    headroom = shared_memory_headroom(16, 128, 64, 2, SCHEMES["W4A16"])
    assert headroom["freed_bytes"] > 0
    assert headroom["scheme_fits"]


def test_gemm2_stages_one_b_tile_not_two():
    """SILU_EPILOGUE applies to GEMM1 only; GEMM2 keeps a single B tile."""
    from LLaDA_Quant.analysis import kernel_shared_memory_bytes

    gemm1 = kernel_shared_memory_bytes(16, 128, 64, 2, b_tiles=2)
    gemm2 = kernel_shared_memory_bytes(16, 128, 64, 2, b_tiles=1)
    assert gemm1 > gemm2


def test_shared_memory_rejects_degenerate_tiles():
    from LLaDA_Quant.analysis import kernel_shared_memory_bytes

    with pytest.raises(ValueError, match=">= 1"):
        kernel_shared_memory_bytes(0, 128, 64, 2)
