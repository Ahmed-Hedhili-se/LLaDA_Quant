"""Correctness and speed for the grouped-expert W8A16 MoE kernel.

The reference is **dequantize-then-matmul through the inference repository's own
``fused_moe``** -- not a hand-written matmul. That matters: the question this
kernel has to answer is "does moving the dequantize into the K-loop change the
answer", so the two arms must differ in exactly that and nothing else. Both use
the same ``moe_align_block_size``, the same routing, the same activation.

Bit-exactness is impossible here and is not asserted. The reference rounds each
weight to bf16 after dequantizing, then accumulates; this kernel scales an int8
in fp32 and rounds once. Same values, different rounding points. What is
asserted is that the difference stays at the level of that rounding, which for
bf16 accumulation over K=2048 lands around 1e-2 relative -- the same tolerance
``test_w8a16_kernel.py`` uses for the standalone GEMM.

Needs a CUDA GPU with Triton and the inference repository importable
(``LLADA_INFERENCE_REPO``, or already on ``sys.path``). Skips otherwise.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped W8A16 kernel needs a CUDA GPU"
)


def _ensure_repo() -> bool:
    repo = os.environ.get("LLADA_INFERENCE_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import model_update.fused_moe_triton  # noqa: F401
    except ImportError:
        return False
    return True


E, TOPK, H, EI, GROUP = 8, 4, 512, 256, 128


def _build(M: int, seed: int = 0, dims=None):
    """Expert weights in the exact layout ``TritonFusedMoEBlock`` declares."""
    from LLaDA_Quant.algorithms.symmetric import quantize_tensor

    E, TOPK, H, EI = dims if dims is not None else (globals()["E"], globals()["TOPK"],
                                                    globals()["H"], globals()["EI"])
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(M, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.05
    w1 = torch.randn(E, 2 * EI, H, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, H, EI, generator=g, device="cuda", dtype=torch.bfloat16) * 0.02

    logits = torch.randn(M, E, generator=g, device="cuda", dtype=torch.float32) * 0.05
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
    topk_w, topk_ids = torch.topk(weights, TOPK, dim=-1)

    q1 = quantize_tensor(w1, bits=8, group_size=GROUP)
    q2 = quantize_tensor(w2, bits=8, group_size=GROUP)
    return hidden, w1, w2, q1, q2, topk_w.to(torch.bfloat16), topk_ids.to(torch.int32)


def _reference(hidden, q1, q2, topk_w, topk_ids):
    """Dequantize to BF16, then the unmodified fused_moe -- the current path."""
    import model_update.fused_moe_triton as fmt

    w1 = q1.dequantize(dtype=torch.bfloat16)
    w2 = q2.dequantize(dtype=torch.bfloat16)
    # fuse_silu=False so both arms run the identical activation sequence; the
    # kernel under test deliberately does not fuse the epilogue.
    return fmt.fused_moe(hidden, w1, w2, topk_w, topk_ids, fuse_silu=False)


@pytest.mark.skipif(not _ensure_repo(), reason="inference repository not importable")
@pytest.mark.parametrize("M", [1, 8, 32, 64, 256, 512])
def test_matches_dequantize_then_matmul(M):
    from LLaDA_Quant.runtime.kernels.w8a16_moe import fused_moe_w8a16

    hidden, _, _, q1, q2, topk_w, topk_ids = _build(M, seed=M)

    want = _reference(hidden, q1, q2, topk_w, topk_ids)
    got = fused_moe_w8a16(hidden, q1.q, q1.scale, q2.q, q2.scale,
                          topk_w, topk_ids, quant_group=GROUP)

    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    assert got.dtype == want.dtype

    rel = (got.float() - want.float()).norm() / want.float().norm().clamp(min=1e-12)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    )
    print(f"  M={M:4d}  rel_L2={rel:.3e}  cos={cos:.8f}")
    assert rel < 2e-2, f"M={M}: relative L2 {rel:.3e} is beyond bf16 rounding"
    assert cos > 0.999, f"M={M}: cosine {cos:.6f}"


@pytest.mark.skipif(not _ensure_repo(), reason="inference repository not importable")
def test_rejects_packed_int4():
    """INT4 needs an unpack in the K-loop; failing loudly beats wrong numbers."""
    from LLaDA_Quant.algorithms.symmetric import quantize_tensor
    from LLaDA_Quant.runtime.kernels.w8a16_moe import fused_moe_w8a16

    hidden, _, _, q1, q2, topk_w, topk_ids = _build(8, seed=1)
    w1 = q1.dequantize(dtype=torch.bfloat16)
    q1_int4 = quantize_tensor(w1, bits=4, group_size=GROUP)

    with pytest.raises(TypeError, match="INT8"):
        fused_moe_w8a16(hidden, q1_int4.q, q1_int4.scale, q2.q, q2.scale,
                        topk_w, topk_ids, quant_group=GROUP)


@pytest.mark.skipif(not _ensure_repo(), reason="inference repository not importable")
def test_rejects_block_k_straddling_a_group():
    """A K-block spanning two groups would silently use one group's scale."""
    from LLaDA_Quant.runtime.kernels.w8a16_moe import default_config, fused_moe_w8a16

    hidden, _, _, q1, q2, topk_w, topk_ids = _build(8, seed=2)
    cfg = default_config(8, E, GROUP)
    cfg["BLOCK_SIZE_K"] = 96  # does not divide 128

    with pytest.raises(ValueError, match="must divide"):
        fused_moe_w8a16(hidden, q1.q, q1.scale, q2.q, q2.scale,
                        topk_w, topk_ids, quant_group=GROUP, config=cfg)


def _bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


@pytest.mark.skipif(not _ensure_repo(), reason="inference repository not importable")
def test_benchmark_against_bf16_and_dequant(capsys):
    """Three arms: BF16 (the speed target), dequantize-then-matmul (today), this.

    Uses the REAL checkpoint geometry (E=64, top-8, H=2048, EI=1024), not the
    small shapes the correctness tests use. That is not incidental: at the
    correctness scale the expert weights are ~6 MiB and sit entirely in the
    A6000's 6 MiB L2, so there is no HBM traffic for in-register dequantization
    to save and all three arms measure the same launch-bound ~0.9 ms. The win
    only exists past L2 -- 805 MiB of BF16 experts here -- which is the regime
    the model actually runs in.
    """
    import model_update.fused_moe_triton as fmt
    from LLaDA_Quant.runtime.kernels.w8a16_moe import fused_moe_w8a16

    dims = (64, 8, 2048, 1024)  # FULL_CFG: E, top_k, H, EI
    e, topk, h, ei = dims
    with capsys.disabled():
        print("")
        print(f"  Real geometry: E={e} top_k={topk} H={h} EI={ei} "
              f"({(e * 2 * ei * h + e * h * ei) * 2 / 2**20:.0f} MiB of BF16 experts)")
        print(f"  {'M':>5}  {'M/expert':>8}  {'BF16':>10}  {'dequant+GEMM':>14}  "
              f"{'W8A16 fused':>13}  {'vs BF16':>9}")
        for M in (1, 8, 32, 64, 128, 256, 512, 1024):
            hidden, w1, w2, q1, q2, topk_w, topk_ids = _build(M, seed=M, dims=dims)

            t_bf16 = _bench(lambda: fmt.fused_moe(hidden, w1, w2, topk_w, topk_ids,
                                                  fuse_silu=False), iters=10, warmup=3)
            t_deq = _bench(lambda: _reference(hidden, q1, q2, topk_w, topk_ids),
                           iters=10, warmup=3)
            t_w8 = _bench(lambda: fused_moe_w8a16(hidden, q1.q, q1.scale, q2.q,
                                                  q2.scale, topk_w, topk_ids,
                                                  quant_group=GROUP), iters=10, warmup=3)
            ratio = t_bf16 / t_w8
            tag = "FASTER" if ratio > 1 else "slower"
            print(f"  {M:5d}  {M * topk / e:8.1f}  {t_bf16:9.3f}ms  {t_deq:13.3f}ms  "
                  f"{t_w8:12.3f}ms  {ratio:6.2f}x {tag}")
            del hidden, w1, w2, q1, q2, topk_w, topk_ids
            torch.cuda.empty_cache()
