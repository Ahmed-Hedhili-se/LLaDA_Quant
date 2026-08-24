"""Wiring the fused W8A16 kernel into a PACKED expert block.

The load-bearing test here is :func:`test_matches_dequantize_per_access`. The
fused forward restates the block's routing (gate -> fp32 softmax -> top-k ->
optional TP masking) because ``TritonFusedMoEBlock.forward`` offers no seam to
call it separately, and a restatement that drifts would change *which experts
run* -- a failure no numerical tolerance would catch as a tolerance problem.
Comparing the two forwards on identical weights turns that risk into an
assertion: the only difference between the arms is where dequantization
happens, so a routing drift shows up immediately as a large error.

Needs a CUDA GPU with Triton and the inference repository importable
(``LLADA_INFERENCE_REPO``). Skips otherwise.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused W8A16 block needs a CUDA GPU"
)


def _ensure_repo() -> bool:
    repo = os.environ.get("LLADA_INFERENCE_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import model_update.model  # noqa: F401
    except ImportError:
        return False
    return True


needs_repo = pytest.mark.skipif(
    not _ensure_repo(), reason="inference repository not importable"
)


def _build_model(bits: int = 8):
    """A small LLaDA-MoE with fused expert blocks, quantized in PACKED mode."""
    from model_update.model import Cfg, LLaDAMoEKV

    from LLaDA_Quant.api import quantize_model
    from LLaDA_Quant.config import QuantConfig

    cfg = Cfg(H=512, NH=4, KVH=4, NL=2, NE=8, TOPK=4, EI=256, VS=157184)
    torch.manual_seed(0)
    model = LLaDAMoEKV(cfg, use_fused_moe=True)
    for layer in model.layers:
        torch.nn.init.normal_(layer.mlp.w1, std=0.02)
        torch.nn.init.normal_(layer.mlp.w2, std=0.02)
    model = model.to(torch.bfloat16).cuda().eval()

    quantize_model(model, QuantConfig(bits=bits, group_size=128,
                                      targets=("expert",),
                                      execution_mode="packed",
                                      expect_expert_blocks=cfg.NL))
    return model, cfg


@needs_repo
def test_install_switches_int8_blocks():
    from LLaDA_Quant.runtime import fused_block

    model, cfg = _build_model(bits=8)
    assert not any(fused_block.is_fused_block(m) for m in model.modules())

    switched = fused_block.install(model)
    assert len(switched) == cfg.NL, f"expected {cfg.NL} blocks, switched {switched}"
    assert all(fused_block.is_fused_block(layer.mlp) for layer in model.layers)


@needs_repo
@torch.no_grad()
def test_the_packed_int8_buffers_are_what_the_kernel_reads():
    """Mutate the INT8 buffer; the output must change.

    Every other test here compares the fused path against another quantized
    path, so all of them would still pass if the fused forward were quietly
    reading a BF16 copy from somewhere. This one cannot: it perturbs the packed
    integers and requires the output to follow.
    """
    from LLaDA_Quant.runtime import fused_block

    model, cfg = _build_model(bits=8)
    fused_block.install(model, strict=True)
    block = model.layers[0].mlp

    # Nothing to fall back to: PACKED deletes the BF16 Parameters.
    resident = dict(block.named_parameters())
    assert "w1" not in resident and "w2" not in resident, (
        f"a BF16 expert Parameter is still resident: {sorted(resident)}"
    )
    assert block._qw1.dtype == torch.int8 and block._qw2.dtype == torch.int8

    torch.manual_seed(0)
    x = torch.randn(1, 16, cfg.H, dtype=torch.bfloat16, device="cuda")
    before = block(x).clone()

    original = block._qw1.clone()
    try:
        block._qw1.zero_()
        after = block(x)
        assert not torch.equal(before, after), (
            "zeroing the packed INT8 weights changed nothing -- the fused kernel "
            "is not reading them"
        )
    finally:
        block._qw1.copy_(original)

    restored = block(x)
    assert torch.equal(before, restored), "restoring the buffer did not restore the output"


@needs_repo
def test_leaves_packed_int4_alone():
    """INT4 has no fused path; silently serving it wrong would be the bug."""
    from LLaDA_Quant.runtime import fused_block

    model, _ = _build_model(bits=4)
    switched = fused_block.install(model)
    assert switched == [], f"INT4 blocks must not be switched, got {switched}"

    with pytest.raises(ValueError, match="INT8"):
        fused_block.install(model, strict=True)


@needs_repo
@torch.no_grad()
@pytest.mark.parametrize("shape", [(1, 8), (2, 32), (1, 128)])
def test_matches_dequantize_per_access(shape):
    """Fused forward vs the PACKED dequantize-per-access forward, same weights.

    Identical quantized values on both sides, so any routing drift in the
    restated forward surfaces here.
    """
    from LLaDA_Quant.runtime import fused_block

    B, T = shape
    model, cfg = _build_model(bits=8)
    torch.manual_seed(7)
    x = (torch.randn(B, T, cfg.H, device="cuda", dtype=torch.bfloat16) * 0.05)

    block = model.layers[0].mlp
    want = block(x)                       # dequantize-per-access
    fused_block.install(model)
    got = model.layers[0].mlp(x)          # packed straight into the kernel

    assert got.shape == want.shape
    assert got.dtype == want.dtype
    rel = (got.float() - want.float()).norm() / want.float().norm().clamp(min=1e-12)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    )
    print(f"  B={B} T={T}  rel_L2={rel:.3e}  cos={cos:.8f}")
    assert rel < 2e-2, f"relative L2 {rel:.3e} — routing drift or kernel bug"
    assert cos > 0.999, f"cosine {cos:.6f}"


@needs_repo
@torch.no_grad()
def test_full_model_forward_still_runs():
    """End to end through LLaDAMoEKV, not just one block."""
    from LLaDA_Quant.runtime import fused_block

    model, cfg = _build_model(bits=8)
    ids = torch.randint(0, 1000, (2, 24), device="cuda")

    want, _ = model(ids, position_offset=0)
    fused_block.install(model)
    got, _ = model(ids, position_offset=0)

    assert got.shape == want.shape
    rel = (got.float() - want.float()).norm() / want.float().norm().clamp(min=1e-12)
    print(f"  full-model logits rel_L2={rel:.3e}")
    assert rel < 5e-2, f"full-model relative L2 {rel:.3e}"
