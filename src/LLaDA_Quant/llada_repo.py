"""Build the inference repository's own BF16 model, and quantize it in place.

These two helpers were written for ``benchmarks/bench_bf16_vs_int4.py`` and
then needed again by ``tools/quantize_checkpoint.py``, so they live here rather
than being imported from one script into another.

Everything here **imports** the inference repository and never modifies it. It
is the only module in the package that knows the repo's module layout
(``model_update.model``, ``src.model.load_weights``); the rest of
``LLaDA_Quant`` works on any ``nn.Module``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

import torch

from .config import QuantConfig


def timer(label: str) -> Callable[[], None]:
    """Print a stage banner now and its duration when the returned fn is called."""
    print(f"  {label} ...", flush=True)
    start = time.perf_counter()

    def done() -> None:
        print(f"  {label}: {time.perf_counter() - start:.1f}s", flush=True)

    return done


def build_bf16_model(repo: str, weight_dir: str, build_device: str = "cpu"):
    """Build the fused-MoE BF16 model, the way the inference repo does.

    ``build_device`` decides where construction happens, and it is a pure
    speed/VRAM trade:

    * ``"cpu"`` (default): safe anywhere. Random-initialising 6.4B parameters
      and fusing 16 blocks costs ~100 s of host CPU.
    * ``"cuda:0"``: the same work on the GPU takes seconds, but the unfused
      model is ~14 GB of VRAM before fusing. Only worth it with roughly 40 GB
      or more, or when nothing else needs to be resident at the same time.
    """
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from model_update.model import LLaDAMoEKV, TritonFusedMoEBlock
    from src.model import load_weights

    index = os.path.join(weight_dir, "model.safetensors.index.json")
    if not os.path.exists(index):
        raise FileNotFoundError(
            f"no checkpoint index at {index}. Cloning the inference repo does not "
            "bring the ~14 GB of weights; fetch them first, e.g.  "
            "huggingface-cli download inclusionAI/LLaDA-MoE-7B-A1B-Instruct "
            f"--local-dir {weight_dir}"
        )

    # Construct directly in BF16. The unfused model is 3072 expert Linears
    # (64 experts x 3 projections x 16 layers); at the default fp32 that is
    # ~25.6 GB of randomly initialised memory, plus another 12.8 GB for the
    # .to(bfloat16) copy. On a host with less RAM than that it swaps, and
    # construction takes tens of minutes instead of a couple.
    step = timer(f"allocating unfused model (BF16) on {build_device}")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(build_device):
            model = LLaDAMoEKV(use_fused_moe=False)
    finally:
        torch.set_default_dtype(previous_dtype)
    model = model.to(torch.bfloat16).eval()
    step()

    step = timer("loading weights")
    load_weights(model, weight_dir, verbose=False)
    step()

    # Fuse AFTER loading: load_state_dict_from_unfused writes w1[i] in place,
    # which is precisely what PACKED mode cannot accept.
    step = timer(f"fusing {len(model.layers)} MoE blocks on {build_device}")
    for index, layer in enumerate(model.layers):
        with torch.device(build_device):
            fused = TritonFusedMoEBlock(layer.mlp.cfg)
        fused = fused.to(device=build_device, dtype=torch.bfloat16)
        fused.load_state_dict_from_unfused(layer.mlp)
        layer.mlp = fused
        if build_device == "cpu":
            print(f"    layer {index + 1}/{len(model.layers)} fused", flush=True)
    step()
    return model.eval()


def quantize_experts_streaming(model, config: QuantConfig, device: str, verbose: bool = True):
    """Quantize expert blocks one at a time on ``device``.

    MSE scale search is ``search_grid`` passes over every weight, each
    allocating GB-scale temporaries. On 6.4B parameters that is tens of minutes
    of CPU. On a GPU it is seconds -- but a whole second BF16 model does not
    always fit next to the reference, so blocks are moved across one at a time.

    Bypasses ``api.quantize_model`` because that quantizes the whole tree at
    once, so the match-count assertion is re-applied here by hand.
    """
    from .adapters.llada_moe import find_expert_blocks, quantize_llada_experts
    from .api import QuantizationResult, TargetingError

    blocks = find_expert_blocks(model, config)
    expected = config.expect_expert_blocks
    if expected is not None and len(blocks) != expected:
        raise TargetingError(
            f"expected {expected} expert block(s), matched {len(blocks)}: "
            f"{[name for name, _, _ in blocks]}"
        )
    if not blocks:
        raise TargetingError("quantization matched no expert blocks")

    records = []
    for index, (name, block, _shape) in enumerate(blocks, start=1):
        block.to(device)
        for record in quantize_llada_experts(block, config):
            record.name = name or record.name
            records.append(record)
        if verbose:
            print(f"    {index}/{len(blocks)} {name} quantized on {device}", flush=True)
    return QuantizationResult(config=config, targets=records)
