"""Quantized checkpoint save/load on top of safetensors.

Layout on disk::

    <dir>/
        model-int8.safetensors      packed ints + scales + untargeted tensors
        quantization.json           manifest: config, targeting audit, totals
        source-checkpoint.json      pointer/hash of the unquantized source

**No redundant BF16.** A quantized checkpoint never stores the dequantized
expert weights, because the load path reconstructs them from the packed
buffers anyway; storing both made the artifact larger than the unquantized
model it came from. Tensors that are re-derivable are dropped at save time
and their absence at load time is expected, not an error.
"""

from __future__ import annotations

import glob
import os
from dataclasses import replace
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from ..config import ExecutionMode, QuantConfig
from .manifest import (
    MANIFEST_FILENAME,
    SOURCE_FILENAME,
    QuantizationManifest,
    write_source_checkpoint_meta,
)

WEIGHTS_GLOB = "model-int*.safetensors"


def weights_filename(bits: int) -> str:
    return f"model-int{bits}.safetensors"


def derivable_tensor_names(manifest: QuantizationManifest) -> Set[str]:
    """Tensors reconstructable from the packed buffers, so never stored.

    For a fused expert block these are the BF16 ``w1``/``w2`` — present only
    in REFERENCE mode, absent already in PACKED mode, and redundant in both.
    """
    names: Set[str] = set()
    for target in manifest.targets:
        if target.kind == "expert":
            prefix = f"{target.name}." if target.name else ""
            names.update({f"{prefix}w1", f"{prefix}w2"})
    return names


def collect_quantized_state(
    model: nn.Module, manifest: Optional[QuantizationManifest] = None
) -> dict[str, torch.Tensor]:
    """State dict for the artifact, with re-derivable BF16 copies removed."""
    skip = derivable_tensor_names(manifest) if manifest is not None else set()
    return {
        name: tensor.cpu().detach()
        for name, tensor in model.state_dict().items()
        if isinstance(tensor, torch.Tensor) and name not in skip
    }


def save_quantized_checkpoint(
    model: nn.Module,
    manifest: QuantizationManifest,
    directory: str,
    weights_file: Optional[str] = None,
) -> str:
    """Write the full quantized artifact. Returns the weights file path."""
    os.makedirs(directory, exist_ok=True)
    bits = manifest.config.bits if manifest.config else 8
    path = os.path.join(directory, weights_file or weights_filename(bits))
    save_file(collect_quantized_state(model, manifest), path)
    manifest.save(directory)
    write_source_checkpoint_meta(directory, manifest.source_checkpoint)
    return path


def find_weights_file(directory: str) -> str:
    matches = sorted(glob.glob(os.path.join(directory, WEIGHTS_GLOB)))
    if not matches:
        raise FileNotFoundError(f"no {WEIGHTS_GLOB} in {directory!r}")
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous checkpoint, multiple weight files: {matches}")
    return matches[0]


def checkpoint_size_bytes(directory: str) -> int:
    """Total on-disk size of the artifact (weights + manifests)."""
    return sum(
        os.path.getsize(os.path.join(directory, f))
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    )


def load_quantized_checkpoint(directory: str) -> Tuple[dict[str, torch.Tensor], QuantizationManifest]:
    """Load packed tensors + manifest. Does not mutate any model."""
    manifest = QuantizationManifest.from_dict(_read_json(directory, MANIFEST_FILENAME))
    state = load_file(find_weights_file(directory), device="cpu")
    return state, manifest


def _read_json(directory: str, filename: str) -> dict:
    import json

    with open(os.path.join(directory, filename)) as f:
        return json.load(f)


def _register_missing_buffers(model: nn.Module, state: dict[str, torch.Tensor]) -> List[str]:
    """Add persistent buffers found in ``state`` but absent from ``model``.

    Needed so a plain (unquantized) model can absorb a quantized checkpoint
    that adds packed expert buffers (``_qw1``, ``_sw1``, ...).
    """
    existing = set(model.state_dict().keys())
    registered: List[str] = []
    for name, tensor in state.items():
        if name in existing:
            continue
        parent_path, _, leaf = name.rpartition(".")
        try:
            module = model.get_submodule(parent_path) if parent_path else model
        except AttributeError as exc:
            raise RuntimeError(
                f"state dict mismatch: checkpoint holds {name!r} but the model has no "
                f"submodule {parent_path!r} ({exc}). The checkpoint was produced from a "
                "different architecture."
            ) from exc
        module.register_buffer(leaf, tensor.clone(), persistent=True)
        registered.append(name)
    return registered


def load_quantized_weights(
    model: nn.Module,
    directory: str,
    strict: bool = True,
    execution_mode: Optional[str] = None,
) -> QuantizationManifest:
    """Load a quantized checkpoint into ``model`` (plain or already quantized).

    Missing packed buffers are registered on the fly. The BF16 expert weights
    are *expected* to be absent from the checkpoint; they are re-derived after
    loading, so their absence never counts as a missing key. Any other missing
    or unexpected key is a real mismatch and raises when ``strict``.

    ``execution_mode`` overrides the residency mode recorded in the manifest.
    The artifact holds the same bytes either way -- ``derivable_tensor_names``
    strips the BF16 experts in REFERENCE mode and they are already gone in
    PACKED -- so residency is a property of *this* run, not of the file. A
    checkpoint written for deployment can therefore be loaded in REFERENCE for
    an accuracy run without rewriting it. The returned manifest reflects the
    mode actually applied.
    """
    state, manifest = load_quantized_checkpoint(directory)
    config: Optional[QuantConfig] = manifest.config
    if execution_mode is not None:
        if config is None:
            raise ValueError(
                "cannot override execution_mode: the manifest carries no quantization "
                "config, so there is nothing to restore expert access with."
            )
        config = replace(config, execution_mode=ExecutionMode(execution_mode).value)
        manifest = replace(manifest, config=config)
    _register_missing_buffers(model, state)

    expected_missing = derivable_tensor_names(manifest)
    incompatible, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible if key not in expected_missing]
    if strict and (missing or unexpected):
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={list(unexpected)}")

    if config is not None:
        from ..adapters.llada_moe import restore_llada_experts_from_buffers

        restore_llada_experts_from_buffers(model, config)
    return manifest
