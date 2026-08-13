"""Quantized checkpoint save/load on top of safetensors.

Layout on disk::

    <dir>/
        model-int8.safetensors      packed ints + scales (+ materialized params)
        quantization.json           QuantizationManifest (provenance + config)
        source-checkpoint.json      pointer/hash of the unquantized source

Buffers whose name starts with ``_q`` are the packed integer weights and
``_s`` are scales; the original parameter tensors are stored too so the
checkpoint can be loaded into a plain model or an already-quantized one.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .manifest import MANIFEST_FILENAME, SOURCE_FILENAME, QuantizationManifest, write_source_checkpoint_meta

WEIGHTS_FILENAME = "model-int8.safetensors"


def collect_quantized_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict containing packed buffers and live params of quantized modules."""
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            state[name] = tensor.cpu().detach()
    return state


def save_quantized_checkpoint(
    model: nn.Module,
    manifest: QuantizationManifest,
    directory: str,
    weights_filename: str = WEIGHTS_FILENAME,
) -> None:
    """Write the full quantized artifact (tensors + manifests)."""
    os.makedirs(directory, exist_ok=True)
    state = collect_quantized_state(model)
    save_file(state, os.path.join(directory, weights_filename))
    manifest.save(directory)
    write_source_checkpoint_meta(directory, manifest.source_checkpoint)


def load_quantized_checkpoint(directory: str) -> tuple[dict[str, torch.Tensor], QuantizationManifest]:
    """Load packed tensors + manifest. Does not mutate any model."""
    manifest = QuantizationManifest.from_dict(_read_json(directory, MANIFEST_FILENAME))
    state = load_file(os.path.join(directory, WEIGHTS_FILENAME), device="cpu")
    return state, manifest


def _read_json(directory: str, filename: str) -> dict:
    import json

    with open(os.path.join(directory, filename)) as f:
        return json.load(f)


def _register_missing_buffers(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    """Add persistent buffers found in ``state`` but absent from ``model``.

    Needed so a plain (unquantized) model can absorb a quantized checkpoint
    that adds packed expert buffers (``_qw1``, ``_sw1``, ...).
    """
    existing = set(model.state_dict().keys())
    registered: list[str] = []
    for name, tensor in state.items():
        if name in existing:
            continue
        module = model
        parts = name.split(".")
        for part in parts[:-1]:
            module = module.get_submodule(part)
        module.register_buffer(parts[-1], tensor.clone(), persistent=True)
        registered.append(name)
    return registered


def load_quantized_weights(
    model: nn.Module,
    directory: str,
    strict: bool = True,
) -> QuantizationManifest:
    """Load a quantized checkpoint into ``model`` (may be the plain model).

    Missing quantized buffers (packed expert weights/scales) are registered
    on the fly; after loading, LLaDA expert blocks re-materialize their live
    ``w1``/``w2`` Parameters from the packed buffers, so the model is
    immediately runnable.
    """
    state, manifest = load_quantized_checkpoint(directory)
    _register_missing_buffers(model, state)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and strict:
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    if manifest.config is not None:
        from ..adapters.llada_moe import restore_llada_experts_from_buffers

        restore_llada_experts_from_buffers(model, manifest.config)
    return manifest