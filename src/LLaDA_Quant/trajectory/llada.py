"""Bind Mode B to the *production* LLaDA decoder instead of copying it.

A measurement built on a reimplemented decoder is worse than no measurement:
it drifts, and it drifts silently, so the numbers keep looking plausible
while describing something that never runs. This module therefore imports the
real commit rule out of the inference repository —

    model_update/generate.py :: add_gumbel_noise
                                get_num_transfer_tokens
                                select_transfer_indices

— and assembles an ``advance_fn`` from those exact functions. Nothing about
LLaDA's decoding semantics is restated here; if the inference repo changes
its rule, this follows automatically.

The inference repository is never imported at module load and never modified.
Point :func:`load_llada_decoder` at its path when you want Mode B against the
real thing; everything else in :mod:`LLaDA_Quant.trajectory` works without it.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state import DiffusionState

LLADA_MASK_ID = 156895
"""From test_llada/src/model.py. Passed explicitly everywhere; this is a default."""


@dataclass
class LLaDADecoder:
    """The production commit rule, as imported (not reimplemented)."""

    add_gumbel_noise: Callable[..., torch.Tensor]
    get_num_transfer_tokens: Callable[..., torch.Tensor]
    select_transfer_indices: Callable[..., torch.Tensor]
    source: str

    def describe(self) -> str:
        return f"production LLaDA decoder from {self.source}"


def load_llada_decoder(repo_path: str, module: str = "model_update.generate") -> LLaDADecoder:
    """Import the real decoding primitives from the inference repository.

    Args:
        repo_path: Path to the inference repo root (the directory containing
            ``model_update/``).
        module: Dotted module holding the commit rule.

    Raises:
        ImportError: With the searched path, when the repo is not importable.
    """
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        generate = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on external repo
        raise ImportError(
            f"could not import {module!r} from {repo_path!r}: {exc}. "
            "Mode B against the production decoder needs the inference "
            "repository on the path; the rest of LLaDA_Quant does not."
        ) from exc
    missing = [
        name
        for name in ("add_gumbel_noise", "get_num_transfer_tokens", "select_transfer_indices")
        if not hasattr(generate, name)
    ]
    if missing:
        raise ImportError(f"{module!r} is missing expected decoder functions: {missing}")
    return LLaDADecoder(
        add_gumbel_noise=generate.add_gumbel_noise,
        get_num_transfer_tokens=generate.get_num_transfer_tokens,
        select_transfer_indices=generate.select_transfer_indices,
        source=f"{repo_path}::{module}",
    )


def make_llada_advance_fn(
    decoder: LLaDADecoder,
    steps: int,
    *,
    mask_token_id: int = LLADA_MASK_ID,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
) -> Callable[[DiffusionState, torch.Tensor], Optional[DiffusionState]]:
    """An ``advance_fn`` whose every decision comes from ``decoder``.

    Mirrors the structure of ``_generate_block_cached``'s inner loop: gumbel
    noise, argmax, confidence at masked positions, then the production
    top-k position selection. The per-step budget is computed once from the
    initial mask, exactly as the real loop does.

    ``temperature=0.0`` is strongly recommended for measurement. With
    stochastic sampling, two models draw independent gumbel noise and the
    resulting divergence is sampling, not quantization — see Phase 10 of the
    project spec and :mod:`~LLaDA_Quant.trajectory.report`.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    budget: dict[int, torch.Tensor] = {}

    def advance(state: DiffusionState, logits: torch.Tensor) -> Optional[DiffusionState]:
        mask_index = state.mask_positions.to(logits.device).bool()
        if not bool(mask_index.any()):
            return None
        if not budget:
            budget[0] = decoder.get_num_transfer_tokens(mask_index, steps)
        step_index = min(state.step, budget[0].shape[1] - 1)
        num_transfer_step = budget[0][:, step_index]

        noisy = decoder.add_gumbel_noise(logits, temperature)
        x0 = noisy.argmax(dim=-1)
        if remasking == "low_confidence":
            probs = F.softmax(logits.float(), dim=-1)
            confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
            confidence = torch.rand(x0.shape, device=logits.device)
        else:
            raise ValueError(f"unknown remasking {remasking!r}")
        confidence = torch.where(
            mask_index, confidence, torch.full_like(confidence, -float("inf"))
        )

        transfer_index = decoder.select_transfer_indices(confidence, num_transfer_step)
        if not bool(transfer_index.any()):
            return None

        input_ids = state.input_ids.clone()
        input_ids[transfer_index] = x0[transfer_index]
        next_mask = mask_index & ~transfer_index
        return DiffusionState(
            step=state.step + 1,
            input_ids=input_ids,
            mask_positions=next_mask,
            attention_mask=state.attention_mask,
            label=f"step {state.step + 1}",
        )

    return advance


class RouterCapture:
    """Recover each MoE block's top-k expert selection without touching the model.

    ``TritonFusedMoEBlock`` computes ``topk_ids`` inside ``forward`` and does
    not expose it, so a plain forward hook cannot see it. This registers a
    *pre*-hook to stash each block's input and recomputes the routing exactly
    as the block does::

        x_flat          = x.reshape(-1, H)
        router_logits   = block.gate(x_flat)
        routing_weights = softmax(router_logits, dim=-1, dtype=float32)
        topk_ids        = topk(routing_weights, TOPK).indices

    The recomputation is bit-identical because it is the same ops on the same
    tensors — and because ``gate`` is excluded from quantization, its weights
    are identical in both models. Any routing difference therefore comes purely
    from the hidden states diverging, which is the causal chain worth measuring:
    quantized experts → different hidden state → different expert choice →
    different logits → a different token committed.

    Global ids are captured, i.e. before the tensor-parallel local-expert
    remap, so two ranks' routings stay comparable.
    """

    def __init__(self, model: nn.Module, top_k: Optional[int] = None) -> None:
        self.model = model
        self._top_k = top_k
        self._topk_ids: Dict[str, torch.Tensor] = {}
        self._gates: Dict[str, torch.Tensor] = {}
        self._handles: List[Any] = []
        self._attach()

    def _attach(self) -> None:
        from ..adapters.llada_moe import is_fused_expert_block

        for name, block in self.model.named_modules():
            if not is_fused_expert_block(block) or not hasattr(block, "gate"):
                continue
            self._handles.append(block.register_forward_pre_hook(self._make_hook(name)))
        if not self._handles:
            raise RuntimeError(
                "RouterCapture found no fused expert block with a `gate` submodule. "
                "Build the model with use_fused_moe=True before attaching."
            )

    def _resolve_top_k(self, block: nn.Module, num_experts: int) -> int:
        if self._top_k is not None:
            return self._top_k
        cfg = getattr(block, "cfg", None)
        top_k = getattr(cfg, "TOPK", None) if cfg is not None else None
        if top_k is None:
            raise RuntimeError(
                "cannot determine top_k: the block has no cfg.TOPK, so pass "
                "top_k=... to attach_router_capture()"
            )
        return min(int(top_k), num_experts)

    def _make_hook(self, name: str):
        def hook(block: nn.Module, args: tuple):
            if not args:
                return None
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None
            with torch.no_grad():
                x_flat = x.reshape(-1, x.shape[-1])
                # A no-op in the real model, where the gate and the hidden
                # state are both BF16. It matters only so that a dtype
                # mismatch surfaces as the model's own error on its own
                # forward, rather than as a crash inside this hook — which
                # would misattribute the failure to the measurement tooling.
                gate_dtype = block.gate.weight.dtype
                if x_flat.dtype != gate_dtype:
                    x_flat = x_flat.to(gate_dtype)
                router_logits = block.gate(x_flat)
                weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                top_k = self._resolve_top_k(block, weights.shape[-1])
                self._topk_ids[name] = weights.topk(top_k, dim=-1).indices.detach().cpu()
                self._gates[name] = router_logits.detach().float().cpu()
            return None

        return hook

    @property
    def topk_ids(self) -> Dict[str, torch.Tensor]:
        """Per-layer ``[tokens, top_k]`` expert ids from the most recent forward."""
        return dict(self._topk_ids)

    @property
    def gates(self) -> Dict[str, torch.Tensor]:
        """Per-layer raw router logits ``[tokens, num_experts]``."""
        return dict(self._gates)

    @property
    def layer_names(self) -> List[str]:
        return sorted(self._topk_ids)

    def clear(self) -> None:
        self._topk_ids.clear()
        self._gates.clear()

    def remove(self) -> None:
        """Detach every hook. Safe to call twice."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "RouterCapture":
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


def attach_router_capture(model: nn.Module, top_k: Optional[int] = None) -> RouterCapture:
    """Register router pre-hooks on every fused expert block of ``model``."""
    return RouterCapture(model, top_k=top_k)


def router_fn_for(*captures: RouterCapture) -> Callable[..., Optional[dict]]:
    """A ``router_fn`` that returns whichever capture belongs to the model asked for.

    ``capture_shared`` calls ``router_fn(model, state)`` immediately after
    ``logits_fn(model, state)``, so each capture holds that model's own routing
    from its own forward. One capture per model keeps them from overwriting
    each other — which is why dispatch is by model identity rather than a
    single shared registry.
    """
    lookup = {id(capture.model): capture for capture in captures}

    def router_fn(model: nn.Module, state: DiffusionState) -> Optional[dict]:
        capture = lookup.get(id(model))
        return capture.topk_ids if capture is not None else None

    return router_fn


def gates_fn_for(*captures: RouterCapture) -> Callable[..., Optional[dict]]:
    """Same dispatch, returning raw router logits for margin/entropy stats."""
    lookup = {id(capture.model): capture for capture in captures}

    def gates_fn(model: nn.Module, state: DiffusionState) -> Optional[dict]:
        capture = lookup.get(id(model))
        return capture.gates if capture is not None else None

    return gates_fn


def assert_matches_production_decoder(
    decoder: LLaDADecoder,
    logits: torch.Tensor,
    initial_state: DiffusionState,
    steps: int,
    *,
    mask_token_id: int = LLADA_MASK_ID,
) -> None:
    """Prove one step of :func:`make_llada_advance_fn` equals the real rule.

    Runs the imported primitives directly, in the order
    ``_generate_block_cached`` uses them, and requires the adapter to produce
    identical ids and mask. Greedy only — with temperature the two draws are
    independent and the comparison would be meaningless.

    Raises:
        AssertionError: If the adapter and the production primitives disagree.
    """
    advance = make_llada_advance_fn(decoder, steps, mask_token_id=mask_token_id, temperature=0.0)
    produced = advance(initial_state, logits)

    mask_index = initial_state.mask_positions.bool()
    num_transfer = decoder.get_num_transfer_tokens(mask_index, steps)
    x0 = decoder.add_gumbel_noise(logits, 0.0).argmax(dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, -float("inf")))
    transfer_index = decoder.select_transfer_indices(confidence, num_transfer[:, 0])
    expected_ids = initial_state.input_ids.clone()
    expected_ids[transfer_index] = x0[transfer_index]

    if produced is None:
        raise AssertionError("adapter returned None where the production rule committed tokens")
    if not torch.equal(produced.input_ids, expected_ids):
        raise AssertionError(
            "adapter ids diverge from the production decoder: "
            f"{(produced.input_ids != expected_ids).sum().item()} position(s) differ"
        )
    if not torch.equal(produced.mask_positions.bool(), mask_index & ~transfer_index):
        raise AssertionError("adapter mask diverges from the production decoder")
