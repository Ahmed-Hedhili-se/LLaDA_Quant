"""Denoising-state description and callback protocols for diffusion LMs.

This module owns *no* decoding logic. A masked diffusion LM's forward pass,
router internals and unmasking rule all live in the model repository; here we
only describe **what a denoising state looks like** (`DiffusionState`) and
**what the caller must hand us** to probe one (`LogitsFn`, `RouterFn`,
`AdvanceFn`). That keeps the boundary from the README intact: LLaDA_Quant
never imports a decoder, it accepts callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = [
    "DiffusionState",
    "LogitsFn",
    "RouterFn",
    "AdvanceFn",
    "mask_positions_from_ids",
    "make_masked_states",
    "fully_masked_state",
]


@dataclass(frozen=True)
class DiffusionState:
    """One point on a denoising trajectory.

    Attributes:
        step: Index along the schedule. 0 is the fully masked start; larger
            means further denoised. Only used for ordering and reporting.
        input_ids: ``[B, L]`` token ids, with the mask token at every
            unresolved position.
        mask_positions: ``[B, L]`` bool, True where a token is still masked.
        attention_mask: Optional ``[B, L]`` mask forwarded to the model.
        label: Human-readable tag for reports (e.g. ``"75% masked"``).
    """

    step: int
    input_ids: torch.Tensor
    mask_positions: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(self.input_ids.shape)}")
        if tuple(self.mask_positions.shape) != tuple(self.input_ids.shape):
            raise ValueError(
                f"mask_positions {tuple(self.mask_positions.shape)} does not match "
                f"input_ids {tuple(self.input_ids.shape)}"
            )
        if self.attention_mask is not None and tuple(self.attention_mask.shape) != tuple(
            self.input_ids.shape
        ):
            raise ValueError("attention_mask must have the same shape as input_ids")
        # A state split across devices is always a bug, and it surfaces far from
        # its cause: construction succeeds, then the caller's indexing fails.
        for name, tensor in (
            ("mask_positions", self.mask_positions),
            ("attention_mask", self.attention_mask),
        ):
            if tensor is not None and tensor.device != self.input_ids.device:
                raise ValueError(
                    f"{name} is on {tensor.device} but input_ids is on "
                    f"{self.input_ids.device}; a DiffusionState must live on one device"
                )

    @property
    def num_masked(self) -> int:
        return int(self.mask_positions.sum().item())

    @property
    def mask_ratio(self) -> float:
        """Fraction of *all* positions still masked (prompt included)."""
        return self.num_masked / max(1, self.mask_positions.numel())

    def describe(self) -> str:
        return self.label or f"step {self.step} ({self.num_masked} masked)"


# ``logits_fn(model, state) -> [B, L, vocab]``
LogitsFn = Callable[[nn.Module, DiffusionState], torch.Tensor]

# ``router_fn(model, state) -> top-k expert ids``, either a single tensor or a
# per-layer mapping ``{"layers.0.mlp": ids, ...}``. Return None to skip.
RouterFn = Callable[
    [nn.Module, DiffusionState],
    Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]],
]

# ``advance_fn(state, logits) -> next state`` — the caller's unmasking rule.
# Return None to signal the trajectory is finished.
AdvanceFn = Callable[[DiffusionState, torch.Tensor], Optional[DiffusionState]]


def mask_positions_from_ids(input_ids: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Boolean mask of positions holding ``mask_token_id``."""
    return input_ids == mask_token_id


def fully_masked_state(
    prompt_ids: torch.Tensor,
    gen_length: int,
    mask_token_id: int,
    *,
    step: int = 0,
) -> DiffusionState:
    """Prompt followed by ``gen_length`` mask tokens — the trajectory start."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if gen_length < 1:
        raise ValueError(f"gen_length must be >= 1, got {gen_length}")
    batch = prompt_ids.shape[0]
    gen = torch.full(
        (batch, gen_length), mask_token_id, dtype=prompt_ids.dtype, device=prompt_ids.device
    )
    input_ids = torch.cat([prompt_ids, gen], dim=1)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[:, prompt_ids.shape[1] :] = True
    return DiffusionState(
        step=step, input_ids=input_ids, mask_positions=mask, label="100% masked"
    )


def make_masked_states(
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_token_id: int,
    *,
    ratios: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.1),
    generator: Optional[torch.Generator] = None,
) -> list[DiffusionState]:
    """Build a monotone early/middle/late schedule over the generation region.

    ``completion_ids`` supplies the tokens that get progressively revealed —
    a reference answer, or a BF16 decode of the same prompt. Positions are
    revealed in a fixed random order, so the masked set at ratio 0.25 is a
    strict subset of the one at 0.5, exactly as in a real decode. The prompt
    is never masked.

    Note the honest limitation: revealing *ground-truth* tokens is not the
    same distribution a model conditions on mid-decode, where earlier tokens
    are its own (possibly wrong) predictions. For sensitivity screening this
    is fine; for a headline number, pass a real BF16 decode as
    ``completion_ids`` or use Mode B
    (:func:`~LLaDA_Quant.trajectory.capture.capture_free_running`).
    """
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if completion_ids.dim() == 1:
        completion_ids = completion_ids.unsqueeze(0)
    if prompt_ids.shape[0] != completion_ids.shape[0]:
        raise ValueError(
            f"batch mismatch: prompt {prompt_ids.shape[0]} vs completion {completion_ids.shape[0]}"
        )
    if not ratios:
        raise ValueError("ratios must not be empty")
    if any(not 0.0 <= r <= 1.0 for r in ratios):
        raise ValueError(f"ratios must lie in [0, 1], got {tuple(ratios)}")

    batch, prompt_len = prompt_ids.shape
    gen_length = completion_ids.shape[1]
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

    # One reveal order per row; low rank == masked longest.
    scores = torch.rand((batch, gen_length), generator=generator, device=prompt_ids.device)
    order = scores.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(gen_length, device=order.device).expand(batch, -1))

    states: list[DiffusionState] = []
    for step, ratio in enumerate(ratios):
        n_masked = int(round(ratio * gen_length))
        gen_mask = rank < n_masked  # [B, gen_length]
        mask = torch.zeros((batch, prompt_len + gen_length), dtype=torch.bool, device=order.device)
        mask[:, prompt_len:] = gen_mask
        input_ids = full_ids.clone()
        input_ids[mask] = mask_token_id
        states.append(
            DiffusionState(
                step=step,
                input_ids=input_ids,
                mask_positions=mask,
                label=f"{ratio:.0%} masked",
            )
        )
    return states
