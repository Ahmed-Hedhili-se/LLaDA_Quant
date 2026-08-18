"""Shared fixtures: a model shaped exactly like the LLaDA-MoE target."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 128
INTERMEDIATE = 64
NUM_EXPERTS = 4
VOCAB = 32
MASK_ID = 31


class FusedExpertBlock(nn.Module):
    """Fused w1/w2 layout: w1 [E, 2I, H], w2 [E, H, I] — the real signature."""

    def __init__(self, num_experts=NUM_EXPERTS, hidden=HIDDEN, intermediate=INTERMEDIATE, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # BF16 like the real model, which is built with .to(torch.bfloat16)
        # before the fused blocks are attached.
        self.gate = nn.Linear(hidden, num_experts, bias=False).to(torch.bfloat16)
        self.w1 = nn.Parameter(
            (torch.randn(num_experts, 2 * intermediate, hidden, generator=gen) * 0.02).to(
                torch.bfloat16
            )
        )
        self.w2 = nn.Parameter(
            (torch.randn(num_experts, hidden, intermediate, generator=gen) * 0.02).to(
                torch.bfloat16
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense over all experts — enough to exercise w1/w2 access."""
        w1, w2 = self.w1.float(), self.w2.float()
        i = w2.shape[-1]
        gate = torch.einsum("th,enh->etn", x.float(), w1[:, :i])
        up = torch.einsum("th,enh->etn", x.float(), w1[:, i:])
        return torch.einsum("etk,enk->etn", F.silu(gate) * up, w2).mean(0)


class TinyMoEModel(nn.Module):
    """Two MoE layers plus the pieces that must never be quantized."""

    def __init__(self, layers=2, seed=0):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList()
        for index in range(layers):
            layer = nn.Module()
            layer.mlp = FusedExpertBlock(seed=seed + index)
            layer.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False).to(torch.bfloat16)
            layer.input_layernorm = nn.LayerNorm(HIDDEN).to(torch.bfloat16)
            self.layers.append(layer)
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False).to(torch.bfloat16)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False).to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids).to(torch.bfloat16)
        h = h + h.mean(dim=1, keepdim=True)
        return self.lm_head(h)


@pytest.fixture
def expert_block() -> FusedExpertBlock:
    return FusedExpertBlock()


@pytest.fixture
def moe_model() -> TinyMoEModel:
    torch.manual_seed(0)
    return TinyMoEModel()
