"""Grouped-expert W8A16 MoE: the standalone GEMM's trick, applied to ``fused_moe``.

:mod:`LLaDA_Quant.runtime.kernels.w8a16_gemm` proved that dequantizing inside
the GEMM's K-loop beats dequantizing to HBM -- up to 1.96x faster than BF16 for
a plain matmul. That kernel has no expert routing, so the model could not use
it: 92% of LLaDA-MoE's weights are multiplied inside ``fused_moe``'s
grouped-expert path, not a plain GEMM.

This module closes that gap. It is a token-by-expert grouped GEMM with the same
sorting/padding contract as the inference repository's ``fused_moe_kernel`` --
it reuses that repo's ``moe_align_block_size`` rather than reimplementing it --
but ``B`` arrives as INT8 plus per-group scales and is expanded in registers.
The BF16 expert weight is never materialised.

Traffic, per MoE layer of the real checkpoint (403 M weights)::

    dequantize-then-matmul : 204 read + 768 written + 768 read = 1740 MiB
    this kernel            : 204 read                          =  204 MiB

Scale layout
------------
``quantize_tensor`` groups along the **last** axis, and both expert tensors are
stored with K last::

    w1 [E, 2*EI, H]  ->  qw1 [E, 2*EI, H] int8,  sw1 [E, 2*EI, H // G]
    w2 [E, H,  EI]   ->  qw2 [E, H,  EI]  int8,  sw2 [E, H,  EI // G]

so in both GEMMs the scale is indexed ``[expert, n, k // G]``. Within one
K-block the group index is constant and the scale varies only along N, which is
why a single ``[BLOCK_SIZE_N]`` vector broadcast across the tile is correct --
provided a K-block never straddles a group boundary. :func:`fused_moe_w8a16`
enforces ``G % BLOCK_SIZE_K == 0`` rather than trusting it.

No SiLU epilogue, deliberately
------------------------------
The inference repo's BF16 kernel folds ``silu(gate) * up`` into GEMM1's
epilogue, which needs the gate and up fp32 accumulators live in registers
simultaneously (two B tiles per program). In-register dequantization is a third
claim on the same budget, and sglang -- which ships both features -- asserts
they are mutually exclusive (``fuse_swiglu_interleaved`` refuses any
fp8/int8/int4 call, because the epilogue reads gate/up as halves in-register).
This kernel takes the same position: GEMM1 emits the full ``2*EI`` width and the
activation runs as a separate pass, exactly as the BF16 path does under
``LLADA_MOE_FUSED_SILU=0``. The epilogue is worth +1-10%; the dequantize round
trip costs ~6x. Fusing both is future work, not a prerequisite.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the platform
    HAS_TRITON = False


DEFAULT_GROUP_SIZE = 128


def _moe_align_block_size():
    """The inference repo's own sorting/padding, imported rather than copied.

    Keeping a second copy in sync with a kernel contract is the exact class of
    silent-divergence bug this toolkit exists to catch, so this raises with the
    fix instead of falling back to a vendored duplicate.
    """
    try:
        from model_update.fused_moe_triton import moe_align_block_size
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise ImportError(
            "fused_moe_w8a16 needs the inference repository on sys.path for "
            "moe_align_block_size -- the same import benchmarks/serve_quantized.py "
            "sets up from --repo. Add the repo root to sys.path first."
        ) from exc
    return moe_align_block_size


if HAS_TRITON:

    @triton.jit
    def fused_moe_w8a16_kernel(
            a_ptr, b_ptr, c_ptr, bs_ptr,
            topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
            num_tokens_post_padded_ptr,
            N, K, EM, num_valid_tokens,
            stride_am, stride_ak,
            stride_be, stride_bk, stride_bn,
            stride_cm, stride_cn,
            stride_bse, stride_bsn, stride_bsg,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
            QUANT_GROUP: tl.constexpr,
            MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
            compute_type: tl.constexpr, is_first_gemm: tl.constexpr):
        """Structurally the inference repo's ``fused_moe_kernel`` -- same pid
        grouping, same padded-token masking, same A-indexing split on
        ``is_first_gemm`` -- with one change: ``b`` loads as INT8 and is scaled
        into ``compute_type`` in registers before ``tl.dot``.
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens

        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        if is_first_gemm:
            a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am
                              + offs_k[None, :] * stride_ak)
        else:
            a_ptrs = a_ptr + (offs_token[:, None] * stride_am
                              + offs_k[None, :] * stride_ak)

        off_experts = tl.load(expert_ids_ptr + pid_m)
        b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk
                                                    + offs_bn[None, :] * stride_bn)
        bs_base = bs_ptr + off_experts * stride_bse + offs_bn * stride_bsn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_rem = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
            b_i8 = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0)
            # QUANT_GROUP % BLOCK_SIZE_K == 0 is enforced host-side, so a
            # K-block never straddles two groups and one scale vector serves
            # the whole tile.
            g = (k * BLOCK_SIZE_K) // QUANT_GROUP
            bs = tl.load(bs_base + g * stride_bsg)
            # The point of the module: the INT8 tile expands here, in registers.
            b = (b_i8.to(tl.float32) * bs[None, :]).to(compute_type)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke(A, B_q, B_s, C, topk_weights, topk_ids, sorted_token_ids, expert_ids,
            num_tokens_post_padded, mul_routed_weight, top_k, config, quant_group):
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B_q.shape[1], META["BLOCK_SIZE_N"]),
    )
    kwargs = dict(config)
    num_warps = kwargs.pop("num_warps", 4)
    num_stages = kwargs.pop("num_stages", 3)

    fused_moe_w8a16_kernel[grid](
        A, B_q, C, B_s,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B_q.shape[1], B_q.shape[2], sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        B_q.stride(0), B_q.stride(2), B_q.stride(1),
        C.stride(-2), C.stride(-1),
        B_s.stride(0), B_s.stride(1), B_s.stride(2),
        QUANT_GROUP=quant_group,
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type, is_first_gemm=not mul_routed_weight,
        num_warps=num_warps, num_stages=num_stages,
        **kwargs,
    )


def default_config(M: int, E: int, quant_group: int = DEFAULT_GROUP_SIZE) -> Dict[str, Any]:
    """Tile choice for the quantized path.

    Deliberately not the inference repo's ``get_best_config``: those configs are
    tuned for BF16 tiles, INT8 halves the B-tile bytes so the shared-memory
    economics differ, and ``BLOCK_SIZE_K`` must divide ``quant_group`` -- a
    constraint that tuner knows nothing about. These are the shapes the
    standalone W8A16 GEMM measured well at. Re-tune per GPU before quoting
    numbers.
    """
    block_k = min(128, quant_group)
    while quant_group % block_k:
        block_k //= 2
    if M <= E:
        return {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    return {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}


def fused_moe_w8a16(hidden_states: torch.Tensor,
                    qw1: torch.Tensor, sw1: torch.Tensor,
                    qw2: torch.Tensor, sw2: torch.Tensor,
                    gating_output: torch.Tensor, topk_ids: torch.Tensor,
                    quant_group: int = DEFAULT_GROUP_SIZE,
                    config: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """Drop-in for the inference repo's ``fused_moe`` that consumes INT8 experts.

    Same signature and same return value, with ``w1``/``w2`` replaced by their
    packed values and per-group scales. The activation between the two GEMMs
    stays a separate BF16 pass (see the module docstring for why the SiLU
    epilogue is not fused here).
    """
    if not HAS_TRITON:
        raise RuntimeError("fused_moe_w8a16 requires Triton and a CUDA GPU.")
    if qw1.dtype != torch.int8 or qw2.dtype != torch.int8:
        raise TypeError(
            f"expected INT8 expert weights, got qw1={qw1.dtype} qw2={qw2.dtype}."
        )

    M, K = hidden_states.shape
    E, N, _ = qw1.shape
    top_k = topk_ids.shape[1]

    # A dtype check alone does NOT catch packed INT4: pack_int4 stores two
    # 4-bit values per byte in an int8 tensor, so qw.dtype is int8 either way
    # and the kernel would read half a K-axis of nibbles as whole values --
    # silently wrong output, no error. The structural check is the K extent,
    # which is halved by packing. (Caught by test_rejects_packed_int4, which
    # failed against the dtype-only guard this replaces.)
    if qw1.shape[2] != K:
        raise TypeError(
            f"qw1 K-extent {qw1.shape[2]} != hidden K {K}. Packed INT4 (two "
            "values per byte) halves it and needs an unpack inside the K-loop; "
            "this kernel is INT8-only."
        )
    if qw2.shape[2] != N // 2:
        raise TypeError(
            f"qw2 K-extent {qw2.shape[2]} != EI {N // 2}. Packed INT4 halves "
            "it; this kernel is INT8-only."
        )

    cfg = dict(config) if config is not None else default_config(M, E, quant_group)
    if quant_group % cfg["BLOCK_SIZE_K"]:
        raise ValueError(
            f"BLOCK_SIZE_K={cfg['BLOCK_SIZE_K']} must divide the quantization "
            f"group size {quant_group}, otherwise a K-block straddles two groups "
            "and silently uses the wrong scale."
        )

    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size()(
        topk_ids, cfg["BLOCK_SIZE_M"], E
    )

    cache1 = torch.empty((M, top_k, N), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    _invoke(hidden_states, qw1, sw1, cache1, gating_output, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k, config=cfg, quant_group=quant_group)

    gate, up = cache1.chunk(2, dim=-1)
    cache2 = (F.silu(gate) * up).view(M * top_k, N // 2)

    cache3 = torch.empty((M, top_k, qw2.shape[1]), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    _invoke(cache2, qw2, sw2, cache3, gating_output, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=True, top_k=top_k, config=cfg, quant_group=quant_group)

    return cache3.sum(dim=1)
