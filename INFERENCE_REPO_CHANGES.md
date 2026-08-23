# Changes in `test_llada` that affect this toolkit

Handoff from the inference repository, 2026-08-16. Covers the commit range
`b4872e9..fbc1cae` on `Ahmed-Hedhili-se/test_llada` (mirrored to
`RLS-ResearchLab/LLaDA_infr_hedhili`).

Two things happened that matter here:

1. **The kernel a future INT8/INT4 fused MoE would extend has changed shape.**
   `fused_moe_kernel` now folds the SiLU activation into GEMM1's epilogue,
   which changes the tile structure, the shared-memory budget, the output
   layout, and four function signatures. Anything written against the old
   kernel needs updating.
2. **`Should the kernel be built?` now has measured evidence, not just a
   roofline model.** Some of it supports that section's reasoning; one
   measurement materially weakens the fallback argument for INT4. Details in
   [§3](#3-evidence-for-should-the-kernel-be-built).

Everything below labelled `MEASURED` came from a single NVIDIA A40-24Q
(sm_86, 84 SMs, 24 GB, ~696 GB/s) running the real 7B checkpoint.

---

## Contents

- [1. What changed in the MoE kernel](#1-what-changed-in-the-moe-kernel)
- [2. API deltas you must track](#2-api-deltas-you-must-track)
- [3. Evidence for "Should the kernel be built?"](#3-evidence-for-should-the-kernel-be-built)
- [4. Notes for a low-bit fused kernel](#4-notes-for-a-low-bit-fused-kernel)
- [4b. The SwiGLU-epilogue conflict, confirmed against sglang](#4b-the-swiglu-epilogue-conflict-confirmed-against-sglang)
- [5. The numerical-contract method](#5-the-numerical-contract-method)
- [6. Unrelated changes worth knowing](#6-unrelated-changes-worth-knowing)

---

## 1. What changed in the MoE kernel

### 1.1 SiLU folded into GEMM1's epilogue

Previously `fused_moe` ran:

```
GEMM1 → materialize [M, top_k, 2*EI] → read back → SiLU(gate)*up → [M*top_k, EI] → GEMM2
```

`w1` packs `[gate ; up]` along N, so the old kernel emitted the full `2*EI`
width and a separate elementwise pass consumed it. Now each program computes
**two B tiles** — gate at `offs_bn`, up at `offs_bn + N` — keeps two fp32
accumulators, and applies the activation from registers, writing `EI`-wide
output directly. The wide intermediate is never allocated.

Consequences that matter to a dequantizing kernel:

| | Before | After (`SILU_EPILOGUE=True`) |
|---|---|---|
| `N` kernel arg | `w1.shape[1]` = `2*EI` | **output** width = `EI` |
| B tiles in flight per K step | 1 | **2** |
| N grid | `cdiv(2*EI, BLOCK_SIZE_N)` | `cdiv(EI, BLOCK_SIZE_N)` — halved |
| GEMM1 output | `[M, top_k, 2*EI]` (3-D) | `[M*top_k, EI]` (2-D) |
| Shared memory | `(BM·BK + BK·BN)·stages·2` | `(BM·BK + **2**·BK·BN)·stages·2` |

The `up` columns are addressed as `offs_bn + N` because `N` is now the half
width. Your groupwise scales run along the **last (K) axis**, which is
orthogonal to this gate/up split along N — so the layout stays compatible, but
a dequantizing kernel must fetch scales for *two* N-tiles per K step.

Kill switch: `LLADA_MOE_FUSED_SILU=0` restores the old path. The module global
is `fused_moe_triton.FUSE_SILU`.

### 1.2 The autotuner now follows `FUSE_SILU`

`tuning_fused_moe_triton.py::profile_config` benchmarked the unfused chain,
which after this change would have tuned for a pipeline that no longer runs.
It now branches on `FUSE_SILU` for both the cache allocation and the
`run_full_pipeline` body, and its search-space pruning calls the same
`_shmem_bytes()` helper the runtime guard uses, so the two cannot drift.

**This matters to you directly**: a low-bit kernel has different tile
economics (B tiles are 1 byte/element instead of 2, so more N fits in shared
memory) and will need its own tuning pass. The generated
`moe_tune_config.json` is per-(hardware, kernel-variant) and is **not**
checked in.

---

## 2. API deltas you must track

All in `model_update/fused_moe_triton.py` unless noted.

```python
# NEW module global — env-overridable
FUSE_SILU = os.environ.get("LLADA_MOE_FUSED_SILU", "1") != "0"

# NEW shared helper. NOTE the hardcoded `* 2` (bytes per bf16 element) —
# a low-bit kernel must generalise this.
def _shmem_bytes(bm, bn, bk, ns, silu_epilogue) -> int:
    b_tiles = 2 if silu_epilogue else 1
    return (bm * bk + b_tiles * bk * bn) * ns * 2

# CHANGED: third parameter added
def get_best_config(M: int, E: int, silu_epilogue: bool = False) -> Dict[str, Any]

# CHANGED: silu_epilogue kwarg; n_out derived; C strides now taken from the
# last two axes so both 3-D and 2-D C work
def invoke_fused_moe_kernel(A, B, C, topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded,
                            mul_routed_weight, top_k, config,
                            silu_epilogue: bool = False) -> None

# CHANGED: fuse_silu kwarg (None = follow FUSE_SILU)
def fused_moe(hidden_states, w1, w2, gating_output, topk_ids,
              fuse_silu: Optional[bool] = None)

# CHANGED (Triton): new constexpr, last positional
fused_moe_kernel(..., is_first_gemm: tl.constexpr, SILU_EPILOGUE: tl.constexpr)
```

Two things that will bite:

- **`_shmem_bytes` assumes 2 bytes per element.** For INT8 B tiles it is
  `bm*bk*2 + b_tiles*bk*bn*1`, and for packed INT4 `*0.5`. Both the runtime
  guard and the tuner's pruning read this one function, so generalising it
  fixes both at once — but leaving it will over-estimate your budget and
  silently reject configs that would have fit.
- **`get_best_config` picks one config used by both GEMMs.** GEMM1 (with two
  B tiles) is the more expensive of the two, so the guard sizes for it. A
  low-bit variant that dequantizes only in GEMM1, or uses different tiles per
  GEMM, breaks that assumption.

Also changed, outside the MoE path: `LLaDAMoEKV.forward` gained
`num_logits: Optional[int] = None` (`model_update/model.py`). Default `None`
preserves prior behaviour; it narrows the final norm + `lm_head` to the rows
the caller actually consumes. Relevant to your memory story — see
[§6](#6-unrelated-changes-worth-knowing).

---

## 3. Evidence for "Should the kernel be built?"

Your README says to run `bench_moe_regime.py` *before* writing the kernel.
Here is the measured side, from the real model.

### 3.1 The bandwidth-bound premise is confirmed `MEASURED`

Nsight Compute on `fused_moe_kernel` (batch 32, `--set full`):

| Metric | Value |
|---|---:|
| L2 cache throughput | **96.35%** |
| DRAM throughput | 66.40% |
| Compute (SM) throughput | 41.25% |

And an end-to-end check from a `torch.profiler` trace (batch 11, 32 tokens,
32 steps, block 32 → 34 forward passes):

| | |
|---|---:|
| Expert weights per layer (`w1` 537 MB + `w2` 268 MB) | 805 MB |
| × 16 layers × 34 forwards | **438 GB** |
| At ~696 GB/s, floor | 626 ms |
| `fused_moe_kernel` measured | **770 ms** (58.75% of all GPU time) |
| **Fraction of theoretical weight-streaming peak** | **81%** |

So the kernel really is streaming expert weights at near hardware limit, and
every expert is touched every forward (352 tokens × top-8 = 2,816 assignments
across 64 experts). **Halving weight bytes should translate close to directly
into time in this regime.** That is the strongest argument for the kernel, and
it is now measured rather than modelled.

One correction to a claim your README quotes from the inference repo: the
"~96% of peak memory bandwidth" figure that circulated is **L2, not DRAM**.
Nsight's Speed-of-Light row reports the most-utilised memory subsystem. It
matters: read as a DRAM ceiling it says "nothing left to gain", read as an L2
ceiling it says "remove intermediate traffic" — which is what produced the
SiLU epilogue in §1.1. The inference repo's README has been corrected.

### 3.2 Your roofline model agrees with the measurement `MEASURED`

At the traced workload M = 11 × 32 = 352, so M/expert = 352 × 8 / 64 = **44**,
against your BF16 crossover of 202 → predicted memory-bound. Measured 81% of
bandwidth peak. **Your model called it correctly.**

Direct arithmetic-intensity check, same workload: 35.4 GFLOP against 805 MB of
weights = 44 FLOP/byte, versus an A40 balance of ~215 FLOP/byte. Same verdict
by a second route.

### 3.3 The capacity → batch → throughput argument is weaker than assumed `MEASURED`

This is the finding I would most want you to have.

Your verdict for the throughput regime is that the win is capacity: INT4 frees
memory, which allows a larger batch, which raises throughput — "a different
mechanism, and should be measured as one." **It has now been measured, and it
saturates early.**

Throughput vs `BATCH_MAX_SIZE`, single A40-24Q, 128 tokens / steps=128 /
block=32, concurrency = batch:

| `BATCH_MAX_SIZE` | Tok/s | Δ throughput | Δ batch | p50 latency |
|---:|---:|---:|---:|---:|
| 8 | 150.3 | — | — | 6.77 s |
| 16 | 204.8 | +36.3% | +100% | 9.93 s |
| 24 | 224.8 | +9.8% | +50% | 13.56 s |
| **32** | **243.2** | +8.2% | +33% | 16.71 s |
| 48 | **0/96 requests succeeded** | — | — | — |

8 → 32 is 4× the batch for **1.62×** the throughput, and the final step bought
8.2%. The curve is past its knee by batch 32.

**Implication for INT4's capacity argument**: on a 24 GB card, freeing memory
to roughly double the batch would be worth **single-digit percent** throughput,
not a multiple. The capacity win is real for *fitting* a model or a longer
context; it is weak as a *throughput* mechanism on this hardware. On a 48 GB
A6000 the knee will sit at a different batch, but the shape of the curve —
front-loaded, saturating — should be expected to hold, because it is the same
mechanism your own `M/expert` analysis describes: past the crossover, extra
tokens stop riding along free.

### 3.4 Where that leaves the verdict

Reconciling your table with the measurements, at the inference repo's actual
operating point (batch 32, block 32 → step-call M = 1024, M/expert = 128):

- **BF16**: 128 < 202 → memory-bound. Confirmed at 81% of peak.
- **W8A16**: 128 > 101 → compute-bound *once quantized*. The weight-only
  kernel would deliver well under its 2× ceiling here.
- **Batch 1–16** (M/expert 4–64): memory-bound before *and* after quantizing.
  This is where a W8A16/W4A16 fused kernel gets close to its full ratio.

So the trap your README already flags — quantizing halves the crossover, so a
bandwidth-bound workload can become compute-bound once quantized — is exactly
what happens at the throughput operating point. **The kernel's case is
strongest for latency-oriented, small-batch serving, and weakest for the
high-batch throughput regime**, which inverts the intuition that quantization
is a throughput play.

If the deployment target moves to H100 (planned), re-run this: ~3.35 TB/s
against a much higher INT8 TOPS shifts every crossover, and native fp8 changes
the arithmetic entirely.

---

## 4. Notes for a low-bit fused kernel

Concrete things the §1.1 restructure implies.

1. **Two dequants per K step, not one.** Each program now loads B tiles at
   `offs_bn` and `offs_bn + N`. Dequant *volume* is unchanged (half the
   programs, twice the work each) but **register pressure roughly doubles** —
   two fp32 accumulators plus two dequantized tiles live simultaneously. With
   the tuner currently favouring `BLOCK_SIZE_M=16` and `BLOCK_SIZE_N` up to
   128, budget accordingly.

2. **Scale indexing.** Groups run along K with `group_size=128`; tuned
   `BLOCK_SIZE_K` values observed on the A40 are 32/64/128. All divide or are
   divided by 128, so `scale_idx = (k_block * BLOCK_SIZE_K) // group_size` is
   exact — but nothing enforces that, and a tuner free to pick `BK=96` would
   break it silently. Constrain the search space, or assert it.

3. **`w2` has no epilogue partner.** `SILU_EPILOGUE` applies to GEMM1 only.
   GEMM2 keeps `MUL_ROUTED_WEIGHT=True` and one B tile, so an INT8 path needs
   two dequant variants, not one.

4. **`is_first_gemm` controls A-operand indexing**, not the activation:
   GEMM1 uses `offs_token // top_k` (A is `[M, K]`), GEMM2 uses `offs_token`
   directly (A is `[M*top_k, EI]`). Preserve that distinction.

5. **The vestigial parameters are a trap.** `fused_moe_kernel` declares
   `a_scale_ptr`, `b_scale_ptr`, `use_fp8_w8a8`, `use_int8_w8a16` — all
   inherited from the vLLM code this was derived from, all passed
   `None`/`False`, and **none referenced in the kernel body**. They look like
   working plumbing and are not. Either implement them or delete them; do not
   assume a partial path exists.

6. **The routing/alignment stage is already vectorized and shared.**
   `moe_align_block_size` is bit-identical to its pre-vectorization reference
   (`eval/test_moe_align_block_size.py`) and is called once per `fused_moe`,
   with the resulting `sorted_token_ids` / `expert_ids` reused by both GEMMs.
   A low-bit kernel should reuse it rather than re-deriving.

7. **`topk_ids` for `expert_token_stats`.** Your README asks for real routing
   to replace the balanced-routing assumption. It is available at
   `TritonFusedMoEBlock.forward` immediately after
   `torch.topk(routing_weights, self.cfg.TOPK, dim=-1)` in
   `model_update/model.py`. Worth capturing — this checkpoint's router is
   documented as near-uniform (top-1 weight ~1.7–5%), so imbalance is probably
   mild, but that is an assumption nobody has measured.

---

## 4b. The SwiGLU-epilogue conflict, confirmed against sglang

`FROM sglang mainline` — relevant now that `runtime/kernels/w8a16_gemm.py`
exists and the remaining work is wiring it into the grouped-expert path.

sglang has the identical SiLU-epilogue optimization (`fuse_swiglu_interleaved`,
`srt/layers/moe/moe_runner/triton_utils/fused_moe.py`), and it **refuses to
combine it with quantization**:

```python
assert (
    activation == "silu" and is_gated
    and not (use_fp8_w8a8 or use_int8_w8a8 or use_int8_w8a16 or use_int4_w4a16)
    and hidden_states.dtype == torch.bfloat16
), "fuse_swiglu_interleaved set on an incompatible fused_moe call"
```

A production engine with mature versions of both features treats them as
mutually exclusive. Their own comment explains why: the epilogue applies
`silu(gate) * up` **in-register**, so a standalone activation kernel "would read
them as halves and be silently wrong."

**What this means here.** `test_llada` now ships that epilogue by default
(`LLADA_MOE_FUSED_SILU=1`). A W8A16 grouped-expert kernel has to resolve a
three-way interaction the standalone GEMM never faced:

- the epilogue needs both the gate and up **fp32 accumulators live in-register**
  at the same time — two B tiles per program, doubling register pressure before
  any dequant state is added;
- dequant wants per-group scales applied during the K-loop, which is where the
  accumulators are being built;
- `test_llada`'s epilogue is bit-exact by deliberately rounding accumulators to
  bf16 *first*. A W8A16 path cannot preserve that contract — it is a different
  numeric by construction — so the acceptance test has to move from Tier B
  (bit-exactness) to Tier A (token identity against the dequantize-then-matmul
  reference). See section 5.

**Three options, in increasing order of work:**

1. **Follow sglang** — assert-and-disable. Quantized MoE runs with
   `LLADA_MOE_FUSED_SILU=0`. Costs the epilogue's measured +1–10%, which is
   small against a projected 2–3x from the fused dequant. Lowest risk, and it is
   what a production engine chose.
2. **Fuse both** — two dequantized B tiles plus two accumulators in-register.
   Strictly better if the registers are there; nobody appears to have shipped it.
3. **Quantize GEMM2 only** — GEMM2 has no epilogue partner and carries `w2`,
   a third of the expert bytes. Captures part of the win with none of the
   conflict. A useful staging step.

Option 1 is the one to build first, because it makes the W8A16 win measurable
end-to-end without also debugging a novel register-pressure problem.

## 5. The numerical-contract method

The SiLU epilogue had to be **bit-exact** against the unfused path, and the
method transfers directly to validating a low-bit kernel against your
dequantize-then-matmul reference.

**Reproduce the reference's op order, not the more accurate one.** The
temptation was to compute the activation straight from the fp32 accumulators.
That is *more* precise and therefore wrong for this purpose. The epilogue
instead rounds accumulators to bf16 first — reproducing the intermediate store
the unfused path performed — then evaluates `x/(1+exp(-x))` in fp32 and rounds
back to bf16 before the multiply, matching ATen's bf16 SiLU (`opmath_t=float`)
and bf16 mul elementwise:

```python
gate = accumulator.to(compute_type).to(tl.float32)
up   = accumulator_up.to(compute_type).to(tl.float32)
silu = gate / (1.0 + tl.exp(-gate))
accumulator = silu.to(compute_type).to(tl.float32) * up
```

**Measure what you cannot prove.** One property could not be established by
construction: that Triton's `tl.exp` lowers to the same instruction as ATen's
`expf`. A last-ulp fp32 difference would usually vanish in the round to bf16 —
but this checkpoint's near-uniform router turns bf16-level noise into discrete
top-8 expert flips, so "usually" was not good enough. The test measures it and
reports `n_differing` / `max_rel` rather than asserting and hoping. Result:
bit-exact at every shape M=1…2048.

**Two-tier assertions.** `eval/test_fused_silu_epilogue.py` separates:

- *Tier A, hard assert*: generated token sequences identical to a frozen copy
  of the pre-change path. This is what must hold.
- *Tier B, reported*: elementwise bit-exactness, with a real diagnostic on
  failure rather than an opaque assert.

For a low-bit kernel Tier B will legitimately fail — the whole point is a
different numeric. But Tier A generalises: your reference path defines the
tokens, and a kernel that reproduces the *reference's* tokens is validated
even though it does not reproduce BF16's. That is a much stronger check than
cosine similarity on logits, and it is cheap once the harness exists.

**Gate on the real task.** GSM8K n=50 seed=42 scored 88.0% (44/50) before and
after, unchanged through a full kernel retune. That is the acceptance gate any
accuracy-affecting change should clear. Note the caveat recorded in the
inference README: `_grade_gsm8k` falls back to "last number anywhere in the
response" when the extracted span has no digits, and 3 of those 44 rest on
that fallback — so treat 88.0% as a *comparison* baseline, not an absolute.

---

## 6. Unrelated changes worth knowing

**Narrowed `lm_head`.** Every forward previously projected its entire input
through `lm_head` (2048 → 157,184, ~644 MB of weights) and callers discarded
most of it — the cache-prime and block-finalize passes discarded *all* of it.
`num_logits` narrows it to the consumed rows. Bit-exact. Relevant here because
it removes an **~800 MB peak allocation** at `gen_length=1024`, which is real
headroom on a 24 GB card and shifts the memory arithmetic your capacity story
depends on.

**Tuning is the dominant single win, and it is not checked in.** Running
`tuning_fused_moe_triton.py` on a machine that had never had a config
generated was worth more than either kernel change: the full MoE pipeline went
8.08 → 3.64 ms at M=2048 (**2.2×**) and 4.29 → 2.45 ms at M=1024 (**1.8×**).
`moe_tune_config.json` is hardware-specific and deliberately untracked. Any
low-bit kernel needs its own.

**Retuning is not numerically neutral.** The tuner validates its winner at
`cos_sim > 0.999`, not bit-equality, and `BLOCK_SIZE_K` changes regroup the
fp32 K-loop accumulation. Generation output can legitimately shift after a
retune. (In practice GSM8K did not move at all, but do not assume that.)

**Headline numbers, for calibration.** 6.46× single-request vs the unfused/
uncached baseline; 54× total-pipeline throughput (4.5 → 243.2 tok/s), which
decomposes as 6.46× engine × 8.7× batching. The baseline is single-request by
construction — `src/generate.py` hardcodes the batch dimension to 1.

---

## Reproduction

```bash
# bit-exactness + per-M benchmark of the SiLU epilogue
python -m eval.test_fused_silu_epilogue

# kernel-time ranking from a profiler trace (new in this range)
PROFILE_BATCHES=1 BATCH_MAX_SIZE=16 bash start.sh --backend fast_dense --weight-dir weights
python -m eval.analyze_trace

# throughput vs batch curve
python -m eval.throughput.run_throughput --base-url http://localhost:8000 \
    --concurrency 32 --n-requests 64 --fixed-prompt \
    --max-tokens 128 --steps 128 --block-length 32
```

Known gap: `run_throughput.py` sets no `aiohttp.ClientTimeout`, so it inherits
the 300 s default and cannot benchmark a serialized backend past ~10 requests.
