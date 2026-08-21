# LLaDA_Quant

Quantization stack for **LLaDA-MoE diffusion LLM inference**: packed low-bit
expert weights, a measured memory story, and a trajectory-aware evaluation
layer that explains what quantization does to diffusion generation. Decoding,
KV caching and serving stay in the inference repository; this package is
consumed as a dependency.

**Every claim below is labelled.** `MEASURED` numbers were produced by the
scripts in `benchmarks/` on the machine that ran them. `IMPLEMENTED` means the
code exists and is tested. `FUTURE` means it does not exist yet and nothing
here reports a benefit from it.

---

## Status at a glance

| Capability | State |
|---|---|
| Symmetric groupwise INT8 weight quantization | IMPLEMENTED |
| **Genuinely packed INT4** (two values per byte) | IMPLEMENTED, MEASURED |
| MSE-optimal scale search (INT4 error, zero extra bytes) | IMPLEMENTED, MEASURED |
| Real resident-memory reduction (`PACKED` mode) | IMPLEMENTED, MEASURED |
| Self-contained checkpoints, no redundant BF16 | IMPLEMENTED, MEASURED |
| Structural expert targeting + audit trail | IMPLEMENTED |
| Storage & numerical-error benchmarks | IMPLEMENTED, MEASURED |
| MoE roofline / tokens-per-expert analysis | IMPLEMENTED, MEASURED |
| Trajectory capture, trace, offline replay, noise floor | IMPLEMENTED |
| Router top-k capture from the real fused block | IMPLEMENTED |
| BF16 vs INT4 experiment on the real checkpoint | IMPLEMENTED, **MEASURED** |
| GSM8K accuracy, BF16 vs INT4, same machine | **MEASURED** (n=50, not significant) |
| **Fused INT8/INT4 Triton MoE kernel** | **FUTURE — does not exist** |
| Latency or throughput speedup | **NOT CLAIMED, NOT MEASURED** |

> **There is no speedup here.** Quantized execution is dequantize-then-matmul,
> which is *slower* than BF16. What exists today is a capacity win and a
> correctness/measurement layer. See
> [Should the kernel be built?](#should-the-kernel-be-built) for whether the
> fast path is even worth writing.

---

## Table of contents

- [What actually happens when you quantize](#what-actually-happens-when-you-quantize)
- [Execution modes](#execution-modes)
- [Measured results](#measured-results)
- [Quick start](#quick-start)
- [Targeting is explicit](#targeting-is-explicit)
- [Checkpoint format](#checkpoint-format)
- [Integrating with the inference repository](#integrating-with-the-inference-repository)
- [Should the kernel be built?](#should-the-kernel-be-built)
- [Trajectory validation](#trajectory-validation)
- [Benchmarks](#benchmarks)
- [API reference](#api-reference)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Changes from v0.1](#changes-from-v01)
- [Legal](#legal)

---

## What actually happens when you quantize

```
              QUANTIZATION                          EXECUTION
   BF16 w1/w2 ──► symmetric groupwise ──► packed INT8   ──► dequantize ──► BF16 GEMM
                  (zero-point-free)       packed INT4        per access     (current)
                            │                   │
                            └── scales ─────────┘                    ┌──► INT8 Triton
                                                                     │    fused MoE
                                            future kernel consumes ──┤    (FUTURE)
                                            packed weights directly  └──► INT4 Triton
                                                                          fused MoE
                                                                          (FUTURE)
```

The reference path (dequantize-then-matmul) defines the numerical contract the
future kernel must reproduce. It is a *memory* optimization, not a speed one:
weights are stored small and expanded on use.

---

## Execution modes

Nothing picks between these implicitly, and only one of them saves memory.

### `ExecutionMode.PACKED` (default)

The BF16 `w1`/`w2` Parameters are **deleted**; only packed integers and scales
stay resident. `block.w1` still works — it is served by a property that
dequantizes from the packed buffers — so the model repository needs no
changes and `fused_moe(x, self.w1, self.w2, ...)` keeps running.

- Resident memory really drops. `state_dict()` contains no BF16 copy.
- Every access reconstructs, so it is **slower than BF16**.
- Transient peak during a call is one layer's BF16 weight on top of the
  packed set — resident savings are model-wide, the peak is per layer.
- `block.w1 = ...` raises. A silent write into a temporary is the exact
  failure this mode exists to prevent.

### `ExecutionMode.REFERENCE`

Packed buffers *and* dequantized BF16 Parameters both resident.

- Uses **~1.5x the memory of not quantizing at all**.
- Runs at BF16 speed with BF16-shaped writable Parameters.
- For validation only. `QuantizationResult.summary()` says so out loud, and
  `MemoryComparison.describe()` prints `REGRESSION`.

---

## Measured results

`MEASURED` — `benchmarks/bench_storage.py --num-experts 16 --hidden 1024 --intermediate 512 --layers 2`, BF16 baseline 96.00 MiB:

| bits | mode | packed | resident MiB | vs BF16 | checkpoint MiB | vs BF16 |
|---|---|---|---|---|---|---|
| 8 | packed | no | 49.50 | **0.516x** | 49.50 | 0.516x |
| 8 | reference | no | 145.50 | 1.516x ⚠ larger | 49.50 | 0.516x |
| 4 | packed | **yes** | 25.50 | **0.266x** | 25.50 | 0.266x |
| 4 | reference | yes | 121.50 | 1.266x ⚠ larger | 25.50 | 0.266x |

Resident bytes come from walking the live module tree
(`LLaDA_Quant.memory.resident_memory`), not from the theoretical size of a
packed tensor.

**INT4 really saves storage**: 0.266x vs INT8's 0.516x — the value bytes are
exactly halved, and the residual above 0.25x is the fp32 scales, which are
identical in both.

`MEASURED` — `benchmarks/bench_numerical.py --num-experts 16 --hidden 512 --intermediate 256`,
Gaussian weights / Student-t(3) weights (`--heavy-tailed`, closer to real LLM tails):

| bits | scale search | weight rel. L2 | output cosine | bytes vs BF16 | error gain |
|---|---|---|---|---|---|
| 8 | amax | 0.0065 / 0.0138 | 0.99993 / 0.99971 | 0.516x | — |
| 8 | mse | 0.0065 / 0.0138 | 0.99993 / 0.99971 | 0.516x | 0.0% |
| 4 | amax | 0.1174 / 0.2248 | 0.97950 / 0.92650 | 0.266x | — |
| 4 | **mse** | **0.1011 / 0.1978** | **0.98461 / 0.94128** | 0.266x | **13.9% / 12.0%** |

Error only. Every row executes the same BF16 matmul, so no timing is reported.

**INT4 is the weak link, and scale search narrows it for free.** `s = amax /
Qmax` spends the whole grid accommodating the single largest weight in a
group; with 256 levels that is nearly free, with 16 it is not. Searching a
clipping ratio that minimises per-group squared error cuts INT4 weight error
by **12–14% at zero extra bytes** — the storage columns above are identical,
because only the *value* of the scale changes. Enable it with
`QuantConfig(scale_search="mse")`.

Read the absolute numbers too, not just the gain: even improved, INT4 weight
error is ~15x INT8's, and output cosine drops to 0.94 on heavy-tailed weights.
Scale search makes INT4 meaningfully better; it does not make it obviously
safe. Establishing that needs the trajectory layer against the real
checkpoint, and probably mixed precision.

### On the real checkpoint

`MEASURED` — `bench_bf16_vs_int4.py`, RTX A6000 48 GB, LLaDA-MoE-7B-A1B-Instruct,
INT4 group 128 with MSE scale search, one GSM8K prompt through the chat
template, 128 tokens / 128 steps, greedy (temperature 0), seed 42.

| | BF16 | INT4-MSE |
|---|---|---|
| expert weights | 12288 MiB | **3264 MiB (0.266x)** |
| whole model resident | 14032 MiB | **5008 MiB (0.357x)** |

| quantity | INT4 | BF16-vs-BF16 floor |
|---|---|---|
| Mode A mean top-1 agreement | 0.8861 | 1.0000 |
| Mode A mean tie fraction | 0.9733 | 0.0000 |
| Mode B final token agreement | **0.4766** | 1.0000 |
| Mode B first divergence step | **34** of 128 | never |
| Mode B commit-order agreement | **0.2109** | 1.0000 |
| amplification (Mode B / Mode A) | **4.59x** | — |

**The trajectory diverges hard; the answer does not.** Fewer than half the
committed tokens match, the commit *order* barely agrees at all, and end-to-end
divergence is 4.6x the per-step injected error — errors genuinely compound along
the schedule. Yet both decodes reach `oxed{72}`, correctly, with coherent
reasoning, differing only in how they write the intermediate step
(`rac{48}{2}` versus `rac{1}{2} 	imes 48`).

This is the concrete case for why text equality is the wrong gate. It would
score this run a failure at 47.7% token agreement, while the task is solved
identically. It is equally the case against declaring success from one prompt:
n=1 shows the failure mode is *survivable here*, not that it is safe.

Read Mode A's 0.886 next to its 0.973 tie fraction: per-step error is small and
almost entirely lands on positions where the reference had no real preference.
The damage is not per-step — it is in what those coin-tosses compound into.

**Routing imbalance is not mild.** Measured max/mean load per expert across the
16 layers: **2.50x to 6.48x** (A6000, 179 tokens; the A40-24Q gave 2.65-7.12x at
two shorter lengths). The inference repo's note that a near-uniform router
probably means mild imbalance does not survive measurement. With mean load ~20
rows per expert, the busiest sees ~130 — above the A6000's W4A16 crossover of 50
while the average is well below it, so the critical-path expert may be
compute-bound where the average one is not. That weakens the case for a
weight-only kernel and belongs in the regime analysis.

### GSM8K accuracy

`MEASURED` — RTX A6000, LLaDA-MoE-7B-A1B-Instruct, n=50 seed=42,
`max_tokens=1024 steps=512 block_length=64 confidence_threshold=0.9` (the
inference repo's recommended config), INT4 group 128 with MSE scale search in
REFERENCE mode. Both arms served through the same launcher, differing only in
quantization.

| n | BF16 | INT4-MSE g128 | delta | p (unpaired) |
|---|---|---|---|---|
| 50 | 70.0% (35/50) | 64.0% (32/50) | **-6.0 pt** | 0.523 |
| 200 | 75.5% (151/200) | 69.5% (139/200) | **-6.0 pt** | 0.179 |

The effect is stable -- exactly -6.0 points at both sample sizes -- and still
not statistically established. Detecting a 6-point gap at 80% power needs
roughly **864 questions per arm** (~1.8 h each at 7.3 s/question); the full
1319-item GSM8K test set would settle it.

Paired breakdown at n=50, since the two arms answered the same questions:

| | items |
|---|---|
| both correct | 28 |
| both wrong | 11 |
| **INT4 fixed what BF16 got wrong** | **4** |
| **INT4 broke what BF16 got right** | **7** |

**McNemar exact test on the 11 discordant items: p = 0.549.** The 6-point gap
is not statistically detectable at n=50 — one standard error on a 70% rate here
is +/-6.5 points, wider than the gap itself. INT4 is not *shown* to be worse;
neither is it shown to be safe.

At n=200 the same -6.0 point gap reaches only p = 0.179, by an *unpaired*
two-proportion test. The paired test would be sharper, but the harness's result
JSON stores only the aggregate, not per-item outcomes, so pipe stdout through
``tee`` if you want McNemar at that size.

The number that does mean something is the churn: **22% of items changed
outcome**, in both directions. That is the task-level shadow of what the
trajectory layer measured directly — 47.7% token divergence and 4.59x
amplification. Aggregate accuracy is roughly preserved while individual answers
are not stable. If per-response reproducibility matters for a deployment, that
churn is the finding, not the 6 points.

Note the baseline: 70.0%, not the 88.0% the inference repo recorded on an
A40-24Q at this exact config (and 77.4% on a third machine). An 18-point spread
across three GPUs with identical code and seed means **only a same-machine
BF16-vs-INT4 delta is interpretable**, never a comparison against a number
measured elsewhere.

Resolving a 6-point difference needs roughly n=200; at ~7.4 s/question that is
about 25 minutes per arm.

### Extrapolated to the real model

`MEASURED` from config (`LLaDA_Quant.analysis.LLADA_MOE_7B_A1B`): expert
weights are **6.44 B elements** (16 layers x 64 experts), which is most of the
7B model.

| | expert weights | free on a 24 GB A40-24Q | free on a 48 GB card |
|---|---|---|---|
| BF16 | 12.88 GB | 11.12 GB | 35.12 GB |
| INT8 | 6.44 GB | 17.56 GB | 41.56 GB |
| INT4 | 3.22 GB | 20.78 GB | 44.78 GB |

On the 24 GB card this is the difference between fitting comfortably and not —
batch 48 currently fails outright. See
[Should the kernel be built?](#should-the-kernel-be-built) for why that
capacity does **not** convert into proportional throughput.

---

## Quick start

```python
from LLaDA_Quant import QuantConfig, QuantizationManifest, quantize_and_measure, save_quantized_checkpoint

config = QuantConfig(
    bits=8,                      # or 4 — genuinely packed
    group_size=128,
    targets=("expert",),
    execution_mode="packed",     # the mode that reduces memory
    expect_expert_blocks=16,     # fail loudly if the match count is wrong
)

quantized, result, memory = quantize_and_measure(model, config)
print(result.summary())
print(memory.describe())         # measured resident delta, says REGRESSION if it grew

save_quantized_checkpoint(
    quantized,
    QuantizationManifest(
        source_checkpoint="hf://inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        config=config,
        targets=result.targets,  # the audit trail travels with the artifact
    ),
    "llada-moe-7b-int8-g128",
)
```

---

## Targeting is explicit

A quantizer that converts the wrong module produces a model that looks fine
and is wrong, so nothing is matched by guesswork.

**Experts are matched structurally.** Four shape relations pin the fused
layout; the module's *name* is never consulted:

```
w1.shape[0] == w2.shape[0]        same expert count E
w1.shape[1] == 2 * w2.shape[2]    w1 is Gate+Up stacked over I
w1.shape[2] == w2.shape[1]        both agree on hidden H
both are 3-D
```

A module called `mlp` that is a plain `nn.Linear` is not touched; a correctly
shaped block with an unexpected name is not missed.

**Linears must be named.** `linear_include` has no implicit default — an empty
tuple quantizes nothing, even with `"linear"` in `targets`:

```python
QuantConfig(targets=("linear",), linear_include=("q_proj", "k_proj", "v_proj", "o_proj"))
```

**Exclusions are component globs**, applied per dot-separated path component,
so `gate` does not silently knock out `gate_proj`. Default:
`("router", "gate", "*norm*", "embed_tokens", "lm_head")`.

**Failures are loud.** `TargetingError` is raised when zero modules match
(a silent no-op is indistinguishable from success), or when the count
disagrees with `expect_expert_blocks` / `expect_linears`. Re-quantizing an
already-quantized block raises.

Every converted module is recorded in `QuantizationResult.targets` and written
into the manifest with its shapes, bits, group size, execution mode and
before/after byte counts.

| Component | Default | Rationale |
|---|---|---|
| MoE expert `w1`, `w2` | INT8 or INT4 | ~6.4 B of the ~7 B parameters |
| Attention projections | opt-in via `linear_include` | must be named explicitly |
| Router / gate | BF16 (excluded) | near-uniform scores; tiny changes flip top-8 |
| Norms | BF16 (excluded) | numerically sensitive, negligible memory |
| Embeddings / LM head | BF16 (excluded) | large but quality-sensitive |
| KV cache | BF16 | belongs to the inference repo |

Numerical contract, per group along the last (K) axis:

```
s   = max(|W_group|) / Qmax
W_q = clamp(round(W / s), -Qmax - 1, Qmax)
W  ~= W_q * s          (arithmetic in fp32, then cast)
```

INT4 packs element `2i` into the low nibble and `2i+1` into the high nibble,
two's complement. Group boundaries stay byte-aligned because every legal
group size is even — enforced by `QuantConfig`, which rejects an odd
`group_size` at `bits=4`.

---

## Checkpoint format

```
llada-moe-7b-int8-g128/
├─ model-int8.safetensors      packed ints + scales + untargeted tensors
├─ quantization.json           config, targeting audit, measured totals
└─ source-checkpoint.json      pointer + SHA-256 of the unquantized source
```

The artifact is self-contained and **contains no BF16 copy of a quantized
weight** in either execution mode. Those tensors are re-derived on load, so
storing them made the "quantized" checkpoint 1.52x the size of the
unquantized one. Their absence at load time is expected and never counts as a
missing key; any *other* missing or unexpected key still raises.

Loading into a plain model works: packed buffers are registered on the fly and
expert access is re-installed. The effective group size is recovered from the
scale tensor's shape rather than the config, because `quantize_tensor` falls
back to per-tensor scaling when the group size does not divide K — trusting
the config there would dequantize with the wrong grouping and produce garbage
of exactly the right shape.

---

## Integrating with the inference repository

Verified against `test_llada` by inspection at `fbc1cae` (not yet executed —
that needs a GPU with Triton and the weights).

**What matches.** `TritonFusedMoEBlock` (`model_update/model.py:103`) declares
`w1 [num_local_experts, 2*EI, H]` and `w2 [num_local_experts, H, EI]`, which
satisfies all four structural relations, so detection finds it with no name
matching. Its path is `layers.{i}.mlp`; `mlp` is not excluded, while the
block's own `gate` (the router) is. `forward` passes `w1=self.w1, w2=self.w2`
into `fused_moe`, so the PACKED property is transparent there, and the kernel
wants BF16, which is what `compute_dtype` produces.

**Order matters.** `load_state_dict_from_unfused` does `self.w1[i].copy_(...)`
— an in-place write that PACKED mode would send into a dequantized temporary.
Quantize *after* the fused block is populated:

```python
MODEL = LLaDAMoEKV(use_fused_moe=False).to(torch.bfloat16).eval()
# ... load HF weights, build fused blocks, call load_state_dict_from_unfused ...
quantize_model(MODEL, QuantConfig(bits=8, group_size=128, targets=("expert",),
                                  execution_mode="packed", expect_expert_blocks=16))
```

Getting this backwards raises rather than silently doing nothing: the packed
subclass shadows every method in `WEIGHT_MUTATING_METHODS` with an error that
names the fix.

**Cost on the real model.** Each layer holds 402.7 M expert elements. A BF16
step reads 805 MB of expert weights per layer; PACKED INT8 reads 403 MB of
int8, writes 805 MB of BF16, then the GEMM reads it back — about **2.5x the
memory traffic**. That is the price of the capacity win until a kernel consumes
packed weights directly, and it is why PACKED is a *fit-the-model* tool today,
not a speed one.

**The kernel a future low-bit path would extend has changed.** As of
`b4872e9..fbc1cae` the inference repo folds SiLU into GEMM1's epilogue. This
does **not** affect anything above — `w1 [E, 2I, H]`, `w2 [E, H, I]`, the
in-place loader and `w1=self.w1` in `forward` are unchanged, so detection, both
residency modes and the checkpoint format are unaffected. It does change what a
dequantizing kernel must do:

| | Before | After (`SILU_EPILOGUE=True`) |
|---|---|---|
| `N` kernel arg | `2*EI` | `EI` (output width) |
| B tiles in flight per K step | 1 | **2** (gate at `offs_bn`, up at `offs_bn + N`) |
| GEMM1 output | `[M, top_k, 2*EI]` | `[M*top_k, EI]` |
| Shared memory | `(BM·BK + BK·BN)·stages·2` | `(BM·BK + 2·BK·BN)·stages·2` |

Groups run along K, orthogonal to the gate/up split along N, so the scale
layout stays compatible — but a dequantizing kernel fetches scales for *two*
N-tiles per K step and register pressure roughly doubles. `SILU_EPILOGUE`
applies to GEMM1 only, so an INT8 path needs two dequant variants, not one.

Two helpers exist for this, because both are silent-failure risks:

```python
from LLaDA_Quant.algorithms.symmetric import validate_block_k_alignment, aligned_block_k_values
from LLaDA_Quant.analysis import kernel_shared_memory_bytes

validate_block_k_alignment(block_k=96, group_size=128)   # raises: straddles a group
aligned_block_k_values(128, [32, 64, 96, 128, 256])      # -> [32, 64, 128, 256]
kernel_shared_memory_bytes(16, 128, 64, 2, weight_bytes=0.5, b_tiles=2)  # packed INT4
```

`validate_block_k_alignment` exists because nothing in a Triton kernel enforces
that `BLOCK_SIZE_K` and `group_size` divide each other; an autotuner free to
pick `BK=96` against `group_size=128` produces a kernel that dequantizes part
of every tile with the wrong scale and returns plausible, wrong numbers.
`kernel_shared_memory_bytes` generalises the inference repo's `_shmem_bytes`,
which hardcodes 2 bytes per element — using that for INT8/INT4 over-estimates
the budget and silently rejects configs that would have fit.

Also: the kernel declares `a_scale_ptr`, `b_scale_ptr`, `use_fp8_w8a8` and
`use_int8_w8a16`, all inherited from vLLM, all passed `None`/`False`, and none
referenced in the body. They look like working plumbing and are not.

**Not covered.** `use_fused_moe=False` (the unfused `MoEBlock`/`ExpertMLP`
path) exposes `gate_proj`/`up_proj`/`down_proj` Linears, not fused tensors —
the expert adapter finds nothing there and raises `TargetingError`; quantize it
with `linear_include` instead. Under tensor parallelism each rank holds
`num_local_experts`, so detection works per rank but checkpoints become
rank-specific. `TritonFusedMoEBlock.forward` raises on CPU, so Mode A/B against
the real model needs a GPU. A low-bit kernel needs its own
`moe_tune_config.json`: it is hardware- and variant-specific, deliberately
untracked, and on a machine that had never been tuned it was worth more (2.2x
at M=2048) than either kernel change.

---

## Should the kernel be built?

`MEASURED` — analytic side from `benchmarks/bench_moe_regime.py`; the measured
side from the inference repository, recorded in
[INFERENCE_REPO_CHANGES.md](INFERENCE_REPO_CHANGES.md) on a single **NVIDIA
A40-24Q** (sm_86, 24 GB, ~696 GB/s) running the real 7B checkpoint.

Weight-only quantization buys latency only where the GEMM is bandwidth-bound.
For a top-k MoE the deciding quantity is **tokens per expert per step** =
`M * top_k / E`, which for LLaDA-MoE is `M / 8`. LLaDA's cached decoder
forwards `x[:, block_start:]`, so `M = batch x suffix_length`.

A40-24Q balance: 215 flops/byte BF16, 430 INT8.

| workload | tokens/step | M/expert | BF16 | W8A16 | W4A16 |
|---|---|---|---|---|---|
| batch=1, first block (L=128) | 128 | 16 | memory | memory | memory |
| batch=1, last block (L=32) | 32 | 4 | memory | memory | memory |
| batch=4, first block | 512 | 64 | memory | memory | **compute** |
| batch=16, first block | 2048 | 256 | **compute** | **compute** | **compute** |
| batch=32, last block | 1024 | 128 | memory | **compute** | **compute** |
| batch=57, first block | 7296 | 912 | **compute** | **compute** | **compute** |

Crossover M/expert (bandwidth-bound below): **BF16 215, W8A16 108, W4A16 54,
W8A8 215.**

### The bandwidth-bound premise is confirmed `MEASURED`

Nsight Compute on `fused_moe_kernel` at batch 32: L2 throughput 96.35%, DRAM
66.40%, SM 41.25%. End to end from a profiler trace (batch 11, 32 tokens, 32
steps): 438 GB of expert weights streamed, a 626 ms floor at ~696 GB/s against
**770 ms measured** — **81% of theoretical weight-streaming peak**, and 58.75%
of all GPU time. Every expert is touched every forward (352 tokens x top-8 =
2,816 assignments over 64 experts).

**Halving weight bytes should translate close to directly into time in this
regime.** That is the strongest argument for the kernel, and it is measured
rather than modelled.

The model called it correctly, by two independent routes: at that traced
workload M/expert = 352 x 8 / 64 = **44**, well below the BF16 crossover of
215 → predicted memory-bound, measured at 81% of peak. A direct
arithmetic-intensity check gives 35.4 GFLOP / 805 MB = 44 FLOP/byte against a
215 FLOP/byte balance — same verdict.

> One figure that circulated as "~96% of peak memory bandwidth" is **L2, not
> DRAM**. Nsight's Speed-of-Light row reports the most-utilised memory
> subsystem. Read as a DRAM ceiling it says "nothing left to gain"; read as an
> L2 ceiling it says "remove intermediate traffic" — which is what produced
> the inference repo's SiLU epilogue.

### The capacity → throughput argument does *not* hold `MEASURED`

A previous version of this section claimed that in the compute-bound regime
the win is capacity: free memory → larger batch → more throughput. **That has
now been measured and it saturates early.** Throughput vs `BATCH_MAX_SIZE`,
A40-24Q, 128 tokens / steps=128 / block=32:

| `BATCH_MAX_SIZE` | Tok/s | Δ throughput | Δ batch | p50 latency |
|---:|---:|---:|---:|---:|
| 8 | 150.3 | — | — | 6.77 s |
| 16 | 204.8 | +36.3% | +100% | 9.93 s |
| 24 | 224.8 | +9.8% | +50% | 13.56 s |
| 32 | 243.2 | +8.2% | +33% | 16.71 s |
| 48 | 0/96 requests succeeded | — | — | — |

8 → 32 is **4x the batch for 1.62x the throughput**, and the last step bought
8.2%. The curve is past its knee by batch 32. So freeing memory to roughly
double the batch is worth **single-digit percent** throughput, not a multiple.

Capacity is still a real win for *fitting* a model or a longer context — on a
24 GB card BF16 experts alone are 12.88 GB, and batch 48 fails outright — but
it is weak as a *throughput* mechanism. The knee moves on a 48 GB card; the
shape of the curve should not, because it is the same mechanism the `M/expert`
analysis describes: past the crossover, extra tokens stop riding along free.

### Verdict

At the inference repo's actual operating point (batch 32, block 32 → M = 1024,
M/expert = 128):

- **BF16**: 128 < 215 → memory-bound. Confirmed at 81% of peak.
- **W8A16**: 128 > 108 → compute-bound *once quantized*. The weight-only
  kernel would deliver well under its 2x ceiling here.
- **Batch 1–16** (M/expert 4–64): memory-bound before *and* after quantizing.
  This is where a W8A16/W4A16 fused kernel gets close to its full ratio.

The trap this section already flagged — quantizing halves the crossover, so a
bandwidth-bound workload can become compute-bound once quantized — is exactly
what happens at the throughput operating point.

**So: build the kernel for latency-oriented, small-batch serving. Its case is
weakest in the high-batch throughput regime**, which inverts the common
intuition that quantization is a throughput play. If the target moves to H100,
re-run all of this: ~3.35 TB/s against much higher INT8 throughput shifts every
crossover, and native FP8 changes the arithmetic entirely.

### Routing balance is still unmeasured

Rows assume perfectly balanced routing, which is optimistic — the slowest
expert bounds the step. The checkpoint's router is documented as near-uniform
(top-1 weight ~1.7–5%), so imbalance is probably mild, but nobody has measured
it. `topk_ids` is available in `TritonFusedMoEBlock.forward` immediately after
`torch.topk(routing_weights, self.cfg.TOPK, dim=-1)`:

```bash
python benchmarks/bench_moe_regime.py --routing-file topk_ids.pt
```

or call `LLaDA_Quant.analysis.expert_token_stats(topk_ids, num_experts)`
in-process for min/p50/mean/p90/p99/max and the imbalance ratio.

---

## Trajectory validation

```
MODEL ──► CAPTURE ──► TRACE ──► METRICS ──► REPORT
          (GPU)       (JSON)    (offline)   (offline)
```

Only `trajectory.capture` touches a model. Everything downstream runs from a
JSON trace, so metrics can be recomputed and unit-tested without rerunning
anything.

### Two modes, never conflated

| Mode | What it measures | What it cannot show |
|---|---|---|
| **A — shared state** (`capture_shared`) | Error *injected per step*. Both models see byte-identical states, so every difference is quantization. | Amplification, by construction. |
| **B — free running** (`capture_free_running`) | *Amplified* end-to-end divergence. Each model advances itself through your commit rule. | Per-step error — once inputs drift, a logit distance conflates two causes. |

Mode B stores no pairwise logit scalars for exactly that reason.

### The noise floor is mandatory

Before comparing BF16 against INT8, compare **BF16 against BF16**. Non-
determinism, batch composition and kernel selection move that floor off zero.
`TrajectoryReport.to_table()` renders a `BF16 floor` column next to every
result, and derives `per_step_signal` (Mode A above floor) and
`amplification` (Mode B ÷ Mode A).

For stochastic decoding use `temperature=0.0`, or share RNG draws. Two
independent gumbel draws produce divergence that has nothing to do with
quantization.

### Traces stay compact and honestly labelled

Full logits for LLaDA are `[B, L, 157184]` — ~40 MB per step per model. So
anything needing the full vocabulary is reduced **on device during capture**
and stored as a scalar; everything else is stored top-k truncated. Every
scalar carries a `MetricPrecision`:

- `EXACT` — reduced over the full tensor (cosine, KL, top-1 agreement, tie
  fraction, router overlap)
- `TOPK` — computed from the stored top-k slice (`topk_set_overlap`,
  `topk_kl_lower_bound`, which is a *lower bound*, not KL)
- `SAMPLED` — estimated from a subset

`verify_replay()` cross-checks replayed numbers against the exact ones
captured on device, so the offline path and the capture code cannot drift
apart unnoticed.

### Metrics

Beyond error and cosine: `top1_agreement`, `kl_divergence`,
`predictive_entropy`, `router_overlap`, **`router_margin`** (gap between the
k-th and (k+1)-th gate — whether quantization noise can flip expert
selection), **`unmask_selection_agreement`** (do both models unmask the *same
position* next — invisible to top-1 agreement, but it changes the context
every later step conditions on), `commit_order_agreement`, and
**`tie_fraction`**.

> **Never quote an agreement number without its tie fraction.** `tie_fraction`
> is the share of positions where the reference's own top-2 margin is smaller
> than the shift quantization introduced — the argmax may flip, but there was
> no preference to destroy. On the toy model in the test suite INT8 scores
> `top1_agreement = 0.0` at the fully masked state with `tie_fraction = 1.0`,
> because the reference's margin is ~1e-5 against ~5e-3 of INT8 noise. Alone,
> the first number looks catastrophic and means nothing.

### Two-tier acceptance, and what the reference is

Method borrowed from the inference repo's SiLU-epilogue validation
([INFERENCE_REPO_CHANGES.md](INFERENCE_REPO_CHANGES.md) §5), which transfers
directly to validating a future low-bit kernel against this package's
dequantize-then-matmul path:

- **Tier A, hard assert** — generated token sequences identical to a frozen
  reference run. This is what must hold.
- **Tier B, reported** — elementwise bit-exactness, printed with a real
  diagnostic (`n_differing`, `max_rel`) on failure rather than an opaque
  assert.

The subtlety worth stealing: for a low-bit kernel **Tier B legitimately
fails** — a different numeric is the whole point — but Tier A generalises once
you pick the right reference. A packed kernel should be gated on reproducing
the *reference path's* tokens, not BF16's. That is a far stronger check than
cosine similarity on logits, and Mode B already produces exactly the artifact
it needs: run `capture_free_running(reference_path_model, kernel_model, ...)`
and require `final_token_agreement == 1.0`.

Gate on the real task as well. GSM8K n=50 seed=42 scored 88.0% before and
after that change; that is the acceptance bar an accuracy-affecting change
should clear. Treat it as a *comparison* baseline rather than an absolute —
3 of the 44 correct answers rest on a last-number-in-the-response grading
fallback.

### Getting routing out of the real block

`TritonFusedMoEBlock` computes `topk_ids` inside `forward` and never returns
it, so router overlap needs `RouterCapture`: a forward **pre**-hook stashes
each block's input and recomputes the routing with the block's own operations —

```python
x_flat   = x.reshape(-1, H)
routing  = softmax(block.gate(x_flat), dim=-1, dtype=float32)
topk_ids = routing.topk(TOPK).indices
```

— which is bit-identical because it is the same ops on the same tensors. The
router is an excluded BF16 `nn.Linear` running *before* the experts, so its
weights are identical in both models and **every overlap below 1.0 is
attributable to the hidden state drifting upstream**, never to the router
itself being damaged. A test pins that.

Attach one capture per model and dispatch by identity — `capture_shared` runs
both models, so a single shared registry would let the second overwrite the
first:

```python
ref_cap, qnt_cap = attach_router_capture(bf16), attach_router_capture(int4)
capture_shared(bf16, int4, states, logits_fn,
               router_fn_for(ref_cap, qnt_cap), gates_fn_for(ref_cap, qnt_cap))
```

### The decoder is not reimplemented

`trajectory.llada` imports `add_gumbel_noise`, `get_num_transfer_tokens` and
`select_transfer_indices` from the inference repo's `model_update/generate.py`
and assembles `advance_fn` from those exact functions. No LLaDA decoding
semantics are restated. `assert_matches_production_decoder()` proves one step
of the adapter equals the production primitives for a deterministic case — a
measurement built on a drifted reimplementation is worse than no measurement.

```python
from LLaDA_Quant.trajectory import load_llada_decoder, make_llada_advance_fn, capture_free_running

decoder = load_llada_decoder("/path/to/test_llada")     # imports, never modifies
advance = make_llada_advance_fn(decoder, steps=32, temperature=0.0)
capture = capture_free_running(reference, quantized, start, logits_fn, advance)
capture.save("traces/")
```

---

## Benchmarks

Three categories, each stating what it measures and what it does not.

| Script | Category | Measures | Explicitly does **not** measure |
|---|---|---|---|
| `bench_storage.py` | A — storage | resident tensor bytes, checkpoint bytes | any latency |
| `bench_numerical.py` | B — numerical | weight and output quantization error | latency — both paths run the same BF16 matmul |
| `bench_moe_regime.py` | decision | tokens/expert, GEMM shapes, roofline side | wall-clock anything |
| `bench_bf16_vs_int4.py` | validation | BF16 vs INT4-MSE on the **real** checkpoint: router overlap, token commits, final output, against a BF16-vs-BF16 floor | latency; needs a GPU, never yet run |
| `bench_generation_latency.py` | C' — cost | wall clock of dequantize-then-matmul INT4 vs BF16, forward and full generation | a fused kernel; INT4 is expected to be **slower** |
| *(none yet)* | C — kernel | BF16 vs INT8 vs INT4 fused MoE | **the kernel does not exist** |
| *(none yet)* | D — end to end | tokens/s, latency, memory, batch, steps | needs the inference repo and a GPU |

`v0.1`'s `bench_experts.py` was **deleted**: it dequantized to BF16 and then
timed the same BF16 computation twice, reporting the difference as an INT8
result, and computed "47% memory saving" from packed tensors while the BF16
copies were still resident.

---

## API reference

### Configuration

```python
QuantConfig(
    bits=8,                      # 8 or 4 (4 is genuinely packed)
    group_size=128,              # -1 = per-tensor; must be even at bits=4
    targets=("expert",),         # "expert", "linear"
    execution_mode="packed",     # "packed" (saves memory) or "reference" (validation)
    linear_include=(),           # explicit globs; empty matches nothing
    exclude=("router", "gate", "*norm*", "embed_tokens", "lm_head"),
    compute_dtype="bfloat16",
    scale_dtype="float32",
    scale_search="amax",       # or "mse": lower INT4 error, same bytes
    search_grid=24,            # candidate clipping ratios; 8 captures most of it
    source_checkpoint=None,
    expect_expert_blocks=None,   # assert the match count
    expect_linears=None,
    allow_no_matches=False,      # tests only
)
```
`.mode`, `.reduces_memory`, `.to_dict()`, `.from_dict()`, `.to_json()`.

### Quantization

| Function | Description |
|---|---|
| `quantize_model(model, config) -> QuantizationResult` | In place. Raises `TargetingError` on an unexpected match set. |
| `quantized_model(model, config) -> nn.Module` | Deep-copies first. |
| `quantize_and_measure(model, config) -> (model, result, MemoryComparison)` | Quantize a clone and measure the resident delta against the original. |

`QuantizationResult`: `targets`, `names`, `expert_blocks`, `linears`,
`source_bytes`, `quantized_bytes`, `weight_ratio`, `summary()`.

### Memory

| Function | Description |
|---|---|
| `resident_memory(module) -> MemoryReport` | Live tensors, shared storages counted once. |
| `compare_resident_memory(baseline, quantized) -> MemoryComparison` | `ratio`, `saved_bytes`, `is_saving`, `describe()`. |

### Checkpoints

`save_quantized_checkpoint`, `load_quantized_weights`,
`load_quantized_checkpoint`, `checkpoint_size_bytes`, `find_weights_file`,
`derivable_tensor_names`.

### Algorithms

`quantize_tensor(w, bits, group_size, scale_dtype, pack=True)`,
`dequantize_tensor(q, scale, bits, group_size, dtype, packed)`,
`search_group_scale` (MSE-optimal clipping),
`pack_int4`, `unpack_int4`, `validate_int4_layout`, `storage_bytes`,
`qmax_for_bits`, `qmin_for_bits`, `block_k_is_scale_aligned`,
`validate_block_k_alignment`, `aligned_block_k_values`. `QuantResult` carries `q`, `scale`, `bits`,
`group_size`, `packed`, `logical_shape`, `dequantize()`, `storage_bytes()`.

### Runtime

`QuantLinear` (drop-in for `nn.Linear`, INT8 and packed INT4),
`QuantExpertWeights`, `attach_packed_buffers`,
`install_packed_expert_access`, `is_packed_expert_block`,
`quant_result_from_buffers`, `materialize_expert_params`.

### Analysis

`MoEShape`, `LLADA_MOE_7B_A1B`, `Machine`, `A40_24Q` (default), `A40`,
`RTX_A6000`, `A100_80GB`, `H100_SXM`, `SCHEMES`, `Workload`,
`expert_token_stats`, `ideal_tokens_per_expert`, `gemm_regime`,
`crossover_m`, `regime_sweep`, `suffix_lengths_for_schedule`,
`kernel_shared_memory_bytes`, `shared_memory_headroom`.

### Trajectory

`DiffusionState`, `make_masked_states`, `fully_masked_state`;
`RouterCapture`, `attach_router_capture`, `router_fn_for`, `gates_fn_for`;
`capture_shared`, `capture_free_running`; `Trace`, `TraceStep`,
`ScalarMetric`, `MetricPrecision`; `replay_shared`, `replay_free_running`,
`verify_replay`; `TrajectoryReport`; `load_llada_decoder`,
`make_llada_advance_fn`, `assert_matches_production_decoder`.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests
```

`MEASURED`: **198 passed, 1 skipped** (the skip binds to the real inference
repo, which needs Triton and CUDA).

| File | Covers |
|---|---|
| `test_symmetric.py` | quantization math, error budget, per-tensor fallback |
| `test_int4.py` | packing roundtrip, sign extension, nibble order, group and expert alignment, half-of-INT8 storage, adapter integration |
| `test_scale_search.py` | MSE search beats amax on its own objective, INT4 gains and INT8 does not, storage layout and dequantize formula unchanged, config wiring and checkpoint roundtrip |
| `test_memory.py` | resident-memory regression guards for the v0.1 bug |
| `test_targeting.py` | structural detection, component globs, loud failures, audit trail |
| `test_checkpoint_format.py` | no BF16 duplicates, bit-exact roundtrip, group-size recovery, manifest |
| `test_quantlinear.py` | `QuantLinear` vs `nn.Linear` |
| `test_llada_moe_adapter.py` | both modes, per-access dequantization, restore |
| `test_api.py` | result surface, modes, non-destructive variants |
| `test_validation.py` | tensor metrics, component comparison |
| `test_trajectory.py` | states, masked-token and routing metrics, Mode A/B capture |
| `test_trace_replay.py` | trace IO, compactness, offline replay, precision labels, noise floor |
| `test_moe_regime.py` | roofline math, crossovers, routing statistics |
| `test_llada_binding.py` | delegation to the production decoder, drift detection |
| `test_router_capture.py` | router recomputation is bit-identical to the block's own, per-model isolation, hook lifecycle, top-k resolution |

---

## Roadmap

1. **v0.2 (current)** — packed INT8/INT4, honest memory modes, self-contained
   checkpoints, safe targeting, split benchmarks, trajectory trace/replay,
   MoE regime analysis.
2. **v0.3** — *(trajectory, noise floor and routing statistics on the real
   checkpoint are now measured; see [On the real checkpoint](#on-the-real-checkpoint))*.
   What remains is the verdict that matters: GSM8K n=50 seed=42 against the
   inference repo's 88.0% baseline. One prompt reaching the right answer is not
   an accuracy result. This is also what decides whether INT4 is usable at all: scale
   search narrows the gap, sensitivity-driven mixed precision (INT4 where a
   layer tolerates it, INT8 where it does not) is the likely next step, and a
   data-aware method (GPTQ/AWQ) is the fallback if neither suffices.
3. **v0.4** — *conditional on v0.3 evidence*: a fused Triton MoE kernel
   consuming packed weights. Build the W8A16/W4A16 path only if the target is
   small-batch latency; the large-batch regime needs a capacity-to-throughput
   measurement instead.
4. **v1.0** — reproducible end-to-end evaluation, frozen checkpoint format.

---

## Changes from v0.1

Four defects were found by measurement and fixed structurally.

| Was | Now |
|---|---|
| Expert quantization kept BF16 `w1`/`w2` resident beside the packed buffers: **1.52x memory**, zero speedup, plus error | Two explicit modes; `PACKED` deletes the Parameters and measures **0.516x** (INT8) / **0.266x** (INT4) |
| `bits=4` called `quantize_tensor` but never `pack_int4` — identical byte count to INT8 | INT4 is packed two per byte end to end, with alignment validation and tests |
| Checkpoints stored BF16 *and* packed copies, though load recomputed the BF16 | Re-derivable tensors are dropped at save; absence is expected at load |
| Targeting was `"expert" in name or "mlp" in name` plus "3-D w1/w2 with an even second dim" | Structural four-relation match, explicit `linear_include`, component globs, `TargetingError` |
| `bench_experts.py` timed two identical BF16 matmuls and called the difference an INT8 result | Deleted; replaced by labelled storage / numerical / decision benchmarks |

Breaking API changes: `quantize_model` returns `QuantizationResult` instead of
`list[str]`; `config.matches()` is replaced by `matches_linear()` and
structural expert detection; `validation.diffusion` / `validation.trajectory`
moved to the `trajectory` package; `WEIGHTS_FILENAME` became
`weights_filename(bits)`.

---

## Legal

This repository contains no code from the LLaDA inference engine's `dInfer`
directory. `trajectory/llada.py` imports the inference repo's decoder at
runtime when you point it there; it neither vendors nor modifies it. Verify
the source model's license before publishing derived quantized weights.
