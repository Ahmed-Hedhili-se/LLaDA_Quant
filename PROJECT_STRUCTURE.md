# LLaDA_Quant — architecture and file-by-file reference

Companion to [README.md](README.md), which holds the public API and the
measured numbers. This document explains **how the project is organised, what
each part is responsible for, and what every individual file does**.

- [1. Orientation](#1-orientation)
- [2. The parts](#2-the-parts)
- [3. Part A — Entry layer](#3-part-a--entry-layer)
- [4. Part B — Numerical core](#4-part-b--numerical-core)
- [5. Part C — Execution layer](#5-part-c--execution-layer)
- [6. Part D — Model integration](#6-part-d--model-integration)
- [7. Part E — Persistence](#7-part-e--persistence)
- [8. Part F — Measurement](#8-part-f--measurement)
- [9. Part G — Trajectory validation](#9-part-g--trajectory-validation)
- [10. Part H — Benchmarks](#10-part-h--benchmarks)
- [11. Part I — Tests](#11-part-i--tests)
- [12. End-to-end walkthroughs](#12-end-to-end-walkthroughs)
- [13. Where to make common changes](#13-where-to-make-common-changes)
- [14. Invariants to respect](#14-invariants-to-respect)

---

## 1. Orientation

The project does two separable jobs. Keeping them separate is the main
structural decision in the codebase.

```
JOB 1 — QUANTIZE                          JOB 2 — VALIDATE
make the weights small, correctly         explain what that did to generation

BF16 w1/w2                                LLaDA execution (GPU)
    │                                          │
 algorithms/  symmetric INT8 / packed INT4     │  trajectory/capture.py
    │         + per-group scales               ↓
 runtime/     residency + reference exec   trace (JSON, compact)
    │                                          ↓
 adapters/    find the right modules       trajectory/metrics + replay
    │                                          ↓
 formats/     write a self-contained ckpt  trajectory/report.py
    │                                          (Mode A | Mode B | BF16 floor)
    ↓
 future fused Triton kernel  (does not exist)
```

Decoding, KV caching, serving and evaluation execution stay in the inference
repository (`test_llada`). This package is consumed as a dependency and never
modifies it — `trajectory/llada.py` *imports* the real decoder at runtime, and
that is the only coupling.

**Dependency direction.** Nothing lower imports anything higher:

```
config ──► algorithms ──► runtime ──► adapters ──► api
                │                        │          │
                └──► formats ◄───────────┘          │
                                                    │
memory, validation, analysis  (leaf, no core deps) ─┘
trajectory  (depends only on validation.metrics)
```

---

## 2. The parts

| Part | Directory | Responsibility | Depends on |
|---|---|---|---|
| **A. Entry layer** | `__init__.py`, `api.py`, `config.py` | Public surface, run policy, targeting rules, audit trail | everything |
| **B. Numerical core** | `algorithms/` | The quantization contract and the storage format | nothing |
| **C. Execution layer** | `runtime/` | How quantized weights are held in memory and used | algorithms |
| **D. Model integration** | `adapters/` | Finding the right modules and converting them | runtime, config, formats |
| **E. Persistence** | `formats/` | Self-contained checkpoints and the provenance record | config |
| **F. Measurement** | `memory.py`, `validation/`, `analysis/` | Every number the project reports | torch only |
| **G. Trajectory** | `trajectory/` | What quantization does to diffusion generation | validation.metrics |
| **H. Benchmarks** | `benchmarks/` | Runnable, category-labelled measurement scripts | A, F |
| **I. Tests** | `tests/unit/` | 244 tests, several of them regression guards for real past bugs | all |
| **J. Tools** | `tools/` | Non-measurement executables: offline quantization, verification, eval drivers | A, E |

---

## 3. Part A — Entry layer

**Role.** Everything a caller touches, plus the two policies that make a run
trustworthy: *which* modules get converted, and *whether* the result actually
saves memory.

### `src/LLaDA_Quant/config.py` (143 lines)

The policy file. Two ideas live here and both exist because of specific past
failures.

| Symbol | Role |
|---|---|
| `ExecutionMode` | `PACKED` (BF16 params deleted, memory really drops) vs `REFERENCE` (BF16 kept beside packed, ~1.5× memory, validation only). An enum rather than a bool so its docstring can carry the warning that a bool cannot. |
| `QuantConfig` | Frozen dataclass holding the entire run: bits, group size, targets, execution mode, `linear_include`, excludes, dtypes, provenance, and the `expect_*` assertions. |
| `is_excluded()` | `fnmatch` glob applied **per dot-separated path component**. |
| `matches_linear()` | Explicit include list, no implicit fallback. |
| `DEFAULT_EXCLUDES` | `("router", "gate", "*norm*", "embed_tokens", "lm_head")` |
| `FORMAT_VERSION = 2` | Bumped from 1 when the config gained execution modes. |

**Contracts enforced in `__post_init__`:** bits ∈ {8, 4}; `group_size > 0` or
`-1`; **`bits=4` requires an even `group_size`** (odd groups would straddle a
packed byte); targets must be known; execution mode must parse.

**Why component-wise globs.** The old code did raw substring matching, so
excluding `gate` also silently killed `gate_proj`. Matching each component
separately means `gate` hits `layers.0.mlp.gate` and leaves `gate_proj` alone,
while `*norm*` still catches `input_layernorm`.

**Why `linear_include` has no default.** An empty tuple quantizes nothing even
with `"linear"` in `targets`. There is deliberately no "everything not an
expert" rule — that rule is how a quantizer converts the LM head by accident.

`.mode` and `.reduces_memory` are the properties callers should branch on;
`reduces_memory` is `True` only for `PACKED`.

### `src/LLaDA_Quant/api.py` (177 lines)

The three entry points plus the guard that makes a run fail loudly.

| Symbol | Role |
|---|---|
| `quantize_model(model, config)` | In place. Returns `QuantizationResult`. |
| `quantized_model(model, config)` | Deep-copies first, leaves the original untouched. |
| `quantize_and_measure(model, config)` | Quantizes a clone and returns `(model, result, MemoryComparison)` — the only call that cannot be wrong about which tensors are alive, because both models exist at once and both are measured. |
| `QuantizationResult` | The audit trail: `targets`, `names`, `expert_blocks`, `linears`, `source_bytes`, `quantized_bytes`, `weight_ratio`, `summary()`. |
| `TargetingError` | Raised when the match set is not what was asked for. |
| `_validate_targeting()` | Zero matches → error (a silent no-op is indistinguishable from success). Count ≠ `expect_expert_blocks` / `expect_linears` → error. |

`summary()` deliberately prints a different final line per mode — `PACKED`
says resident memory drops and latency rises; `REFERENCE` says the model is
**larger than unquantized** and is validation only. `weight_ratio` is
documented as a *representation* ratio, not a resident-memory claim, and
points at `memory.compare_resident_memory` for that question.

### `src/LLaDA_Quant/__init__.py` (57 lines)

Re-exports the public surface. Its docstring is a scope statement: what is
implemented, what is measured, and the explicit sentence that **no kernel
consumes packed weights yet, so nothing here reports a speedup**.

---

## 4. Part B — Numerical core

**Role.** Define the arithmetic and the byte layout that the reference path
implements today and a future Triton kernel must reproduce exactly. Depends on
nothing but torch, and is testable with no model in sight.

### `src/LLaDA_Quant/algorithms/symmetric.py` (352 lines)

The contract, per group along the **last (K) axis**:

```
s   = max(|W_group|) / Qmax
W_q = clamp(round(W / s), -Qmax - 1, Qmax)
W  ~= W_q * s            (arithmetic in fp32, then cast to compute dtype)
```

Because groups run along the last axis, an expert-stacked `[E, N, K]` tensor is
quantized per expert and per group with **no special casing of the expert
dimension** — that is why the same function serves `w1`, `w2` and `nn.Linear`.

| Symbol | Role |
|---|---|
| `QuantResult` | `q`, `scale`, `bits`, `group_size`, **`packed`**, **`logical_shape`**. `dequantize()`, `storage_bytes()`. The last two fields make a stored tensor self-describing. |
| `quantize_tensor(w, bits, group_size, scale_dtype, pack=True)` | The forward direction. Packs at `bits=4` by default. |
| `dequantize_tensor(q, scale, bits, group_size, dtype, packed)` | The inverse, with the same reshape logic. |
| `pack_int4` / `unpack_int4` | Two values per byte: even index → low nibble, odd → high nibble, two's complement. |
| `validate_int4_layout(shape, group_size)` | Even K, even group size. |
| `storage_bytes(numel, bits, group_size, scale_bytes)` | The formula tests cross-check against measurement. |
| `search_group_scale(w_g, bits, grid, max_shrink)` | MSE-optimal per-group scale: searches clipping ratios instead of taking `amax`. |
| `qmax_for_bits` / `qmin_for_bits` | `2^(b-1)-1` and `-2^(b-1)`. |

**The packing invariant.** 2 divides every legal group size, so a group never
straddles a byte and a kernel can address a group without splitting one. This
is enforced twice — in `QuantConfig.__post_init__` and in
`validate_int4_layout`.

**The per-tensor fallback trap.** When `group_size` does not divide K,
`quantize_tensor` silently falls back to per-tensor scaling and records
`group_size = -1` in the result. Any code reconstructing from stored tensors
must therefore read the *scales*, not the config — see
`runtime.moe.quant_result_from_buffers`.

**`pack_int4` validates its input range** ([-8, 7]) rather than truncating,
because a silently wrapped nibble produces plausible garbage.

**Scale choice is separable from the storage contract.** ``scale_search="mse"``
changes only which ``s`` is stored: the dequantize formula, the packed layout
and the byte count are byte-for-byte identical to ``"amax"``. That is why it
needs no cooperation from the checkpoint format or a future kernel, and why
the tests assert equal `storage_bytes()` between the two. It is specifically an
INT4 tool — 256 levels absorb an outlier, 16 do not, so INT8 gains ~0%.

### `algorithms/calibration.py`, `algorithms/outliers.py` — empty by decision

Both hold an argument instead of code, so the reasoning does not have to be
re-derived. They are not unfinished work.

**`calibration.py`.** The scale is `s = max(abs(W_group)) / Qmax` — there is no
activation term, so calibration data cannot move a single scale. A
`calibrate(model, batches)` here would consume data, run forwards, and emit a
bit-identical checkpoint. It becomes real only under data-aware weight
quantization (GPTQ's Hessian, AWQ's per-channel search) or activation
quantization. The docstring carries the measured case for when that is worth
building: INT4 runs ~15x INT8's weight error for −6 GSM8K points, clipping
search recovered 13%, AWQ-class methods typically recover 2–4x — so it matters
only if INT4 specifically is needed, which the memory numbers do not currently
justify.

**`outliers.py`.** This one was actively misleading before: it implied nothing
handles outliers. Two things do. Groupwise scaling contains them structurally —
at `group_size=128` one large weight inflates the step for its own 128
neighbours and no others, which is exactly what per-tensor scaling fails to do.
And clipping search in `symmetric.search_group_scale` remediates the rest,
measured at 12–14% lower INT4 error for zero extra bytes. What is genuinely
absent is SmoothQuant-style per-channel migration (only relevant once
activations are quantized) and outlier-level mixed precision, which needs a
sensitivity measurement that does not exist yet.

---

## 5. Part C — Execution layer

**Role.** Decide how quantized weights are *held* and how they are *used* in a
forward pass. This is where the memory saving is actually realised or thrown
away.

### `src/LLaDA_Quant/runtime/moe.py` (271 lines)

The most subtle file in the project. Layout it targets:

```
w1: [local_experts, 2 * intermediate, hidden]   Gate+Up stacked
w2: [local_experts, hidden, intermediate]       Down
```

| Symbol | Role |
|---|---|
| `QuantExpertWeights` | One layer's fused pair as two `QuantResult`s. `quantize()`, `dequantize()`, `storage_bytes()`. |
| `quantize_fused_experts()` | Convenience wrapper. |
| `quant_result_from_buffers(q, scale, bits)` | Rebuild a `QuantResult` from stored tensors alone, **deriving the group size from the scale shape**. |
| `attach_packed_buffers(block, weights, dtype)` | Registers `_qw1/_sw1/_qw2/_sw2` plus `_llada_quant_meta`. |
| `install_packed_expert_access(block)` | Deletes the BF16 Parameters and installs the property. |
| `is_packed_expert_block(module)` | Mode predicate. |
| `materialize_expert_params()` | REFERENCE-mode write-back; **raises** on a packed block. |
| `expert_storage_bytes(block)` | Packed bytes resident on one block. |
| `PACKED_BUFFERS`, `EXPERT_PARAM_NAMES`, `QUANT_META_ATTR` | Names used consistently across adapter and formats. |
| `WEIGHT_MUTATING_METHODS` | Methods shadowed with a loud error in PACKED mode. |

**How PACKED residency works.** `install_packed_expert_access` pops `w1`/`w2`
out of `block._parameters` and repoints `block.__class__` at a generated
subclass (cached per base class in `_PACKED_CLASS_CACHE`) whose `w1`/`w2` are
`property` objects calling `_packed_expert_weight`. A `property` is a data
descriptor found on the type, so it wins over `nn.Module.__getattr__` and the
model's own `fused_moe(x, self.w1, self.w2, ...)` keeps working untouched.
`state_dict()` no longer contains a BF16 copy, and `resident_memory` sees the
saving.

This is the one piece of deliberate magic in the codebase. It is contained in
a single function so it can be reasoned about and tested.

**Three guards around that magic:**

1. **Assignment.** `block.w1 = ...` raises `AttributeError` — a `property`
   without a setter. A silent write into a temporary is the exact failure this
   mode exists to prevent.
2. **In-place loaders.** `WEIGHT_MUTATING_METHODS` currently lists
   `load_state_dict_from_unfused`, which LLaDA's real `TritonFusedMoEBlock`
   uses (`self.w1[i].copy_(...)`) at build time. In PACKED mode that write
   would vanish into a dequantized temporary *without error*, so the generated
   subclass shadows the method with a `RuntimeError` naming the fix: **load
   weights first, quantize second.**
3. **Write-back.** `materialize_expert_params` refuses to run on a packed
   block rather than pretending to.

**Why `quant_result_from_buffers` ignores the config's group size.** See the
per-tensor fallback trap above: trusting the config would dequantize with the
wrong grouping and produce garbage of exactly the right shape — the worst kind
of bug. The number of groups is recoverable from `scale.shape[-1]`, so it is
read from there.

### `src/LLaDA_Quant/runtime/linear.py` (131 lines)

`QuantLinear`, a drop-in for `nn.Linear` with identical call semantics.
Storage mirrors `nn.Linear`: `qweight` is `[out, in]` int8 at 8 bits and
`[out, in//2]` at 4 bits; `scale` is `[out, num_groups]`.

`forward` is dequantize-then-matmul. That trade is stated in the module
docstring rather than hidden: **resident memory always drops** (the BF16
weight is gone) and **latency always rises** (the weight is rebuilt every
call). `from_linear()` converts an existing module; `storage_bytes()` reports
what it costs; `extra_repr()` shows bits/group/packed in a `print(model)`.

### `src/LLaDA_Quant/runtime/kernels/__init__.py` (1 line)

Empty namespace reserved for the fused Triton kernels. It contains nothing,
and nothing in the repo claims a benefit from it.

---

## 6. Part D — Model integration

**Role.** Find exactly the right modules in someone else's model and convert
them, without requiring a single change to that model's code.

### `src/LLaDA_Quant/adapters/llada_moe.py` (184 lines)

| Symbol | Role |
|---|---|
| `ExpertBlockShape` | `num_experts`, `hidden`, `intermediate`, plus `numel` and `describe()`. |
| `describe_fused_expert_block(module)` | Returns the shape if the module *is* a fused expert block, else `None`. |
| `is_fused_expert_block(module)` | Boolean wrapper. |
| `find_expert_blocks(model, config)` | All structurally matching, non-excluded blocks. |
| `quantize_llada_experts(model, config)` | Converts them, honouring `execution_mode`; returns `TargetedModule` records. |
| `restore_llada_experts_from_buffers(model, config)` | Re-establishes access after a checkpoint load, in either mode. |

**Detection is structural, never name-based.** Four relations pin the layout:

```
both w1 and w2 are 3-D
w1.shape[0] == w2.shape[0]        same expert count E
w1.shape[1] == 2 * w2.shape[2]    w1 is Gate+Up stacked over I
w1.shape[2] == w2.shape[1]        both agree on hidden H
```

A module called `mlp` that happens to be an `nn.Linear` is not touched; a
correctly shaped block with an unexpected name is not missed. The predecessor
matched `"expert" in name or "mlp" in name` plus "3-D `w1`/`w2` with an even
second dim", which is a guess in both directions.

`describe_fused_expert_block` accepts a block that is *already* in PACKED mode
(where `w1` is a property returning a plain tensor, not a `Parameter`), so
detection stays idempotent for inspection while `quantize_llada_experts`
refuses to re-quantize.

Each record captures measured `source_bytes` (the BF16 Parameters before
conversion) and `quantized_bytes`, which is what makes the manifest's memory
totals checkable rather than asserted.

### `src/LLaDA_Quant/adapters/torch.py` (58 lines)

`find_linears` and `replace_linears` for explicitly named `nn.Linear` modules,
producing the same `TargetedModule` record type so both kinds of target flow
through one audit trail. Walks to the parent module by path and `setattr`s the
`QuantLinear` in place.

---

## 7. Part E — Persistence

**Role.** Produce an artifact that is self-contained, smaller than the source,
and carries enough provenance to audit a memory claim after the fact.

### `src/LLaDA_Quant/formats/manifest.py` (144 lines)

| Symbol | Role |
|---|---|
| `TargetedModule` | The targeting audit record: `name`, `kind`, `module_type`, `shapes`, `bits`, `group_size`, `packed`, `execution_mode`, `source_bytes`, `quantized_bytes`, `compression_ratio`. |
| `QuantEntry` | Per-tensor metadata (shape, dtypes, source tensor, sha256). |
| `QuantizationManifest` | `format_version`, `framework_version`, `created_at`, `source_checkpoint`, `config`, `entries`, `targets`, and derived `source_bytes` / `quantized_bytes`. `to_dict()` also emits a `totals` block. |
| `tensor_hash(t)` | Value-deterministic SHA-256. |
| `write_source_checkpoint_meta()` | Writes `source-checkpoint.json` with a hash of the unquantized source when it is a local file. |

The `targets` list is the difference between "this checkpoint claims to be
INT4" and "this checkpoint records that it converted these 16 blocks, from
these many bytes to these many".

### `src/LLaDA_Quant/formats/safetensors.py` (167 lines)

| Symbol | Role |
|---|---|
| `weights_filename(bits)` / `WEIGHTS_GLOB` | `model-int8.safetensors` / `model-int4.safetensors`; load globs so the name stays informative. |
| `derivable_tensor_names(manifest)` | The BF16 `w1`/`w2` of every expert target — reconstructable, therefore never stored. |
| `collect_quantized_state(model, manifest)` | State dict with those dropped. |
| `save_quantized_checkpoint(...)` | Weights + `quantization.json` + `source-checkpoint.json`. Returns the weights path. |
| `find_weights_file(dir)` | Resolves the glob; raises on missing or ambiguous. |
| `checkpoint_size_bytes(dir)` | Makes artifact size a measured quantity. |
| `load_quantized_checkpoint(dir)` | Raw tensors + manifest, no model touched. |
| `load_quantized_weights(model, dir, strict, execution_mode)` | Model-level load. `execution_mode` overrides what the manifest recorded. |
| `_register_missing_buffers()` | Lets a *plain* model absorb a quantized checkpoint, on the parent module's device. |

**The redundancy fix.** The predecessor saved the BF16 weights *and* the packed
buffers even though the load path recomputed the former from the latter,
making a "quantized" checkpoint 1.52× the unquantized one. Now they are
dropped at save time, in **both** execution modes.

**Residency is not a property of the file.** Both execution modes write
byte-identical artifacts — the BF16 experts are re-derivable and dropped in
either case — so `execution_mode=` at load time selects PACKED or REFERENCE
without rewriting anything. One artifact serves the deployment run and the
accuracy run. A test asserts the two modes produce identical bytes, and
another that an override never touches the file on disk.

**The device fix.** safetensors reads to CPU. `_register_missing_buffers`
registered those tensors as-is, so inside a CUDA model the packed integers
stayed on the host. Nothing upstream noticed — dequantization moves data, so
resident accounting and numerics were both correct — and only the fused
kernel, which consumes the packed buffers directly, saw the split, at request
time. Buffers now land on their parent module's device.

**The strictness consequence.** Those keys are then legitimately missing at
load. `load_quantized_weights` calls `load_state_dict(strict=False)` and
subtracts `derivable_tensor_names` from the missing set, so expected absences
pass while any *other* missing or unexpected key still raises. A checkpoint
from a different architecture raises a `RuntimeError` naming the offending
tensor rather than an opaque `AttributeError` from `get_submodule`.

---

## 8. Part F — Measurement

**Role.** Produce every number the project reports. Leaf modules: they depend
on torch and nothing else in the package, so they cannot be biased by the code
they measure.

### `src/LLaDA_Quant/memory.py` (133 lines)

| Symbol | Role |
|---|---|
| `MemoryReport` | `parameters`, `buffers`, `by_dtype`, `tensor_count`, `total`. |
| `resident_memory(module)` | Walks parameters and buffers of a live module tree. |
| `MemoryComparison` | `ratio`, `saved_bytes`, `is_saving`, `describe()`, `to_dict()`. |
| `compare_resident_memory(baseline, quantized, label)` | The before/after. |

Two details matter. **Shared storages are counted once**, keyed by
`untyped_storage().data_ptr()`, so a tied embedding is not double-billed.
And **`describe()` prints `REGRESSION`** when a "quantized" model grew — the
single line that would have caught the original bug immediately.

This module is the reason the README can state a memory number at all: it is
derived from tensors that are actually alive, never from the theoretical size
of a packed representation.

### `src/LLaDA_Quant/validation/metrics.py` (189 lines)

Tensor-level and masked-token metrics.

| Group | Functions |
|---|---|
| Error | `max_abs_error`, `mean_abs_error`, `max_rel_error`, `cosine_similarity`, `summarize_metrics` |
| Routing | `router_overlap` — fraction of (token, rank) slots where two top-k routings agree |
| Masked-token | `top1_agreement`, `kl_divergence`, `unmask_selection_agreement`, `top2_margin`, `tie_fraction` |

The masked-token group takes an optional boolean `positions` mask because only
the masked slots matter — a diffusion decoder discards predictions at resolved
positions.

Two of these carry real weight. **`unmask_selection_agreement`** asks whether
both models would unmask the *same position* next: two models can agree on
every predicted token yet commit in a different order, which changes the
context every later step conditions on, and top-1 agreement cannot see it.
**`tie_fraction`** reports the share of positions where the reference's own
top-2 margin is *smaller than the shift quantization introduced* — a flipped
argmax there is a coin toss, not damage. It exists because on the toy model
INT8 scores `top1_agreement = 0.0` at the fully masked state with
`tie_fraction = 1.0`.

### `src/LLaDA_Quant/validation/compare.py` (50 lines)

`compare_models(reference, quantized, components, get_inputs_fn, forward_fn,
router_fn, dtype)` — feeds identical inputs to named submodules of two models
and returns a `ComponentReport` per component. The narrow, single-forward-pass
question; the trajectory package handles the wider one.

### `src/LLaDA_Quant/analysis/moe_regime.py` (465 lines)

The roofline analysis that decides whether the fused kernel is worth writing.

| Symbol | Role |
|---|---|
| `MoEShape`, `LLADA_MOE_7B_A1B` | Geometry. The constant mirrors `test_llada/src/model.py`: E=64, top-k=8, H=2048, I=1024, 16 layers. |
| `Machine`, `RTX_A6000`, `A100_80GB`, `H100_SXM` | Peak flops, bandwidth, capacity, and `balance()` = the roofline ridge point. |
| `Scheme`, `SCHEMES` | BF16, W8A16, W4A16, W8A8, W4A8 — weight bytes plus compute dtype. |
| `Workload`, `suffix_lengths_for_schedule` | One decoding configuration. The helper encodes that LLaDA forwards `x[:, block_start:]`, so M shrinks block by block. |
| `ideal_tokens_per_expert` | `M * top_k / E` — for LLaDA, `M/8`. |
| `expert_token_stats(topk_ids, E)` | The *measured* alternative: per-expert counts, active experts, percentiles, `imbalance`. |
| `gemm_regime`, `crossover_m` | Classification and the ridge point per scheme. |
| `regime_sweep`, `RegimeReport` | The table, with `to_dict()` and `to_table()`. |

**The arithmetic.** For a weight-stationary GEMM with few rows, intensity is
`2*M*N*K / (N*K*weight_bytes) = 2M / weight_bytes`; activations are negligible
while M is small, which is exactly the regime in question. Memory-bound while
that sits below the machine balance, giving `M_cross = balance *
weight_bytes / 2`.

**Two honesty features.** `SCHEMES` models `W4A16`/`W4A8` as *expanded before
the MMA*, because no mainstream kernel stack emits Ampere INT4 tensor-core
math — no scheme claims native 4-bit arithmetic. And `RegimeReport.to_dict()`
carries a `routing_note` stating that rows assume perfect balance and that
`measured_routing` is `None` unless real `topk_ids` were supplied.

---

## 9. Part G — Trajectory validation

**Role.** Answer *how quantization changes diffusion generation* — the
question a single-forward-pass metric structurally cannot reach.

```
MODEL ──► CAPTURE ──► TRACE ──► METRICS ──► REPORT
          (GPU)       (JSON)    (offline)   (offline)
```

Only `capture.py` touches a model. Everything downstream runs from JSON, so
metrics can be recomputed, re-cut and unit-tested without rerunning anything.

### `trajectory/state.py` (182 lines)

| Symbol | Role |
|---|---|
| `DiffusionState` | `step`, `input_ids [B,L]`, `mask_positions [B,L]`, optional `attention_mask`, `label`; `num_masked`, `mask_ratio`, `describe()`. Validates shapes on construction. |
| `fully_masked_state(prompt, gen_length, mask_id)` | The trajectory start. |
| `make_masked_states(prompt, completion, mask_id, ratios, generator)` | A **monotone** early/middle/late schedule. |
| `mask_positions_from_ids()` | Helper. |
| `LogitsFn`, `RouterFn`, `AdvanceFn` | The callback protocols. |

`make_masked_states` draws one reveal order per row and masks the first
`ratio * gen_length` of it, so the masked set at 25% is a strict subset of the
one at 50% — positions are only ever revealed, exactly as in a real decode.
Its docstring states the limitation honestly: revealing *ground-truth* tokens
is not the distribution a model conditions on mid-decode, so it is a
screening tool, not a headline number.

The protocol aliases are why this package contains **no decoding logic**: the
caller supplies the callables and the framework never imports a decoder.

### `trajectory/trace.py` (201 lines)

| Symbol | Role |
|---|---|
| `MetricPrecision` | `EXACT` / `TOPK` / `SAMPLED`. |
| `ScalarMetric` | `value`, `precision`, `note`, `is_exact`. |
| `LayerStats` | `hidden_norm`, `hidden_absmax`, `router_margin`, `router_gate_entropy`, optional `router_topk_ids`. |
| `TraceStep` | One step: masked/committed positions, committed tokens, top-k ids and log-probs, per-layer stats, scalars. |
| `Trace` | Steps plus context (`mode`, `seed`, `mask_token_id`, lengths, `top_k_stored`, `meta`), `scalar_series()`, `save()`, `load()`, `size_estimate_bytes()`. |

**Two rules the format enforces.** *Never store full tensors* — LLaDA logits
are `[B, L, 157184]`, about 40 MB per step per model, so anything needing the
full vocabulary is reduced to a scalar on device during capture and everything
else is top-k truncated. *Never call an approximation exact* — every scalar
carries a `MetricPrecision`, so a KL computed from 8 stored log-probs is
labelled `TOPK`, because it is a lower bound, not the KL.

`Trace.from_dict` rejects an unknown `format_version` rather than guessing.

### `trajectory/capture.py` (308 lines)

The only GPU-dependent module.

| Symbol | Role |
|---|---|
| `capture_shared(...) -> SharedCapture` | **Mode A**: both models fed byte-identical states. |
| `capture_free_running(...) -> FreeRunCapture` | **Mode B**: each model advances itself through the caller's `advance_fn`. |
| `GatesFn` | Optional callback yielding raw router scores, enabling router-margin capture. |
| `_step_record`, `_topk_slice`, `_layer_stats` | Reduce a step to a `TraceStep`. |

**Mode A** stores pairwise quantities as `pair.*` EXACT scalars on the
quantized trace: `logit_cosine`, `max_abs_error`, `kl_masked`,
`top1_agreement`, `tie_fraction`, `unmask_agreement`, and per-layer
`router_overlap.<layer>` plus their mean. These need both models' full logits
at once, so they are reduced here and nowhere else.

**Mode B** deliberately records **no** `pair.*` scalars. Once the trajectories
differ the two models are looking at different inputs, and a logit distance
would conflate quantization error with input drift. What it records is what
actually diverged: `committed_positions` and `committed_tokens` per step.

### `trajectory/metrics.py` (126 lines)

Re-exports the tensor metrics so callers have one import, and adds the
diffusion-MoE-specific ones:

| Function | Role |
|---|---|
| `predictive_entropy` | How undecided the model is at masked slots. |
| `router_margin(gates, top_k)` | Gap between the k-th and (k+1)-th gate — whether quantization noise can flip expert selection. The routing analogue of `tie_fraction`. |
| `router_gate_entropy` | How spread the routing is. |
| `topk_kl_lower_bound` | KL restricted to the stored top-k support. Named as a bound because that is what it is. |
| `commit_order_agreement` | Fraction of steps committing the same position set. |

### `trajectory/replay.py` (198 lines)

| Symbol | Role |
|---|---|
| `replay_shared(ref, qnt)` | Mode A metrics recomputed offline. |
| `replay_free_running(ref, qnt)` | Mode B: cumulative token agreement, same-commit-set, disagreement counts, first divergence, commit-order agreement. |
| `verify_replay(report, tolerance)` | Cross-checks replayed values against the exact ones captured on device. |
| `ReplayedStep`, `ReplayReport` | Results, separating `replayed` from `stored_exact`. |

Every `ReplayedStep` keeps the two sources apart: numbers *it* derived from
the top-k slice (labelled `TOPK`) and the EXACT scalars carried in the trace.
`verify_replay` returns a list of human-readable discrepancies — run it in CI
and the trace format cannot drift away from the capture code unnoticed.

### `trajectory/report.py` (112 lines)

`TrajectoryReport` holds `mode_a`, `mode_b`, `noise_floor_a`, `noise_floor_b`
and derives `per_step_signal` (Mode A above the floor) and `amplification`
(Mode B ÷ Mode A, `nan` without both).

Its purpose is to prevent three specific confusions: reading Mode A as
end-to-end damage (understates amplification), reading Mode B as per-step
error (overstates it, because one early flip drags every later step), and
quoting either without the BF16-vs-BF16 floor. `to_table()` renders a
`BF16 floor` column structurally — the table cannot be produced without space
for it.

### `trajectory/llada.py` (335 lines)

| Symbol | Role |
|---|---|
| `LLaDADecoder` | Holds the three imported primitives plus their `source`. |
| `load_llada_decoder(repo_path, module)` | Imports `add_gumbel_noise`, `get_num_transfer_tokens`, `select_transfer_indices` from the inference repo. |
| `make_llada_advance_fn(decoder, steps, ...)` | Assembles an `advance_fn` from those exact functions. |
| `assert_matches_production_decoder(...)` | Proves one step of the adapter equals the production primitives. |
| `RouterCapture` / `attach_router_capture` | Recovers each fused block's top-k routing. |
| `router_fn_for` / `gates_fn_for` | Dispatch a capture by model identity. |
| `LLADA_MASK_ID = 156895` | Default, always passable explicitly. |

**Router capture.** `TritonFusedMoEBlock` computes `topk_ids` inside `forward`
and never returns it, so a plain forward hook cannot see it and router overlap
would be unmeasurable. `RouterCapture` registers a forward *pre*-hook, stashes
each block's input, and recomputes `topk(softmax(gate(x_flat), float32))` — the
block's own operations on the block's own tensors, so it is bit-identical.

Because the router is an excluded BF16 `nn.Linear` running *before* the
experts, its weights are identical in both models. Every overlap below 1.0 is
therefore attributable to the hidden state drifting upstream, never to the
router being damaged — that attribution is the point of the measurement, and a
test pins it.

Attach one capture per model. `capture_shared` runs both, so a single shared
registry would let the second model's forward overwrite the first's; hence
dispatch by `id(model)` rather than by layer name alone.

**No LLaDA decoding semantics are restated here.** A measurement built on a
reimplemented decoder drifts silently, so the numbers keep looking plausible
while describing something that never runs. The inference repository is never
imported at module load and never modified; point `load_llada_decoder` at it
only when you want Mode B against the real thing.

`make_llada_advance_fn` documents that `temperature=0.0` is required for
measurement: with sampling, two models draw independent gumbel noise and the
divergence is sampling, not quantization.

---

## 10. Part H — Benchmarks

**Role.** Runnable measurement, each script stating in its output what it
measures **and what it does not**.

| File | Category | Measures | Explicitly not |
|---|---|---|---|
| `bench_storage.py` (151) | A — storage | resident tensor bytes + checkpoint bytes, both bit widths × both modes | any latency |
| `bench_numerical.py` (116) | B — numerical | weight and expert-output quantization error, measured vs formula storage | latency — both paths run the same BF16 matmul |
| `bench_moe_regime.py` (98) | decision | tokens/expert, GEMM shapes, roofline side, capacity headroom | wall-clock anything |
| `bench_bf16_vs_int4.py` | validation | BF16 vs INT4-MSE on the real checkpoint against a BF16-vs-BF16 floor | latency; needs a GPU, not yet run |

`bench_storage.py` prints `<- LARGER than BF16` next to every REFERENCE row.
`bench_moe_regime.py` accepts `--routing-file` to replace the ideal-balance
assumption with a real `topk_ids` tensor.

The predecessor, `bench_experts.py`, was deleted: it dequantized to BF16 and
then timed the same BF16 computation twice, reporting the difference as an
INT8 result.

> **This section is behind the code.** `runtime/kernels/w8a16_gemm.py`,
> `runtime/kernels/w8a16_moe.py`, `runtime/fused_block.py`,
> `benchmarks/bench_fused_e2e.py` and `benchmarks/serve_quantized.py` all exist
> and are not described anywhere in this document. README.md and RESULTS.md
> carry the current numbers; this file has not caught up.

---

## 10b. Part J — Tools

**Role.** Executables that produce artifacts rather than measurements.

### `tools/quantize_checkpoint.py`

Quantizes the real weights **once**, offline, into a standalone artifact.
Everything else in the repository quantizes at startup.

| Step | Detail |
|---|---|
| build | `llada_repo.build_bf16_model` — the inference repo's own model, imported |
| quantize | `llada_repo.quantize_experts_streaming` on the GPU; 13.1 s for 6.44 B weights |
| write | `save_quantized_checkpoint` |
| verify | re-reads the file and compares it against memory tensor by tensor, and asserts the re-derivable BF16 experts are absent |

Measured on the real checkpoint: 13.71 → 7.89 GiB (0.576×), and the result is
**bit-identical** to a startup-time quantization — 211 tensors, 0 mismatches,
checked from a fresh process that rebuilt and requantized the model.

`serve_quantized.py --quantized-checkpoint DIR` loads it. `--bits`,
`--group-size`, `--scale-search` and `--search-grid` then come from the
manifest, and a command line that contradicts it is rejected rather than
ignored: a server started with `--bits 4` against an INT8 artifact would
otherwise serve INT8 and label itself INT4, which is the same class of bug
that once produced two identical GSM8K result files from one unchanged server.

**What it does not buy.** Accuracy and inference speed are unchanged, and the
BF16 weight read still happens — model construction belongs to the inference
repository, which this project does not modify, so the artifact saves the
scale search and nothing else.

### `tools/check_determinism.py`

Rebuilds the model in a fresh process, requantizes from the artifact's own
manifest, and compares tensor by tensor. Exists because the claim it checks —
that the scale search is deterministic, so an artifact equals a startup-time
quantization — fails **silently** when false: different scales still load,
still run, and still produce plausible text.

### `tools/run_gsm8k_comparison.sh`

Both GSM8K arms in one command: starts each server, waits for `/health`,
records what `/v1/quantization` reports, grades, stops the server.

Two guards, both from real failures. It **aborts if the two arms report the
same label** — forgetting the restart once produced two byte-identical result
files that read as a clean "quantization changed nothing" finding. And below
n=100 it refuses to print the interpretation, because a smoke run at n=8 shows
a 12.5-point delta per question and that reads like a result.

### `src/LLaDA_Quant/llada_repo.py`

`build_bf16_model` and `quantize_experts_streaming`, shared by
`bench_bf16_vs_int4.py` and `tools/quantize_checkpoint.py`. The only module in
the package that knows the inference repository's layout (`model_update.model`,
`src.model.load_weights`); everything else works on any `nn.Module`.

---

## 11. Part I — Tests

`tests/unit/conftest.py` (74) supplies `FusedExpertBlock` — the real
`w1 [E,2I,H]` / `w2 [E,H,I]` layout with a `gate` — and `TinyMoEModel`, two MoE
layers plus router, norms, embeddings and LM head that must never be
quantized.

| File | Lines | Covers |
|---|---|---|
| `test_symmetric.py` | 74 | Quantization math, error budget, per-tensor fallback |
| `test_int4.py` | 178 | Packing roundtrip, sign extension, nibble order, group + expert alignment, half-of-INT8 storage, adapter integration |
| `test_scale_search.py` | 200 | MSE search vs amax, storage-contract invariance, config wiring, checkpoint roundtrip |
| `test_memory.py` | 156 | Resident-memory regression guards, mode semantics, blocked in-place loaders |
| `test_targeting.py` | 200 | Structural detection, component globs, loud failures, audit trail |
| `test_checkpoint_format.py` | 241 | No BF16 duplicates, bit-exact roundtrip, group-size recovery, manifest |
| `test_quantlinear.py` | 51 | `QuantLinear` vs `nn.Linear` |
| `test_llada_moe_adapter.py` | 138 | Both modes, per-access dequantization, restore |
| `test_api.py` | 108 | Result surface, modes, non-destructive variants |
| `test_validation.py` | 62 | Tensor metrics, component comparison |
| `test_trajectory.py` | 342 | States, masked-token and routing metrics, Mode A/B capture |
| `test_trace_replay.py` | 241 | Trace IO, compactness, offline replay, precision labels, noise floor |
| `test_moe_regime.py` | 218 | Roofline math, crossovers, routing statistics |
| `test_llada_binding.py` | 154 | Delegation to the production decoder, drift detection |
| `test_router_capture.py` | 205 | Router recomputation exactness, per-model isolation, hook lifecycle, top-k resolution |

Several are **regression guards for bugs that actually shipped** — the resident
BF16 duplicate, cosmetic INT4, redundant checkpoints, substring targeting.
Each names the bug in its docstring so a future edit cannot reintroduce it by
accident.

---

## 12. End-to-end walkthroughs

### Quantizing a model

```
api.quantize_model(model, config)
  └─ adapters.llada_moe.quantize_llada_experts
       ├─ find_expert_blocks          config.is_excluded + structural match
       ├─ QuantExpertWeights.quantize algorithms.symmetric.quantize_tensor  (+ pack_int4 at 4 bits)
       ├─ attach_packed_buffers       register _qw1/_sw1/_qw2/_sw2 + metadata
       ├─ PACKED   → install_packed_expert_access   delete Parameters, install property
       │  REFERENCE→ materialize_expert_params      write dequantized values back
       └─ record TargetedModule       measured source/quantized bytes
  └─ adapters.torch.replace_linears   config.matches_linear → runtime.linear.QuantLinear
  └─ api._validate_targeting          TargetingError on an unexpected match set
```

### Saving and loading

```
save_quantized_checkpoint(model, manifest, dir)
  ├─ derivable_tensor_names(manifest)   expert w1/w2 → dropped
  ├─ save_file(...)                     model-int{bits}.safetensors
  ├─ manifest.save(dir)                 quantization.json (config + targets + totals)
  └─ write_source_checkpoint_meta(dir)  source-checkpoint.json

load_quantized_weights(model, dir)
  ├─ load_quantized_checkpoint          tensors + manifest
  ├─ _register_missing_buffers          plain model gains _qw1/... 
  ├─ load_state_dict(strict=False)      expected-missing = derivable names
  └─ restore_llada_experts_from_buffers
       └─ quant_result_from_buffers     group size derived from scale shape
```

### Measuring a trajectory

```
capture_shared(ref, qnt, states, logits_fn, router_fn, gates_fn)   Mode A  ─┐
capture_free_running(ref, qnt, start, logits_fn, advance_fn)       Mode B  ─┤
capture_shared(ref, ref, ...) / capture_free_running(ref, ref, ...) floor  ─┤
                                                                            │
   Trace.save()  ──►  JSON on disk  ──►  Trace.load()   (no model needed)   │
                                                                            │
replay_shared / replay_free_running  ──►  ReplayReport  ◄───────────────────┘
   verify_replay()          replayed vs on-device exact
TrajectoryReport(mode_a, mode_b, noise_floor_a, noise_floor_b).to_table()
```

---

## 13. Where to make common changes

| Goal | Files to touch |
|---|---|
| Add a bit width | `algorithms/symmetric.py` (`qmax_for_bits`, packing), `config.py` validation, `runtime/linear.py` buffer shapes |
| Add a quantization algorithm (GPTQ/AWQ) | new module in `algorithms/`, wire through `runtime/moe.py`; `calibration.py` becomes real |
| Support a new model family | new module in `adapters/`, dispatch in `api.quantize_model` |
| Write the fused kernel | `runtime/kernels/`, consumed by a new execution mode in `config.ExecutionMode`; add benchmark category C |
| Add a trajectory metric | `trajectory/metrics.py`; if it needs full tensors, reduce it in `capture.py` as an EXACT scalar; otherwise compute it in `replay.py` labelled `TOPK` |
| Change the trace format | `trajectory/trace.py`, bump `TRACE_FORMAT_VERSION`, update `replay.py` |
| Retarget a different GPU | add a `Machine` in `analysis/moe_regime.py` |

---

## 14. Invariants to respect

1. **No claim without a measurement.** Memory from `memory.resident_memory`,
   storage from `os.path.getsize`. Never report the theoretical size of a
   packed tensor as a runtime saving.
2. **No speed claim without an executed kernel.** Two paths running the same
   BF16 matmul are not a quantization comparison.
3. **Never match a module by guessing its name.** Structure, or an explicit
   list, and fail loudly when the match set is unexpected.
4. **A config option must do what its name says**, or raise. `bits=4` that
   saves nothing is worse than no `bits=4` at all.
5. **Mode A and Mode B are not interchangeable**, and neither means anything
   without the BF16-vs-BF16 floor.
6. **Approximations are labelled.** From a top-k slice it is `TOPK`, not
   `EXACT`.
7. **Do not reimplement the decoder.** Import it, and prove equivalence.
8. **A silent no-op is a bug.** Zero matches, a discarded in-place write, a
   dequantize with the wrong grouping — each of these has a guard, and new
   code should add one rather than rely on the caller reading a docstring.
