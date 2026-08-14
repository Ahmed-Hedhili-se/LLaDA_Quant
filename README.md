# LLaDA_Quant

Independent quantization framework for LLaDA-style MoE models (and any PyTorch
model made of linear layers). It handles **weight quantization only** — no
diffusion decoding, KV caching, serving, or orchestration. Your model /
inference application stays a separate repository and consumes this framework
as a dependency.

**Current status (v0.1):** symmetric groupwise **INT8** reference implementation
(dequantize-then-matmul), generic `QuantLinear`, fused-expert `w1`/`w2`
adapter, versioned checkpoint format, validation metrics, benchmarks, and
[diffusion-trajectory validation](#diffusion-trajectory-validation) —
divergence measured *across denoising steps* rather than on a single forward
pass. Triton INT8/INT4 kernels that consume packed weights directly are
planned for v0.3.

---

## Table of contents

- [Design goals](#design-goals)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Quantization strategy](#quantization-strategy)
- [Public API reference](#public-api-reference)
  - [Configuration](#configuration)
  - [Model quantization](#model-quantization)
  - [Checkpoint save / load](#checkpoint-save--load)
  - [Low-level algorithms](#low-level-algorithms)
  - [Runtime modules](#runtime-modules)
  - [Adapters](#adapters)
  - [Validation](#validation)
- [Diffusion-trajectory validation](#diffusion-trajectory-validation)
- [Checkpoint format](#checkpoint-format)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [Legal](#legal)

---

## Design goals

- **Explicit, serializable configuration.** Every checkpoint records bits,
  group size, scale type, packing layout, compute dtype, and provenance.
- **Reference-first correctness.** A dequantize-then-matmul path defines the
  numerical contract that future Triton kernels must reproduce.
- **Drop-in, zero model changes.** The LLaDA adapter quantizes expert weights
  and stores packed int8 buffers beside them; the model's `fused_moe` calls
  keep working on the materialized BF16 tensors.
- **Sensitive layers stay in BF16.** Router, norms, embeddings and the LM head
  are excluded by default — tiny BF16 differences can flip top-8 expert
  selection, which makes quality comparisons noisy.

---

## Installation

Requires Python >= 3.10, `torch >= 2.0`, `safetensors`.

```bash
# Development (editable, recommended):
git clone https://github.com/Ahmed-Hedhili-se/LLaDA_Quant.git
cd LLaDA_Quant
pip install -e .

# Or directly from GitHub:
pip install git+https://github.com/Ahmed-Hedhili-se/LLaDA_Quant.git

# With dev/test extras:
pip install -e ".[dev]"
```

---

## Quick start

```python
from LLaDA_Quant import (
    QuantConfig,
    QuantizationManifest,
    quantize_model,
    save_quantized_checkpoint,
    load_quantized_weights,
)

# 1. Configure: INT8, group_size=128, experts + attention linears,
#    router / norms / embeddings / LM head excluded.
config = QuantConfig(
    bits=8,
    group_size=128,
    targets=("expert", "linear"),
    exclude=("router", "norm", "embed_tokens", "lm_head"),
)

# 2. Quantize a model in place.
quantize_model(model, config)

# 3. Save a versioned, provenance-tracked checkpoint (original weights
#    are never touched).
save_quantized_checkpoint(
    model,
    QuantizationManifest(
        source_checkpoint="hf://inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        config=config,
    ),
    "llada-moe-7b-int8-g128",
)

# 4. Later, in another process, load into a plain (unquantized) model.
#    Missing packed buffers are registered automatically and w1/w2 are
#    re-materialized from them.
load_quantized_weights(model, "llada-moe-7b-int8-g128")
```

### How `targets` maps to actions

| `targets` value | Action |
|---|---|
| `("expert",)` | Every fused expert block (3-D `w1`/`w2` Parameters, as in `TritonFusedMoEBlock`) gets persistent packed buffers `_qw1`/`_sw1`/`_qw2`/`_sw2`; the live `w1`/`w2` Parameters are replaced by dequantized BF16 values. Router and gating are untouched, so top-k routing is bit-identical. |
| `("linear",)` | Every matching non-expert `nn.Linear` is swapped for a `QuantLinear` module with the same name, in/out features and bias. |
| `("expert", "linear")` | Both of the above. |

---

## Quantization strategy

| Component | v0.1 default | Rationale |
|---|---|---|
| MoE expert `w1`, `w2` | INT8 (groupwise) | Dominates compute and memory |
| Attention Q/K/V/O projections | INT8 via `QuantLinear` | Straightforward `nn.Linear` replacement |
| Router / gate | BF16 (excluded) | Near-uniform scores; tiny changes flip top-k |
| RMSNorm | BF16 (excluded) | Numerical sensitivity, negligible memory |
| Embeddings / LM head | BF16 (excluded) | Large but quality-sensitive |
| KV cache | BF16 | Changes runtime accuracy; separate phase |

Algorithm (symmetric, zero-point-free, per group along K):

```
W_q = clamp(round(W / s), -Qmax - 1, Qmax)      s = max(|W_group|, 0) / Qmax
```

Groups run along the last axis of a weight tensor, so an expert-stacked tensor
of shape `[E, N, K]` is quantized per expert and per group with no special
casing of the expert dimension. If `group_size` does not divide the last
dimension (or `group_size=-1`), it falls back to per-tensor scaling.

---

## Public API reference

### Configuration

`LLaDA_Quant.config.QuantConfig`

```python
QuantConfig(
    bits=8,                    # 8 or 4
    group_size=128,            # -1 = per-tensor fallback
    targets=("expert",),       # ("expert",), ("linear",) or both
    exclude=("router", "norm", "embed_tokens", "lm_head"),
    compute_dtype="bfloat16",  # activation/accumulation dtype
    scale_dtype="float32",     # scale storage dtype
    source_checkpoint=None,    # provenance (path or HF id)
)
```

Methods: `to_dict()`, `from_dict(data)`, `to_json()`. Every checkpoint
embeds the full config so a run is reproducible.

### Model quantization

`LLaDA_Quant.api`

| Function | Description |
|---|---|
| `quantize_model(model, config) -> list[str]` | Quantize `model` in place. Returns the names of quantized modules/blocks. |
| `quantized_model(model, config) -> nn.Module` | Non-destructive variant: deep-copies `model` first, then quantizes the clone. |
| `save_quantized_checkpoint(model, manifest, directory)` | Write `model-int8.safetensors` + `quantization.json` + `source-checkpoint.json`. |
| `load_quantized_weights(model, directory, strict=True) -> QuantizationManifest` | Load a checkpoint into a model (plain or already-quantized). Registers missing packed buffers and re-materializes `w1`/`w2`. |

### Checkpoint save / load

`LLaDA_Quant.formats.safetensors`

| Function | Description |
|---|---|
| `save_quantized_checkpoint(model, manifest, directory)` | Full artifact (tensors + manifests). |
| `load_quantized_checkpoint(directory) -> (state_dict, manifest)` | Raw tensors + manifest, without touching any model. |
| `load_quantized_weights(model, directory, strict=True)` | Model-level load (see above). |

`LLaDA_Quant.formats.manifest`

- `QuantizationManifest(format_version, framework_version, created_at, source_checkpoint, config, entries)` — `to_dict()`, `from_dict()`, `to_json()`, `save(directory)`.
- `QuantEntry(tensor_name, shape, bits, group_size, storage_dtype, compute_dtype, source_tensor, sha256)` — per-tensor metadata.
- `tensor_hash(tensor) -> str` — deterministic SHA-256 of tensor values.

### Low-level algorithms

`LLaDA_Quant.algorithms.symmetric`

| Function | Description |
|---|---|
| `quantize_tensor(w, bits=8, group_size=128, scale_dtype=float32) -> QuantResult` | Quantize along the last axis. |
| `dequantize_tensor(q, scale, bits, group_size, dtype=float32)` | Reconstruct `w ~= q * scale` (arithmetic in fp32, then cast — preserves scale precision in BF16). |
| `qmax_for_bits(bits) -> int` | Largest positive representable value. |
| `pack_int4(q8) -> int8` / `unpack_int4(packed)` | Two's-complement nibble packing along the last axis. |

`QuantResult` (`q`, `scale`, `bits`, `group_size`) exposes `dequantize(dtype)`.

### Runtime modules

`LLaDA_Quant.runtime.linear.QuantLinear`

```python
QuantLinear(in_features, out_features, bits=8, group_size=128,
            compute_dtype=torch.bfloat16, bias=False, scale_dtype=torch.float32)
# from an existing module:
QuantLinear.from_linear(linear, bits=8, group_size=128, compute_dtype=torch.bfloat16)
# access the reconstructed weight:
w = qlin.dequantize_weight()
```

Call semantics are identical to `nn.Linear`: `out = qlin(x)` returns a
BF16 tensor of shape `(..., out_features)`. Storage: `qweight`
`[out_features, in_features]` int8, `scale` `[out_features, num_groups]`.

`LLaDA_Quant.runtime.moe`

| Function / class | Description |
|---|---|
| `QuantExpertWeights(w1: QuantResult, w2: QuantResult)` | Container for one layer's packed fused `w1`/`w2`. |
| `QuantExpertWeights.quantize(w1, w2, bits, group_size, scale_dtype)` | Quantize a fused pair. |
| `QuantExpertWeights.dequantize(dtype) -> (w1, w2)` | Reconstruct both tensors. |
| `quantize_fused_experts(w1, w2, bits=8, group_size=128)` | Convenience wrapper. |
| `materialize_expert_params(module, weights, compute_dtype)` | Copy dequantized values into `module.w1` / `module.w2`. |

### Adapters

`LLaDA_Quant.adapters.llada_moe` (LLaDA-specific, targets `TritonFusedMoEBlock`)

| Function | Description |
|---|---|
| `is_fused_expert_block(module) -> bool` | Detects a block with 3-D `w1`/`w2` Parameters (w1 with even second dim). |
| `quantize_llada_experts(model, config) -> list[str]` | Quantize all fused expert blocks in place; registers `_qw1/_sw1/_qw2/_sw2` buffers. |
| `restore_llada_experts_from_buffers(model, config)` | Re-materialize `w1`/`w2` from the packed buffers. |

`LLaDA_Quant.adapters.torch` (generic)

| Function | Description |
|---|---|
| `replace_linears(model, config) -> list[str]` | Swap matching non-excluded `nn.Linear` modules for `QuantLinear`. |

### Validation

`LLaDA_Quant.validation.metrics`

| Function | Description |
|---|---|
| `max_abs_error(a, b)` / `mean_abs_error(a, b)` | Absolute error statistics. |
| `max_rel_error(a, b, eps=1e-6)` | Relative error (flattened). |
| `cosine_similarity(a, b)` | Cosine similarity of flattened tensors. |
| `router_overlap(topk_ids_a, topk_ids_b) -> float` | Fraction of (token, rank) slots where two routings agree — the key diagnostic for this codebase. |
| `summarize_metrics(a, b) -> dict` | All of the above in one dict. |
| `top1_agreement(a, b, positions=None) -> float` | Fraction of positions predicting the same token id. |
| `kl_divergence(ref, other, positions=None) -> float` | Mean `KL(ref ‖ other)` in nats. |
| `unmask_selection_agreement(a, b, mask_positions, k=1) -> float` | Overlap of the positions each model would unmask next. |
| `top2_margin(logits, positions=None) -> float` | How decisive the reference is. |
| `tie_fraction(ref, other, positions=None) -> float` | Share of positions where the reference's margin is below the quantization shift. |

`LLaDA_Quant.validation.compare.compare_models(reference, quantized, components, get_inputs_fn, forward_fn, router_fn, dtype) -> dict[str, ComponentReport]`

Compares named submodules of two models on identical inputs; `ComponentReport`
carries the metrics plus `router_overlap`. `forward_fn(module, inputs)` and
`router_fn(module, inputs)` are supplied by you.

> Do not use text equality as your only gate: small numerical perturbations
> change diffusion trajectories. Track logit similarity and per-layer router
> overlap instead.

---

## Diffusion-trajectory validation

`compare_models` answers *how wrong is this layer on one forward pass* — the
standard PTQ question, and one that is structurally blind to how a masked
diffusion LM actually fails. A small logit shift changes **which position gets
unmasked**; that position becomes context for every later step; the error
compounds along the schedule. `LLaDA_Quant.validation.trajectory` measures
that directly.

The boundary from [Design goals](#design-goals) is unchanged: **no decoding
logic lives here.** You pass callables, and your inference repository keeps
owning the forward pass, the router internals and the unmasking rule.

### Building denoising states

`LLaDA_Quant.validation.diffusion`

| Function | Description |
|---|---|
| `DiffusionState(step, input_ids, mask_positions, attention_mask=None, label="")` | One point on a trajectory. `num_masked` / `mask_ratio` / `describe()`. |
| `fully_masked_state(prompt_ids, gen_length, mask_token_id)` | Prompt + all-mask generation region: the trajectory start. |
| `make_masked_states(prompt_ids, completion_ids, mask_token_id, ratios=..., generator=None)` | A **monotone** early/middle/late schedule — positions are only ever revealed, never re-masked, so state *i+1*'s masked set is a strict subset of state *i*'s. |
| `mask_positions_from_ids(input_ids, mask_token_id)` | Boolean mask helper. |

### The two entry points

| Function | What it isolates |
|---|---|
| `compare_trajectory(reference, quantized, states, logits_fn, router_fn=None, unmask_k=1, masked_positions_only=True)` | **Teacher-forced.** Both models see byte-identical inputs, so every difference is quantization alone. Shows per-step sensitivity; by construction cannot show compounding. |
| `compare_free_running(reference, quantized, initial_state, logits_fn, advance_fn, max_steps=64)` | **Self-driven.** Each model denoises with its own logits through *your* unmasking rule. Inputs drift apart — that is the point. This is where compounding becomes visible. |

Callbacks you supply:

```python
logits_fn(model, state)   -> Tensor [B, L, vocab]
router_fn(model, state)   -> Tensor | {layer_name: Tensor} | None   # top-k expert ids
advance_fn(state, logits) -> DiffusionState | None                  # your unmasking rule
```

```python
from LLaDA_Quant import (
    QuantConfig, quantized_model, make_masked_states, compare_trajectory,
)

reference = load_llada_moe()                       # your BF16 model
quantized = quantized_model(reference, QuantConfig(bits=8, group_size=128))

states = make_masked_states(prompt_ids, completion_ids, mask_token_id=MASK_ID,
                            ratios=(1.0, 0.75, 0.5, 0.25, 0.1))

report = compare_trajectory(
    reference, quantized, states,
    logits_fn=lambda m, s: m(s.input_ids).logits,
    router_fn=lambda m, s: collect_topk_ids(m),    # from your own hooks
    unmask_k=block_size,                           # match your decoder
)
print(report.to_table())
json.dump(report.to_dict(), open("trajectory-int8.json", "w"))
```

`TrajectoryReport` gives `states`, `series(key)` (the degradation *shape* in
step order), `worst_state`, `min_router_overlap`, `to_table()`, `to_dict()`.
`FreeRunReport` gives `steps`, `first_divergence_step`, `final_token_agreement`
and the two final id tensors.

### Reading the numbers

Per state: `top1_agreement`, `unmask_agreement`, `kl_masked`, per-layer
`router_overlap`, plus the standard logit metrics restricted to masked slots.

Two of these carry the load. **`unmask_agreement`** catches models that agree
on every predicted token yet unmask in a different order — invisible to top-1
agreement, but it changes the context all later steps condition on.
**`tie_fraction`** guards against over-reading the rest: it reports the share
of positions where the *reference's own* top-2 margin is smaller than the
logit shift quantization introduced. Disagreement there is a coin-toss on a
tie, not damage.

That guard is not hypothetical. On the toy model in the test suite, INT8
scores `top1_agreement = 0.0` at the fully masked state — with
`tie_fraction = 1.0`, because the reference's margin is ~1e-5 against ~5e-3 of
INT8 noise. Reported alone, the first number looks catastrophic and means
nothing. **Never quote an agreement number without its tie fraction.**

---

## Checkpoint format

```
llada-moe-7b-int8-g128/
├─ model-int8.safetensors      packed ints + scales + materialized params
├─ quantization.json           manifest: config, entries, framework version
└─ source-checkpoint.json      pointer + SHA-256 of the unquantized source
```

The original Hugging Face checkpoint is never mutated. Loading into a *plain*
model works too: missing packed buffers are registered automatically and
`w1`/`w2` are re-materialized from them.

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests
```

Test layers:

| Level | What is checked |
|---|---|
| Unit | quantize/dequantize error bounds, int4 pack roundtrip, `QuantLinear` vs `nn.Linear`, adapter buffer registration, manifest JSON roundtrip, checkpoint save/load, router-overlap metric |
| Component | one fused MoE layer output error vs BF16 reference |
| Trajectory | monotone schedule construction, masked-token metrics, teacher-forced and free-running divergence on a toy masked diffusion LM |
| Regression / E2E | planned: fixed prompts + seeds, routing overlap, task accuracy, memory, tokens/sec |

---

## Benchmarks

```bash
python benchmarks/bench_experts.py --num-experts 64 --hidden 2048 --intermediate 1024
```

Produces a JSON report with BF16 vs INT8 latency and weight-memory footprint
(INT8 weight memory is ~47% of BF16). v0.1 measures the reference
(dequantize-then-matmul) path; the packed Triton kernel will be benchmarked
with the same CLI in v0.3.

Planned matrix (fixed schedule + hardware):

| Row | Config |
|---|---|
| 1 | BF16 baseline |
| 2 | INT8 experts only |
| 3 | INT8 experts + attention |
| 4 | INT4 experts only |
| 5 | INT4 experts + attention |

Metrics: GPU memory, prefill/generation latency, throughput, router overlap, task accuracy.

---

## Roadmap

1. **v0.1** (current): INT8 reference implementation, `QuantLinear`, LLaDA
   expert adapter, metadata, tests.
2. **v0.2**: full model-side integration in the inference project (editable
   install), BF16-fallback toggle, attention projection quantization, and a
   measured trajectory report on the real LLaDA-MoE checkpoint. If INT8 router
   overlap turns out flat across the schedule, that is the finding — it points
   the work at INT4 rather than at more diffusion-specific machinery.
3. **v0.3**: Triton INT8 (W8A16) fused-MoE kernel consuming the packed buffers
   directly, with benchmarks vs BF16.
4. **v0.4**: groupwise INT4 + packing, activation calibration, outlier handling.
5. **v1.0**: reproducible evaluation suite, documentation, frozen checkpoint
   format.

---

## Legal

This repository contains no code from the LLaDA inference engine's
`dInfer` directory. Verify the source model's license before publishing any
derived quantized weights.
