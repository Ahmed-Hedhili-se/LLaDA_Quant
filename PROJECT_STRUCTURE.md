# LLaDA_Quant — Project Overview & File-by-File Reference

## What this project is

**LLaDA_Quant** is an independent, framework-style **weight-quantization library**
for LLaDA-style Mixture-of-Experts (MoE) models — and, by extension, any
PyTorch model made of linear layers. It is **weight-only quantization**: it
does not include diffusion decoding, KV caching, serving, or model
orchestration. Your model / inference application stays a separate repository
and consumes this package as a dependency.

**Current state (v0.1):** a symmetric groupwise **INT8** reference
implementation built on the *dequantize-then-matmul* strategy:

- a generic drop-in `QuantLinear` module,
- a LLaDA-specific adapter that quantizes fused MoE expert tensors (`w1`/`w2`)
  in place, without touching the router,
- a versioned checkpoint format (safetensors + JSON manifests) with
  provenance tracking,
- validation metrics (numerical errors + router-overlap diagnostics),
- diffusion-trajectory validation: divergence measured across denoising steps,
  teacher-forced and free-running, with no decoding logic in this repo,
- a benchmark script comparing BF16 vs INT8 latency and memory.

Planned future versions add Triton INT8/INT4 kernels (v0.3), INT4 packing +
activation calibration (v0.4).

---

## Architecture at a glance

```
LLaDA_Quant/
├─ pyproject.toml                     packaging / dependencies / pytest config
├─ README.md                          user-facing documentation
├─ PROJECT_STRUCTURE.md               this file
├─ benchmarks/                        performance measurement scripts
├─ src/LLaDA_Quant/                   the actual library
│  ├─ __init__.py                     public API exports (from .api)
│  ├─ api.py                          high-level entry points
│  ├─ config.py                       serializable QuantConfig + matching rules
│  ├─ adapters/                       model-type-specific integration
│  ├─ algorithms/                     the quantization math
│  ├─ formats/                        checkpoint serialization (disk layout)
│  ├─ runtime/                        runnable quantized modules
│  ├─ validation/                     error/routing metrics, component + trajectory harnesses
└─ tests/unit/                        pytest suite
```

The dependency flow is strict and one-directional:

```
config  ──►  algorithms (symmetric quant math)
                  ▲
                  │
runtime (QuantLinear, QuantExpertWeights)  ──►  adapters ──►  api
formats (safetensors + manifest)            ────────────────────►  api
validation (metrics, compare)                                  (independent)
```

---

## File-by-file role

### Root files

| File | Role |
|---|---|
| `pyproject.toml` | Build config (setuptools, src layout via `[tool.setuptools.packages.find] where=["src"]`), dependencies (`torch>=2.0`, `numpy>=1.24`, `safetensors>=0.4`), optional `dev` extra (pytest, pytest-cov), and pytest settings (`testpaths=["tests"]`, `pythonpath=["src"]`). |
| `README.md` | User-facing documentation: design goals, installation, quick start, API reference, checkpoint format, testing/benchmark instructions, roadmap. |
| `.gitignore` | Ignored files for git. |

### `src/LLaDA_Quant/` — the library

| File | Role |
|---|---|
| `__init__.py` | Re-exports the public API (`QuantConfig`, `QuantizationManifest`, `quantize_model`, `quantized_model`, `save_quantized_checkpoint`, `load_quantized_weights`) and defines `__version__`. |
| `api.py` | High-level entry points. `quantize_model(model, config)` dispatches to the adapters according to `config.targets`; `quantized_model(model, config)` is the non-destructive variant (deep-copies first); re-exports save/load from `formats.safetensors`. |
| `config.py` | Defines the frozen dataclass `QuantConfig` (bits, group_size, targets, exclude patterns, compute/scale dtypes, source provenance) with `to_dict`/`from_dict`/`to_json` so every checkpoint is reproducible. Also the matching policy: `matches()`, `is_excluded()`, `target_matches_targets()` decide which modules get quantized, plus `FORMAT_VERSION` and the default exclude list (router, norm, embed_tokens, lm_head, gate). |

### `src/LLaDA_Quant/algorithms/` — quantization math

| File | Role |
|---|---|
| `symmetric.py` | The core algorithm: **symmetric, zero-point-free, groupwise** quantization. `quantize_tensor()` (W_q = clamp(round(W/s))) with per-group scales along the last axis, per-tensor fallback; `dequantize_tensor()` (fp32 arithmetic then cast, to preserve scale precision in BF16); `qmax_for_bits()`; `pack_int4()`/`unpack_int4()` two's-complement nibble packing (ready for INT4). `QuantResult` dataclass bundles `(q, scale, bits, group_size)` with a `dequantize()` method. This file is the numerical contract future Triton kernels must reproduce. |
| `calibration.py` | Placeholder — activation calibration is planned for v0.4; a no-op in v0.1 since quantization is weight-only and symmetric. |
| `outliers.py` | Placeholder — outlier remediation strategies planned for v0.4 (per-channel handling for large groups or INT4). |
| `__init__.py` | Empty namespace marker. |

### `src/LLaDA_Quant/adapters/` — model integration

| File | Role |
|---|---|
| `llada_moe.py` | LLaDA-specific adapter for `TritonFusedMoEBlock`. Detects fused blocks (`is_fused_expert_block`: 3-D `w1`/`w2` Parameters, even second dim). `quantize_llada_experts()` quantizes both fused tensors, registers **persistent** packed buffers `_qw1/_sw1/_qw2/_sw2` on the block, and materializes dequantized BF16 values back into the live `w1`/`w2` Parameters — the model code stays untouched and the router remains bit-identical. `restore_llada_experts_from_buffers()` re-materializes `w1`/`w2` from the packed buffers after loading a checkpoint. |
| `torch.py` | Generic PyTorch adapter: `replace_linears()` walks `model.named_modules()`, matches names against `config` (respecting `exclude`), and swaps each qualified `nn.Linear` for a `QuantLinear` in place. |
| `__init__.py` | Empty namespace marker. |

### `src/LLaDA_Quant/formats/` — checkpoint serialization

| File | Role |
|---|---|
| `safetensors.py` | Disk layout: `<dir>/model-int8.safetensors` + `quantization.json` + `source-checkpoint.json`. `save_quantized_checkpoint()` writes tensors + manifests; `load_quantized_checkpoint()` reads them without touching any model; `collect_quantized_state()` gathers packed buffers + live params; `_register_missing_buffers()` lets a *plain* (unquantized) model absorb the extra `_q*`/`_s*` buffers; `load_quantized_weights()` is the model-level loader (registers missing buffers, loads state, then re-materializes expert weights). |
| `manifest.py` | Provenance and metadata layer. `QuantizationManifest` (format version, framework version, created-at, source checkpoint, full `QuantConfig`, per-tensor `QuantEntry` list) with dict/JSON roundtrip; `QuantEntry` records tensor name, shape, bits, group size, storage/compute dtypes, source tensor, SHA-256; `tensor_hash()` deterministic value hash; `write_source_checkpoint_meta()` records the unquantized source's path + SHA-256 so artifacts are provably traceable. |
| `__init__.py` | Empty namespace marker. |

### `src/LLaDA_Quant/runtime/` — runnable quantized modules

| File | Role |
|---|---|
| `linear.py` | `QuantLinear`, a drop-in replacement for `nn.Linear` with identical call semantics (`out = qlin(x)`, same in/out features and bias). Storage: `qweight` `[out, in]` int8 + `scale` `[out, num_groups]`. `from_linear()` builds it from an existing module; `dequantize_weight()` reconstructs the weight; `forward()` does dequantize-then-matmul in the compute dtype. This is the correctness reference that the future Triton kernel must match — the public API is designed not to change. |
| `moe.py` | Quantized MoE expert-weight interface. `QuantExpertWeights` (dataclass over two `QuantResult`s for the fused `w1` [E, 2·I, H] and `w2` [E, H, I] tensors) with `quantize()`/`dequantize()`; `quantize_fused_experts()` convenience wrapper; `materialize_expert_params()` copies dequantized values into a module's `w1`/`w2` Parameters (the reference-runtime contract that keeps router behavior untouched). |
| `kernels/__init__.py` | Empty namespace marker — reserved for the Triton kernels planned in v0.3. |
| `__init__.py` | Empty namespace marker. |

### `src/LLaDA_Quant/validation/` — correctness diagnostics

| File | Role |
|---|---|
| `metrics.py` | Numerical + routing metrics: `max_abs_error`, `mean_abs_error`, `max_rel_error`, `cosine_similarity`, `router_overlap` (fraction of (token, rank) slots where two top-k routings agree — the key diagnostic for MoE quantization), `summarize_metrics()` bundling them, plus masked-token metrics for diffusion LMs: `top1_agreement`, `kl_divergence`, `unmask_selection_agreement` (do both models unmask the *same position* next), `top2_margin` and `tie_fraction`. |
| `compare.py` | `compare_models()` harness: given a reference and a quantized model, feeds identical inputs to named submodules (via user-supplied `get_inputs_fn`/`forward_fn`/`router_fn`) and returns a `ComponentReport` per component with metrics + router overlap. |
| `diffusion.py` | Denoising-state description and callback protocols. `DiffusionState` (step, `input_ids`, `mask_positions`), `fully_masked_state()`, `make_masked_states()` (monotone early/middle/late schedule — positions are only ever revealed), `mask_positions_from_ids()`, and the `LogitsFn`/`RouterFn`/`AdvanceFn` type aliases. Contains **no decoding logic**: the caller supplies the callables, so the framework never imports a decoder. |
| `trajectory.py` | Trajectory-level divergence. `compare_trajectory()` is teacher-forced — identical inputs at each state, so differences are quantization alone; it shows per-step sensitivity but cannot show compounding. `compare_free_running()` lets each model denoise through the caller's `advance_fn`, which is where compounding becomes visible. Reports: `StateReport`/`TrajectoryReport`, `FreeRunStep`/`FreeRunReport`, both with `to_dict()` and `to_table()`. |
| `__init__.py` | Re-exports the whole validation surface (`compare_models`, `compare_trajectory`, `compare_free_running`, states and metrics). |

### `src/LLaDA_Quant.egg-info/` — generated

| File | Role |
|---|---|
| `*.egg-info/*` | Auto-generated metadata by `pip install -e .` (PKG-INFO, requires.txt, SOURCES.txt, top_level.txt, dependency_links.txt). Do not edit — regenerate instead. |

### `tests/unit/` — pytest suite

| File | What it checks |
|---|---|
| `test_symmetric.py` | Quantization math: qmax values, zero tensors, roundtrip error budget (< ~1/127), scale == amax/qmax, per-tensor fallback, int4 pack/unpack roundtrip, BF16 inputs. |
| `test_quantlinear.py` | `QuantLinear` vs `nn.Linear` within tolerance, weight/scale layout, per-tensor fallback shape, state_dict roundtrip. |
| `test_llada_moe_adapter.py` | Fused-block detection, expert quantization preserving router/shapes, reproducibility, `QuantExpertWeights` shapes. |
| `test_api.py` | `matches()`/excludes policy, in-place vs non-destructive quantization, MoE-like container quantization. |
| `test_checkpoint_format.py` | Manifest JSON roundtrip, full save/load roundtrip, loading quantized weights into a plain model (buffer registration + w1/w2 re-materialization). |
| `test_validation.py` | Router-overlap metric (incl. shape-mismatch error), metrics values, expert-quantized MoE output error bounds. |
| `test_trajectory.py` | Diffusion-trajectory layer against a toy masked LM: monotone/prompt-safe schedule construction, `DiffusionState` shape validation, masked-token metrics, teacher-forced comparison (identical models, INT8, router-key and logit-shape errors), free-running divergence (no drift for identical models, compounding for a perturbed one, `max_steps` and early-stop handling), and a regression guard that `tie_fraction` flags the degenerate fully-masked state where `top1_agreement` reads 0.0 for no real reason. |

### `benchmarks/`

| File | Role |
|---|---|
| `bench_experts.py` | CLI benchmark (`python benchmarks/bench_experts.py --num-experts 64 --hidden 2048 --intermediate 1024`). Builds synthetic MoE expert weights, quantizes them INT8, and reports a JSON report with BF16 vs INT8 reference-path latency (ms) and weight-memory footprint (~47% of BF16). v0.1 measures the reference dequantize-then-matmul path; the same CLI will benchmark the packed Triton kernel in v0.3. |

---

## How the pieces fit together (typical run)

1. **Config** (`config.py`) describes *what* to quantize: `QuantConfig(bits=8, group_size=128, targets=("expert", "linear"), exclude=(...))`.
2. **Quantize** (`api.quantize_model`):
   - experts → `adapters/llada_moe.py` uses `algorithms/symmetric.py` (via `runtime/moe.py`) to produce packed int8 + scales, stores them as persistent buffers, and replaces live weights with dequantized BF16;
   - linears → `adapters/torch.py` swaps `nn.Linear` → `runtime/linear.QuantLinear`.
3. **Save** (`formats/safetensors.save_quantized_checkpoint`) writes `model-int8.safetensors` (packed buffers + materialized params), `quantization.json` (full manifest incl. config), and `source-checkpoint.json` (provenance + SHA-256).
4. **Load** (`load_quantized_weights`) works on plain *or* already-quantized models: missing `_q*`/`_s*` buffers are registered on the fly, then `w1`/`w2` are re-materialized from the packed buffers — the model is immediately runnable.
5. **Validate** (`validation/`) at two altitudes:
   - `compare.py` compares named submodules on identical inputs — error metrics and, crucially, router overlap;
   - `trajectory.py` compares whole models *across denoising steps*, which is the only way to see the failure mode specific to a masked diffusion LM: a logit shift changes which position is unmasked, that position becomes context, and the error compounds. Always read an agreement number next to its `tie_fraction`.

## Design principles to respect when editing

- **Reference-first correctness**: `symmetric.py` and the dequantize-then-matmul path define the numerical contract; future kernels must reproduce it.
- **Drop-in, zero model changes**: the LLaDA adapter never modifies model code; router/norms/embeddings/LM head stay BF16 by default.
- **Explicit, serializable configuration**: every checkpoint embeds the full config so runs are reproducible.
- **Original weights never mutated**: checkpoints store packed buffers *alongside* materialized params, so the source checkpoint stays pristine.