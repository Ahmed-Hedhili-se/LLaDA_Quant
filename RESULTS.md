# LLaDA-MoE quantization: measured results

Everything here was measured on the real `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`
checkpoint through the inference repository's own model, decoder, server and
grader. Nothing is extrapolated unless it says so.

**Headline:** INT8 halves the expert weights at no measurable accuracy cost.
INT4 saves twice as much again for a probable ~6-point accuracy loss. As
*served* today, both are ~6x *slower* per forward, because the grouped-expert
MoE path still dequantizes into HBM before an unchanged BF16 GEMM — so the
deployed model is a **capacity win, not a speed win**.

That is a property of the wiring, not of quantization, and the kernel side of it
is now closed. `fused_moe_w8a16` ([section 5](#5-speed)) is a grouped-expert MoE
GEMM that consumes INT8 experts directly and is **1.25–1.95x faster than BF16**
at the real checkpoint geometry — and ~10x faster than the dequantize-then-matmul
path it replaces. What remains is wiring it into the served block so the
end-to-end number can be measured.

- [1. Setup](#1-setup)
- [2. Memory](#2-memory)
- [3. Numerical error](#3-numerical-error)
- [4. Accuracy](#4-accuracy)
- [5. Speed](#5-speed)
- [6. Trajectory divergence](#6-trajectory-divergence)
- [7. Routing imbalance](#7-routing-imbalance)
- [8. Should the fused kernel be built?](#8-should-the-fused-kernel-be-built)
- [9. Hardware portability](#9-hardware-portability)
- [10. Corrections made during the investigation](#10-corrections-made-during-the-investigation)
- [11. What is not established](#11-what-is-not-established)

---

## 1. Setup

| | |
|---|---|
| Model | LLaDA-MoE-7B-A1B-Instruct — 16 layers, 64 experts, top-8, H=2048, I=1024 |
| Expert weights | 6.44 B elements (~92% of the model) |
| GPU | NVIDIA RTX A6000, 48 GB, 768 GB/s, 154.8 BF16 TFLOPS |
| Backend | `fast_dense` — the fused Triton MoE path, unmodified |
| Quantized | MoE experts (`w1`, `w2`) only. Router, attention, norms, embeddings and LM head stay BF16 |

Both arms of every comparison were served through the same launcher
(`benchmarks/serve_quantized.py`), so the only difference between them is
quantization — not the serving path. The inference repository was never
modified; it is imported.

---

## 2. Memory

`MEASURED` on the real model, from live tensors.

| | expert weights | vs BF16 |
|---|---|---|
| BF16 | 12288 MiB | — |
| INT8 g128 | **6336 MiB** | **0.516x** |
| INT4 g128 | **3264 MiB** | **0.266x** |

Whole model in PACKED mode: **14032 → 5008 MiB (0.357x)**. The ratio is higher
than 0.266 because embeddings and the LM head (~1.3 GB) stay BF16 by design.

INT4 is genuinely packed two values per byte. The residual above 0.25x is the
fp32 group scales, identical in both widths.

### The two residency modes are not interchangeable

| mode | resident | speed | use for |
|---|---|---|---|
| `PACKED` | **0.357x** | ~6x slower per forward | deployment, memory measurement |
| `REFERENCE` | **1.452x** (larger than BF16) | BF16 speed | accuracy evaluation |

`REFERENCE` keeps the dequantized BF16 alongside the packed buffers. It is
numerically identical to `PACKED` — a test asserts the reconstructed weights are
`torch.equal` — so it gives INT4/INT8 *numerics* at BF16 *speed*, which is what
made a 25-minute GSM8K run possible instead of a multi-hour one. It uses more
memory than not quantizing at all and must never be reported as a saving.

---

## 3. Numerical error

`MEASURED` on synthetic weights, Gaussian / Student-t(3) (heavier tails, closer
to real LLM weight distributions).

| bits | scale search | weight rel. L2 | output cosine | bytes vs BF16 |
|---|---|---|---|---|
| 8 | amax | 0.0065 / 0.0138 | 0.99993 / 0.99971 | 0.516x |
| 8 | mse | 0.0065 / 0.0138 | 0.99993 / 0.99971 | 0.516x |
| 4 | amax | 0.1174 / 0.2248 | 0.97950 / 0.92650 | 0.266x |
| 4 | **mse** | **0.1011 / 0.1978** | **0.98461 / 0.94128** | 0.266x |

**INT4 weight error is ~15x INT8's.** That gap is the whole story of section 4.

**MSE-optimal scale search cuts INT4 error 12–14% for free.** `s = amax/Qmax`
spends the entire grid accommodating the single largest weight in a group; with
256 levels that is nearly free, with 16 it is not. Searching a clipping ratio
that minimises per-group squared error recovers part of it at **zero extra
bytes** — the storage columns are identical, because only the *value* of the
scale changes, not the format. INT8 gains ~0%, as expected: it is an INT4 tool.

---

## 4. Accuracy

`MEASURED` — GSM8K, n=200, seed 42, `max_tokens=1024 steps=512 block_length=64
confidence_threshold=0.9` (the inference repo's recommended config), 4-shot,
chat-templated, greedy.

| config | accuracy | vs BF16 | significance | item churn |
|---|---|---|---|---|
| BF16 | **75.5%** (151/200) | — | — | — |
| INT8 g128 | **73.5%** (147/200) | −2.0 pt | McNemar **p = 0.585** | 15% |
| INT4-MSE g128 | **69.5%** (139/200) | −6.0 pt | unpaired **p = 0.179** | 22%* |

\* churn measured at n=50; per-item results were not captured at n=200 for INT4.

### INT8 is the deployable configuration

13 items fixed, 17 broken, p = 0.585. A paired McNemar cannot distinguish that
from chance. **Half the expert bytes at no measurable accuracy cost.**

### The loss tracks weight precision, not a routing threshold

This was the open question. The checkpoint's router is near-uniform (top-1
weight ~1.7–5%), so top-8 membership sits on a razor-thin boundary that any
numerical noise can flip. If INT4's −6 points came from that mechanism, INT8 —
with 15x lower weight error — should have lost roughly the same amount.

It lost a third as much. So **more bits genuinely buy accuracy here**, and
mixed precision is a real lever rather than a workaround for a cliff.

The relationship is strongly sublinear: **15x the weight error costs 3x the
accuracy**. Most of INT4's error is absorbed.

### Aggregate accuracy hides per-answer instability

15% of answers change under INT8, 22% under INT4 — in **both directions**. At
n=50, INT4 fixed 4 questions BF16 got wrong and broke 7 it got right. If
per-response reproducibility matters for a deployment, that churn is the
finding, not the points.

### The n=50 result was a fluke, in both directions

INT4 scored 64.0% at n=50 and 69.5% at n=200 — the same model on a superset of
the same questions. A 5.5-point swing from sample size alone is a direct
demonstration of why n=50 cannot resolve a 6-point difference.

---

## 5. Speed

`MEASURED` — A6000, batch 1, dense forward (no KV cache).

### Forward latency

| seq | BF16 | INT4-PACKED | ratio | **INT4 − BF16** |
|---|---|---|---|---|
| 32 | 34.31 ms | 282.75 ms | 8.24x | **248.4 ms** |
| 64 | 34.95 ms | 283.22 ms | 8.10x | **248.3 ms** |
| 128 | 43.32 ms | 292.45 ms | 6.75x | **249.1 ms** |
| 256 | 47.47 ms | 298.59 ms | 6.29x | **251.1 ms** |

**The overhead is constant at ~249 ms regardless of token count.** That is the
whole mechanism: dequantization cost scales with *weight* size, not with tokens.
More tokens add GEMM work; the dequantization is already paid.

### Generation throughput

| | seconds | tok/s | vs BF16 |
|---|---|---|---|
| BF16 | 6.08 | **21.06** | — |
| INT4-PACKED | 37.96 | **3.37** | **0.16x** |

### Where the 249 ms goes

Per MoE layer (403 M weights, 768 MiB BF16 → 204 MiB INT4):

| | time |
|---|---|
| touch the BF16 weights | 2.38 ms |
| `unpack_int4` alone | 4.36 ms |
| **full dequantize** | **15.58 ms** (6.5x a BF16 touch) |
| × 16 layers | **249.3 ms** — against 249.0 ms measured |

Written out, the trade is absurd:

> INT4 reads **204 MiB** instead of 768 MiB, saving **0.7 ms** of reading, then
> spends **15.6 ms** expanding it back. Net **+13.2 ms per layer**.

The expansion materialises 768 MiB of BF16 in HBM that the GEMM then reads back,
so total traffic is 204 read + 768 written + 768 read ≈ **1740 MiB against BF16's
768 MiB** — more than double, spread over several eager kernels each with its own
round trip.

**Tensor cores are not the issue.** `fused_moe` receives BF16 weights in both
cases and runs on tensor cores identically. INT4 never reaches a tensor core as
INT4. The entire gap is dequantization sitting in front of an unchanged GEMM.

### Solved: the grouped-expert W8A16 kernel

`MEASURED` -- A6000, real checkpoint geometry (E=64, top-8, H=2048, EI=1024;
768 MiB of BF16 experts, past L2), `tests/unit/test_w8a16_moe.py`.

`runtime/kernels/w8a16_moe.py` applies the standalone GEMM's trick to the
grouped-expert path: same sorting/padding contract as the inference repo's
`fused_moe_kernel`, reusing its `moe_align_block_size`, but `B` loads as INT8
and expands in registers. The BF16 expert weight never reaches HBM.

| M | M/expert | BF16 | dequant+GEMM | **fused W8A16** | vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1 | 0.927 ms | 12.094 ms | 0.928 ms | 1.00x |
| 8 | 1.0 | 1.560 ms | 12.661 ms | **1.249 ms** | **1.25x** |
| 32 | 4.0 | 1.884 ms | 12.999 ms | **1.252 ms** | **1.50x** |
| 64 | 8.0 | 1.920 ms | 13.054 ms | **1.286 ms** | **1.49x** |
| 128 | 16.0 | 2.521 ms | 13.693 ms | **1.341 ms** | **1.88x** |
| 256 | 32.0 | 2.714 ms | 13.807 ms | **1.395 ms** | **1.95x** |
| 512 | 64.0 | 2.854 ms | 14.006 ms | **1.735 ms** | **1.64x** |
| 1024 | 128.0 | 3.321 ms | 14.427 ms | **2.291 ms** | **1.45x** |

Faster than BF16 at every batch except M=1, where the kernel is launch-bound
rather than weight-bound and there is nothing to win. Against the
dequantize-then-matmul path it replaces it is **~10x faster** -- that column is
flat at ~12-14 ms because it is dominated by the dequantize, which does the same
work regardless of how many tokens consume it.

**Correctness.** Compared against dequantize-then-matmul through the inference
repo's own `fused_moe`, so the two arms differ in *only* where the dequantize
happens: relative L2 between 0 and 1.7e-4, cosine 1.0000, at M = 1 to 512.
Bit-exactness is impossible by construction -- the reference rounds each weight
to bf16 then accumulates, this kernel scales an int8 in fp32 and rounds once --
so the assertion is that the difference stays at bf16 rounding level, which it
does by two orders of magnitude.

**Two things this measurement nearly got wrong.** The first benchmark used the
correctness tests' small shapes (E=8, H=512, EI=256), where the experts are
~6 MiB and sit entirely in the A6000's 6 MiB L2. All three arms measured a flat
~0.9 ms and the kernel looked worthless. The win only exists past L2. Second, a
dtype-only guard did not reject packed INT4: `pack_int4` stores two 4-bit values
per byte *in an int8 tensor*, so `dtype == torch.int8` either way, and the kernel
would have read nibbles as whole values with no error. The guard now checks the
K-extent, which packing halves.

**Not yet claimed:** an end-to-end served speedup. This is the kernel in
isolation. Wiring it into `TritonFusedMoEBlock.forward` under
`ExecutionMode.PACKED`, then re-running GSM8K and the throughput harness, is the
next step.

---

### Fixing it: two solutions, measured

`MEASURED` on the A6000, expert-GEMM shape K=2048 N=16384 (64 MiB weight, past
L2 so it streams from HBM as a real forward does), L2 flushed between timings.

| M/expert | BF16 | eager INT8 | **compile** | **fused kernel** |
|---|---|---|---|---|
| 4 | 1.00x | 8.68x slower | 2.21x slower | **1.96x FASTER** |
| 16 | 1.00x | 9.40x slower | 2.37x slower | **1.93x FASTER** |
| 64 | 1.00x | 9.15x slower | 2.28x slower | **1.56x FASTER** |
| 128 | 1.00x | 8.25x slower | 2.17x slower | **1.10x FASTER** |
| 256 | 1.00x | 6.70x slower | 1.93x slower | 0.78x slower |

**Solution 1 — `torch.compile` the dequantize** (`QuantConfig(compile_dequant=True)`).
Fuses the four elementwise kernels into one: 6.1x faster than eager at INT8,
10.1x at INT4, and **bit-identical** on both. But it tops out at 1.22 ms against
a 1.19 ms floor, and is still ~2.2x slower than BF16. It cannot win: the
expanded weight must still land in HBM for ``fused_moe`` to read back, so the
compiled path moves 1280 MiB where BF16 moves 512.

**Solution 2 — dequantize inside the GEMM's K-loop**
(`runtime/kernels/w8a16_gemm.py`). The int8 tile is expanded in registers,
consumed by one `tl.dot`, discarded. No BF16 weight ever exists: 32 KB in
registers instead of 512 MiB in HBM. **1.96x faster than BF16**, 4.3x ahead of
solution 1, rel L2 3.05e-05 against the reference.

**Only solution 2 ever beats the baseline.** Solution 1 is damage limitation —
useful when PACKED mode's memory is needed today, turning ~3.4 tok/s into
~13 tok/s against BF16's 21, but never a win.

The fused kernel crosses back over between M=128 and M=256 (parity near ~170).
The roofline predicted 101, so it is conservative by about 1.7x — close enough
that the regime table can be planned against, and in the safe direction.

**Caveat:** that kernel is a plain GEMM. No `moe_align_block_size`, no routing,
no top-k weighting, **no SiLU epilogue** — and the epilogue holds two B tiles in
flight, roughly doubling register pressure, which is where a 2x can evaporate.
It proves in-register dequantization wins; it is not a drop-in for `fused_moe`.
The realistic path is porting the `use_int8_w8a16` branch back into that kernel
from upstream vLLM, whose signature it already carries, with 3.05e-05 as the
numerical bar.

---

## 6. Trajectory divergence

`MEASURED` — one GSM8K prompt, 128 tokens / 128 steps, greedy, seed 42, BF16 vs
INT4-MSE.

| quantity | INT4 | BF16-vs-BF16 floor |
|---|---|---|
| Mode A mean top-1 agreement | 0.8861 | 1.0000 |
| Mode A mean tie fraction | 0.9733 | 0.0000 |
| Mode B final token agreement | **0.4766** | 1.0000 |
| Mode B first divergence step | **34** of 128 | never |
| Mode B commit-order agreement | **0.2109** | 1.0000 |
| amplification (Mode B / Mode A) | **4.59x** | — |

The noise floor is exactly clean, so the comparison is valid: at temperature 0
`add_gumbel_noise` returns its input unchanged and the whole trajectory is a
deterministic function of the logits.

**Small per-step error, large end-to-end divergence.** Fewer than half the
committed tokens match, commit *order* barely agrees, and divergence is 4.6x the
per-step injected error — errors genuinely compound along the schedule.

**Yet both decodes answered correctly.** Both reached `\boxed{72}` with coherent
reasoning, differing only in how they wrote the intermediate step
(`\frac{48}{2}` versus `\frac{1}{2} \times 48`).

This is the concrete case against text equality as a correctness gate: it would
score this run a failure at 47.7% token agreement while the task is solved
identically.

**Mode A was saturated and is not interpretable here.** A tie fraction of 0.9733
means that at 97% of masked positions the reference's own top-2 margin is
smaller than the perturbation INT4 introduced. The 0.886 top-1 agreement is
almost entirely coin-tosses, not damage. The useful reading is the *gap* between
the two modes, not either number alone.

---

## 7. Routing imbalance

`MEASURED` — per-layer max/mean expert load, recovered by recomputing each
block's `topk(softmax(gate(x)))` through a forward pre-hook.

| GPU | tokens | max/mean across 16 layers |
|---|---|---|
| A40-24Q | 64 | 2.75 – 7.12 |
| A40-24Q | 160 | 2.65 – 6.75 |
| A6000 | 179 | **2.50 – 6.48** |

The inference repository's note that a near-uniform router probably implies mild
imbalance **does not survive measurement**. Two GPUs, three token counts,
consistent result — so this is not small-sample noise.

With mean load ~20 rows per expert, the busiest sees ~130. That matters for
section 8: the busiest expert crosses into compute-bound well before the average
one does.

---

## 8. Should the fused kernel be built?

Weight-only quantization buys latency only where the expert GEMM is
bandwidth-bound. For a top-k MoE the deciding quantity is **tokens per expert per
step** = `M · top_k / E`, which for LLaDA-MoE is `M/8`.

A6000 roofline crossovers (bandwidth-bound below): **BF16 202, W8A16 101,
W4A16 50, W8A8 202** rows per expert.

| workload | M/expert | BF16 | W8A16 | W4A16 |
|---|---|---|---|---|
| batch 1 | 4 – 16 | memory | memory | memory |
| batch 4 | 16 – 64 | memory | memory | **compute** |
| batch 16 | 64 – 256 | **compute** | **compute** | **compute** |
| batch 57 | 228 – 912 | **compute** | **compute** | **compute** |

The BF16 forward is confirmed memory-bound in practice: touching weights runs at
**629 GB/s of the A6000's 768 peak** (82%), and Nsight on the A40 measured the
fused MoE kernel at **81% of theoretical weight-streaming peak**.

### Verdict

**Build it for low-batch latency serving. Do not expect it to help high-batch
throughput.**

A fused kernel reads the 204 MiB packed weights into shared memory, dequantizes
inside the GEMM's inner loop, and never materialises BF16 in HBM — weight traffic
drops **768 → 204 MiB, a 3.8x reduction**. At batch 1, where ~19 of the 34 ms
forward is weight reading, that is a plausible **2–3x speedup**.

Three things cap it:

1. Past the crossover the GEMM is compute-bound and the win vanishes.
2. Quantizing *halves* the crossover (202 → 101 for W8A16), so a workload that
   was bandwidth-bound in BF16 can be compute-bound once quantized.
3. The measured 6.48x routing imbalance pushes the busiest expert over the line
   first, and the slowest expert bounds the step.

Given INT8 now shows no measurable accuracy cost, **a fused INT8 kernel is the
higher-value target than INT4**.

---

## 9. Hardware portability

Identical code, identical seed, identical config, GSM8K n=50:

| GPU | accuracy |
|---|---|
| A40-24Q | 88.0% |
| "machine A" (partial run) | 77.4% |
| **A6000** | **70.0%** |

An 18-point spread across three GPUs. The cause is documented upstream: the
near-uniform router means bf16-level numerical noise flips top-8 expert
membership, and different hardware produces different noise.

**Consequence for this work: only a same-machine BF16-vs-quantized delta is
interpretable.** Comparing a quantized number against an accuracy measured on
different hardware is meaningless. Every result in section 4 was produced on one
machine in one session for exactly this reason.

---

## 10. Corrections made during the investigation

Recording these because several were wrong in ways that would have produced
confident false conclusions.

| Claim | Correction |
|---|---|
| "INT4 will be ~2.5x slower" (traffic model) | Measured **8.9x**. `unpack_int4` allocated ~2 GB of int16 temporaries per expert tensor. Rewritten with int8 arithmetic shifts: 7.2x faster, verified bit-identical over all 256 byte values. |
| "Both models answer ` 27`, the prompt must be wrong" | **Display bug, not the model.** `final_tokens` concatenated in *commit* order; diffusion commits by confidence, so `72` printed as `27` and `Natalia` as `iaNatal`. The model was correct throughout. |
| "Routing imbalance is probably small-sample noise" | Reproduced at **2.50–6.48x** on a second GPU at 3x the token count. Real. |
| "The GEMM is compute-bound like prefill, so weight-only quantization is the wrong lever" | At batch 1 it is firmly **memory-bound** — routing scatters tokens 8-way across 64 experts, so M/expert is 4–16. |
| GSM8K "88.0% baseline" | Not reproducible on this hardware. The baseline here is **75.5%**. |
| Two GSM8K runs labelled bf16 / int4 | **Both were BF16** — the server was never restarted. Caught because the outputs were byte-identical, which is impossible for models that diverge 47.7%. A `/v1/quantization` endpoint now makes the served model checkable over HTTP. |

Tooling-level failures caught by the toolkit's own guards: `verify_replay`
detected a 6-point drift between on-device capture and offline replay (`argmax`
vs `topk` tie-breaking); `tie_fraction` prevented reading Mode A's 0.886 as
damage.

---

## 11. What is not established

- **Statistical significance.** "No detectable cost" is not "no cost". p = 0.585
  (INT8) and p = 0.179 (INT4) mean the tests cannot resolve these gaps at n=200.
  Settling INT4's −6 points needs ~864 questions per arm; the full 1319-item
  GSM8K test set would do it.
- **Any speedup.** No kernel consumes packed weights. Every speed number here is
  a cost.
- **End-to-end serving throughput.** All latency was measured on the dense
  forward at batch 1, not through `generate_cached` with KV caching and batching.
  The ~249 ms tax is per forward, so the cached path pays it more often, not
  less.
- **Whether capacity converts to throughput.** INT4 frees ~9 GB, which permits a
  larger batch. On the A40 that curve saturated by batch 32 (4x the batch bought
  1.62x throughput). Unmeasured on the A6000's 48 GB.
- **Generality.** The toolkit is architecture-agnostic by construction but
  validated on exactly one model. The `QuantLinear` path for dense models is
  unit-tested and never exercised on real weights.
- **Mixed precision.** Now motivated by the section-4 finding that loss tracks
  precision, but not implemented or tested.

---

## Reproduction

```bash
# storage and numerical error (no GPU needed)
python benchmarks/bench_storage.py --num-experts 64 --hidden 2048 --intermediate 1024 --layers 1
python benchmarks/bench_numerical.py --heavy-tailed
python benchmarks/bench_moe_regime.py --machine a6000

# trajectory: routing, commits, output
python benchmarks/bench_bf16_vs_int4.py --repo ~/test_llada \
    --weight-dir ~/test_llada/weights --gen-length 128 --steps 128 \
    --scale-search mse --search-grid 8 --chat-template \
    --build-device cuda:0 --out traces/

# latency
python benchmarks/bench_generation_latency.py --repo ~/test_llada \
    --weight-dir ~/test_llada/weights --build-device cuda:0

# accuracy: serve, verify, evaluate
python benchmarks/serve_quantized.py --repo ~/test_llada \
    --weight-dir ~/test_llada/weights --backend fast_dense \
    --bits 8 --execution-mode reference --port 8000
curl -s http://localhost:8000/v1/quantization      # confirm before evaluating
python eval/correctness/run_math_reasoning_code.py --task gsm8k \
    --limit 200 --seed 42 --max-tokens 1024 --steps 512 --block-length 64 \
    --confidence-threshold 0.9 --base-url http://localhost:8000 \
    --output results/gsm8k-int8-n200.json | tee gsm8k-int8-n200.txt
```
