# Re-running every measurement

Each block states the **expected result**, so a run can be judged pass/fail
instead of merely "it produced numbers". Values come from the A6000 runs
recorded in `RESULTS.md`; treat a deviation as a finding, not a rounding error.

Paths assume the layout on the GPU box:

```bash
export REPO=~/test_llada
export WEIGHTS=$REPO/weights
export ART=~/llada-moe-int8-g128        # the pre-quantized artifact
cd ~/LLaDA_Quant && source ~/venv/bin/activate
```

Sections 1–4 need no GPU and no weights. Sections 5–9 need both.

- [1. Unit tests](#1-unit-tests)
- [2. Storage](#2-storage)
- [3. Numerical error](#3-numerical-error)
- [4. MoE regime](#4-moe-regime)
- [5. Offline quantization](#5-offline-quantization)
- [6. Determinism](#6-determinism-offline--startup)
- [7. Throughput](#7-throughput)
- [8. Correctness — GSM8K](#8-correctness--gsm8k)
- [9. Trajectory divergence](#9-trajectory-divergence)

---

## 1. Unit tests

**Set `PYTHONPATH` to the inference repo.** Without it 15 tests skip with
"inference repository not importable" — including every fused-kernel and
fused-block test, which are the ones most worth running.

```bash
PYTHONPATH=$REPO python -m pytest tests -q
```

**Expect: `259 passed`, zero skips.**

Without `PYTHONPATH`, `244 passed, 15 skipped` — correct, but it has not tested
the kernel. On a machine with no GPU, expect two further skips (the CUDA
device-placement tests in `test_checkpoint_format.py`).

The kernel tests print a table as they run; every row should say `FASTER` up to
about M=128.

---

## 2. Storage

```bash
python benchmarks/bench_storage.py
```

**Expect:** INT8 `0.516x`, INT4 `0.266x` of BF16 expert bytes, and every
REFERENCE row flagged `<- LARGER than BF16` (that mode is not a saving).

---

## 3. Numerical error

```bash
python benchmarks/bench_numerical.py --heavy-tailed
```

**Expect**, relative weight L2:

| bits | search | Gaussian / Student-t(3) |
|---|---|---|
| 8 | amax | 0.0065 / 0.0138 |
| 4 | amax | 0.1174 / 0.2248 |
| 4 | mse | **0.1011 / 0.1978** |

The MSE row must beat the amax row at 4 bits by 12–14%, and by ~0% at 8 bits.
Storage bytes must be **identical** between the two — the search changes the
value of the scale, never the format.

---

## 4. MoE regime

```bash
python benchmarks/bench_moe_regime.py --machine RTX_A6000
```

**Expect:** M/expert of 4–16 at batch 1, verdict **memory-bound**, and a
roofline crossover near M=101. (Measured crossover is ~M=170, so the roofline
is conservative by ~1.7x — see `RESULTS.md` section 8.)

---

## 5. Offline quantization

```bash
python tools/quantize_checkpoint.py \
    --repo $REPO --weight-dir $WEIGHTS \
    --out $ART \
    --bits 8 --group-size 128 --scale-search mse \
    --build-device cuda:0 --quantize-device cuda:0 --overwrite
```

**Expect:**

```
quantizing on cuda:0: ~13 s
converted weights: 12288.00 MiB -> 6336.00 MiB (0.516x)
211 tensors verified bit-exact, 32 re-derivable tensors correctly absent
source checkpoint :    13.71 GiB
quantized artifact:     7.89 GiB (0.576x)
```

The read-back check runs automatically. `32 re-derivable tensors correctly
absent` is the guard against the old bug where the "quantized" checkpoint came
out 1.52x the size of the unquantized one.

For INT4 instead: `--bits 4` (expect `3264 MiB`, `0.266x`).

### Load it back

```bash
python benchmarks/serve_quantized.py \
    --repo $REPO --weight-dir $WEIGHTS \
    --quantized-checkpoint $ART --execution-mode packed --fused --port 8000
```

**Expect** `resident: 13.70 GiB -> 7.89 GiB (0.576x)` and
`PACKED + fused W8A16: 16 expert blocks`. Then, from another shell:

```bash
curl -s http://localhost:8000/v1/quantization | python3 -m json.tool
```

**Expect** `"fused_kernel": true`, `"fused_blocks": 16`,
`"weights_from": "pre-quantized artifact"`.

Two guards worth re-exercising:

```bash
# must be REJECTED: --bits 4 against an INT8 artifact
python benchmarks/serve_quantized.py --repo $REPO --weight-dir $WEIGHTS \
    --quantized-checkpoint $ART --bits 4 --execution-mode packed --port 8001

# must load the same file at 1.452x, not 0.576x
python benchmarks/serve_quantized.py --repo $REPO --weight-dir $WEIGHTS \
    --quantized-checkpoint $ART --execution-mode reference --port 8002
```

---

## 6. Determinism: offline == startup

The claim that an artifact is interchangeable with startup-time quantization
fails **silently** if it is wrong — different scales still load, still run, and
still produce plausible text. So it gets checked, not assumed.

```bash
python tools/check_determinism.py $ART \
    --repo $REPO --weight-dir $WEIGHTS
```

**Expect:**

```
compared      : 211 tensors (64 packed expert buffers)
mismatched    : 0
VERDICT       : IDENTICAL - offline == startup
```

Exit code is non-zero on any mismatch, so this is CI-safe.

---

## 7. Throughput

### End to end, three arms in one process

```bash
python benchmarks/bench_fused_e2e.py --repo $REPO --weight-dir $WEIGHTS \
    --gen-length 128 --steps 128 --block-length 32 --runs 3
```

**Expect:** BF16 as the reference, PACKED (dequantize-per-access) ~6x slower,
PACKED+fused **~1.08x faster than BF16** — quantized and faster at the same
time, on 0.58x the memory.

If the fused arm comes out *slower*, check the weight working set is larger
than L2 and that L2 is flushed between runs; an 8 MiB tile fits in the A6000's
cache and produced a false loss once already.

### Forward and generation latency, INT4 dequantize path

```bash
python benchmarks/bench_generation_latency.py --repo $REPO --weight-dir $WEIGHTS \
    --build-device cuda:0 --seq-lengths 32,64,128,256 --json latency.json
```

**Expect** a **constant ~249 ms** per-forward overhead independent of sequence
length — that is the signature of a cost that scales with weight size rather
than token count. Generation: BF16 ~21.1 tok/s, INT4-PACKED ~3.4 tok/s (0.16x).

This benchmark measures the **unfused** path on purpose. It is the "before"
number that section 7's fused arm is compared against.

---

## 8. Correctness — GSM8K

Two arms, **each needing its own server restart**. Forgetting to restart once
produced two byte-identical result files that looked like a clean "quantization
changed nothing" finding — hence the `/v1/quantization` check below.

Use `--execution-mode reference` for accuracy: numerically identical to
`packed` (a test asserts the reconstructed weights are `torch.equal`) but at
BF16 speed, which turns an hour into ~25 minutes. It costs 1.452x memory, which
48 GB absorbs.

### Arm 1 — BF16 baseline

```bash
python benchmarks/serve_quantized.py --repo $REPO --weight-dir $WEIGHTS \
    --no-quantize --port 8000
```

### Arm 2 — INT8

```bash
python benchmarks/serve_quantized.py --repo $REPO --weight-dir $WEIGHTS \
    --quantized-checkpoint $ART --execution-mode reference --port 8000
```

### Verify which model is actually up, then grade

```bash
curl -s http://localhost:8000/v1/quantization | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d["label"])'

cd $REPO && python eval/correctness/run_math_reasoning_code.py \
    --task gsm8k --base-url http://localhost:8000 \
    --limit 200 --seed 42 \
    --max-tokens 1024 --steps 512 --block-length 64 \
    --confidence-threshold 0.9 \
    --output results/gsm8k-bf16.json      # rename per arm
```

**Expect:** BF16 **75.5%** (151/200), INT8 **73.5%** (147/200), −2.0 pt,
McNemar **p = 0.585** — not distinguishable from chance. INT4-MSE lands near
**69.5%**.

Two things worth reading beyond the headline:

- **Item churn.** ~15% of answers change under INT8 in *both* directions (13
  fixed, 17 broken). If per-response reproducibility matters, that churn is the
  finding, not the two points.
- **Sample size.** INT4 scored 64.0% at n=50 and 69.5% at n=200 on a superset
  of the same questions. n=50 cannot resolve a 6-point difference; settling
  INT4 needs the full 1319-item set.

The 82.41% in the harness and the 88% seen earlier are **not** reproducible on
this hardware; the baseline here is 75.5%. Only same-machine BF16-vs-quantized
deltas are interpretable.

---

## 9. Trajectory divergence

Three runs, and all three are needed to read any of them — the BF16-vs-BF16
noise floor is not optional.

```bash
# sanity-check the harness first, quantizing nothing
python benchmarks/bench_bf16_vs_int4.py --repo $REPO --weight-dir $WEIGHTS \
    --build-device cuda:0 --gen-length 32 --steps 32 --noise-floor-only

# the real comparison
python benchmarks/bench_bf16_vs_int4.py --repo $REPO --weight-dir $WEIGHTS \
    --build-device cuda:0 --gen-length 32 --steps 32 \
    --scale-search mse --seed 42 --out traces/
```

**Expect** the floor run to be *exactly* clean: `mean_top1_agreement == 1.0`,
`mean_tie_fraction == 0.0`, `first_divergence_step == -1`. If it is not, the
harness is non-deterministic and the comparison run means nothing.

Read a trace back without a GPU:

```bash
python benchmarks/decode_trace.py traces/modeB-quantized.json --weight-dir $WEIGHTS
```

Tokens print in **position** order. `--commit-order` shows confidence order
instead — useful, but it is what once made `72` print as `27`.

---

## Cleanup

```bash
pkill -f "[s]erve_quantized"
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect ~1 MiB
```

The bracketed `[s]` stops the pattern from matching the killing shell itself.
