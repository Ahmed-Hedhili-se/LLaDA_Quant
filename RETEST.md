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

**Run that block first, in every new shell.** Without the venv there is no
`python` on this box (only `python3`, which lacks torch), and without the
exports `--repo $REPO` expands to `--repo --weight-dir`. Confirm with:

```bash
which python && python -c "import torch; print(torch.__version__)" && echo "$REPO"
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
python benchmarks/bench_numerical.py                  # Gaussian weights
python benchmarks/bench_numerical.py --heavy-tailed   # Student-t(3), closer to real weights
```

One distribution per invocation. **Expect**, relative weight L2:

| bits | search | Gaussian | Student-t(3) |
|---|---|---|---|
| 8 | amax | 0.0065 | 0.0139 |
| 8 | mse | 0.0065 (gain 0.0%) | 0.0139 (gain 0.0%) |
| 4 | amax | 0.1174 | 0.2240 |
| 4 | mse | **0.1011** | **0.1969** (gain **12.1%**) |

The `gain` column must show 12-14% at 4 bits and ~0% at 8 -- the search is an
INT4 tool, because 256 levels already absorb an outlier unaided. The `bytes`
columns must be **identical** between the amax and mse rows: the search
changes the value of the scale, never the format.

---

## 4. MoE regime

```bash
python benchmarks/bench_moe_regime.py --machine a6000
```

Machine names are lowercase: `a100`, `a40`, `a40-24q`, `a6000`, `h100`.

**Expect** at batch 1, M/expert of **4.0** (last block) to **16.0** (first
block), every scheme **memory-bound**, and:

```
crossover M/expert (bandwidth-bound below this): BF16=202, W8A16=101, W4A16=50
expert weights BF16 :  12.88 GB   INT8 : 6.44 GB   INT4 : 3.22 GB
```

W8A16's predicted crossover is M=101; the fused kernel measured ~M=170, so
the roofline is conservative by ~1.7x. Rows assume perfect routing balance,
which is **false** on this checkpoint (measured 2.50-6.48x imbalance); pass
`--routing-file` with real `topk_ids` to replace the assumption.

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

**Expect** (reproduced 2026-08-24):

```
  arm                             time     tok/s   vs BF16    resident
  BF16                           5.52s    23.17     1.00x     14032 MiB
  PACKED (dequant/access)       29.48s     4.34     0.19x      8088 MiB
  PACKED + fused W8A16           5.01s    25.53     1.10x      8088 MiB

  PACKED vs PACKED+fused produce identical tokens: True
```

Three things have to hold together, and only together do they mean anything:
**faster than BF16** (1.10x), **on 0.58x the memory** (8088 vs 14032 MiB),
**and bit-identical output** to the unfused quantized path. A fused kernel
that were faster but not identical would be measuring a different model.

`RESULTS.md` records 1.08x; 1.08-1.10x is run-to-run variance, a drop below
1.0x is not.

If the fused arm comes out *slower*, check the weight working set exceeds L2
and that L2 is flushed between runs -- an 8 MiB tile fits in the A6000's cache
and produced a false loss once already.

This arm quantizes at startup on `cuda:0`; it does not read the artifact.

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

### Use the script — detached

```bash
cd ~/LLaDA_Quant
nohup bash tools/run_gsm8k_comparison.sh > ~/gsm8k.log 2>&1 &
tail -f --pid=$! ~/gsm8k.log      # returns by itself when the run ends
```

**Use `--pid=$!`.** A plain `tail -f` never exits, so a finished run leaves
the terminal sitting on a complete log looking hung. `--pid` makes the tail
return the moment the job does.

**Re-running a finished comparison takes seconds** — it reuses both arms and
only reprints the result. That is not a hang. To actually re-grade:

```bash
FORCE=1 nohup bash tools/run_gsm8k_comparison.sh > ~/gsm8k.log 2>&1 &
```

**Run it detached.** A dropped ssh connection SIGHUPs the foreground process
group and takes the script, the grader and the server with it. That already
happened: an n=200 run died at question 127 and lost 25 minutes of grading
with nothing on disk. The script warns if you start it on a TTY with
`LIMIT >= 100`.

For a quick plumbing check first (~4 min, safe in the foreground):

```bash
LIMIT=8 bash tools/run_gsm8k_comparison.sh
```

It activates the venv itself, so it works from a bare login shell and does
not depend on the `export` block at the top of this file. Overridable:
`REPO`, `WEIGHTS`, `ART`, `PORT`, `LIMIT`, `SEED`, `MODE`, `OUT`. For INT4,
pass `ART=~/llada-moe-int4-g128`.

**Re-running resumes.** An arm whose result already covers `LIMIT` questions
is reused rather than re-graded, so a crash costs the arm in flight, not the
finished one. The count has to match and the JSON has to parse, so neither a
leftover smoke run nor a truncated file from a killed grader can stand in for
a real result. Output goes to `~/gsm8k-results/n<LIMIT>/`.

The script also reads `/v1/quantization` on each arm and **aborts if both
report the same label**. Two arms serving the same model produce a
meaningless delta — that happened once, and was caught only because the
outputs were byte-identical.

Below n=100 it prints the delta but refuses the interpretation: at n=8 one
question is 12.5 points, which reads like a finding and is not one.

### Or by hand, one command per shell

Three steps per arm, and the server must be **stopped between arms**. Run
the server in one shell and the grader in another, or background the server.

Use `--execution-mode reference` for accuracy: numerically identical to
`packed` (a test asserts the reconstructed weights are `torch.equal`) but at
BF16 speed, which turns an hour into ~25 minutes. It costs 1.452x memory,
which 48 GB absorbs. Never report REFERENCE as a memory saving.

```bash
# shell 1 -- arm 1, BF16 baseline. Leave it running.
cd ~/LLaDA_Quant && source ~/venv/bin/activate
python benchmarks/serve_quantized.py --repo $REPO --weight-dir $WEIGHTS \
    --no-quantize --port 8000
```

```bash
# shell 2 -- confirm what is serving, then grade
cd ~/LLaDA_Quant && source ~/venv/bin/activate
curl -s http://localhost:8000/v1/quantization | python -c \
  'import json,sys; print(json.load(sys.stdin)["label"])'

cd $REPO && python eval/correctness/run_math_reasoning_code.py \
    --task gsm8k --base-url http://localhost:8000 \
    --limit 200 --seed 42 \
    --max-tokens 1024 --steps 512 --block-length 64 \
    --confidence-threshold 0.9 \
    --output ~/gsm8k-results/gsm8k-bf16.json
```

Then stop the server (`Ctrl-C`, or `pkill -f "[s]erve_quantized"`), start arm 2, and
repeat with a different `--output`:

```bash
python benchmarks/serve_quantized.py --repo $REPO --weight-dir $WEIGHTS \
    --quantized-checkpoint $ART --execution-mode reference --port 8000
```

**Check the label changes between arms.** If both print the same thing, the
server was never restarted and the two result files describe one model.

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
