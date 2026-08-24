#!/usr/bin/env bash
#
# GSM8K, BF16 vs quantized, both arms end to end, one command.
#
# The manual procedure is three commands per arm that cannot be pasted as a
# block, because the server runs in the foreground and never returns. Doing it
# by hand also has a failure mode that already happened once: forgetting to
# restart the server between arms produced two byte-identical result files that
# looked like a clean "quantization changed nothing" finding. This script
# starts each server, waits for it, records which model is actually serving,
# runs the grader, stops the server, and refuses to report a comparison if the
# two arms served the same model.
#
# Usage -- run it DETACHED. A dropped ssh connection SIGHUPs the foreground
# process group and takes the run with it; that already cost one n=200 arm at
# question 127 of 200.
#
#     nohup bash tools/run_gsm8k_comparison.sh > ~/gsm8k.log 2>&1 &
#     tail -f --pid=$! ~/gsm8k.log
#
# The --pid matters: plain 'tail -f' never exits, so when the run finishes the
# terminal just sits there and looks hung. With --pid the tail returns as soon
# as the job does.
#
# Other forms:
#
#     LIMIT=8 bash tools/run_gsm8k_comparison.sh            # smoke run, ~4 min
#     ART=~/llada-moe-int4-g128 bash tools/...              # grade INT4 instead
#
# Re-running after a crash reuses any arm whose result already covers LIMIT
# questions, so an interrupted run resumes rather than restarting. Results go
# to ~/gsm8k-results/n<LIMIT>/, keyed by size so a smoke run can never be
# mistaken for a real one.
#
# Everything is overridable by environment variable; the defaults match the
# layout on the A6000 box.

set -euo pipefail

REPO="${REPO:-$HOME/test_llada}"
WEIGHTS="${WEIGHTS:-$REPO/weights}"
QUANT="${QUANT:-$HOME/LLaDA_Quant}"
VENV="${VENV:-$HOME/venv}"
ART="${ART:-$HOME/llada-moe-int8-g128}"
PORT="${PORT:-8000}"
LIMIT="${LIMIT:-200}"
SEED="${SEED:-42}"
OUT="${OUT:-$HOME/gsm8k-results/n$LIMIT}"
# FORCE=1 re-grades even when a matching result exists. Without it, a repeat
# run of a finished comparison completes in seconds and only reprints -- which
# looks indistinguishable from a hang if you are watching with plain "tail -f".
FORCE="${FORCE:-0}"

# The inference repo's recommended config -- the one section 4 of RESULTS.md
# was measured with. Changing these makes the numbers incomparable.
MAX_TOKENS="${MAX_TOKENS:-1024}"
STEPS="${STEPS:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
CONFIDENCE="${CONFIDENCE:-0.9}"

# REFERENCE, not PACKED: numerically identical (a test asserts the
# reconstructed weights are torch.equal) but at BF16 speed. PACKED would pay
# the ~250 ms per-forward dequantization tax across ~12,800 forwards.
MODE="${MODE:-reference}"

# FUSED=1 grades the fused W8A16 kernel instead of the dequantize path. This is
# a genuinely different computation, not just a different residency: the kernel
# dequantizes inside the GEMM's K-loop and accumulates in fp32, where REFERENCE
# and PACKED both dequantize to bf16 in HBM and hand cuBLAS a bf16 matrix. Same
# weights, different arithmetic -- so its accuracy has to be measured, not
# inherited from the REFERENCE run.
FUSED="${FUSED:-0}"
if [ "$FUSED" = "1" ]; then
    MODE="packed"
    QUANT_EXTRA="--fused"
    OUT="${OUT:-$HOME/gsm8k-results/n$LIMIT-fused}"
else
    QUANT_EXTRA=""
fi

mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$QUANT"

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

stop_server() {
    # The bracket stops the pattern from matching this script's own pkill.
    pkill -f "[s]erve_quantized" 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -f "[s]erve_quantized" >/dev/null || return 0
        sleep 1
    done
    fail "a serve_quantized process survived pkill; stop it by hand"
}

wait_for_server() {
    local logfile="$1" arm="$2"
    for _ in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        # Do not wait out the full timeout on a server that already died.
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -30 "$logfile" >&2
            fail "$arm: the server exited during startup (log above)"
        fi
        sleep 5
    done
    tail -30 "$logfile" >&2
    fail "$arm: the server did not become healthy within 15 minutes"
}

# run_arm <name> <extra serve_quantized args...>
run_arm() {
    local arm="$1"; shift
    local logfile="$OUT/serve-$arm.log"
    local result="$OUT/gsm8k-$arm.json"

    # Resume. An arm takes ~25 min at n=200; losing a finished one because the
    # next stage died -- or because an ssh connection dropped -- is the
    # expensive failure. A result is only reused if it graded the same number
    # of questions, so a leftover smoke run never stands in for a real one.
    if [ -f "$result" ] && [ "$FORCE" != "1" ]; then
        local done_n
        done_n=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['total'])"                  "$result" 2>/dev/null || echo 0)
        if [ "$done_n" = "$LIMIT" ]; then
            log "$arm: reusing $result ($done_n questions already graded)"
            echo "   nothing to run. FORCE=1 re-grades from scratch."
            return 0
        fi
        log "$arm: discarding $result -- it graded $done_n, this run wants $LIMIT"
    fi

    log "$arm: starting the server"
    stop_server
    python -u benchmarks/serve_quantized.py \
        --repo "$REPO" --weight-dir "$WEIGHTS" --port "$PORT" "$@" \
        > "$logfile" 2>&1 &
    SERVER_PID=$!
    wait_for_server "$logfile" "$arm"

    # What is actually being served, asked over HTTP rather than assumed from
    # the command line. This is the check that catches a missed restart.
    local label
    label=$(curl -s "http://localhost:$PORT/v1/quantization" \
            | python -c 'import json,sys; print(json.load(sys.stdin)["label"])')
    printf '   serving: %s\n' "$label"
    echo "$label" > "$OUT/label-$arm.txt"

    log "$arm: grading GSM8K (limit=$LIMIT) -- this takes ~25 min at n=200"
    ( cd "$REPO" && python eval/correctness/run_math_reasoning_code.py \
        --task gsm8k --base-url "http://localhost:$PORT" \
        --limit "$LIMIT" --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" --steps "$STEPS" \
        --block-length "$BLOCK_LENGTH" --confidence-threshold "$CONFIDENCE" \
        --config-name "$label" \
        --output "$result" ) 2>&1 | tee "$OUT/eval-$arm.log"

    stop_server
    [ -f "$result" ] || fail "$arm: the grader wrote no $result"
}

log "config"
cat <<EOF
  repo        $REPO
  weights     $WEIGHTS
  artifact    $ART
  limit       $LIMIT   seed $SEED
  generation  max_tokens=$MAX_TOKENS steps=$STEPS block_length=$BLOCK_LENGTH threshold=$CONFIDENCE
  quant mode  $MODE
  output      $OUT
EOF

[ -d "$ART" ] || fail "no artifact at $ART -- run tools/quantize_checkpoint.py first"

# A dropped ssh connection sends SIGHUP to the foreground process group and
# takes the whole run with it, servers included. At n=200 that is ~50 minutes
# of GPU time lost to a network blip -- which has already happened once, at
# question 127 of 200.
if [ -t 1 ] && [ "$LIMIT" -ge 100 ]; then
    printf '
[33m%s[0m
' "WARNING: attached to a terminal, and this run takes ~50 minutes."
    echo "  A dropped ssh connection will kill it. Prefer:"
    echo
    echo "      nohup bash tools/run_gsm8k_comparison.sh > ~/gsm8k.log 2>&1 &"
    echo "      tail -f ~/gsm8k.log"
    echo
    echo "  Re-running after a crash reuses whichever arms already finished."
    echo
fi

run_arm bf16 --no-quantize
# shellcheck disable=SC2086
run_arm quant --quantized-checkpoint "$ART" --execution-mode "$MODE" $QUANT_EXTRA

# Two arms that served the same model produce a meaningless delta. This is not
# hypothetical: it happened, and was only caught because the outputs were
# byte-identical, which is impossible for models that actually diverge.
if [ "$(cat "$OUT/label-bf16.txt")" = "$(cat "$OUT/label-quant.txt")" ]; then
    fail "both arms served '$(cat "$OUT/label-bf16.txt")' -- the delta would be meaningless"
fi

log "comparison"
python - "$OUT/gsm8k-bf16.json" "$OUT/gsm8k-quant.json" <<'PY'
import json, sys

# Schema written by run_math_reasoning_code.py: accuracy is a fraction,
# alongside integer correct/total.
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))

if a['total'] != b['total']:
    raise SystemExit(
        f"arms graded different question counts ({a['total']} vs {b['total']}); "
        "the delta is not a paired comparison"
    )

pa, pb = a['accuracy'] * 100, b['accuracy'] * 100
print(f"  BF16       {pa:6.2f}%  ({a['correct']}/{a['total']})")
print(f"  quantized  {pb:6.2f}%  ({b['correct']}/{b['total']})")
print(f"  delta      {pb - pa:+6.2f} pt")
print()

# A small run is a plumbing check, not a measurement. n=8 moves 12.5 points
# per question, and INT4 already demonstrated the trap: 64.0% at n=50 and
# 69.5% at n=200 on a superset of the same questions.
if a['total'] < 100:
    step = 100 / a['total']
    print(f"  *** n={a['total']} is a SMOKE RUN, not a result.")
    print(f"      One question is worth {step:.1f} points here, so this delta")
    print("      says nothing about accuracy -- it says the harness runs.")
    print("      Resolving a 2-point gap needs the full 1319-item set;")
    print("      a 6-point gap needs roughly 864 questions per arm.")
    raise SystemExit(0)
print()
print("  Expected at n=200: BF16 75.5%, INT8 73.5% -- a -2.0 pt gap that",
      "McNemar puts at p = 0.585,")
print("  i.e. not distinguishable from chance. RESULTS.md section 4.")
print()
print("  The aggregate hides the finding that matters for deployment: ~15% of",
      "answers change")
print("  under INT8, in BOTH directions (13 fixed, 17 broken). If per-response",
      "reproducibility")
print("  matters, that churn is the result, not the two points.")
PY

log "done -- results in $OUT"
