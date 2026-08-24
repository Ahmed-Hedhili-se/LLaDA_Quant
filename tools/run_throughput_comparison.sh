#!/usr/bin/env bash
#
# Served throughput, every arm, one command.
#
# Answers "what do I actually get after quantizing the weights", measured over
# HTTP against the real server rather than in-process. Three arms:
#
#   bf16     the speed target
#   packed   quantized, dequantize-per-access -- what shipped before the kernel
#   fused    quantized, packed straight into the W8A16 kernel
#
# All three are served from the same launcher, so the only difference is how
# the expert GEMM gets its weights. The quantized arms load the pre-quantized
# artifact, so this also confirms that an artifact serves at the same speed as
# a startup-time quantization -- the tensors are bit-identical, but "should be
# identical" is not a measurement.
#
# Run it DETACHED; the packed arm is slow and a dropped connection kills the run:
#
#     nohup bash tools/run_throughput_comparison.sh > ~/throughput.log 2>&1 &
#     tail -f --pid=$! ~/throughput.log
#
#     ARMS="bf16 fused" bash tools/run_throughput_comparison.sh   # skip the slow arm
#     MAX_TOKENS=128 bash tools/run_throughput_comparison.sh      # shorter generations

set -euo pipefail

REPO="${REPO:-$HOME/test_llada}"
WEIGHTS="${WEIGHTS:-$REPO/weights}"
QUANT="${QUANT:-$HOME/LLaDA_Quant}"
VENV="${VENV:-$HOME/venv}"
ART="${ART:-$HOME/llada-moe-int8-g128}"
PORT="${PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUNS="${RUNS:-2}"
ARMS="${ARMS:-bf16 packed fused}"
OUT="${OUT:-$HOME/throughput-results}"

mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$QUANT"

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

stop_server() {
    pkill -f "[s]erve_quantized" 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -f "[s]erve_quantized" >/dev/null || return 0
        sleep 1
    done
    fail "a serve_quantized process survived pkill; stop it by hand"
}
trap stop_server EXIT

serve_args() {
    case "$1" in
        bf16)   echo "--no-quantize" ;;
        packed) echo "--quantized-checkpoint $ART --execution-mode packed" ;;
        fused)  echo "--quantized-checkpoint $ART --execution-mode packed --fused" ;;
        *)      fail "unknown arm '$1' (expected bf16, packed or fused)" ;;
    esac
}

run_arm() {
    local arm="$1"
    local logfile="$OUT/serve-$arm.log"
    local result="$OUT/throughput-$arm.json"

    log "$arm: starting the server"
    stop_server
    # shellcheck disable=SC2046
    python -u benchmarks/serve_quantized.py \
        --repo "$REPO" --weight-dir "$WEIGHTS" --port "$PORT" $(serve_args "$arm") \
        > "$logfile" 2>&1 &
    local pid=$!

    for _ in $(seq 1 180); do
        curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -30 "$logfile" >&2
            fail "$arm: the server exited during startup (log above)"
        fi
        sleep 5
    done
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 \
        || fail "$arm: server never became healthy"

    # Resident memory, straight from the launcher's own accounting, so the
    # speed number is never reported without the memory it cost.
    grep -E "^resident:|serving BF16 unmodified" "$logfile" | tail -1 || true

    log "$arm: measuring throughput"
    python benchmarks/bench_served_throughput.py \
        --base-url "http://localhost:$PORT" \
        --max-tokens "$MAX_TOKENS" --runs "$RUNS" \
        --json "$result" 2>&1 | tee "$OUT/bench-$arm.log"

    stop_server
}

log "config"
cat <<EOF
  repo        $REPO
  artifact    $ART
  arms        $ARMS
  max_tokens  $MAX_TOKENS   runs $RUNS
  output      $OUT
EOF

for arm in $ARMS; do
    if [ "$arm" != "bf16" ] && [ ! -d "$ART" ]; then
        fail "no artifact at $ART -- run tools/quantize_checkpoint.py first"
    fi
    run_arm "$arm"
done

log "summary"
python - "$OUT" $ARMS <<'PY'
import json, os, sys

out, arms = sys.argv[1], sys.argv[2:]
rows = []
for arm in arms:
    path = os.path.join(out, f"throughput-{arm}.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        rows.append((arm, json.load(f)))

if not rows:
    raise SystemExit("no results to summarise")

base = next((d["aggregate_tok_s"] for a, d in rows if a == "bf16"), None)

print(f"  {'arm':8} {'tok/s':>9} {'vs BF16':>9}   label")
print("  " + "-" * 68)
for arm, d in rows:
    rate = d["aggregate_tok_s"]
    ratio = f"{rate / base:.2f}x" if base else "--"
    print(f"  {arm:8} {rate:9.2f} {ratio:>9}   {d['label']}")

print()
print("  Measured over HTTP against the real server, tokens/second from the")
print("  server's own timing field. The quantized arms were served from a")
print("  pre-quantized artifact, not a startup-time quantization.")
PY

log "done -- results in $OUT"
