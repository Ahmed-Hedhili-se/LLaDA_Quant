"""Throughput of a *running* server, measured over HTTP.

``bench_fused_e2e.py`` measures ``generate_cached`` in-process. This measures
what a client gets: real requests to the real server, through the batch
collector and the FastAPI layer, which is the number a deployment is judged on.

The server reports ``timing.generation_seconds`` and ``usage.completion_tokens``
per response, so tokens/second is read from the server's own accounting rather
than inferred from wall clock. Wall clock is reported alongside it; the gap is
queueing and HTTP overhead.

    python benchmarks/bench_served_throughput.py --base-url http://localhost:8000

It does not start a server. Point it at one, and check ``/v1/quantization``
first to know which model you are timing -- this script prints that label so
the number is never orphaned from the configuration that produced it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

# Fixed prompts so every arm does the same work. Deliberately arithmetic: the
# model answers them without wandering, so completion length stays comparable
# across arms instead of one arm looking fast by generating less.
PROMPTS = [
    "What is 17 times 4? Answer with just the number.",
    "A shop sells pens for $3 each. How much do 12 pens cost? Show your work briefly.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    "Sarah has 24 apples and gives away one third. How many are left?",
    "What is the sum of the first 10 positive integers?",
]


def post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str, timeout: int = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2,
                        help="passes over the prompt set, after one warmup pass")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", default=None, help="write the summary here")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    try:
        label = get(f"{base}/v1/quantization")["label"]
    except (urllib.error.URLError, KeyError, TimeoutError):
        # A plain start.sh server has no such endpoint. Say so rather than
        # reporting a throughput number with no idea what produced it.
        label = "UNKNOWN (no /v1/quantization -- not served by serve_quantized.py)"
    print(f"serving: {label}")
    print(f"prompts: {len(PROMPTS)}  max_tokens: {args.max_tokens}  runs: {args.runs}\n")

    endpoint = f"{base}/v1/chat/completions"

    def one(prompt: str) -> tuple[float, int, float]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
        }
        start = time.perf_counter()
        body = post(endpoint, payload, args.timeout)
        wall = time.perf_counter() - start
        tokens = body["usage"]["completion_tokens"]
        seconds = body.get("timing", {}).get("generation_seconds", wall)
        return seconds, tokens, wall

    print("  warmup ...", flush=True)
    for prompt in PROMPTS:
        one(prompt)

    rates, walls, total_tokens, total_seconds = [], [], 0, 0.0
    for run in range(args.runs):
        for index, prompt in enumerate(PROMPTS):
            seconds, tokens, wall = one(prompt)
            rate = tokens / seconds if seconds else 0.0
            rates.append(rate)
            walls.append(wall)
            total_tokens += tokens
            total_seconds += seconds
            print(f"  run {run + 1} prompt {index + 1}: {tokens:4d} tok in "
                  f"{seconds:6.2f}s = {rate:6.2f} tok/s  (wall {wall:6.2f}s)",
                  flush=True)

    aggregate = total_tokens / total_seconds if total_seconds else 0.0
    summary = {
        "label": label,
        "max_tokens": args.max_tokens,
        "requests": len(rates),
        "total_completion_tokens": total_tokens,
        "aggregate_tok_s": round(aggregate, 3),
        "mean_tok_s": round(statistics.mean(rates), 3),
        "median_tok_s": round(statistics.median(rates), 3),
        "mean_wall_s": round(statistics.mean(walls), 3),
    }
    print("\n" + "=" * 62)
    print(f"  {label}")
    print(f"  aggregate : {aggregate:6.2f} tok/s  "
          f"({total_tokens} tokens / {total_seconds:.2f}s)")
    print(f"  per-request mean {summary['mean_tok_s']:.2f} tok/s, "
          f"median {summary['median_tok_s']:.2f} tok/s")
    print("=" * 62)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
