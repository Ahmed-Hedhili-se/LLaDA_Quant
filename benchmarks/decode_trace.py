"""Decode a saved free-running trace to text. No model, no GPU.

The point of storing traces: capture is expensive and needs the hardware, but
every question afterwards should be answerable from JSON. This one is "what did
the model actually say", which needs nothing but a tokenizer.

    python benchmarks/decode_trace.py traces/int4-modeB-reference.json \\
        --weight-dir ~/test_llada/weights

    # both sides of a comparison, diffed
    python benchmarks/decode_trace.py traces/int4-modeB-{reference,quantized}.json \\
        --weight-dir ~/test_llada/weights
"""

from __future__ import annotations

import argparse

from LLaDA_Quant.trajectory import Trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="trace JSON files")
    parser.add_argument("--weight-dir", required=True, help="tokenizer location")
    parser.add_argument("--commit-order", action="store_true",
                        help="also show tokens in the order the decoder committed "
                             "them, which is NOT reading order")
    parser.add_argument("--raw", action="store_true", help="keep special tokens")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)

    for path in args.traces:
        trace = Trace.load(path)
        tokens = trace.final_tokens
        text = tok.decode(tokens, skip_special_tokens=not args.raw)
        print(f"=== {path}")
        print(f"    label={trace.label!r} mode={trace.mode!r} "
              f"steps={len(trace)} committed={len(tokens)}")
        print(f"    {text!r}")
        if args.commit_order:
            scrambled = tok.decode(
                trace.commit_order_tokens, skip_special_tokens=not args.raw
            )
            print(f"    commit order (not reading order): {scrambled!r}")
        print()


if __name__ == "__main__":
    main()
