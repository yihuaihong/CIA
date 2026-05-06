"""Standalone post-processor: read raw hint_mmlu_false_<MODEL>_<seed>_results.jsonl,
call HintAIAgent (Gemini) to label `acknowledge_hint_ai`, and write
`<input_stem>_with_ai_label.jsonl`.

Mirrors the in-process labeling call in `evaluation_utils.run_hint_acc_evaluation`
(which is gated by `gemini_labeler=True` and not exposed via cpf_evaluation CLI).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import jsonlines

sys.path.insert(0, "${REPO_ROOT}")
from CPF_utils.gemini_caller_utils import HintAIAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Raw hint results jsonl (output of cpf_evaluation Hint_MMLU)")
    ap.add_argument("--output", default=None,
                    help="Output path. Default: <input_stem>_with_ai_label.jsonl")
    ap.add_argument("--gemini_model", default="gemini-1.5-flash-latest")
    args = ap.parse_args()

    in_path = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_name(f"{in_path.stem}_with_ai_label.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = list(jsonlines.open(in_path))
    print(f"Loaded {len(samples)} samples from {in_path}")
    print(f"Calling HintAIAgent (model={args.gemini_model}) ...")
    agent = HintAIAgent(model_name=args.gemini_model)
    labeled = agent.label_acknowledgment(samples)

    with jsonlines.open(out_path, "w") as w:
        for r in labeled:
            w.write(r)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
