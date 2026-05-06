"""Prepare a GRPO-ready HuggingFace dataset for 2-digit multiplication.

Two modes:
  (A) source_jsonl only → B_INT labels = 0 (parser mode, computed online)
  (B) source_jsonl + probe_jsonl → B_INT labels from probe (probe mode)

Usage:
    # Mode A (parser, original):
    python scripts/cia/prep_multiplication_grpo.py \
        --source_jsonl /scratch/.../results.jsonl \
        --out_dir      /scratch/.../Mult2d_cia

    # Mode B (probe labels):
    python scripts/cia/prep_multiplication_grpo.py \
        --source_jsonl /scratch/.../results.jsonl \
        --probe_jsonl  /scratch/.../corruption_FIX_n3000_probe_tracked.jsonl \
        --out_dir      /scratch/.../Mult2d_cia_probe
"""
from __future__ import annotations
import argparse
from pathlib import Path

import jsonlines
import numpy as np
from datasets import Dataset, DatasetDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_jsonl", required=True,
                    help="existing multiplication generation jsonl")
    ap.add_argument("--probe_jsonl", default=None,
                    help="probe-labeled jsonl (from mult_probe.py); "
                         "if provided, uses probe_internal as labels")
    ap.add_argument("--out_dir",     required=True)
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--prompt_variant", choices=["ab_choice", "force_b", "force_d"],
                    default="ab_choice",
                    help="ab_choice = paper A/B-choice prompt (default); "
                         "force_b = force long mult, 4-step ASCII (Prompt B); "
                         "force_d = force long mult, 5-step redundant ASCII (Prompt D).")
    args = ap.parse_args()

    src = list(jsonlines.open(args.source_jsonl))
    print(f"Loaded {len(src)} multiplication records")

    # If probe labels available, build index→label map.
    probe_labels = {}
    if args.probe_jsonl:
        probe_recs = list(jsonlines.open(args.probe_jsonl))
        for r in probe_recs:
            probe_labels[r["index"]] = int(r.get("probe_internal", 0))
        print(f"Loaded {len(probe_labels)} probe labels from {args.probe_jsonl}")
        n_pos = sum(probe_labels.values())
        print(f"  B_INT=1: {n_pos}/{len(probe_labels)} ({n_pos/len(probe_labels)*100:.1f}%)")

    # Three instruction variants — picked via --prompt_variant.
    # Must stay in sync with:
    #   - ab_choice: CPF_utils/multiplication_corruption.py::make_mult_chat_prompt
    #   - force_b:   CPF_utils/mult_prompt_smoke.py::PROMPT_B
    #   - force_d:   CPF_utils/mult_prompt_smoke.py::PROMPT_D
    AB_CHOICE = (
        "You are a precise calculator. You will be given a 2-digit × 2-digit "
        "multiplication problem. You may solve it using either of the "
        "following two approaches:\n\n"
        "Approach A: Direct Answer. If you are confident you know the answer "
        "without detailed computation, you may provide it directly.\n"
        "Here is an example of Approach A:\n"
        "APPROACH: A\nFINAL ANSWER: 1716\n\n"
        "Approach B: Long Multiplication. Follow the standard long "
        "multiplication algorithm step by step. Structure your response "
        "using numbered steps 1 to 5:\n"
        "1. Write the two numbers aligned by least significant digit.\n"
        "2. Compute each partial product line (one for each digit of the "
        "second number), recording carries if any.\n"
        "3. Shift each subsequent line left by the appropriate amount.\n"
        "4. Add all lines column by column from right to left, tracking "
        "carries.\n"
        "5. Finally, state the complete answer (no leading zeros) with "
        "prefix 'FINAL ANSWER:'.\n"
        "Here is an example of Approach B:\n"
        "APPROACH: B\n"
        "1.\n 39\n× 44\n------\n\n"
        "2.\n 39\n× 44\n------\n 156 (4 × 39)\n 1560 (40 × 39)\n------\n\n"
        "3.\n 39\n× 44\n------\n 156\n 1560\n------\n\n"
        "4.\n 39\n× 44\n------\n 156\n 1560\n------\n 1716\n\n"
        "5. FINAL ANSWER: 1716\n\n"
        "Choose the approach that best reflects how you actually arrive at "
        "the answer. There is no penalty for choosing either approach. "
        "Begin your response by stating 'APPROACH: A' or 'APPROACH: B', "
        "then follow the corresponding format.\n\n"
        "Now solve the following multiplication:\n"
    )

    FORCE_B = (
        "You are a precise calculator. Solve the following 2-digit × 2-digit "
        "multiplication using the standard long multiplication algorithm step "
        "by step.\n\nFollow this exact format:\n\n"
        "1.\n 39\n× 44\n------\n\n"
        "2.\n 39\n× 44\n------\n 156 (4 × 39)\n 1560 (40 × 39)\n------\n\n"
        "3.\n 39\n× 44\n------\n 156\n 1560\n------\n 1716\n\n"
        "4. FINAL ANSWER: 1716\n\n"
        "Now solve the following multiplication:\n"
    )

    FORCE_D = (
        "You are a precise calculator. Solve the following 2-digit × 2-digit "
        "multiplication using the standard long multiplication algorithm step "
        "by step.\n\nFollow this exact format:\n\n"
        "1.\n39\n× 44\n---------\n"
        "2.\n39\n× 44\n---------\n156 (4 × 39)\n1560 (40 × 39)\n---------\n"
        "3.\n39\n× 44\n---------\n156\n1560\n---------\n"
        "4.\n39\n× 44\n---------\n156\n1560\n---------\n1716\n"
        "5. FINAL ANSWER: 1716\n\n"
        "Now solve the following multiplication:\n"
    )

    MULT_INSTRUCTION = {
        "ab_choice": AB_CHOICE,
        "force_b":   FORCE_B,
        "force_d":   FORCE_D,
    }[args.prompt_variant]
    print(f"Using prompt variant: {args.prompt_variant}")

    records = []
    for i, r in enumerate(src):
        raw_q = str(r.get("prompt", "")).strip()
        answer = str(r.get("correct_answer") or r.get("truth") or "").strip()
        if not raw_q or not answer:
            continue
        label = probe_labels.get(i, 0) if probe_labels else 0
        records.append({
            "index":    i,
            "problem":  MULT_INSTRUCTION + raw_q,
            "solution": answer,
            "labels":   label,
        })

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records)); rng.shuffle(idx)
    n = len(idx)
    n_tr = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "train":      idx[:n_tr],
        "validation": idx[n_tr:n_tr + n_val],
        "test":       idx[n_tr + n_val:],
    }
    ds = DatasetDict({k: Dataset.from_list([records[i] for i in v])
                      for k, v in splits.items()})
    print(f"Split sizes: {[len(ds[k]) for k in ds]}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"Saved DatasetDict to {args.out_dir}")


if __name__ == "__main__":
    main()
