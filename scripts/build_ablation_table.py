"""Build the GRPO ablation results table.

For each (task, model_tag, variant) trained checkpoint, reads:
  * stats.json from the eval pass (--eval_acc) for task accuracy
  * labeled jsonl (TwoHop only for now, until Hint/Mult CIA pipelines run)
    -> compute_cia for CIA macro-F1

Run AFTER the eval array (sbatch/ablation_eval_array.sbatch) completes.
For Hint/Mult CIA, need to run hint_probe / multiplication_corruption
separately — those columns will print 'pending' until then.

Usage:
    python scripts/build_ablation_table.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

# Bring CPF_utils into path.
sys.path.insert(0, "${REPO_ROOT}")
from CPF_utils import metrics  # type: ignore

RESULTS = Path("${SCRATCH}/results/open-r1")

# tag -> HF-style model name used in base eval filenames.
BASE_MODEL = {
    "llama3_8b":  "Meta-Llama-3-8B-Instruct",
    "llama31_8b": "Llama-3.1-8B-Instruct",
    "gemma_9b":   "gemma-2-9b-it",
    "qwen3_8b":   "Qwen3-8B",
}

# Llama-3.1 mult ablation checkpoints predate the grpo_ablation naming
# convention — they live under mult_llama31_8b_force_b_rdelta9_<variant>.
def _run_dir(task: str, model_tag: str, variant: str) -> str:
    if task == "mult" and model_tag == "llama31_8b":
        return f"mult_llama31_8b_force_b_rdelta9_{variant}"
    return f"{task}_{model_tag}_grpo_ablation_{variant}"

# (task, model_tag, variant); used to derive RUN_DIR_NAME and lookup paths.
# variant="base" reads from the unfine-tuned base-model eval files.
ROWS = [
    ("two_hop", "llama3_8b", "base"),
    ("two_hop", "llama3_8b", "acc_only"),
    ("two_hop", "llama3_8b", "faith_only"),
    ("two_hop", "llama3_8b", "acc_faith"),
    ("two_hop", "gemma_9b",  "base"),
    ("two_hop", "gemma_9b",  "acc_only"),
    ("two_hop", "gemma_9b",  "faith_only"),
    ("two_hop", "gemma_9b",  "acc_faith"),
    ("two_hop", "qwen3_8b",  "base"),
    ("two_hop", "qwen3_8b",  "acc_only"),
    ("two_hop", "qwen3_8b",  "faith_only"),
    ("two_hop", "qwen3_8b",  "acc_faith"),
    ("hint",    "llama3_8b", "base"),
    ("hint",    "llama3_8b", "acc_only"),
    ("hint",    "llama3_8b", "faith_only"),
    ("hint",    "llama3_8b", "acc_faith"),
    ("hint",    "gemma_9b",  "base"),
    ("hint",    "gemma_9b",  "acc_only"),
    ("hint",    "gemma_9b",  "faith_only"),
    ("hint",    "gemma_9b",  "acc_faith"),
    ("hint",    "qwen3_8b",  "base"),
    ("hint",    "qwen3_8b",  "acc_only"),
    ("hint",    "qwen3_8b",  "faith_only"),
    ("hint",    "qwen3_8b",  "acc_faith"),
    ("mult",    "llama3_8b", "base"),
    ("mult",    "llama3_8b", "acc_only"),
    ("mult",    "llama3_8b", "faith_only"),
    ("mult",    "llama3_8b", "acc_faith"),
    ("mult",    "gemma_9b",  "base"),
    ("mult",    "gemma_9b",  "acc_only"),
    ("mult",    "gemma_9b",  "faith_only"),
    ("mult",    "gemma_9b",  "acc_faith"),
    ("mult",    "qwen3_8b",  "base"),
    ("mult",    "qwen3_8b",  "acc_only"),
    ("mult",    "qwen3_8b",  "faith_only"),
    ("mult",    "qwen3_8b",  "acc_faith"),
    # ---- Llama-3.1 (primary model post 2026-05-01) ----
    ("two_hop", "llama31_8b", "base"),
    ("two_hop", "llama31_8b", "acc_only"),
    ("two_hop", "llama31_8b", "faith_only"),
    ("two_hop", "llama31_8b", "acc_faith"),
    ("hint",    "llama31_8b", "base"),
    ("hint",    "llama31_8b", "acc_only"),
    ("hint",    "llama31_8b", "faith_only"),
    ("hint",    "llama31_8b", "acc_faith"),
    ("mult",    "llama31_8b", "base"),
    ("mult",    "llama31_8b", "acc_only"),
    ("mult",    "llama31_8b", "faith_only"),
    ("mult",    "llama31_8b", "acc_faith"),
]


def stats_path(task: str, run_dir: str) -> Path:
    if task == "two_hop":
        # actual pattern: TwoHopFact_<run>_8888_results_8888_stats.json
        return RESULTS / f"twohop_results/TwoHopFact_{run_dir}_8888_results_8888_stats.json"
    if task == "hint":
        return RESULTS / f"hint_mmlu_results/hint_mmlu_false_{run_dir}_8888_results_8888_stats.json"
    if task == "mult":
        return RESULTS / f"math_results/2-digit-Multiplication_{run_dir}_use-cot-is-True_8888_results_8888_stats.json"
    raise ValueError(task)


def results_jsonl_path(task: str, run_dir: str) -> Path:
    """Per-sample eval jsonl (used as fallback to compute acc when stats absent)."""
    if task == "two_hop":
        return RESULTS / f"twohop_results/TwoHopFact_{run_dir}_8888_results.jsonl"
    if task == "hint":
        return RESULTS / f"hint_mmlu_results/hint_mmlu_false_{run_dir}_8888_results.jsonl"
    if task == "mult":
        return RESULTS / f"math_results/2-digit-Multiplication_{run_dir}_use-cot-is-True_8888_results.jsonl"
    raise ValueError(task)


def labeled_path(task: str, run_dir: str) -> Path | None:
    if task == "two_hop":
        return RESULTS / f"twohop_results/TwoHopFact_{run_dir}_8888_results_linear_probe_labeled.jsonl"
    if task == "hint":
        # probe_labeled has BOTH B_INT (probe_internal) and B_CoT (acknowledge_hint_ai);
        # fall back to with_ai_label if probe step pending.
        for pat in (f"*{run_dir}*probe_labeled.jsonl", f"*{run_dir}*with_ai_label.jsonl"):
            candidates = list((RESULTS / "hint_mmlu_results").glob(pat))
            if candidates:
                return candidates[0]
        return None
    if task == "mult":
        candidates = list((RESULTS / "math_results").glob(f"*{run_dir}*corruption_labeled.jsonl"))
        return candidates[0] if candidates else None
    return None


def base_paths(task: str, model_tag: str) -> tuple[Path | None, Path | None, Path | None]:
    """Return (stats, results_jsonl, labeled_jsonl) for a base model."""
    name = BASE_MODEL[model_tag]
    if task == "two_hop":
        return (
            RESULTS / f"twohop_results/TwoHopFact_{name}_8888_results_8888_stats.json",
            RESULTS / f"twohop_results/TwoHopFact_{name}_8888_results.jsonl",
            RESULTS / f"twohop_results/TwoHopFact_{name}_8888_results_linear_probe_labeled.jsonl",
        )
    if task == "hint":
        return (
            RESULTS / f"hint_mmlu_results/hint_mmlu_false_{name}_8888_results_8888_stats.json",
            RESULTS / f"hint_mmlu_results/hint_mmlu_false_{name}_8888_results.jsonl",
            RESULTS / f"hint_mmlu_results/hint_mmlu_false_{name}_8888_probe_labeled.jsonl",
        )
    if task == "mult":
        # Base mult corruption results live as corruption_FIX_force_b_rdelta9_n3000.jsonl.
        return (
            None,  # acc comes from corruption jsonl (full_correct field)
            None,
            RESULTS / f"math_results/2-digit-Multiplication_{name}_8888_corruption_FIX_force_b_rdelta9_n3000.jsonl",
        )
    raise ValueError(task)


def fmt_acc(stats: dict, task: str) -> str:
    if not stats: return "  --  "
    for k in ("accuracy", "full_accuracy", "acc"):
        if k in stats:
            return f"{100 * stats[k]:5.2f}%"
    return "  ??  "


def acc_from_jsonl(task: str, jsonl: Path) -> float | None:
    """Fallback: compute accuracy directly from per-sample jsonl when no stats file."""
    if not jsonl.exists():
        return None
    import jsonlines
    correct, total = 0, 0
    with jsonlines.open(str(jsonl), 'r') as r:
        for rec in r:
            total += 1
            if task == "mult":
                if rec.get("full_correct") or rec.get("correct"):
                    correct += 1
            elif task == "two_hop":
                # is_correct or correct field
                if rec.get("is_correct") or rec.get("correct"):
                    correct += 1
            elif task == "hint":
                # Hint records have pred_unbiased (model's answer w/o hint) +
                # correct_answer; accuracy is unbiased correctness.
                pu = rec.get("pred_unbiased")
                ca = rec.get("correct_answer")
                if pu is not None and ca is not None and pu == ca:
                    correct += 1
    return correct / total if total else None


def main():
    print(f"\n{'task':>8s} {'model':>10s} {'variant':>11s}  {'acc':>7s}  {'CIA':>7s}  {'F1+':>5s}  {'F1-':>5s}  {'n':>5s}  src")
    print("-" * 92)
    for task, model_tag, variant in ROWS:
        if variant == "base":
            sp, jp_override, lp_override = base_paths(task, model_tag)
        else:
            sp = stats_path(task, _run_dir(task, model_tag, variant))
            jp_override = lp_override = None
        run_dir = _run_dir(task, model_tag, variant)
        stats = {}
        if sp and sp.exists():
            try:
                stats = json.loads(sp.read_text())
            except Exception:
                stats = {}
        # For TwoHop base, use stats final_acc directly.
        if variant == "base" and task == "two_hop" and stats:
            acc_str = f"{100 * stats.get('final_acc', 0):5.2f}%"
        elif variant == "base" and task == "hint" and stats:
            for k in ("accuracy", "full_accuracy", "acc"):
                if k in stats:
                    acc_str = f"{100 * stats[k]:5.2f}%"; break
            else:
                acc_str = "  --  "
        else:
            acc_str = fmt_acc(stats, task)
        if acc_str.strip() in ("--", "??"):
            # fallback: compute from jsonl
            jp = jp_override if jp_override is not None else results_jsonl_path(task, run_dir)
            # For base mult: acc is computed from the corruption jsonl (full_correct).
            if variant == "base" and task == "mult" and lp_override is not None:
                jp = lp_override
            # For base hint: 8888_results.jsonl may be missing — fall back to probe_labeled.
            if variant == "base" and task == "hint" and (jp is None or not jp.exists()) and lp_override is not None:
                jp = lp_override
            a = acc_from_jsonl(task, jp) if jp else None
            if a is not None:
                acc_str = f"{100 * a:5.2f}%"
            else:
                print(f"{task:>8s} {model_tag:>10s} {variant:>11s}  {'pending':>7s}  {'pending':>7s}  {'--':>5s}  {'--':>5s}  {'--':>5s}  no eval yet")
                continue

        # CIA
        lp = lp_override if lp_override is not None else labeled_path(task, run_dir)
        if lp and lp.exists():
            try:
                task_metric = "two_hop" if task == "two_hop" else ("hint" if task == "hint" else "multiplication")
                cia = metrics.compute_cia_from_jsonl(str(lp), task=task_metric)
                cia_str = f"{cia['cia']:.3f}"
                f1p = f"{cia['f1_pos']:.3f}"
                f1n = f"{cia['f1_neg']:.3f}"
                n = str(cia['n'])
                src = lp.name
            except Exception as e:
                cia_str = "err"
                f1p = f1n = n = "--"
                src = f"ERROR: {e}"
        else:
            cia_str = "pending" if task != "two_hop" else "no_lbl"
            f1p = f1n = n = "--"
            src = "labeling pipeline pending" if task != "two_hop" else "missing"

        print(f"{task:>8s} {model_tag:>10s} {variant:>11s}  {acc_str:>7s}  {cia_str:>7s}  {f1p:>5s}  {f1n:>5s}  {n:>5s}  {src}")


if __name__ == "__main__":
    main()
