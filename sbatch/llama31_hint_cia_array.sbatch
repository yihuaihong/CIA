#!/bin/bash
# Llama-3.1 hint CIA pipeline (DeepSeek AI label + hint_probe). Depends on
# llama31_hint_eval (writes _8888_results.jsonl).
#
#SBATCH --job-name=llama31_hint_cia
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e
mkdir -p logs

source ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate open_r1
cd ${REPO_ROOT}
source sbatch/_keepalive_init.sh

: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY in env}"

VARIANTS=(acc_only faith_only acc_faith)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
RUN_DIR=hint_llama31_8b_grpo_ablation_${VARIANT}

INPUT=${SCRATCH}/results/open-r1/hint_mmlu_results/hint_mmlu_false_${RUN_DIR}_8888_results.jsonl
PROBE_LABELED=${SCRATCH}/results/open-r1/hint_mmlu_results/hint_mmlu_false_${RUN_DIR}_8888_probe_labeled.jsonl

if [[ ! -f "$INPUT" ]]; then
    echo "Eval jsonl missing: $INPUT"; exit 1
fi

echo "=== Llama-3.1 hint CIA: $RUN_DIR ==="

# 1) Train probe + B_INT label (writes probe_labeled.jsonl, no AI labels yet).
python -m CPF_utils.hint_probe train_and_label \
    --gen_jsonl "$INPUT" \
    --model_dir ${SCRATCH}/open-r1/cia \
    --model_name "$RUN_DIR" \
    --seed 8888 --tau 0.1 \
    --output "$PROBE_LABELED"

# 2) DeepSeek AI label B_CoT in-place (rewrites probe_labeled.jsonl).
python ${REPO_ROOT}/scripts/hint_ai_label_fixup_deepseek.py \
    --input "$PROBE_LABELED" \
    --workers 32

echo "=== DONE: $RUN_DIR ==="
