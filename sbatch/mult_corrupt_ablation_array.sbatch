#!/bin/bash
# Phase 2 Mult CIA: run multiplication_corruption on each trained mult
# checkpoint to produce a corruption_labeled jsonl with B_INT
# (follows_partial_products) and B_CoT (approach=B). Uses the trained
# checkpoint's already-existing eval jsonl as the source of generations.
#
# Each task runs the corruption rollout (corrupt pp1, corrupt pp2, regen
# summation, check whether final differs) — ~30-60 min/task on H100.
#
# Array indices map to trained mult checkpoints:
#   0=Llama3 acc_only   1=Llama3 faith_only   2=Llama3 acc_faith
#   3=Gemma  acc_only   4=Gemma  faith_only   5=Gemma  acc_faith
#   6=Qwen3  acc_only   7=Qwen3  faith_only
#
#SBATCH --job-name=mult_corrupt_ablation
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e
mkdir -p logs

source ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate open_r1
cd ${REPO_ROOT}
source sbatch/_keepalive_init.sh

CONFIGS=(
    "llama3_8b:acc_only"
    "llama3_8b:faith_only"
    "llama3_8b:acc_faith"
    "gemma_9b:acc_only"
    "gemma_9b:faith_only"
    "gemma_9b:acc_faith"
    "qwen3_8b:acc_only"
    "qwen3_8b:faith_only"
)
TUPLE=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
IFS=':' read -r MODEL_TAG VARIANT <<< "$TUPLE"
RUN_DIR=mult_${MODEL_TAG}_grpo_ablation_${VARIANT}
TRAINED_PATH=${SCRATCH}/open-r1/cia/${RUN_DIR}

if [[ ! -f "${TRAINED_PATH}/config.json" ]]; then
    echo "Trained model not found at $TRAINED_PATH"; exit 0
fi

# Source eval jsonl: produced by ablation_eval_array (--eval_acc) earlier.
INPUT=${SCRATCH}/results/open-r1/math_results/2-digit-Multiplication_${RUN_DIR}_use-cot-is-True_8888_results.jsonl
OUTPUT=${SCRATCH}/results/open-r1/math_results/2-digit-Multiplication_${RUN_DIR}_8888_corruption_labeled.jsonl

if [[ ! -f "$INPUT" ]]; then
    echo "Eval jsonl missing: $INPUT — run ablation_eval_array first."; exit 1
fi

echo "============================================="
echo "Mult corruption: $RUN_DIR"
echo "  trained: $TRAINED_PATH"
echo "  input:   $INPUT"
echo "  output:  $OUTPUT"
echo "============================================="

python -m CPF_utils.multiplication_corruption \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model_dir ${SCRATCH}/open-r1/cia \
    --model_name "$RUN_DIR" \
    --prompt_variant force_b \
    --resume

echo "=== DONE: $RUN_DIR ==="
