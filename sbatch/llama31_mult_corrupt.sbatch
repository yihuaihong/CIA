#!/bin/bash
# Llama-3.1 mult corruption labeling for the 3 trained checkpoints. Depends on
# llama31_eval (mult acc eval) producing the source jsonl first.
#
#SBATCH --job-name=llama31_mult_corrupt
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

VARIANTS=(acc_only faith_only acc_faith)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
RUN_DIR=mult_llama31_8b_force_b_rdelta9_${VARIANT}
TRAINED=${SCRATCH}/open-r1/cia/${RUN_DIR}

INPUT=${SCRATCH}/results/open-r1/math_results/2-digit-Multiplication_${RUN_DIR}_use-cot-is-True_8888_results.jsonl
OUTPUT=${SCRATCH}/results/open-r1/math_results/2-digit-Multiplication_${RUN_DIR}_8888_corruption_labeled.jsonl

if [[ ! -f "${TRAINED}/config.json" ]]; then
    echo "Trained model missing: $TRAINED"; exit 2
fi
if [[ ! -f "$INPUT" ]]; then
    echo "Eval jsonl missing: $INPUT — wait for acc eval to finish."; exit 1
fi

echo "=== Mult corruption: $RUN_DIR ==="
python -m CPF_utils.multiplication_corruption \
    --input "$INPUT" \
    --model_dir ${SCRATCH}/open-r1/cia \
    --model_name "$RUN_DIR" \
    --output "$OUTPUT" \
    --device 0 \
    --prompt_variant force_b

echo "=== DONE: $RUN_DIR ==="
