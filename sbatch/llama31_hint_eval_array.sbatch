#!/bin/bash
# Llama-3.1 hint acc eval (3 variants). Depends on 7707021_[3-5] training.
#
#SBATCH --job-name=llama31_hint_eval
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
RUN_DIR=hint_llama31_8b_grpo_ablation_${VARIANT}
TRAINED=${SCRATCH}/open-r1/cia/${RUN_DIR}

if [[ ! -f "${TRAINED}/config.json" ]]; then
    echo "Trained model missing: $TRAINED — likely training not done yet"; exit 2
fi

export PROBE_BASE_MODEL=Llama-3.1-8B-Instruct
export SKIP_PROBE_CACHE_HIT=1

echo "=== Llama-3.1 hint eval: $RUN_DIR ==="
python cpf_evaluation.py --eval_acc --use_cot_prompt \
    --model_dir ${SCRATCH}/open-r1/cia \
    --model_name "$RUN_DIR" \
    --dataset_name Hint_MMLU \
    --batch_size 64 --seed 8888

echo "=== DONE: $RUN_DIR ==="
