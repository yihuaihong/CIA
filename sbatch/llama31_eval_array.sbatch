#!/bin/bash
# Llama-3.1 ablation eval: acc eval + (TwoHop) cpf eval. Mult acc eval is
# separate because the mult llama31 checkpoints live under a different
# naming convention (mult_llama31_8b_force_b_rdelta9_*).
#
# Array index -> (task, variant):
#   0 = TwoHop acc_only       1 = TwoHop faith_only       2 = TwoHop acc_faith
#   3 = Mult   acc_only       4 = Mult   faith_only       5 = Mult   acc_faith
#
#SBATCH --job-name=llama31_eval
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e
mkdir -p logs

source ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate open_r1
cd ${REPO_ROOT}
source sbatch/_keepalive_init.sh

CONFIGS=(
    "two_hop:two_hop_llama31_8b_grpo_ablation_acc_only:TwoHopFact"
    "two_hop:two_hop_llama31_8b_grpo_ablation_faith_only:TwoHopFact"
    "two_hop:two_hop_llama31_8b_grpo_ablation_acc_faith:TwoHopFact"
    "mult:mult_llama31_8b_force_b_rdelta9_acc_only:2-digit-Multiplication"
    "mult:mult_llama31_8b_force_b_rdelta9_faith_only:2-digit-Multiplication"
    "mult:mult_llama31_8b_force_b_rdelta9_acc_faith:2-digit-Multiplication"
)
TUPLE=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
IFS=':' read -r TASK RUN_DIR DATASET <<< "$TUPLE"
TRAINED=${SCRATCH}/open-r1/cia/${RUN_DIR}

if [[ ! -f "${TRAINED}/config.json" ]]; then
    echo "ERROR: trained model not found at $TRAINED"; exit 2
fi

# Use Llama-3.1 base probe for cache lookup (the trained checkpoint shares
# the same hidden-state geometry).
export PROBE_BASE_MODEL=Llama-3.1-8B-Instruct
export SKIP_PROBE_CACHE_HIT=1
if [[ "$TASK" == "two_hop" ]]; then
    export INTERP_TOOLS=linear_probe
fi

echo "=== Llama-3.1 eval: $TASK / $RUN_DIR ==="
echo "  trained: $TRAINED"
echo "  dataset: $DATASET"

python cpf_evaluation.py --eval_acc --use_cot_prompt \
    --model_dir ${SCRATCH}/open-r1/cia \
    --model_name "$RUN_DIR" \
    --dataset_name "$DATASET" \
    --batch_size 64 --seed 8888

if [[ "$TASK" == "two_hop" ]]; then
    python cpf_evaluation.py --eval_cpf --use_cot_prompt \
        --model_dir ${SCRATCH}/open-r1/cia \
        --model_name "$RUN_DIR" \
        --dataset_name "$DATASET" \
        --batch_size 64 --seed 8888
fi

echo "=== DONE: $RUN_DIR ==="
