#!/bin/bash
# Llama-3.1 ablation array: 6 GRPO runs (TwoHop x 3 + Hint x 3 reward variants).
# Mult Llama-3.1 already trained at ${SCRATCH}/open-r1/cia/mult_llama31_8b_force_b_rdelta9_*.
#
# Array index -> (task, variant):
#   0=TwoHop acc_only  1=TwoHop faith_only  2=TwoHop acc_faith
#   3=Hint   acc_only  4=Hint   faith_only  5=Hint   acc_faith
#
#SBATCH --job-name=grpo_llama31_ablation
#SBATCH --account=torch_pr_520_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -e
mkdir -p logs

CONFIGS=(
    "ablation/two_hop_llama31_8b_acc_only.yaml"
    "ablation/two_hop_llama31_8b_faith_only.yaml"
    "ablation/two_hop_llama31_8b_acc_faith.yaml"
    "ablation/hint_llama31_8b_acc_only.yaml"
    "ablation/hint_llama31_8b_faith_only.yaml"
    "ablation/hint_llama31_8b_acc_faith.yaml"
)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
ACC_CFG=recipes/accelerate_configs/zero2.yaml

echo "============================================="
echo "Llama-3.1 ablation array task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "============================================="

export CONFIG="recipes/CIA/grpo/${CONFIG}"
export ACC_CFG
exec bash sbatch/grpo.sbatch
