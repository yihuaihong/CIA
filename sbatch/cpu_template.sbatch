#!/bin/bash
# CPU-only template — for tasks that DON'T need a GPU and would otherwise
# get killed by the low-util watchdog: API calls (DeepSeek/Gemini),
# dataset prep, result aggregation, jsonl post-processing.
#
# Use:
#   sbatch --job-name=mytask --wrap "python scripts/whatever.py"   sbatch/cpu_template.sbatch
# or copy this file and edit the block below.
#
#SBATCH --job-name=cpu_task
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=cpu_prem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
mkdir -p logs

source ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate open_r1
cd ${REPO_ROOT}

echo "=== CPU task on $(hostname) ==="

# === edit the command below ===
# Examples:
#   python scripts/hint_ai_label_fixup_deepseek.py --input ... --workers 32
#   python open-r1/scripts/cia/prep_two_hop_grpo.py --labeled_jsonl ... --out_dir ...
#   python scripts/build_ablation_table.py
echo "ERROR: edit cpu_template.sbatch and replace this block"
exit 1
