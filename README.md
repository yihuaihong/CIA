# CIA: CoT-Interpretability Alignment

Code for Paper "Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment". Three tasks (TwoHopFact, MMLU-Hint,
2-Digit Multiplication) × three open-weight 8–9B LLMs.

Core metric: **CIA macro-F1** — agreement between an interpretability-derived
internal label `B_INT` and a CoT-text classifier label `B_CoT`. See
`CPF_utils/metrics.py`.

## Install

```bash
pip install -r requirements.txt
pip install -e open-r1
```

Set paths and API keys:

```bash
export REPO_ROOT=/path/to/this/repo
export SCRATCH=/path/for/results-and-datasets
export DEEPSEEK_API_KEY=sk-...        # AI-judge for Hint B_CoT
```

## Quickstart

```bash
# 1. Base-model task evaluation
python cpf_evaluation.py --eval_acc --use_cot_prompt \
    --model_name Llama-3.1-8B-Instruct \
    --dataset_name TwoHopFact --batch_size 64 --seed 8888

# 2. Mult B_INT via partial-product corruption
python -m CPF_utils.multiplication_corruption \
    --input <gen.jsonl> --output <out.jsonl> \
    --model_name Llama-3.1-8B-Instruct --prompt_variant force_b

# 3. Hint B_CoT (DeepSeek judge) + B_INT (probe)
python scripts/hint_ai_label_fixup_deepseek.py --input <hint_results.jsonl>
python -m CPF_utils.hint_probe train_and_label \
    --gen_jsonl <hint_results_with_ai_label.jsonl> \
    --model_name Llama-3.1-8B-Instruct --output <probe_labeled.jsonl>

# 4. GRPO ablation training (SLURM)
python open-r1/scripts/cia/prep_two_hop_grpo.py \
    --labeled_jsonl <linear_probe_labeled.jsonl> \
    --out_dir       ${SCRATCH}/datasets/TwoHopFact_cia
CONFIG=recipes/CIA/grpo/ablation/two_hop_llama31_8b_acc_faith.yaml \
ACC_CFG=recipes/accelerate_configs/zero2.yaml \
sbatch sbatch/grpo.sbatch

# 5. Aggregate
python scripts/build_ablation_table.py
```

## Layout

```
CPF_utils/        metrics.py (CIA), hint_probe.py, mult_probe.py,
                  multiplication_corruption.py, logitlens_utils.py,
                  evaluation_utils.py, layer_config.py
cpf_evaluation.py + compute_cia_batch.py    eval drivers
open-r1/          fork of huggingface/open-r1; CIA reward funcs in rewards.py
  recipes/CIA/grpo/                         canonical + ablation YAMLs
  scripts/cia/    prep_{two_hop,hint,multiplication}_grpo.py
scripts/          build_ablation_table, AI-label, gpu_keepalive
sbatch/           SLURM templates (NYU HPC; adapt headers for your cluster)
Figures/          reward_ablation*.ipynb, probe_reliability.ipynb,
                  reward_ablation_mult_2x2.py
```

## Notes

* Probe layer / hparams per `(model, task)`: `CPF_utils/layer_config.py`.
* Mult prompt is unified to `force_b` (long-multiplication format).
* Ablation reward variants: `acc_only`, `faith_only`, `acc_faith` (full).
* `gpu_keepalive.py` is an optional watchdog to dodge low-util preempt
  during vLLM rollout / DeepSpeed CPU-offload idle phases.

## License

MIT.
