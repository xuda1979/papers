# Deliberative Compute — Experiments

This folder provides a **reproducible harness** to evaluate test-time (deliberative) compute policies.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Example: GSM8K with DeepSeek-R1-Distill-Llama-8B
```bash
python experiments/run_experiments.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset gsm8k --split test \
  --policies sc_fixed --sc_n 1 2 4 8 16 \
  --temperature 0.7 --max_new_tokens 512 \
  --measure_energy --out_dir results/gsm8k_r1d8b
```

### Analyze & plot
```bash
python experiments/analyze_results.py --glob "results/gsm8k_r1d8b/*.jsonl" \
  --out_csv results/gsm8k_r1d8b/summary.csv
python experiments/plot_curves.py --csv results/gsm8k_r1d8b/summary.csv \
  --out_png results/gsm8k_r1d8b/accuracy_vs_tokens.png
```

## Benchmarks
- **GSM8K** (math word problems)
- **MMLU** (general knowledge) – choose a few representative subjects to keep runtime reasonable
- (Optional) **HumanEval/MBPP** (code), **TruthfulQA/Safety** (safety)

## Policies
- `sc_fixed`: self-consistency with n ∈ {1,2,4,8,16} (majority vote)
- `adaptive_margin`: adaptively halt by vote-entropy margin (SPRT-inspired)

## Metrics
- Accuracy / Pass@1 (task-dependent)
- Tokens generated (total and per-instance)
- Wall-clock time
- **GPU energy** (NVML if available; otherwise best-effort via `nvidia-smi`)

## Notes
- Models: prefer open models supporting deliberate reasoning; e.g. `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
- Reproducibility: seeds fixed; all runs logged as JSONL rows.
- Ethical/energy: we log energy/time to expose the trade-offs; consider energy-aware policies in production.
