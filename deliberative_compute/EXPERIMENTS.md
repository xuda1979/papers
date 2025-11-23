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

### Hosted models: OpenAI & Gemini
Use `run_api_frontiers.py` to exercise the same policies against managed models.

```bash
export OPENAI_API_KEY=...
python experiments/run_api_frontiers.py \
  --provider openai --model gpt-4o-mini \
  --dataset gsm8k --max_examples 32 \
  --policies sc_fixed adaptive_margin --sc_n 1 2 4 --adaptive_max_n 4 8 \
  --out_dir results/api_runs

export GEMINI_API_KEY=...
python experiments/run_api_frontiers.py \
  --provider gemini --model gemini-1.5-pro \
  --dataset gsm8k --max_examples 32 --out_dir results/api_runs
```

The script logs JSONL rows compatible with `analyze_results.py` / `plot_curves.py` so you can overlay open-source vs API models on the same Budget–Performance Frontier.

## Experiment design: demonstrating test-time compute gains

1. **Budget sweep.** Run both `sc_fixed` and `adaptive_margin` with increasing sample counts (e.g., 1/2/4/8) on a held-out GSM8K slice (32–128 questions). Each policy run yields accuracy, tokens, and latency.
2. **Frontier comparison.** Combine the JSONL logs with `analyze_results.py` to compute frontier area (FA) and marginal value of compute (MVC). Plot accuracy vs. tokens to highlight where additional sampling delivers outsized accuracy jumps.
3. **Provider diversity.** Repeat the sweep with OpenAI (e.g., `gpt-4o-mini`) and Gemini (e.g., `gemini-1.5-pro`). Consistent FA/MVC improvements across both providers provide evidence that the algorithms are provider-agnostic.
4. **Adaptive advantage.** Inspect the adaptive_margin runs: they should match or exceed the accuracy of fixed-`n` at comparable or lower token budgets because early-stopping triggers when consensus is reached. Highlight this by overlaying adaptive vs. fixed curves.
5. **Reproducibility.** All runs are random-seeded; limit `--max_examples` to control API cost while preserving relative comparisons.

This structure yields a clean demonstration that test-time compute policies (self-consistency and adaptive halting) outperform single-shot decoding at similar or lower budgets across multiple providers.

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
