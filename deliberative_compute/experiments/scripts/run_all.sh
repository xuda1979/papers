#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
OUT="${2:-results/grid}"
mkdir -p "$OUT"

python experiments/run_experiments.py --model "$MODEL" --dataset gsm8k \
  --split test --policies sc_fixed adaptive_margin \
  --sc_n 1 2 4 8 16 --adaptive_max_n 8 16 --margin_th 0.6 0.7 0.8 \
  --temperature 0.7 --max_new_tokens 512 --measure_energy \
  --out_dir "$OUT/gsm8k"

python experiments/analyze_results.py --glob "$OUT/gsm8k/*.jsonl" \
  --out_csv "$OUT/gsm8k/summary.csv"

python experiments/plot_curves.py --csv "$OUT/gsm8k/summary.csv" \
  --out_png "$OUT/gsm8k/accuracy_vs_tokens.png"
