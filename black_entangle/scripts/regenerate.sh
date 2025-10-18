#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

print_step() {
  printf '%s\n' "$1"
}

print_step "[1/2] Building paper PDF…"
latexmk -pdf -halt-on-error -interaction=nonstopmode paper/black_entangle.tex

print_step "[2/2] Running toy simulator…"
python3 code/simulate_tpchshr.py --spin 0.6 --bmin 3.0 --bmax 8.0 --samples 40

print_step "Done."
