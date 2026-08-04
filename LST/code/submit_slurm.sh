#!/usr/bin/env bash
#SBATCH --job-name=lst_npu
#SBATCH --partition=npu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=npu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lst_npu_%j.log
#SBATCH --error=logs/lst_npu_%j.err
#
# Submit with:  sbatch submit_slurm.sh
#
# Runs all LST paper experiments on a single Ascend NPU.
# Results land in code/results/ and figures in code/figs/.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results figs

echo "=== LST slurm NPU job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Date: $(date)"
echo

srun bash submit_npu_job.sh
