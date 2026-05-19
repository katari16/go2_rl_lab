#!/bin/bash
#SBATCH --job-name=abl-analyze
#SBATCH --output=slurm_logs/ablation_analyze_%j.out
#SBATCH --error=slurm_logs/ablation_analyze_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=10G

# ── Run analysis on all static_360 eval results ────────────────────────────
# Finds all static_360_* directories under data/eval/ and runs analysis on each.
# Submit AFTER ablation_static_360_eval.sh has completed.
#
# Usage:
#   sbatch --dependency=afterok:<eval_job_id> scripts/cluster/ablation_static_360_analyze.sh

module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

cd /cluster/home/habaumann/go2_rl_lab

echo "========================================="
echo "Running analysis on all static_360 eval results"
echo "Starting at: $(date)"
echo "========================================="

TOTAL=0
FAILED=0

for dir in data/eval/*/static_360_*; do
    if [ -d "$dir" ] && [ -f "$dir/raw_data.json" ] && [ ! -f "$dir/analysis.pdf" ]; then
        echo "Analyzing: $dir"
        python scripts/rsl_rl/eval/analyze_static_360.py "$dir" || FAILED=$((FAILED + 1))
        TOTAL=$((TOTAL + 1))
    else
        [ -f "$dir/analysis.pdf" ] && echo "Skipping (already analyzed): $dir"
    fi
done

echo "========================================="
echo "Analysis complete at $(date)"
echo "Processed: $TOTAL directories, $FAILED failures"
echo "========================================="
