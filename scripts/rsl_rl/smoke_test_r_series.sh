#!/bin/bash
# Smoke tests for R1 and R2: 50 iters each, headless, verify no crash.
set -e
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Smoke test: R1 (6D, no est-acc reward)"
echo "=========================================="
python scripts/rsl_rl/train.py --task Go2-Ablation-R1-v0 --num_envs 64 --max_iterations 50 --headless

echo "=========================================="
echo "Smoke test: R2 (6D, est-acc reward w=50)"
echo "=========================================="
python scripts/rsl_rl/train.py --task Go2-Ablation-R2-v0 --num_envs 64 --max_iterations 50 --headless

echo "=========================================="
echo "R-series smoke tests passed."
echo "=========================================="
