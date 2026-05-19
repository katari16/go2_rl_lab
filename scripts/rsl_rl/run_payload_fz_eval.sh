#!/bin/bash
# Payload Fz estimation eval: walk forward on flat terrain with different payload masses.
# Saves JSON with Fz estimates per payload condition.
#
# P18: 6D big TCN, trained with payload_link + 0-4kg mass randomization (same architecture as deployed R1)

TASK="Go2-Ablation-P18-v0"
CKPT="logs/rsl_rl/ablation_P18_h30_6d_big_tcn_norec_payload/2026-04-25_23-28-49/model_9500.pt"
OUTDIR="data/payload_fz_eval"
mkdir -p "$OUTDIR"

COMMON="--task $TASK --checkpoint $CKPT --num_envs 16 --terrain flat --slope_deg 0 --walk_only --walk_duration 20 --walk_speed 0.5 --no_force --headless"

echo "=== Payload Fz Eval (P18, flat, no ext force) ==="
echo "Task: $TASK"
echo "Checkpoint: $CKPT"
echo ""

for MASS in 0.0 1.0 2.0 3.0 4.0 5.0 6.0; do
    echo "--- Running payload_mass=${MASS}kg ---"
    python scripts/rsl_rl/slope_eval.py $COMMON --payload_mass $MASS --save_json "$OUTDIR/p18_payload_${MASS}kg.json"
    echo ""
done

echo "=== All runs complete. Results in $OUTDIR/ ==="
