#!/bin/bash
# Static-360 directional sweep on the best estimator (R1: h30, 6D, big net, TCN, no rec).
# FIXED 20 N horizontal pull, only the azimuth direction rotates (elevation 0 -> pure Fx/Fy,
# no Fz component). Plus the standard torque-only sweep, since R1 is 6D. Produces:
#   - <out>/raw_data.json          : full per-trial timeseries
#   - <out>/direction_metrics.json : per-direction MAE / angular-error (mean & std over trials)
#   - <out>/polar_*.png            : polar plots of per-direction MAE / Fx / Fy / Fz / τ_yaw / angular
#
# Use direction_metrics.json to study whether the estimator is biased / more accurate
# in particular directions.

set -e
cd "$(dirname "$0")/../.."

R1_CKPT=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/model_9500.pt
FORCE_N=20      # fixed pull magnitude (N)

if [ ! -f "$R1_CKPT" ]; then
    echo "!! R1 checkpoint not found: $R1_CKPT"
    exit 1
fi

echo "=========================================="
echo "Static-360 directional sweep — R1  (fixed ${FORCE_N} N, azimuth-only)"
echo "  checkpoint: $R1_CKPT"
echo "  start: $(date)"
echo "=========================================="

python scripts/rsl_rl/eval/static_360_eval.py \
    --task Go2-Ablation-R1-v0 \
    --checkpoint "$R1_CKPT" \
    --force_magnitudes $FORCE_N \
    --elevation_angles 0 \
    --num_trials 20 \
    --force_hold_s 4.0 --warmup_s 3.0 \
    --metrics_mag_range $((FORCE_N - 1)) $((FORCE_N + 1)) \
    --headless

# Locate the just-created output dir (most recent static_360_* under R1's eval folder)
OUT_DIR=$(ls -td /home/ubuntu/go2_rl_lab/data/eval/ablation_R1_h30_6d_big_tcn_norec/static_360_* 2>/dev/null | head -1)
echo ""
echo "Output dir: $OUT_DIR"

if [ -n "$OUT_DIR" ] && [ -f "$OUT_DIR/raw_data.json" ]; then
    echo "Generating per-direction metrics + polar plots..."
    python scripts/analyze_static_360_directions.py "$OUT_DIR/raw_data.json" --mag_range $((FORCE_N - 1)) $((FORCE_N + 1))
fi

echo ""
echo "=========================================="
echo "Done — $(date)"
echo "  Inspect:  $OUT_DIR/direction_metrics.json"
echo "  Plots:    $OUT_DIR/polar_*.png"
echo "=========================================="
