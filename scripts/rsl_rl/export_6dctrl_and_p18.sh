#!/bin/bash
# Export 6Dctrl curriculum variants + P18 payload policy as JIT for deployment.
set -e
cd "$(dirname "$0")/../.."

run_export() {
    echo "=========================================="
    echo "Exporting: $1"
    echo "=========================================="
    eval "$1"
}

# P18 — payload policy (R1-aligned architecture)
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P18-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_P18_h30_6d_big_tcn_norec_payload/2026-04-25_23-28-49/model_15000.pt"

# 6Dctrl — Total50 curriculum baseline
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-6Dctrl-Total50-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_6Dctrl_curr_total50/2026-04-26_18-23-51/model_9500.pt"

# 6Dctrl — Total50 + est-accuracy reward sweep
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-6Dctrl-Total50-EstAccW10-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_6Dctrl_curr_total50_estacc_w10/2026-04-26_18-23-51/model_9500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-6Dctrl-Total50-EstAccW25-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_6Dctrl_curr_total50_estacc_w25/2026-04-26_18-23-51/model_9500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-6Dctrl-Total50-EstAccW50-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_6Dctrl_curr_total50_estacc_w50/2026-04-26_18-23-51/model_9500.pt"

# P-series — history size sweep (P1 h10, P2 h20, P4 h40; P3 h30 already exported)
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P1-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P1_h10_4d/2026-04-19_11-13-01/model_11500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P2-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P2_h20_4d/2026-04-19_11-12-59/model_11500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P4-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P4_h40_4d/2026-04-19_11-11-09/model_11500.pt"

# P-series — network size (P5 half; P6 double already exported)
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P5-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P5_h30_4d_half/2026-04-19_11-12-50/model_11500.pt"

# P-series — TCN encoder
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P12-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P12_h30_4d_tcn/2026-04-19_11-11-03/model_12500.pt"

# P-series — default PD gains (Kp=25, Kd=0.5)
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P20-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/model_13000.pt"

echo "=========================================="
echo "All exports complete."
echo "=========================================="
