"""Export TB scalars for the PD gain comparison appendix (.dat files)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "r1":  "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/",
    "p20": "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/",
}

TAGS = {
    "train_mean_reward":   "Train/mean_reward",
    "policy_noise_std":    "Policy/mean_noise_std",
    "joint_torques_l2":    "Episode_Reward/joint_torques_l2",
    "action_rate_l2":      "Episode_Reward/action_rate_l2",
}

OUT_DIR = Path("/home/ubuntu/robobarrow_report/data")


TARGET_POINTS = 120


def export(run_key: str, log_path: str) -> None:
    ea = EventAccumulator(log_path, size_guidance={"scalars": 100000})
    ea.Reload()
    for short, tag in TAGS.items():
        if tag not in ea.Tags()["scalars"]:
            print(f"[!] {run_key}: tag missing: {tag}")
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.float64)
        vals = np.array([e.value for e in events], dtype=np.float64)

        # Downsample by block-mean to stay under TeX memory limits
        n = len(steps)
        if n > TARGET_POINTS:
            block = int(np.ceil(n / TARGET_POINTS))
            trim = (n // block) * block
            steps_ds = steps[:trim].reshape(-1, block).mean(axis=1)
            vals_ds = vals[:trim].reshape(-1, block).mean(axis=1)
        else:
            steps_ds, vals_ds = steps, vals

        out_path = OUT_DIR / f"pd_{run_key}_{short}.dat"
        with open(out_path, "w") as fp:
            fp.write("t val\n")
            for s, v in zip(steps_ds, vals_ds):
                fp.write(f"{int(s)} {v:.6f}\n")
        print(f"wrote {out_path}  ({len(steps_ds)} points from {n})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, path in RUNS.items():
        export(key, path)


if __name__ == "__main__":
    main()
