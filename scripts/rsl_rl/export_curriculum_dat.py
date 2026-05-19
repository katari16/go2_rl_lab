"""Export TB scalars for the force curriculum comparison appendix.

Produces .dat files (t val header, downsampled to ~250 points) in the report's
data/ directory, one per (run, scalar) pair. Names use descriptive labels
instead of run codes:

  hardgate    — R1 (force range jumps to full on gate)
  linramp     — R6 (linear ramp 10→30 N over 2500 iters post-gate)
  buckets     — R8 (bucketed 10/20/30 N × 1000 iters post-gate)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "hardgate": "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/",
    "linramp":  "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R6_h30_6d_big_tcn_norec_linramp/2026-05-17_12-53-33/",
    "buckets":  "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R8_h30_6d_big_tcn_norec_buckets/2026-05-17_12-53-33/",
}

TAGS = {
    "reward":      "Train/mean_reward",
    "noise_std":   "Policy/mean_noise_std",
    "force_max":   "Compliant/force_magnitude_mean",
}

OUT_DIR = Path("/home/ubuntu/robobarrow_report/data")
TARGET_POINTS = 250


def export(run_key: str, log_path: str) -> None:
    ea = EventAccumulator(log_path, size_guidance={"scalars": 100000})
    ea.Reload()
    available = ea.Tags()["scalars"]
    for short, tag in TAGS.items():
        if tag not in available:
            print(f"[!] {run_key}: missing tag {tag}")
            continue
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.float64)
        vals = np.array([e.value for e in events], dtype=np.float64)

        n = len(steps)
        if n > TARGET_POINTS:
            block = int(np.ceil(n / TARGET_POINTS))
            trim = (n // block) * block
            steps_ds = steps[:trim].reshape(-1, block).mean(axis=1)
            vals_ds = vals[:trim].reshape(-1, block).mean(axis=1)
        else:
            steps_ds, vals_ds = steps, vals

        out = OUT_DIR / f"curric_{run_key}_{short}.dat"
        with open(out, "w") as fp:
            fp.write("t val\n")
            for s, v in zip(steps_ds, vals_ds):
                fp.write(f"{int(s)} {v:.6f}\n")
        print(f"wrote {out}  ({len(steps_ds)} points from {n})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, path in RUNS.items():
        export(key, path)


if __name__ == "__main__":
    main()
