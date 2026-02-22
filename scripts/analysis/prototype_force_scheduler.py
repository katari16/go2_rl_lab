"""Prototype force curriculum schedulers on logged force_loss_smooth data.

Reads TB logs from a completed (or running) training run and simulates
how different scheduling strategies would have ramped the force magnitude.

Usage:
    python scripts/analysis/prototype_force_scheduler.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ──────────────────────────────────────────────────────────────
LOG_DIR = "/home/ubuntu/go2_rl_lab/logs/rsl_rl/go2_force_estimator/2026-02-22_12-58-48/"
RAMP_STEP = 2.0
MAX_FORCE = 20.0
OUTPUT_PATH = "/home/ubuntu/go2_rl_lab/scripts/analysis/scheduler_comparison.png"


# ── Load TB data ────────────────────────────────────────────────────────
def load_scalar(ea, tag):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


print("Loading TB data...")
ea = EventAccumulator(LOG_DIR)
ea.Reload()

fl_steps, force_loss = load_scalar(ea, "Estimator/force_loss_smooth")
ta_steps, training_active = load_scalar(ea, "Estimator/training_active")
cur_steps, cur_force = load_scalar(ea, "Curriculum/force_curriculum")
xy_steps, xy_reward = load_scalar(ea, "Estimator/xy_tracking_reward")

# Find when estimator training activated
activation_iter = None
for s, v in zip(ta_steps, training_active):
    if v >= 1.0:
        activation_iter = int(s)
        break

print(f"Estimator activated at iter {activation_iter}")
print(f"Force loss data: {len(force_loss)} points, iters {fl_steps[0]}..{fl_steps[-1]}")
print(f"Actual curriculum force: {cur_force[0]} -> {cur_force[-1]} N")


# ── Scheduler simulators ───────────────────────────────────────────────
# Each takes the force_loss timeseries and returns (force_at_each_step,)


def scheduler_plateau(steps, loss, patience, min_delta=0.01):
    """Current implementation: ReduceLROnPlateau-style."""
    force = np.zeros_like(loss)
    current_force = RAMP_STEP  # starts at first ramp
    best_loss = np.inf
    wait = 0

    for i in range(len(loss)):
        # Check improvement
        if loss[i] < best_loss * (1 - min_delta):
            best_loss = loss[i]
            wait = 0
        else:
            wait += 1

        # Ramp
        if wait >= patience:
            current_force = min(current_force + RAMP_STEP, MAX_FORCE)
            best_loss = np.inf
            wait = 0

        force[i] = current_force

    return force


def scheduler_trend(steps, loss, patience, slope_threshold=-1e-4):
    """User's proposal: check for downward trend over patience window.

    After each ramp, wait `patience` iterations, then fit a linear
    regression to the loss over that window.  If the slope is negative
    enough (network is learning), wait another patience period.
    After two periods with no meaningful downward trend, ramp.
    """
    force = np.zeros_like(loss)
    current_force = RAMP_STEP
    ramp_iter = 0
    waited_extra = False  # True if we already gave one extra patience period

    for i in range(len(loss)):
        iters_since_ramp = i - ramp_iter

        if iters_since_ramp >= patience:
            # Fit linear regression over the patience window
            window_start = max(ramp_iter, i - patience)
            window = loss[window_start:i + 1]
            x = np.arange(len(window))
            if len(window) > 2:
                slope = np.polyfit(x, window, 1)[0]
            else:
                slope = 0.0

            if slope < slope_threshold and not waited_extra:
                # Network is learning — give it another patience period
                waited_extra = True
                ramp_iter = i  # reset window
            else:
                # No learning trend (or already waited extra) — ramp
                current_force = min(current_force + RAMP_STEP, MAX_FORCE)
                ramp_iter = i
                waited_extra = False

        force[i] = current_force

    return force


def scheduler_trend_proportional(steps, loss, patience, strong_slope=-2e-4):
    """Variant: scale wait time by how strong the learning signal is.

    - Strong downward trend (slope < strong_slope): wait 2x patience
    - Weak downward trend: wait 1.5x patience
    - No trend or upward: ramp after patience
    """
    force = np.zeros_like(loss)
    current_force = RAMP_STEP
    ramp_iter = 0
    current_patience = patience

    for i in range(len(loss)):
        iters_since_ramp = i - ramp_iter

        if iters_since_ramp >= current_patience:
            # Fit slope over last patience iterations
            window_start = max(0, i - patience)
            window = loss[window_start:i + 1]
            x = np.arange(len(window))
            if len(window) > 2:
                slope = np.polyfit(x, window, 1)[0]
            else:
                slope = 0.0

            if slope < strong_slope:
                # Strong learning — extend patience significantly
                current_patience = int(patience * 2)
                ramp_iter = i
            elif slope < strong_slope / 2:
                # Weak learning — extend a bit
                current_patience = int(patience * 1.5)
                ramp_iter = i
            else:
                # Plateau / no learning — ramp
                current_force = min(current_force + RAMP_STEP, MAX_FORCE)
                ramp_iter = i
                current_patience = patience

        force[i] = current_force

    return force


def scheduler_fixed_timer(steps, loss, interval):
    """Baseline: ramp every N iterations regardless of loss."""
    force = np.zeros_like(loss)
    current_force = RAMP_STEP

    for i in range(len(loss)):
        if i > 0 and i % interval == 0:
            current_force = min(current_force + RAMP_STEP, MAX_FORCE)
        force[i] = current_force

    return force


# ── Run simulations ────────────────────────────────────────────────────
print("\nSimulating schedulers...")

schedulers = {
    "Plateau (patience=500, min_delta=1%)": lambda s, l: scheduler_plateau(s, l, patience=500, min_delta=0.01),
    "Plateau (patience=300)": lambda s, l: scheduler_plateau(s, l, patience=300, min_delta=0.01),
    "Trend (patience=500, 1 extra wait)": lambda s, l: scheduler_trend(s, l, patience=500, slope_threshold=-1e-4),
    "Trend (patience=400, 1 extra wait)": lambda s, l: scheduler_trend(s, l, patience=400, slope_threshold=-1e-4),
    "Trend proportional (patience=400)": lambda s, l: scheduler_trend_proportional(s, l, patience=400, strong_slope=-2e-4),
    "Fixed timer (every 500)": lambda s, l: scheduler_fixed_timer(s, l, interval=500),
    "Fixed timer (every 300)": lambda s, l: scheduler_fixed_timer(s, l, interval=300),
}

results = {}
for name, fn in schedulers.items():
    force_curve = fn(fl_steps, force_loss)
    results[name] = force_curve
    final = force_curve[-1]
    n_ramps = int((final - RAMP_STEP) / RAMP_STEP)
    print(f"  {name:45s} → final={final:5.1f}N  ramps={n_ramps}")


# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
fig.suptitle("Force Curriculum Scheduler Comparison\n(simulated on logged force_loss_smooth)", fontsize=14)

# Panel 1: Force loss
ax1 = axes[0]
ax1.plot(fl_steps, force_loss, "k-", alpha=0.7, linewidth=0.8, label="force_loss_smooth")
ax1.set_ylabel("Force Loss (MSE)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
if activation_iter is not None:
    ax1.axvline(activation_iter, color="green", linestyle="--", alpha=0.5, label="estimator activated")

# Panel 2: XY tracking reward
ax2 = axes[1]
ax2.plot(xy_steps, xy_reward, "b-", alpha=0.7, linewidth=0.8, label="xy_tracking_reward")
ax2.axhline(0.8, color="r", linestyle="--", alpha=0.5, label="threshold (0.8)")
ax2.set_ylabel("XY Tracking Reward")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# Panel 3: Force curves from different schedulers
ax3 = axes[2]
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
for (name, force_curve), color in zip(results.items(), colors):
    ax3.plot(fl_steps, force_curve, label=name, linewidth=1.5, alpha=0.8, color=color)

# Also plot the actual curriculum from the run
ax3.plot(cur_steps, cur_force, "k--", linewidth=2, alpha=0.9, label="Actual (from run)")
ax3.set_ylabel("Force Magnitude (N)")
ax3.set_xlabel("Iteration")
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, MAX_FORCE + 2)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUTPUT_PATH}")
plt.close()
