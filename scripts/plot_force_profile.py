"""Plot illustrative force profile for presentation.

Shows the persistent constant wrench profile: forces held constant for
3-5 seconds then resampled to a new random direction and magnitude.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

dt = 0.02
t_total = 20.0
steps = int(t_total / dt)
t = np.arange(steps) * dt

# Simulate persistent wrench resampling every 3-5s
components = {"$F_x$ (N)": 30.0, "$F_y$ (N)": 30.0, "$F_z$ (N)": 18.0, r"$\tau_{yaw}$ (Nm)": 10.0}
colors     = {"$F_x$ (N)": "#e63946", "$F_y$ (N)": "#9b2226", "$F_z$ (N)": "#5c4db1", r"$\tau_{yaw}$ (Nm)": "#6d3a1e"}

def make_profile(max_val, steps, dt, t_min=3.0, t_max=5.0):
    profile = np.zeros(steps)
    i = 0
    while i < steps:
        hold = int(np.random.uniform(t_min, t_max) / dt)
        val = np.random.uniform(-max_val, max_val)
        end = min(i + hold, steps)
        profile[i:end] = val
        i = end
    return profile

fig, axes = plt.subplots(len(components), 1, figsize=(12, 7), sharex=True)
fig.suptitle("Training Force Profile — Persistent Constant Wrench (resampled every 3–5 s)",
             fontsize=13, fontweight="bold", y=0.98)

for ax, (label, max_val) in zip(axes, components.items()):
    color = colors[label]
    profile = make_profile(max_val, steps, dt)
    ax.step(t, profile, where="post", color=color, linewidth=2.0)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel(label, fontsize=10)
    ax.set_ylim(-max_val * 1.35, max_val * 1.35)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time (s)", fontsize=11)
plt.tight_layout()

out = "scripts/force_profile_illustration.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
