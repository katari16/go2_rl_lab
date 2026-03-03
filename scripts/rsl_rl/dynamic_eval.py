"""Dynamic force evaluation — visualize GT vs NN-estimated force arrows while walking.

Same as static_eval but with non-zero velocity commands so the robot walks around.
This tests the force estimator performance under locomotion.

Two arrows hover above the robot:
  RED  arrow = ground-truth applied force (from wrench composer)
  BLUE arrow = force estimated by the neural network
  GREEN arrow = EMA-filtered estimate

Forces re-randomize every 1–3 seconds. Velocity commands re-sample every 10 seconds.

Usage examples:
    # Auto-detect latest checkpoint:
    python scripts/rsl_rl/dynamic_eval.py --task Go2-Force-Only-v0 --num_envs 1

    # Specific checkpoint:
    python scripts/rsl_rl/dynamic_eval.py --task Go2-Force-Only-v0 \
        --checkpoint logs/rsl_rl/go2_force_only/2026-02-24_11-40-32/model_1400.pt

    # Custom force range + velocity range:
    python scripts/rsl_rl/dynamic_eval.py --task Go2-Force-Only-v0 \
        --force_min 10.0 --force_max 30.0 --vel_max 1.0 --real-time

    # No-filter mode (raw signals only: base vel, cmd vel, GT force, est force):
    python scripts/rsl_rl/dynamic_eval.py --task Go2-Force-Only-v0 --no-filter
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Dynamic force estimation evaluation with arrow visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument("--task", type=str, default="Go2-Force-Only-v0", help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--force_min", type=float, default=10.0, help="Minimum force magnitude per axis (N).")
parser.add_argument("--force_max", type=float, default=20.0, help="Maximum force magnitude per axis (N).")
parser.add_argument("--vel_max", type=float, default=1.0, help="Max linear/angular velocity command magnitude.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration to run in seconds (then save plots and exit).")
parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor for filtered estimate (0=full smooth, 1=no smooth).")
parser.add_argument("--compliance_k", type=float, default=0.0, help="Compliance gain k for v*=v+k*F (0=disabled). Modulates velocity command with filtered force.")
parser.add_argument("--no-filter", action="store_true", default=False, help="Disable EMA filter; plot raw base vel, cmd vel, GT force, estimated force.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


# ── Arrow helpers ─────────────────────────────────────────────────────────────


def _create_arrow_markers(prim_path: str, color: tuple) -> VisualizationMarkers:
    """Create coloured arrow markers for force visualization."""
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _force_to_quat(force_world: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert world-frame force vectors [N,3] to arrow quaternions [N,4].

    The arrow USD asset points along +X.  Returns identity quat for near-zero forces.
    """
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0  # identity (w,x,y,z)

    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        # Arbitrary up perpendicular to x_ax
        up = torch.tensor(
            [0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0],
            device=device,
        )
        z_ax = torch.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)  # [3, 3]
        quats[i] = quat_from_matrix(rot_mat)
    return quats


# ── Plotting ─────────────────────────────────────────────────────────────────


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180] degrees."""
    return (a + 180.0) % 360.0 - 180.0


def _rolling_stats(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean and std with same-length output (edge-padded)."""
    if len(x) < window:
        return np.full_like(x, np.nan), np.full_like(x, np.nan)
    cumsum = np.cumsum(np.insert(x, 0, 0))
    cumsum2 = np.cumsum(np.insert(x**2, 0, 0))
    mean = (cumsum[window:] - cumsum[:-window]) / window
    var = (cumsum2[window:] - cumsum2[:-window]) / window - mean**2
    std = np.sqrt(np.clip(var, 0, None))
    # Pad start with first valid value
    pad = window - 1
    mean = np.concatenate([np.full(pad, mean[0]), mean])
    std = np.concatenate([np.full(pad, std[0]), std])
    return mean, std


def generate_plots(
    log: dict,
    force_min: float,
    force_max: float,
    vel_max: float,
    checkpoint_path: str,
    ema_alpha: float,
) -> None:
    """Generate 3-panel evaluation plot and save next to checkpoint."""
    import matplotlib.pyplot as plt
    from datetime import datetime

    t = np.array(log["time_s"])
    gt_ang = np.array(log["gt_angle"])
    est_ang = np.array(log["est_angle"])
    filt_ang = np.array(log["filt_angle"])
    gt_mag = np.array(log["gt_mag"])
    est_mag = np.array(log["est_mag"])
    filt_mag = np.array(log["filt_mag"])
    ang_err = np.array(log["ang_error"])
    filt_ang_err = np.array(log["filt_ang_error"])
    valid = np.array(log["valid_mask"])
    rerandom = log["rerandom_steps"]

    # Mask invalid (low-force) samples with NaN for plotting
    gt_ang_plot = np.where(valid, gt_ang, np.nan)
    est_ang_plot = np.where(valid, est_ang, np.nan)
    filt_ang_plot = np.where(valid, filt_ang, np.nan)
    ang_err_plot = np.where(valid, ang_err, np.nan)
    filt_ang_err_plot = np.where(valid, filt_ang_err, np.nan)

    # Rolling stats on angular error (1-second window ≈ 50 steps at 50Hz)
    window = max(1, int(1.0 / (t[1] - t[0])) if len(t) > 1 else 50)
    # For rolling stats, fill invalid with 0 to avoid NaN propagation, then mask
    ang_err_filled = np.where(valid, ang_err, 0.0)
    valid_filled = valid.astype(float)
    # Weighted rolling: sum(err*valid)/sum(valid)
    err_mean, err_std = _rolling_stats(ang_err_filled, window)
    valid_mean, _ = _rolling_stats(valid_filled, window)
    valid_mean = np.clip(valid_mean, 1e-8, None)
    err_mean = err_mean / valid_mean
    err_std = err_std / valid_mean

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Force Estimation (Dynamic) — [{force_min:.0f}, {force_max:.0f}] N/axis — vel ±{vel_max:.1f} — EMA α={ema_alpha}",
        fontsize=14,
        fontweight="bold",
    )

    # -- Vertical lines for force re-randomization events --
    for ax in axes:
        for rs in rerandom:
            if rs < len(t):
                ax.axvline(t[rs], color="gray", alpha=0.3, linewidth=0.5, linestyle="--")

    # ── Panel 1: GT angle vs Estimated angle ─────────────────────────────
    ax = axes[0]
    ax.plot(t, gt_ang_plot, color="tab:red", linewidth=0.8, alpha=0.8, label="GT angle")
    ax.plot(t, est_ang_plot, color="tab:blue", linewidth=0.5, alpha=0.4, label="Est raw")
    ax.plot(t, filt_ang_plot, color="tab:green", linewidth=1.2, alpha=0.9, label=f"Est EMA (α={ema_alpha})")
    ax.set_ylabel("Force angle (°)")
    ax.set_ylim(-200, 200)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Force Direction: GT vs Estimated", fontsize=11)

    # ── Panel 2: Angular error with rolling mean ± std ───────────────────
    ax = axes[1]
    ax.plot(t, ang_err_plot, color="tab:orange", linewidth=0.5, alpha=0.3, label="Raw instantaneous")
    ax.plot(t, filt_ang_err_plot, color="tab:green", linewidth=0.8, alpha=0.7, label=f"EMA (α={ema_alpha})")
    ax.plot(t, err_mean, color="tab:orange", linewidth=1.5, label=f"Raw rolling mean ({window}-step)")
    ax.fill_between(t, err_mean - err_std, err_mean + err_std, color="tab:orange", alpha=0.15, label="±1 std")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    # Mean absolute error annotations
    valid_errs = np.abs(ang_err[valid])
    valid_filt_errs = np.abs(filt_ang_err[valid])
    if len(valid_errs) > 0:
        mae = np.mean(valid_errs)
        filt_mae = np.mean(valid_filt_errs)
        ax.axhline(mae, color="tab:orange", linewidth=1, linestyle=":", alpha=0.7)
        ax.axhline(-mae, color="tab:orange", linewidth=1, linestyle=":", alpha=0.7)
        ax.text(t[-1], mae, f"  Raw MAE={mae:.1f}°", va="bottom", fontsize=8, color="tab:orange")
        ax.axhline(filt_mae, color="tab:green", linewidth=1, linestyle=":", alpha=0.7)
        ax.axhline(-filt_mae, color="tab:green", linewidth=1, linestyle=":", alpha=0.7)
        ax.text(t[-1], -filt_mae, f"  EMA MAE={filt_mae:.1f}°", va="top", fontsize=8, color="tab:green")
    ax.set_ylabel("Angular error (°)")
    ax.set_ylim(-180, 180)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Angular Error Over Time", fontsize=11)

    # ── Panel 3: GT magnitude vs Estimated magnitude ─────────────────────
    ax = axes[2]
    ax.plot(t, gt_mag, color="tab:red", linewidth=0.8, alpha=0.8, label="GT |F|")
    ax.plot(t, est_mag, color="tab:blue", linewidth=0.5, alpha=0.4, label="Est raw |F|")
    ax.plot(t, filt_mag, color="tab:green", linewidth=1.2, alpha=0.9, label=f"Est EMA |F| (α={ema_alpha})")
    mag_err = np.abs(gt_mag - est_mag)
    filt_mag_err = np.abs(gt_mag - filt_mag)
    ax.fill_between(t, 0, mag_err, color="tab:gray", alpha=0.1, label="Raw mag error")
    if len(mag_err) > 0:
        mmae = np.mean(mag_err)
        filt_mmae = np.mean(filt_mag_err)
        ax.text(t[-1], mmae + 1, f"  Raw MAE={mmae:.1f}N", va="bottom", fontsize=8, color="tab:gray")
        ax.text(t[-1], filt_mmae - 1, f"  EMA MAE={filt_mmae:.1f}N", va="top", fontsize=8, color="tab:green")
    ax.set_ylabel("Force magnitude (N)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Force Magnitude: GT vs Estimated", fontsize=11)

    plt.tight_layout()

    # Save to force_eval/ folder next to checkpoint
    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dynamic_eval_max_{force_max:.0f}_min_{force_min:.0f}_vel{vel_max:.1f}_ema{ema_alpha}_{timestamp}.png"
    fig_path = os.path.join(eval_dir, filename)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[dynamic_eval] Plot saved: {fig_path}")


def generate_plots_no_filter(
    log: dict,
    force_min: float,
    force_max: float,
    vel_max: float,
    checkpoint_path: str,
) -> None:
    """Generate 4-panel plot: base vel, cmd vel, GT force, estimated force."""
    import matplotlib.pyplot as plt
    from datetime import datetime

    t = np.array(log["time_s"])
    rerandom = log["rerandom_steps"]

    # Velocity data
    base_vx = np.array(log["base_vel_x"])
    base_vy = np.array(log["base_vel_y"])
    base_wz = np.array(log["base_vel_yaw"])
    cmd_vx = np.array(log["cmd_vel_x"])
    cmd_vy = np.array(log["cmd_vel_y"])
    cmd_wz = np.array(log["cmd_vel_yaw"])

    # Force data
    gt_fx = np.array(log["gt_force_x"])
    gt_fy = np.array(log["gt_force_y"])
    est_fx = np.array(log["est_force_x"])
    est_fy = np.array(log["est_force_y"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Dynamic Eval (no filter) — F[{force_min:.0f}, {force_max:.0f}] N — vel ±{vel_max:.1f}",
        fontsize=14,
        fontweight="bold",
    )

    # -- Vertical lines for force re-randomization events --
    for ax in axes:
        for rs in rerandom:
            if rs < len(t):
                ax.axvline(t[rs], color="gray", alpha=0.3, linewidth=0.5, linestyle="--")

    # ── Panel 1: Commanded velocity ──────────────────────────────────────
    ax = axes[0]
    ax.plot(t, cmd_vx, color="tab:red", linewidth=1.0, alpha=0.9, label="cmd vx")
    ax.plot(t, cmd_vy, color="tab:blue", linewidth=1.0, alpha=0.9, label="cmd vy")
    ax.plot(t, cmd_wz, color="tab:green", linewidth=1.0, alpha=0.9, label="cmd yaw")
    ax.set_ylabel("Velocity (m/s, rad/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Commanded Velocity", fontsize=11)

    # ── Panel 2: Base velocity ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, base_vx, color="tab:red", linewidth=0.8, alpha=0.8, label="base vx")
    ax.plot(t, base_vy, color="tab:blue", linewidth=0.8, alpha=0.8, label="base vy")
    ax.plot(t, base_wz, color="tab:green", linewidth=0.8, alpha=0.8, label="base yaw")
    # Overlay commanded as dashed for comparison
    ax.plot(t, cmd_vx, color="tab:red", linewidth=0.5, alpha=0.4, linestyle="--", label="cmd vx")
    ax.plot(t, cmd_vy, color="tab:blue", linewidth=0.5, alpha=0.4, linestyle="--", label="cmd vy")
    ax.plot(t, cmd_wz, color="tab:green", linewidth=0.5, alpha=0.4, linestyle="--", label="cmd yaw")
    ax.set_ylabel("Velocity (m/s, rad/s)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Base Velocity (solid) vs Commanded (dashed)", fontsize=11)

    # ── Panel 3: Applied force (GT) ──────────────────────────────────────
    ax = axes[2]
    ax.plot(t, gt_fx, color="tab:red", linewidth=1.0, alpha=0.9, label="GT Fx")
    ax.plot(t, gt_fy, color="tab:blue", linewidth=1.0, alpha=0.9, label="GT Fy")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Applied Force (Ground Truth)", fontsize=11)

    # ── Panel 4: Estimated force ─────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, est_fx, color="tab:red", linewidth=0.8, alpha=0.7, label="Est Fx")
    ax.plot(t, est_fy, color="tab:blue", linewidth=0.8, alpha=0.7, label="Est Fy")
    # Overlay GT as dashed for comparison
    ax.plot(t, gt_fx, color="tab:red", linewidth=0.5, alpha=0.3, linestyle="--", label="GT Fx")
    ax.plot(t, gt_fy, color="tab:blue", linewidth=0.5, alpha=0.3, linestyle="--", label="GT Fy")
    ax.set_ylabel("Force (N)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Estimated Force (solid) vs GT (dashed)", fontsize=11)

    plt.tight_layout()

    # Save to force_eval/ folder next to checkpoint
    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"dynamic_eval_nofilter_F{force_max:.0f}_vel{vel_max:.1f}_{timestamp}.png"
    fig_path = os.path.join(eval_dir, filename)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[dynamic_eval] Plot saved: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play a trained policy while visualizing GT and estimated forces."""

    # ── Config overrides ──────────────────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Walking robot: velocity commands in a reasonable range
    v = args_cli.vel_max
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-v, v)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-v, v)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-v, v)
    env_cfg.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
    env_cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.02
    env_cfg.commands.base_velocity.rel_heading_envs = 1.0
    env_cfg.commands.base_velocity.heading_command = True
    env_cfg.commands.base_velocity.heading_control_stiffness = 0.5
    env_cfg.commands.base_velocity.debug_vis = True

    # Activate forces immediately with user-specified range
    env_cfg.events.persistent_xy_force.params["force_range"] = (
        args_cli.force_min,
        args_cli.force_max,
    )
    # Re-randomize frequently for a dynamic demo
    env_cfg.events.persistent_xy_force.interval_range_s = (1.0, 3.0)

    # Override the reset event to also apply XY forces on episode start.
    # The default reset event uses apply_external_force_torque which zeroes
    # all forces, leaving a gap until the interval event fires.  Replace it
    # with our apply_persistent_xy_force so forces are present from step 0.
    from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_xy_force

    env_cfg.events.base_external_force_torque.func = apply_persistent_xy_force
    env_cfg.events.base_external_force_torque.params = {
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "force_range": (args_cli.force_min, args_cli.force_max),
    }

    # ── Resolve checkpoint ────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[dynamic_eval] Checkpoint: {resume_path}")

    # ── Create env + runner ───────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    from go2_rl_lab.estimator.force_runner import ForceOnPolicyRunner

    runner = ForceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.eval_mode()

    # Switch to the estimator-wrapped env so policy gets augmented obs
    if hasattr(runner, "_wrapped_env"):
        env = runner._wrapped_env

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── Access robot asset for GT force and pose ──────────────────────────
    isaac_env = env.unwrapped
    device = isaac_env.device
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)

    # ── Create arrow markers ─────────────────────────────────────────────
    no_filter = args_cli.no_filter
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))     # Red
    est_markers = _create_arrow_markers("/World/Visuals/EstForceArrow", (0.0, 0.4, 1.0))   # Blue
    filt_markers = None
    if not no_filter:
        filt_markers = _create_arrow_markers("/World/Visuals/FiltForceArrow", (0.0, 0.8, 0.2))  # Green

    # ── Run loop ──────────────────────────────────────────────────────────
    dt = isaac_env.step_dt
    obs = env.get_observations()
    n = args_cli.num_envs
    step_count = 0
    max_steps = int(args_cli.duration / dt)

    # EMA filter state (body frame XY)
    ema_alpha = args_cli.ema_alpha
    compliance_k = args_cli.compliance_k
    force_filtered = torch.zeros(n, 2, device=device)

    # Data collection for plots (env 0 only)
    if no_filter:
        plot_log: dict[str, list] = {
            "time_s": [],
            "base_vel_x": [],
            "base_vel_y": [],
            "base_vel_yaw": [],
            "cmd_vel_x": [],
            "cmd_vel_y": [],
            "cmd_vel_yaw": [],
            "gt_force_x": [],
            "gt_force_y": [],
            "est_force_x": [],
            "est_force_y": [],
            "rerandom_steps": [],
        }
    else:
        plot_log: dict[str, list] = {
            "time_s": [],
            "gt_angle": [],
            "est_angle": [],
            "filt_angle": [],
            "gt_mag": [],
            "est_mag": [],
            "filt_mag": [],
            "ang_error": [],
            "filt_ang_error": [],
            "valid_mask": [],     # True when |GT| > 1N (angle meaningful)
            "rerandom_steps": [], # step indices where force re-randomized
        }
    prev_gt_xy = None  # for detecting re-randomization

    print(f"\n{'=' * 70}")
    print(f"  Mode        : DYNAMIC {'(no filter)' if no_filter else '(walking with velocity commands)'}")
    print(f"  Vel range   : lin ±{v:.1f} m/s, ang ±{v:.1f} rad/s")
    print(f"  Force range : [{args_cli.force_min:.0f}, {args_cli.force_max:.0f}] N per axis")
    if not no_filter:
        print(f"  EMA alpha   : {ema_alpha}")
        if compliance_k > 0.0:
            print(f"  Compliance  : k={compliance_k:.4f}  (v* = v + k*F̂)")
        else:
            print(f"  Compliance  : DISABLED (use --compliance_k to enable)")
        print(f"  Arrows      : RED = GT   BLUE = raw NN   GREEN = EMA filtered")
    else:
        print(f"  Filter      : DISABLED (--no-filter)")
        print(f"  Arrows      : RED = GT   BLUE = raw NN estimate")
    print(f"  Duration    : {args_cli.duration:.0f}s ({max_steps} steps)")
    print(f"  Envs        : {n}")
    print(f"{'=' * 70}\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # ── Policy step ───────────────────────────────────────
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                # ── GT force (body frame XY from wrench composer) ─────
                gt_force_body_xy = asset.permanent_wrench_composer.composed_force_as_torch[
                    :n, base_idx, :2
                ]
                if gt_force_body_xy.dim() == 3:
                    gt_force_body_xy = gt_force_body_xy.squeeze(1)

                # ── NN estimated force (body frame XY) ────────────────
                force_hat, _ = runner.estimator.get_latent(
                    runner._history_buffer.get_flattened()
                )
                force_hat = force_hat[:n]

                # ── EMA-filtered estimate (skip when --no-filter) ────
                if not no_filter:
                    force_filtered = ema_alpha * force_hat[:, :2] + (1.0 - ema_alpha) * force_filtered

                    # ── Compliance modulation (SAC-Loco) ─────────────
                    if compliance_k > 0.0:
                        obs[:, 6] = obs[:, 6] + compliance_k * force_filtered[:, 0]
                        obs[:, 7] = obs[:, 7] + compliance_k * force_filtered[:, 1]

                # ── Transform to world frame for arrow rendering ──────
                base_pos = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]

                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :2] = gt_force_body_xy
                gt_world = quat_apply(base_quat, gt_3d)

                est_3d = torch.zeros(n, 3, device=device)
                est_3d[:, :2] = force_hat
                est_world = quat_apply(base_quat, est_3d)

                # ── Arrow positions (above the robot base) ────────────
                gt_pos = base_pos.clone()
                gt_pos[:, 2] += 0.55
                est_pos = base_pos.clone()
                est_pos[:, 2] += 0.40

                # ── Arrow orientations ────────────────────────────────
                gt_quats = _force_to_quat(gt_world, device)
                est_quats = _force_to_quat(est_world, device)

                # ── Arrow scales (length ∝ magnitude) ─────────────────
                gt_mag_w = gt_world.norm(dim=-1, keepdim=True)
                est_mag_w = est_world.norm(dim=-1, keepdim=True)
                scale_factor = 0.05  # 20N → length 1.0

                gt_scales = torch.full((n, 3), 0.3, device=device)
                gt_scales[:, 0:1] = (gt_mag_w * scale_factor).clamp(min=0.05)

                est_scales = torch.full((n, 3), 0.3, device=device)
                est_scales[:, 0:1] = (est_mag_w * scale_factor).clamp(min=0.05)

                # ── Render arrows ─────────────────────────────────────
                gt_markers.visualize(gt_pos, gt_quats, gt_scales)
                est_markers.visualize(est_pos, est_quats, est_scales)

                if not no_filter:
                    filt_3d = torch.zeros(n, 3, device=device)
                    filt_3d[:, :2] = force_filtered
                    filt_world = quat_apply(base_quat, filt_3d)
                    filt_mag_w = filt_world.norm(dim=-1, keepdim=True)
                    filt_pos = base_pos.clone()
                    filt_pos[:, 2] += 0.25
                    filt_quats = _force_to_quat(filt_world, device)
                    filt_scales = torch.full((n, 3), 0.3, device=device)
                    filt_scales[:, 0:1] = (filt_mag_w * scale_factor).clamp(min=0.05)
                    filt_markers.visualize(filt_pos, filt_quats, filt_scales)

                # ── Record data for plots (env 0) ────────────────────
                g0 = gt_force_body_xy[0]
                e0 = force_hat[0]
                gm0 = g0.norm().item()
                em0 = e0.norm().item()

                if no_filter:
                    # Read base velocity (body frame) and commanded velocity
                    base_lin_vel = asset.data.root_lin_vel_b[0]  # [3]
                    base_ang_vel = asset.data.root_ang_vel_b[0]  # [3]
                    cmd_vel = isaac_env.command_manager.get_command("base_velocity")[0]  # [vx, vy, yaw]

                    plot_log["time_s"].append(step_count * dt)
                    plot_log["base_vel_x"].append(base_lin_vel[0].item())
                    plot_log["base_vel_y"].append(base_lin_vel[1].item())
                    plot_log["base_vel_yaw"].append(base_ang_vel[2].item())
                    plot_log["cmd_vel_x"].append(cmd_vel[0].item())
                    plot_log["cmd_vel_y"].append(cmd_vel[1].item())
                    plot_log["cmd_vel_yaw"].append(cmd_vel[2].item())
                    plot_log["gt_force_x"].append(g0[0].item())
                    plot_log["gt_force_y"].append(g0[1].item())
                    plot_log["est_force_x"].append(e0[0].item())
                    plot_log["est_force_y"].append(e0[1].item())
                else:
                    f0 = force_filtered[0]
                    fm0 = f0.norm().item()
                    ga0 = math.atan2(g0[1].item(), g0[0].item()) * 180.0 / math.pi
                    ea0 = math.atan2(e0[1].item(), e0[0].item()) * 180.0 / math.pi
                    fa0 = math.atan2(f0[1].item(), f0[0].item()) * 180.0 / math.pi
                    ang_diff = (ga0 - ea0 + 180.0) % 360.0 - 180.0
                    filt_ang_diff = (ga0 - fa0 + 180.0) % 360.0 - 180.0
                    is_valid = gm0 > 1.0

                    plot_log["time_s"].append(step_count * dt)
                    plot_log["gt_angle"].append(ga0)
                    plot_log["est_angle"].append(ea0)
                    plot_log["filt_angle"].append(fa0)
                    plot_log["gt_mag"].append(gm0)
                    plot_log["est_mag"].append(em0)
                    plot_log["filt_mag"].append(fm0)
                    plot_log["ang_error"].append(ang_diff)
                    plot_log["filt_ang_error"].append(filt_ang_diff)
                    plot_log["valid_mask"].append(is_valid)

                # Detect force re-randomization (GT force jumped)
                if prev_gt_xy is not None:
                    delta = (g0 - prev_gt_xy).norm().item()
                    if delta > 1.0:
                        plot_log["rerandom_steps"].append(step_count)
                prev_gt_xy = g0.clone()

                # ── Terminal readout (first env, every 10 steps) ──────
                step_count += 1
                if step_count % 10 == 0:
                    pct = step_count / max_steps
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    elapsed_s = step_count * dt
                    if no_filter:
                        print(
                            f"\r  [{bar}] {elapsed_s:.0f}/{args_cli.duration:.0f}s  "
                            f"GT:[{g0[0]:+6.1f},{g0[1]:+6.1f}]N  "
                            f"Est:[{e0[0]:+6.1f},{e0[1]:+6.1f}]N  "
                            f"MagErr:{abs(gm0 - em0):5.1f}N",
                            end="",
                            flush=True,
                        )
                    else:
                        ang_diff_val = (
                            math.atan2(g0[1].item(), g0[0].item()) - math.atan2(e0[1].item(), e0[0].item())
                        )
                        ang_diff_deg = math.degrees(ang_diff_val)
                        ang_diff_deg = (ang_diff_deg + 180.0) % 360.0 - 180.0
                        angle_str = f"{abs(ang_diff_deg):5.1f}°" if gm0 > 1.0 else "  N/A"
                        print(
                            f"\r  [{bar}] {elapsed_s:.0f}/{args_cli.duration:.0f}s  "
                            f"GT:[{g0[0]:+6.1f},{g0[1]:+6.1f}]N |{gm0:5.1f}|  "
                            f"Est:[{e0[0]:+6.1f},{e0[1]:+6.1f}]N |{em0:5.1f}|  "
                            f"AngErr:{angle_str}  MagErr:{abs(gm0 - em0):5.1f}N",
                            end="",
                            flush=True,
                        )

                if step_count >= max_steps:
                    break

            # Real-time pacing
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n\n[dynamic_eval] Stopping simulation ({type(exc).__name__}).")

    print()

    # Close env first, then plot (same pattern as pace data_collection.py)
    env.close()

    from time import sleep
    sleep(1)

    # ── Save plots ────────────────────────────────────────────────────────
    if len(plot_log["time_s"]) > 10:
        print(f"[dynamic_eval] Collected {len(plot_log['time_s'])} steps. Saving plots...")
        if no_filter:
            generate_plots_no_filter(
                plot_log, args_cli.force_min, args_cli.force_max, args_cli.vel_max, resume_path,
            )
        else:
            generate_plots(
                plot_log, args_cli.force_min, args_cli.force_max, args_cli.vel_max, resume_path, ema_alpha,
            )
    else:
        print("[dynamic_eval] Too few steps collected, skipping plots.")


if __name__ == "__main__":
    main()
    simulation_app.close()
