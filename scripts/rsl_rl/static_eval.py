"""Static force evaluation — standing robot with persistent XY forces.

Spawns a Go2 standing on flat ground with zero velocity commands and persistent
XY body forces that re-randomize every 1-3 seconds.

Supports both runner types:
  - ForceOnPolicyRunner   (Go2-Force-Only-v0)
  - CompliantOnPolicyRunner (Go2-Compliant-v0)

4-panel plot (saved next to checkpoint):
  Panel 1 (RED):    Applied force — GT (solid) vs estimated (dashed), Fx & Fy
  Panel 2 (BLUE):   Base velocity — vx & vy (body frame)
  Panel 3 (GREEN):  Normal velocity command — cmd_vx & cmd_vy
  Panel 4 (YELLOW): Adjusted velocity command — v* = v_cmd + k * F_hat

Arrows above the robot:
  RED  = ground-truth force
  BLUE = NN-estimated force
  GREEN = EMA-filtered estimate

Usage examples:
    # ForceOnPolicyRunner:
    python scripts/rsl_rl/static_eval.py --task Go2-Force-Only-v0 --num_envs 1

    # CompliantOnPolicyRunner:
    python scripts/rsl_rl/static_eval.py --task Go2-Compliant-v0 --num_envs 1 \
        --checkpoint logs/rsl_rl/go2_compliant/2026-XX-XX/model_XXXX.pt

    # Custom force range + compliance:
    python scripts/rsl_rl/static_eval.py --task Go2-Compliant-v0 \
        --force_min 10.0 --force_max 30.0 --compliance_k 0.02 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Static force estimation evaluation with arrow visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
parser.add_argument("--task", type=str, default=None, help="Task name (e.g. Go2-Force-Only-v0 or Go2-Compliant-v0).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--force_min", type=float, default=10.0, help="Minimum force magnitude per axis (N).")
parser.add_argument("--force_max", type=float, default=20.0, help="Maximum force magnitude per axis (N).")
parser.add_argument("--duration", type=float, default=20.0, help="Duration to run in seconds (then save plots and exit).")
parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor for filtered estimate (0=full smooth, 1=no smooth).")
parser.add_argument("--compliance_k", type=float, default=0.0, help="Compliance gain k for v*=v+k*F (0=disabled).")
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
    """Convert world-frame force vectors [N,3] to arrow quaternions [N,4]."""
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0  # identity (w,x,y,z)

    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
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


def generate_plots(
    log: dict,
    force_min: float,
    force_max: float,
    checkpoint_path: str,
    ema_alpha: float,
    compliance_k: float,
) -> None:
    """Generate 4-panel evaluation plot and save next to checkpoint.

    Panel 1 (RED):    Applied force — GT vs estimated (Fx, Fy)
    Panel 2 (BLUE):   Base velocity (vx, vy)
    Panel 3 (GREEN):  Normal velocity command (cmd_vx, cmd_vy)
    Panel 4 (YELLOW): Adjusted velocity command v* = v_cmd + k * F_hat
    """
    import matplotlib.pyplot as plt
    from datetime import datetime

    t = np.array(log["time_s"])
    gt_fx = np.array(log["gt_force_x"])
    gt_fy = np.array(log["gt_force_y"])
    est_fx = np.array(log["est_force_x"])
    est_fy = np.array(log["est_force_y"])
    base_vx = np.array(log["base_vel_x"])
    base_vy = np.array(log["base_vel_y"])
    cmd_vx = np.array(log["cmd_vel_x"])
    cmd_vy = np.array(log["cmd_vel_y"])
    adj_vx = np.array(log["adj_vel_x"])
    adj_vy = np.array(log["adj_vel_y"])
    rerandom = log["rerandom_steps"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    k_str = f"k={compliance_k:.4f}" if compliance_k > 0 else "k=0 (disabled)"
    fig.suptitle(
        f"Static Force Eval — [{force_min:.0f}, {force_max:.0f}] N/axis — EMA \u03b1={ema_alpha} — {k_str}",
        fontsize=14,
        fontweight="bold",
    )

    # Vertical lines for force re-randomization events
    for ax in axes:
        for rs in rerandom:
            if rs < len(t):
                ax.axvline(t[rs], color="gray", alpha=0.3, linewidth=0.5, linestyle="--")

    # ── Panel 1 (RED): Applied force — GT solid, estimated dashed ────────
    ax = axes[0]
    ax.plot(t, gt_fx, color="tab:red", linewidth=1.2, alpha=0.9, label="GT Fx")
    ax.plot(t, gt_fy, color="darkred", linewidth=1.2, alpha=0.9, label="GT Fy")
    ax.plot(t, est_fx, color="tab:red", linewidth=0.8, alpha=0.5, linestyle="--", label="Est Fx")
    ax.plot(t, est_fy, color="darkred", linewidth=0.8, alpha=0.5, linestyle="--", label="Est Fy")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("Applied Force: GT (solid) vs Estimated (dashed)", fontsize=11)

    # ── Panel 2 (BLUE): Base velocity ────────────────────────────────────
    ax = axes[1]
    ax.plot(t, base_vx, color="tab:blue", linewidth=1.0, alpha=0.9, label="base vx")
    ax.plot(t, base_vy, color="navy", linewidth=1.0, alpha=0.9, label="base vy")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Base Velocity (body frame)", fontsize=11)

    # ── Panel 3 (GREEN): Normal velocity command ─────────────────────────
    ax = axes[2]
    ax.plot(t, cmd_vx, color="tab:green", linewidth=1.2, alpha=0.9, label="cmd vx")
    ax.plot(t, cmd_vy, color="darkgreen", linewidth=1.2, alpha=0.9, label="cmd vy")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Normal Velocity Command", fontsize=11)

    # ── Panel 4 (YELLOW): Adjusted velocity command v* = v_cmd + k*F_hat ─
    ax = axes[3]
    ax.plot(t, adj_vx, color="goldenrod", linewidth=1.2, alpha=0.9, label="v* x")
    ax.plot(t, adj_vy, color="darkgoldenrod", linewidth=1.2, alpha=0.9, label="v* y")
    # Overlay normal cmd as dashed for comparison
    ax.plot(t, cmd_vx, color="tab:green", linewidth=0.6, alpha=0.4, linestyle="--", label="cmd vx")
    ax.plot(t, cmd_vy, color="darkgreen", linewidth=0.6, alpha=0.4, linestyle="--", label="cmd vy")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Adjusted Velocity Command: v* = v_cmd + {compliance_k:.4f} * F_hat", fontsize=11)

    plt.tight_layout()

    # Save to force_eval/ folder next to checkpoint
    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"static_eval_F{force_max:.0f}_k{compliance_k:.4f}_ema{ema_alpha}_{timestamp}.png"
    fig_path = os.path.join(eval_dir, filename)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[static_eval] Plot saved: {fig_path}")


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

    # Standing robot: zero velocity commands, disable built-in debug arrows
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Activate forces immediately with user-specified range
    env_cfg.events.persistent_xy_force.params["force_range"] = (
        args_cli.force_min,
        args_cli.force_max,
    )
    # Re-randomize frequently for a dynamic demo
    env_cfg.events.persistent_xy_force.interval_range_s = (1.0, 3.0)

    # Override the reset event to also apply XY forces on episode start.
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
    print(f"[static_eval] Checkpoint: {resume_path}")

    # ── Create env + runner (auto-detect runner class) ──────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    print(f"[static_eval] Runner class: {runner_class_name}")

    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
        runner_mode = "compliant"
    elif runner_class_name == "ForceOnPolicyRunner":
        from go2_rl_lab.estimator.force_runner import ForceOnPolicyRunner
        runner = ForceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner_mode = "force"
    else:
        raise ValueError(f"Unsupported runner class for static_eval: {runner_class_name}")

    runner.load(resume_path)
    runner.eval_mode()

    # Switch to the estimator-wrapped env so policy gets augmented obs
    if hasattr(runner, "_wrapped_env"):
        env = runner._wrapped_env

    # For CompliantOnPolicyRunner, activate the mapping so compliance is visible
    if runner_mode == "compliant" and hasattr(runner, "_mapping_active"):
        runner._mapping_active = True
        isaac_env_raw = env.unwrapped if hasattr(env, "unwrapped") else env
        isaac_env_raw._mapping_active = True
        isaac_env_raw._compliance_alpha = runner._compliance_alpha
        isaac_env_raw._compliance_beta = runner._compliance_beta
        print(
            f"[static_eval] Compliance mapping forced ON: "
            f"alpha={runner._compliance_alpha:.1f}N  beta={runner._compliance_beta:.1f}"
        )

    # For CompliantOnPolicyRunner, determine the raw obs dim (without force estimate)
    if runner_mode == "compliant":
        compliant_raw_obs_dim = runner._num_one_step_obs  # 61

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
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))       # Red = GT force
    cmd_markers = _create_arrow_markers("/World/Visuals/CmdVelArrow", (0.0, 0.8, 0.2))       # Green = normal cmd vel
    adj_markers = _create_arrow_markers("/World/Visuals/AdjVelArrow", (1.0, 0.85, 0.0))        # Yellow = adjusted cmd vel

    # ── Run loop ──────────────────────────────────────────────────────────
    dt = isaac_env.step_dt
    n = args_cli.num_envs

    # Initialize force estimate for CompliantOnPolicyRunner
    if runner_mode == "compliant":
        isaac_env._force_estimate_xy = torch.zeros(n, 2, device=device)

    obs = env.get_observations()
    step_count = 0
    max_steps = int(args_cli.duration / dt)

    # Compliance
    ema_alpha = args_cli.ema_alpha
    compliance_k = args_cli.compliance_k
    force_ema = torch.zeros(n, 2, device=device)  # EMA-filtered force estimate

    # Data collection for plots (env 0 only)
    plot_log: dict[str, list] = {
        "time_s": [],
        "gt_force_x": [],
        "gt_force_y": [],
        "est_force_x": [],
        "est_force_y": [],
        "base_vel_x": [],
        "base_vel_y": [],
        "cmd_vel_x": [],
        "cmd_vel_y": [],
        "adj_vel_x": [],
        "adj_vel_y": [],
        "rerandom_steps": [],
    }
    prev_gt_xy = None

    print(f"\n{'=' * 70}")
    print(f"  Runner      : {runner_class_name} ({runner_mode})")
    print(f"  Mode        : STATIC (zero velocity commands)")
    print(f"  Force range : [{args_cli.force_min:.0f}, {args_cli.force_max:.0f}] N per axis")
    print(f"  EMA alpha   : {ema_alpha}")
    if compliance_k > 0.0:
        print(f"  Compliance  : k={compliance_k:.4f}  (v* = v_cmd + k * F_hat)")
    else:
        print(f"  Compliance  : DISABLED (use --compliance_k to enable)")
    print(f"  Arrows      : RED = GT force   GREEN = cmd vel   YELLOW = adjusted cmd vel")
    print(f"  Duration    : {args_cli.duration:.0f}s ({max_steps} steps)")
    print(f"  Envs        : {n}")
    print(f"{'=' * 70}\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # ── CompliantOnPolicyRunner: update estimator BEFORE acting ──
                if runner_mode == "compliant":
                    raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
                    runner._history_buffer.insert(raw_obs)
                    force_hat_pre, _ = runner.estimator.get_latent(
                        runner._history_buffer.get_flattened()
                    )
                    isaac_env._force_estimate_xy = force_hat_pre
                    # EMA filter the estimate
                    force_ema = ema_alpha * force_hat_pre[:, :2] + (1.0 - ema_alpha) * force_ema

                # ── Inject adjusted velocity command into obs ─────────
                # Policy obs layout: [6:8] = velocity_commands (vx, vy)
                # v* = v_cmd + k * EMA(F_hat)
                if compliance_k > 0.0:
                    obs["policy"][:, 6] = obs["policy"][:, 6] + compliance_k * force_ema[:, 0]
                    obs["policy"][:, 7] = obs["policy"][:, 7] + compliance_k * force_ema[:, 1]

                # ── Policy step ───────────────────────────────────────
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                # ── CompliantOnPolicyRunner: reset history for terminated envs ──
                if runner_mode == "compliant":
                    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        runner._history_buffer.reset(done_ids)

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

                # EMA filter for force runner (compliant already updated above)
                if runner_mode != "compliant":
                    force_ema = ema_alpha * force_hat[:, :2] + (1.0 - ema_alpha) * force_ema

                # ── Read base velocity and velocity command ───────────
                base_lin_vel = asset.data.root_lin_vel_b[:n]  # [n, 3]
                cmd_vel = isaac_env.command_manager.get_command("base_velocity")[:n]  # [n, 3+]

                # Adjusted velocity: v* = v_cmd + k * EMA(F_hat)
                adj_vel_x = cmd_vel[:, 0] + compliance_k * force_ema[:, 0]
                adj_vel_y = cmd_vel[:, 1] + compliance_k * force_ema[:, 1]

                # ── Transform to world frame for arrow rendering ──────
                base_pos = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]

                # GT force arrow (RED)
                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :2] = gt_force_body_xy
                gt_world = quat_apply(base_quat, gt_3d)

                # Normal cmd vel arrow (GREEN)
                cmd_3d = torch.zeros(n, 3, device=device)
                cmd_3d[:, 0] = cmd_vel[:, 0]
                cmd_3d[:, 1] = cmd_vel[:, 1]
                cmd_world = quat_apply(base_quat, cmd_3d)

                # Adjusted cmd vel arrow (YELLOW): v* = v_cmd + k * F_hat
                adj_3d = torch.zeros(n, 3, device=device)
                adj_3d[:, 0] = adj_vel_x
                adj_3d[:, 1] = adj_vel_y
                adj_world = quat_apply(base_quat, adj_3d)

                # ── Arrow positions (above the robot base) ────────────
                gt_pos = base_pos.clone()
                gt_pos[:, 2] += 0.55
                cmd_pos = base_pos.clone()
                cmd_pos[:, 2] += 0.40
                adj_pos = base_pos.clone()
                adj_pos[:, 2] += 0.25

                # ── Arrow orientations ────────────────────────────────
                gt_quats = _force_to_quat(gt_world, device)
                cmd_quats = _force_to_quat(cmd_world, device)
                adj_quats = _force_to_quat(adj_world, device)

                # ── Arrow scales ──────────────────────────────────────
                gt_mag_w = gt_world.norm(dim=-1, keepdim=True)
                cmd_mag_w = cmd_world.norm(dim=-1, keepdim=True)
                adj_mag_w = adj_world.norm(dim=-1, keepdim=True)
                force_scale = 0.05   # 20N -> length 1.0
                vel_scale = 1.0      # 1 m/s -> length 1.0

                gt_scales = torch.full((n, 3), 0.3, device=device)
                gt_scales[:, 0:1] = (gt_mag_w * force_scale).clamp(min=0.05)

                cmd_scales = torch.full((n, 3), 0.3, device=device)
                cmd_scales[:, 0:1] = (cmd_mag_w * vel_scale).clamp(min=0.05)

                adj_scales = torch.full((n, 3), 0.3, device=device)
                adj_scales[:, 0:1] = (adj_mag_w * vel_scale).clamp(min=0.05)

                # ── Render arrows ─────────────────────────────────────
                gt_markers.visualize(gt_pos, gt_quats, gt_scales)
                cmd_markers.visualize(cmd_pos, cmd_quats, cmd_scales)
                adj_markers.visualize(adj_pos, adj_quats, adj_scales)

                # ── Record data for plots (env 0) ────────────────────
                g0 = gt_force_body_xy[0]
                e0 = force_hat[0]

                plot_log["time_s"].append(step_count * dt)
                plot_log["gt_force_x"].append(g0[0].item())
                plot_log["gt_force_y"].append(g0[1].item())
                plot_log["est_force_x"].append(e0[0].item())
                plot_log["est_force_y"].append(e0[1].item())
                plot_log["base_vel_x"].append(base_lin_vel[0, 0].item())
                plot_log["base_vel_y"].append(base_lin_vel[0, 1].item())
                plot_log["cmd_vel_x"].append(cmd_vel[0, 0].item())
                plot_log["cmd_vel_y"].append(cmd_vel[0, 1].item())
                plot_log["adj_vel_x"].append(adj_vel_x[0].item())
                plot_log["adj_vel_y"].append(adj_vel_y[0].item())

                # Detect force re-randomization (GT force jumped)
                if prev_gt_xy is not None:
                    delta = (g0 - prev_gt_xy).norm().item()
                    if delta > 1.0:
                        plot_log["rerandom_steps"].append(step_count)
                prev_gt_xy = g0.clone()

                # ── Terminal readout (first env, every 10 steps) ──────
                step_count += 1
                if step_count % 10 == 0:
                    gm0 = g0.norm().item()
                    em0 = e0.norm().item()
                    pct = step_count / max_steps
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    elapsed_s = step_count * dt
                    print(
                        f"\r  [{bar}] {elapsed_s:.0f}/{args_cli.duration:.0f}s  "
                        f"GT:[{g0[0]:+6.1f},{g0[1]:+6.1f}]N |{gm0:5.1f}|  "
                        f"Est:[{e0[0]:+6.1f},{e0[1]:+6.1f}]N |{em0:5.1f}|  "
                        f"v*:[{adj_vel_x[0]:+5.2f},{adj_vel_y[0]:+5.2f}]",
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
        print(f"\n\n[static_eval] Stopping simulation ({type(exc).__name__}).")

    print()

    # Close env first, then plot
    env.close()

    from time import sleep
    sleep(1)

    # ── Save plots ────────────────────────────────────────────────────────
    if len(plot_log["time_s"]) > 10:
        print(f"[static_eval] Collected {len(plot_log['time_s'])} steps. Saving plots...")
        generate_plots(
            plot_log, args_cli.force_min, args_cli.force_max, resume_path, ema_alpha, compliance_k,
        )
    else:
        print("[static_eval] Too few steps collected, skipping plots.")


if __name__ == "__main__":
    main()
    simulation_app.close()
