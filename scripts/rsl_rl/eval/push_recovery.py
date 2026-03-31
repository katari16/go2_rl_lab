"""Push Recovery Evaluation — measure compliance via path deviation after impulse.

Robot walks forward along a straight line, then a velocity impulse is applied.
Measures how far the robot deviates from its intended path and how long it takes
to return — more compliant = larger deviation but smoother recovery.

Visualization (in Isaac Sim):
  GREEN arrow  = intended walking direction
  RED arrow    = impulse force direction

Metrics per direction:
  - Lateral path deviation (max and over time)
  - Recovery time (return within threshold of original path)
  - Peak joint torque during recovery
  - Mean power consumption during recovery
  - Success rate (doesn't fall)

Generates PDF with: path deviation plots, polar plots, summary table.

Usage:
    python scripts/rsl_rl/eval/push_recovery.py --task Go2-LowLevel-v0 --num_envs 1 \
        --checkpoint logs/rsl_rl/.../model_XXXX.pt --impulse_magnitude 0.5 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Push recovery evaluation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--impulse_magnitude", type=float, default=0.5, help="Impulse velocity magnitude (m/s).")
parser.add_argument("--num_trials", type=int, default=5, help="Trials per direction.")
parser.add_argument("--walk_speed", type=float, default=0.5, help="Forward walking speed before push.")
parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Stage-1 checkpoint (for stage 2).")
parser.add_argument("--estimator_checkpoint", type=str, default=None, help="Estimator checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

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


NUM_DIRECTIONS = 8
DIRECTIONS_DEG = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)
DIRECTIONS_RAD = np.deg2rad(DIRECTIONS_DEG)

WARMUP_S = 3.0
RECOVERY_WINDOW_S = 4.0
PATH_RECOVERY_THRESHOLD_M = 0.05  # 5cm from original line


def _create_arrow_markers(prim_path: str, color: tuple) -> VisualizationMarkers:
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


def _dir_to_quat(dx, dy, device):
    """Quaternion to orient arrow along (dx, dy, 0) direction."""
    quats = torch.zeros(1, 4, device=device)
    quats[0, 0] = 1.0
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 0.01:
        return quats
    angle = math.atan2(dy, dx)
    quats[0, 0] = math.cos(angle / 2)
    quats[0, 3] = math.sin(angle / 2)  # rotation around Z
    return quats


def _step_policy(obs, policy, policy_nn, env, runner, runner_class_name, is_stage2,
                 wrapper, isaac_env, device, n, compliant_raw_obs_dim=None):
    """One policy step — handles both stage1 and stage2."""
    with torch.inference_mode():
        if runner_class_name == "CompliantOnPolicyRunner":
            raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
            runner._history_buffer.insert(raw_obs)
            fhat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
            isaac_env._force_estimate_xy = fhat
        if is_stage2:
            actions = policy(obs)
            obs_next, rew, dones, extras = wrapper.step(actions)
        else:
            actions = policy(obs)
            obs_next, rew, dones, extras = env.step(actions)
        policy_nn.reset(dones)
    return obs_next, dones


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Walking forward
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.walk_speed, args_cli.walk_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Disable force events
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    try:
        force_event = getattr(env_cfg.events, force_event_name)
        force_event.params["force_range"] = (0.0, 0.0)
    except AttributeError:
        pass
    env_cfg.events.push_robot = None

    # Resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[push_recovery] Checkpoint: {resume_path}")

    # Create env + runner
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    print(f"[push_recovery] Runner: {runner_class_name}")

    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.estimator_checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.estimator_checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "ComplianceOnPolicyRunner":
        from go2_rl_lab.estimator.compliance_runner import ComplianceOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.stage1_checkpoint:
            train_cfg.setdefault("compliance", {})["stage1_checkpoint"] = args_cli.stage1_checkpoint
        runner = ComplianceOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    else:
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    runner.eval_mode()

    is_stage2 = runner_class_name == "ComplianceOnPolicyRunner"
    wrapper = runner._compliance_wrapper if is_stage2 else None
    if is_stage2:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy
    else:
        if hasattr(runner, "_wrapped_env"):
            env = runner._wrapped_env
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

    isaac_env = env.unwrapped
    device = isaac_env.device
    asset = isaac_env.scene["robot"]
    dt = isaac_env.step_dt
    n = args_cli.num_envs

    compliant_raw_obs_dim = None
    if runner_class_name == "CompliantOnPolicyRunner":
        force_dim = getattr(runner, "_force_dim", 3)
        isaac_env._force_estimate_xy = torch.zeros(n, force_dim, device=device)
        compliant_raw_obs_dim = runner._num_one_step_obs

    impulse_mag = args_cli.impulse_magnitude
    num_trials = args_cli.num_trials
    warmup_steps = int(WARMUP_S / dt)
    recovery_steps = int(RECOVERY_WINDOW_S / dt)

    # Arrows for visualization
    walk_markers = _create_arrow_markers("/World/Visuals/WalkArrow", (0.0, 0.8, 0.2))    # Green
    push_markers = _create_arrow_markers("/World/Visuals/PushArrow", (1.0, 0.0, 0.0))    # Red

    print(f"\n{'=' * 70}")
    print(f"  Push Recovery — Path Deviation Test")
    print(f"  Impulse: {impulse_mag:.1f} m/s from {NUM_DIRECTIONS} directions")
    print(f"  Trials: {num_trials}/dir, Walk: {args_cli.walk_speed:.1f} m/s")
    print(f"  Arrows: GREEN=walk direction, RED=impulse direction")
    print(f"{'=' * 70}\n")

    # Results
    all_results = {d: [] for d in DIRECTIONS_DEG}

    for trial in range(num_trials):
        for d_idx, (deg, rad) in enumerate(zip(DIRECTIONS_DEG, DIRECTIONS_RAD)):
            impulse_vx = impulse_mag * math.cos(rad)
            impulse_vy = impulse_mag * math.sin(rad)

            print(f"  Trial {trial+1}/{num_trials}  Dir {deg:.0f}°  ", end="", flush=True)

            # Reset
            obs = wrapper.get_observations() if is_stage2 else env.get_observations()

            # ── Phase 1: Warmup walk ──────────────────────────────────
            for step in range(warmup_steps):
                obs, dones = _step_policy(
                    obs, policy, policy_nn, env, runner, runner_class_name,
                    is_stage2, wrapper, isaac_env, device, n, compliant_raw_obs_dim)

                # Show green walk arrow
                if step % 5 == 0:
                    pos = asset.data.root_pos_w[:n].clone()
                    pos[:, 2] += 0.5
                    quat = _dir_to_quat(1.0, 0.0, device)  # forward
                    scale = torch.tensor([[0.5, 0.3, 0.3]], device=device).expand(n, -1)
                    walk_markers.visualize(pos, quat.expand(n, -1), scale)

                if args_cli.real_time:
                    time.sleep(dt)

            # Record position at moment of push (the "intended line" start)
            push_pos = asset.data.root_pos_w[0, :2].cpu().numpy().copy()
            # Intended direction = current heading (forward in world frame)
            heading_quat = asset.data.root_quat_w[0].cpu().numpy()
            # Extract heading from quaternion (yaw angle)
            qw, qx, qy, qz = heading_quat
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            walk_dir = np.array([math.cos(yaw), math.sin(yaw)])

            # ── Phase 2: Apply impulse ────────────────────────────────
            root_vel = asset.data.root_lin_vel_w.clone()
            root_vel[:, 0] += impulse_vx
            root_vel[:, 1] += impulse_vy
            asset.write_root_velocity_to_sim(
                torch.cat([root_vel, asset.data.root_ang_vel_w], dim=-1)
            )

            # ── Phase 3: Measure recovery ─────────────────────────────
            path_xy = []         # world XY position over time
            lateral_devs = []    # lateral deviation from intended line
            torque_log = []
            power_log = []
            fell = False

            for step in range(recovery_steps):
                obs, dones = _step_policy(
                    obs, policy, policy_nn, env, runner, runner_class_name,
                    is_stage2, wrapper, isaac_env, device, n, compliant_raw_obs_dim)

                # Record world position
                pos_xy = asset.data.root_pos_w[0, :2].cpu().numpy()
                path_xy.append(pos_xy.copy())

                # Lateral deviation from intended line
                # Project (pos - push_pos) onto perpendicular of walk_dir
                delta = pos_xy - push_pos
                perp = np.array([-walk_dir[1], walk_dir[0]])  # perpendicular
                lat_dev = abs(np.dot(delta, perp))
                lateral_devs.append(lat_dev)

                # Torque and power
                torques = asset.data.applied_torque[0].cpu().numpy()
                torque_log.append(np.max(np.abs(torques)))
                dq_vals = asset.data.joint_vel[0].cpu().numpy()
                power_log.append(np.sum(np.abs(torques * dq_vals)))

                # Visualize arrows
                if step % 5 == 0:
                    vis_pos = asset.data.root_pos_w[:n].clone()
                    # Green: walk direction
                    walk_pos = vis_pos.clone(); walk_pos[:, 2] += 0.5
                    walk_quat = _dir_to_quat(walk_dir[0], walk_dir[1], device)
                    walk_scale = torch.tensor([[0.5, 0.3, 0.3]], device=device).expand(n, -1)
                    walk_markers.visualize(walk_pos, walk_quat.expand(n, -1), walk_scale)
                    # Red: impulse direction
                    push_vis_pos = vis_pos.clone(); push_vis_pos[:, 2] += 0.35
                    push_quat = _dir_to_quat(impulse_vx, impulse_vy, device)
                    push_scale = torch.tensor([[0.3, 0.2, 0.2]], device=device).expand(n, -1)
                    push_markers.visualize(push_vis_pos, push_quat.expand(n, -1), push_scale)

                if dones[0] > 0:
                    fell = True
                    break

                if args_cli.real_time:
                    time.sleep(dt)

            path_xy = np.array(path_xy) if path_xy else np.zeros((0, 2))
            lateral_devs = np.array(lateral_devs)
            torque_log = np.array(torque_log)
            power_log = np.array(power_log)

            # Compute metrics
            if fell or len(lateral_devs) == 0:
                result = {
                    "success": False, "recovery_time": RECOVERY_WINDOW_S,
                    "max_lateral_dev": 0.0, "peak_torque": 0.0, "mean_power": 0.0,
                    "path_xy": path_xy, "lateral_devs": lateral_devs,
                    "push_pos": push_pos, "walk_dir": walk_dir,
                    "time_axis": np.arange(len(lateral_devs)) * dt,
                }
            else:
                max_dev = np.max(lateral_devs)
                # Recovery time = first time lateral dev returns below threshold after peak
                peak_idx = np.argmax(lateral_devs)
                recovered_after_peak = np.where(lateral_devs[peak_idx:] < PATH_RECOVERY_THRESHOLD_M)[0]
                if len(recovered_after_peak) > 0:
                    recovery_time = (peak_idx + recovered_after_peak[0]) * dt
                else:
                    recovery_time = RECOVERY_WINDOW_S

                result = {
                    "success": True,
                    "recovery_time": recovery_time,
                    "max_lateral_dev": max_dev,
                    "peak_torque": np.max(torque_log) if len(torque_log) > 0 else 0,
                    "mean_power": np.mean(power_log) if len(power_log) > 0 else 0,
                    "path_xy": path_xy,
                    "lateral_devs": lateral_devs,
                    "push_pos": push_pos,
                    "walk_dir": walk_dir,
                    "time_axis": np.arange(len(lateral_devs)) * dt,
                }

            all_results[deg].append(result)
            status = "OK" if result["success"] else "FELL"
            print(f"max_dev={result['max_lateral_dev']:.3f}m  "
                  f"recovery={result['recovery_time']:.2f}s  "
                  f"peak_τ={result['peak_torque']:.1f}  [{status}]")

    env.close()

    # ── Generate PDF ──────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime

    eval_dir = os.path.join(os.path.dirname(resume_path), "eval")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = os.path.join(eval_dir, f"push_recovery_{impulse_mag:.1f}ms_{timestamp}.pdf")

    angles = np.deg2rad(DIRECTIONS_DEG)
    angles_closed = np.append(angles, angles[0])

    with PdfPages(pdf_path) as pdf:
        # ── Page 1: Path deviation time series per direction ──────────
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
        fig.suptitle(f"Lateral Path Deviation After Impulse ({impulse_mag:.1f} m/s)", fontsize=14, fontweight="bold")

        for d_idx, deg in enumerate(DIRECTIONS_DEG):
            ax = axes[d_idx // 4, d_idx % 4]
            for r in all_results[deg]:
                if len(r["lateral_devs"]) > 0:
                    color = "tab:blue" if r["success"] else "tab:red"
                    ax.plot(r["time_axis"], r["lateral_devs"], color=color, alpha=0.5, linewidth=0.8)
            ax.axhline(PATH_RECOVERY_THRESHOLD_M, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_title(f"{deg:.0f}°", fontsize=11)
            ax.set_ylabel("Lateral dev (m)" if d_idx % 4 == 0 else "")
            ax.set_xlabel("Time (s)" if d_idx >= 4 else "")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Top-down path traces per direction ────────────────
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.suptitle(f"Robot Path (Top-Down) — Impulse {impulse_mag:.1f} m/s", fontsize=14, fontweight="bold")

        for d_idx, deg in enumerate(DIRECTIONS_DEG):
            ax = axes[d_idx // 4, d_idx % 4]
            ax.set_aspect("equal")
            for r in all_results[deg]:
                if len(r["path_xy"]) == 0:
                    continue
                # Center path at push position
                centered = r["path_xy"] - r["push_pos"]
                color = "tab:blue" if r["success"] else "tab:red"
                ax.plot(centered[:, 0], centered[:, 1], color=color, alpha=0.4, linewidth=0.8)

                # Draw intended walk line
                line_len = 2.0
                ax.plot([0, r["walk_dir"][0] * line_len], [0, r["walk_dir"][1] * line_len],
                        color="green", linewidth=1.5, alpha=0.6, linestyle="--", label="intended" if r == all_results[deg][0] else "")

            # Impulse direction arrow
            rad = DIRECTIONS_RAD[d_idx]
            ax.annotate("", xy=(0.3 * math.cos(rad), 0.3 * math.sin(rad)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=2))

            ax.plot(0, 0, "ko", markersize=4)  # push point
            ax.set_title(f"{deg:.0f}°", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x (m)" if d_idx >= 4 else "")
            ax.set_ylabel("y (m)" if d_idx % 4 == 0 else "")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: Polar plots ───────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "polar"})
        fig.suptitle(f"Push Recovery Metrics — {impulse_mag:.1f} m/s, {num_trials} trials/dir",
                     fontsize=14, fontweight="bold")

        def polar_plot(ax, key, title, unit, color):
            means = [np.mean([r[key] for r in all_results[d]]) for d in DIRECTIONS_DEG]
            stds = [np.std([r[key] for r in all_results[d]]) for d in DIRECTIONS_DEG]
            m = np.array(means + [means[0]])
            s = np.array(stds + [stds[0]])
            ax.plot(angles_closed, m, color=color, linewidth=2)
            ax.fill_between(angles_closed, np.clip(m - s, 0, None), m + s, alpha=0.2, color=color)
            ax.set_title(f"{title}\n({unit})", fontsize=11, pad=15)
            ax.set_thetagrids(DIRECTIONS_DEG)

        polar_plot(axes[0, 0], "max_lateral_dev", "Max Lateral Deviation", "meters", "tab:blue")
        polar_plot(axes[0, 1], "recovery_time", "Recovery Time", "seconds", "tab:purple")
        polar_plot(axes[1, 0], "peak_torque", "Peak Joint Torque", "N·m", "tab:red")
        polar_plot(axes[1, 1], "mean_power", "Mean Power", "Watts", "tab:orange")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 4: Summary table ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        headers = ["Dir", "Max Dev (m)", "Recovery (s)", "Peak τ (N·m)", "Power (W)", "Success"]
        rows = []
        for d in DIRECTIONS_DEG:
            results = all_results[d]
            rows.append([
                f"{d:.0f}°",
                f"{np.mean([r['max_lateral_dev'] for r in results]):.3f} ± {np.std([r['max_lateral_dev'] for r in results]):.3f}",
                f"{np.mean([r['recovery_time'] for r in results]):.2f} ± {np.std([r['recovery_time'] for r in results]):.2f}",
                f"{np.mean([r['peak_torque'] for r in results]):.1f}",
                f"{np.mean([r['mean_power'] for r in results]):.1f}",
                f"{np.mean([r['success'] for r in results])*100:.0f}%",
            ])
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.5)
        ax.set_title("Push Recovery Summary", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n[push_recovery] Results saved to {pdf_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
