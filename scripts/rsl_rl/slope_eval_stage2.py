"""Slope eval for HAC-LOCO stage 2 — robot walks onto a 12° slope and holds position.

The terrain is a single pyramid slope tile (12° inclination). The robot spawns
at the bottom edge, walks uphill with a forward velocity command, then the
command switches to zero so it must hold position on the slope.

This tests whether the high-level compliance policy correctly compensates for
the gravity-induced force component that the force estimator sees on slopes.

Arrows:
  RED       = GT applied force (body frame)
  TURQUOISE = linear mapping k*F_hat (baseline)
  YELLOW    = learned high-level policy a'

Usage:
    python scripts/rsl_rl/slope_eval_stage2.py --task Go2-HighLevel-NonLinear-R5-v0 \\
        --stage1_checkpoint /path/to/stage1/model.pt \\
        --checkpoint /path/to/stage2/model.pt \\
        --slope_deg 12 --walk_duration 8 --hold_duration 20 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Slope eval for HAC-LOCO stage 2.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--slope_deg", type=float, default=12.0, help="Slope inclination in degrees.")
parser.add_argument("--walk_duration", type=float, default=8.0, help="Seconds to walk uphill before stopping.")
parser.add_argument("--hold_duration", type=float, default=20.0, help="Seconds to hold position on slope.")
parser.add_argument("--walk_speed", type=float, default=0.5, help="Forward velocity command while walking (m/s).")
parser.add_argument("--real-time", action="store_true", default=False, help="Real-time pacing.")
parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Stage-1 checkpoint for frozen low-level.")
parser.add_argument("--compliance_k", type=float, default=0.06, help="Linear mapping gain k for comparison arrow.")
parser.add_argument("--force_min", type=float, default=0.0, help="Min external force (N). Set 0 for slope-only test.")
parser.add_argument("--force_max", type=float, default=0.0, help="Max external force (N). Set 0 for slope-only test.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidSlopedTerrainCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


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


def _force_to_quat(force_world: torch.Tensor, device: torch.device) -> torch.Tensor:
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        up = torch.tensor(
            [0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0],
            device=device,
        )
        z_ax = torch.linalg.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.linalg.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)
        quats[i] = quat_from_matrix(rot_mat)
    return quats


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── Configure slope terrain ──────────────────────────────────────────
    slope_rad = math.radians(args_cli.slope_deg)

    slope_terrain_cfg = TerrainGeneratorCfg(
        size=(12.0, 12.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "slope": HfPyramidSlopedTerrainCfg(
                proportion=1.0,
                slope_range=(slope_rad, slope_rad),
                platform_width=1.5,
                border_width=0.25,
            ),
        },
    )

    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = slope_terrain_cfg
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # Disable terrain curriculum
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None

    # ── Configure velocity commands: forward walk, then zero ─────────────
    # Start with forward command; we'll override to zero after walk_duration
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.walk_speed, args_cli.walk_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.debug_vis = True
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)

    # ── Configure external forces ────────────────────────────────────────
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    force_event = getattr(env_cfg.events, force_event_name)

    if args_cli.force_max > 0:
        force_event.params["force_range"] = (args_cli.force_min, args_cli.force_max)
        force_event.interval_range_s = (3.0, 5.0)
        from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_xyz_force
        env_cfg.events.base_external_force_torque.func = apply_persistent_xyz_force
        env_cfg.events.base_external_force_torque.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (args_cli.force_min, args_cli.force_max),
            "fz_scale": force_event.params.get("fz_scale", 0.6),
        }
    else:
        # No external forces — pure slope test
        force_event.params["force_range"] = (0.0, 0.0)

    # Long episodes
    total_duration = args_cli.walk_duration + args_cli.hold_duration
    env_cfg.episode_length_s = total_duration + 5.0

    # ── Resolve checkpoint ───────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[slope_eval_stage2] Checkpoint: {resume_path}")

    # ── Create env + runner ──────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    from go2_rl_lab.estimator.compliance_runner import ComplianceOnPolicyRunner
    train_cfg = agent_cfg.to_dict()
    if args_cli.stage1_checkpoint:
        train_cfg.setdefault("compliance", {})["stage1_checkpoint"] = args_cli.stage1_checkpoint
    runner = ComplianceOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.eval_mode()

    wrapper = runner._compliance_wrapper
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Access robot
    isaac_env = env.unwrapped
    device = isaac_env.device
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)

    force_dim = 3
    compliance_k = args_cli.compliance_k

    # Arrows
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))       # Red
    lin_markers = _create_arrow_markers("/World/Visuals/LinMapArrow", (0.0, 0.8, 0.8))       # Turquoise
    adj_markers = _create_arrow_markers("/World/Visuals/AdjVelArrow", (1.0, 0.85, 0.0))      # Yellow

    dt = isaac_env.step_dt
    n = args_cli.num_envs
    walk_steps = int(args_cli.walk_duration / dt)
    total_steps = int(total_duration / dt)

    obs = wrapper.get_observations()
    step_count = 0
    phase = "WALK"

    # Data logging
    plot_log = {
        "time_s": [], "phase": [],
        "gt_force_x": [], "gt_force_y": [],
        "est_force_x": [], "est_force_y": [],
        "lin_vel_x": [], "lin_vel_y": [],
        "learned_vel_x": [], "learned_vel_y": [],
        "gravity_bx": [], "gravity_by": [],
    }

    print(f"\n{'=' * 70}")
    print(f"  HAC-LOCO Stage 2 — Slope Eval")
    print(f"  Slope: {args_cli.slope_deg:.1f}°")
    print(f"  Phase 1: WALK uphill at {args_cli.walk_speed:.1f} m/s for {args_cli.walk_duration:.0f}s")
    print(f"  Phase 2: HOLD position on slope for {args_cli.hold_duration:.0f}s")
    if args_cli.force_max > 0:
        print(f"  External forces: [{args_cli.force_min:.0f}, {args_cli.force_max:.0f}] N")
    else:
        print(f"  External forces: NONE (pure slope gravity test)")
    print(f"  Arrows: RED=GT force  TURQUOISE=linear k*F  YELLOW=learned a'")
    print(f"  Linear mapping k={compliance_k:.4f}")
    print(f"{'=' * 70}\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # Switch to hold phase
                if step_count == walk_steps and phase == "WALK":
                    phase = "HOLD"
                    print(f"\n\n  >>> Switching to HOLD phase (zero velocity command) <<<\n")
                    # Override velocity commands to zero
                    cmd = isaac_env.command_manager.get_command("base_velocity")
                    cmd[:] = 0.0

                # In hold phase, keep commands at zero
                if phase == "HOLD":
                    cmd = isaac_env.command_manager.get_command("base_velocity")
                    cmd[:] = 0.0

                # High-level policy
                a_prime = policy(obs)
                obs, rewards, dones, extras = wrapper.step(a_prime)
                policy_nn.reset(dones)

                # GT force
                gt_force_body = asset.permanent_wrench_composer.composed_force_as_torch[
                    :n, base_idx, :force_dim
                ]
                if gt_force_body.dim() == 3:
                    gt_force_body = gt_force_body.squeeze(1)

                # Adjusted velocity (c* = c + a', in hold phase c=0 so c*=a')
                adj_vel = a_prime[:n].clone()

                # Force estimate
                force_hat = wrapper.estimator.get_latent(
                    wrapper.history_buffer.get_flattened()
                )[0][:n]

                # Linear mapping
                lin_vel = torch.zeros(n, 3, device=device)
                lin_vel[:, 0] = compliance_k * force_hat[:, 0]
                lin_vel[:, 1] = compliance_k * force_hat[:, 1]

                # Get projected gravity from obs for logging
                raw_obs = wrapper._last_raw_obs if hasattr(wrapper, '_last_raw_obs') else None

                # Transform to world frame for arrows
                base_pos = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]

                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :force_dim] = gt_force_body
                gt_world = quat_apply(base_quat, gt_3d)

                lin_world = quat_apply(base_quat, lin_vel)

                adj_3d = torch.zeros(n, 3, device=device)
                adj_3d[:, 0] = adj_vel[:, 0]
                adj_3d[:, 1] = adj_vel[:, 1]
                adj_world = quat_apply(base_quat, adj_3d)

                # Arrow positions
                gt_pos = base_pos.clone(); gt_pos[:, 2] += 0.55
                lin_pos = base_pos.clone(); lin_pos[:, 2] += 0.40
                adj_pos = base_pos.clone(); adj_pos[:, 2] += 0.25

                # Arrow orientations
                gt_quats = _force_to_quat(gt_world, device)
                lin_quats = _force_to_quat(lin_world, device)
                adj_quats = _force_to_quat(adj_world, device)

                # Arrow scales
                force_scale = 0.05
                vel_scale = 1.0

                gt_scales = torch.full((n, 3), 0.3, device=device)
                gt_scales[:, 0:1] = (gt_world.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)

                lin_scales = torch.full((n, 3), 0.3, device=device)
                lin_scales[:, 0:1] = (lin_world.norm(dim=-1, keepdim=True) * vel_scale).clamp(min=0.05)

                adj_scales = torch.full((n, 3), 0.3, device=device)
                adj_scales[:, 0:1] = (adj_world.norm(dim=-1, keepdim=True) * vel_scale).clamp(min=0.05)

                # Render
                gt_markers.visualize(gt_pos, gt_quats, gt_scales)
                lin_markers.visualize(lin_pos, lin_quats, lin_scales)
                adj_markers.visualize(adj_pos, adj_quats, adj_scales)

                # Log data (env 0)
                g0 = gt_force_body[0]
                e0 = force_hat[0]
                l0 = lin_vel[0]
                a0 = adj_vel[0]
                plot_log["time_s"].append(step_count * dt)
                plot_log["phase"].append(phase)
                plot_log["gt_force_x"].append(g0[0].item())
                plot_log["gt_force_y"].append(g0[1].item())
                plot_log["est_force_x"].append(e0[0].item())
                plot_log["est_force_y"].append(e0[1].item())
                plot_log["lin_vel_x"].append(l0[0].item())
                plot_log["lin_vel_y"].append(l0[1].item())
                plot_log["learned_vel_x"].append(a0[0].item())
                plot_log["learned_vel_y"].append(a0[1].item())

                # Terminal readout
                step_count += 1
                if step_count % 10 == 0:
                    r0 = rewards[0].item() if rewards.dim() > 0 else rewards.item()
                    pct = step_count / total_steps
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    print(
                        f"\r  [{bar}] {step_count * dt:.0f}/{total_duration:.0f}s [{phase:4s}]  "
                        f"est:[{e0[0]:+5.1f},{e0[1]:+5.1f}]N  "
                        f"lin:[{l0[0]:+5.2f},{l0[1]:+5.2f}]  "
                        f"a':[{a0[0]:+5.2f},{a0[1]:+5.2f}]  "
                        f"r={r0:+.2f}",
                        end="", flush=True,
                    )

                if step_count >= total_steps:
                    break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n\n[slope_eval_stage2] Stopped ({type(exc).__name__}).")

    print()
    env.close()

    from time import sleep
    sleep(1)

    # ── Generate plots ────────────────────────────────────────────────────
    if len(plot_log["time_s"]) > 10:
        import matplotlib.pyplot as plt
        from datetime import datetime

        t = np.array(plot_log["time_s"])
        est_fx = np.array(plot_log["est_force_x"])
        est_fy = np.array(plot_log["est_force_y"])
        lin_vx = np.array(plot_log["lin_vel_x"])
        lin_vy = np.array(plot_log["lin_vel_y"])
        learned_vx = np.array(plot_log["learned_vel_x"])
        learned_vy = np.array(plot_log["learned_vel_y"])
        phases = np.array(plot_log["phase"])

        # Walk/hold boundary
        t_switch = args_cli.walk_duration

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"Stage 2 Slope Eval — {args_cli.slope_deg:.0f}° slope, k={compliance_k:.4f}",
            fontsize=14, fontweight="bold",
        )

        # Panel 1: Estimated force
        ax = axes[0]
        ax.plot(t, est_fx, color="tab:red", linewidth=1.2, label="F_hat_x")
        ax.plot(t, est_fy, color="tab:blue", linewidth=1.2, label="F_hat_y")
        ax.axvline(t_switch, color="gray", linestyle="--", alpha=0.7, label="WALK→HOLD")
        ax.set_ylabel("Estimated Force (N)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title("Force Estimator Output (body frame)", fontsize=11)

        # Panel 2: X velocity
        ax = axes[1]
        ax.plot(t, lin_vx, color="tab:cyan", linewidth=1.2, label="Linear (k*F_hat_x)")
        ax.plot(t, learned_vx, color="tab:orange", linewidth=1.2, label="Learned (a'_x)")
        ax.axvline(t_switch, color="gray", linestyle="--", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Velocity cmd (m/s)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title("X component — should be ~0 on slope (gravity, not real force)", fontsize=11)

        # Panel 3: Y velocity
        ax = axes[2]
        ax.plot(t, lin_vy, color="tab:cyan", linewidth=1.2, label="Linear (k*F_hat_y)")
        ax.plot(t, learned_vy, color="tab:orange", linewidth=1.2, label="Learned (a'_y)")
        ax.axvline(t_switch, color="gray", linestyle="--", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Velocity cmd (m/s)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title("Y component", fontsize=11)

        plt.tight_layout()

        eval_dir = os.path.join(os.path.dirname(resume_path), "eval")
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig_path = os.path.join(
            eval_dir, f"slope_eval_{args_cli.slope_deg:.0f}deg_k{compliance_k:.4f}_{timestamp}.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[slope_eval_stage2] Plot saved: {fig_path}")
    else:
        print("[slope_eval_stage2] Too few steps, skipping plots.")


if __name__ == "__main__":
    main()
    simulation_app.close()
