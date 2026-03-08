"""Static eval for HAC-LOCO stage 2 — visualize high-level compliance policy output.

Standing robot with random XYZ forces. Shows what the high-level policy
outputs as residual velocity commands a' = [Dvx, Dvy, Dwz].

Arrows:
  RED    = GT applied force
  GREEN  = original velocity command (zero in static)
  YELLOW = adjusted velocity command c* = c + a' (from high-level policy)

Usage:
    python scripts/rsl_rl/static_eval_stage2.py --task Go2-HAC-LOCO-Stage2-v0 --num_envs 1 \
        --stage1_checkpoint /path/to/stage1/model.pt \
        --checkpoint /path/to/stage2/model.pt \
        --duration 20 --real-time
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Static eval for HAC-LOCO stage 2.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--force_min", type=float, default=10.0, help="Min force magnitude per axis (N).")
parser.add_argument("--force_max", type=float, default=20.0, help="Max force magnitude per axis (N).")
parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds.")
parser.add_argument("--real-time", action="store_true", default=False, help="Real-time pacing.")
parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Stage-1 checkpoint for frozen low-level.")
parser.add_argument("--compliance_k", type=float, default=0.06, help="Linear mapping gain k for comparison arrow.")
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

    # Static: zero velocity commands
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Activate forces
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    force_event = getattr(env_cfg.events, force_event_name)
    force_event.params["force_range"] = (args_cli.force_min, args_cli.force_max)
    force_event.interval_range_s = (1.0, 3.0)

    # Override reset event
    from go2_rl_lab.tasks.manager_based.go2_rl_lab.mdp.events import apply_persistent_xyz_force
    env_cfg.events.base_external_force_torque.func = apply_persistent_xyz_force
    env_cfg.events.base_external_force_torque.params = {
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "force_range": (args_cli.force_min, args_cli.force_max),
        "fz_scale": force_event.params.get("fz_scale", 0.6),
    }

    # Resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[static_eval_stage2] Checkpoint: {resume_path}")

    # Create env + runner
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    from go2_rl_lab.estimator.compliance_runner import ComplianceOnPolicyRunner
    train_cfg = agent_cfg.to_dict()
    if args_cli.stage1_checkpoint:
        train_cfg.setdefault("compliance", {})["stage1_checkpoint"] = args_cli.stage1_checkpoint
    runner = ComplianceOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.eval_mode()

    # The wrapper is the env the high-level policy sees
    wrapper = runner._compliance_wrapper

    # Get high-level inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Access robot for GT force + arrows
    isaac_env = env.unwrapped
    device = isaac_env.device
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)

    force_dim = 3

    # Linear mapping gain for comparison
    compliance_k = args_cli.compliance_k

    # Arrows
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))       # Red = GT force
    lin_markers = _create_arrow_markers("/World/Visuals/LinMapArrow", (0.0, 0.8, 0.8))       # Turquoise = linear k*F_hat
    adj_markers = _create_arrow_markers("/World/Visuals/AdjVelArrow", (1.0, 0.85, 0.0))      # Yellow = learned a'

    dt = isaac_env.step_dt
    n = args_cli.num_envs
    max_steps = int(args_cli.duration / dt)

    # Get initial high-level obs
    obs = wrapper.get_observations()
    step_count = 0

    # EMA state for linear mapping comparison
    force_ema = torch.zeros(n, force_dim, device=device)
    ema_alpha = 0.1

    print(f"\n{'=' * 70}")
    print(f"  HAC-LOCO Stage 2 — Static Eval")
    print(f"  Force range: [{args_cli.force_min:.0f}, {args_cli.force_max:.0f}] N")
    print(f"  High-level obs: {obs['policy'].shape[-1]} dims")
    print(f"  High-level action: 3 dims [Dvx, Dvy, Dwz]")
    print(f"  Linear mapping k={compliance_k:.4f} (for comparison)")
    print(f"  Arrows: RED=GT force  TURQUOISE=linear k*F  YELLOW=learned a'")
    print(f"  Duration: {args_cli.duration:.0f}s ({max_steps} steps)")
    print(f"{'=' * 70}\n")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # High-level policy outputs a' = [Dvx, Dvy, Dwz]
                a_prime = policy(obs)

                # Step through wrapper (runs frozen low-level internally)
                obs, rewards, dones, extras = wrapper.step(a_prime)
                policy_nn.reset(dones)

                # GT force
                gt_force_body = asset.permanent_wrench_composer.composed_force_as_torch[
                    :n, base_idx, :force_dim
                ]
                if gt_force_body.dim() == 3:
                    gt_force_body = gt_force_body.squeeze(1)

                # The adjusted velocity is c* = c + a'
                # In static eval, c = 0, so c* = a'
                adj_vel = a_prime[:n].clone()

                # Get force estimate from the wrapper's estimator for linear mapping
                force_hat = wrapper.estimator.get_latent(
                    wrapper.history_buffer.get_flattened()
                )[0][:n]  # [n, 3]
                force_ema = ema_alpha * force_hat + (1.0 - ema_alpha) * force_ema

                # Linear mapping: v_lin = k * EMA(F_hat_xy)
                lin_vel = torch.zeros(n, 3, device=device)
                lin_vel[:, 0] = compliance_k * force_ema[:, 0]
                lin_vel[:, 1] = compliance_k * force_ema[:, 1]

                # Transform to world frame for arrows
                base_pos = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]

                # GT force arrow (RED)
                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :force_dim] = gt_force_body
                gt_world = quat_apply(base_quat, gt_3d)

                # Linear mapping arrow (TURQUOISE) — k * F_hat
                lin_world = quat_apply(base_quat, lin_vel)

                # Learned policy arrow (YELLOW) — a' from high-level
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

                # Terminal readout
                step_count += 1
                if step_count % 10 == 0:
                    g0 = gt_force_body[0]
                    a0 = adj_vel[0]
                    l0 = lin_vel[0]
                    r0 = rewards[0].item() if rewards.dim() > 0 else rewards.item()
                    pct = step_count / max_steps
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                    print(
                        f"\r  [{bar}] {step_count * dt:.0f}/{args_cli.duration:.0f}s  "
                        f"GT:[{g0[0]:+5.1f},{g0[1]:+5.1f}]N  "
                        f"lin:[{l0[0]:+5.2f},{l0[1]:+5.2f}]  "
                        f"a':[{a0[0]:+5.2f},{a0[1]:+5.2f}]  "
                        f"r={r0:+.2f}",
                        end="", flush=True,
                    )

                if step_count >= max_steps:
                    break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n\n[static_eval_stage2] Stopped ({type(exc).__name__}).")

    print()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
