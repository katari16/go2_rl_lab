"""Impulse 360-degree evaluation — brief force from all directions while walking.

Robot walks forward at a fixed speed, then a brief force impulse (0.5s) is applied
from 10 directions (36-degree increments). Measures decay time, peak lateral
deviation, success rate, and power consumption during recovery.

Inspired by Deep Compliant Control (Hartmann et al.) Fig. 4b — decay time polar
plot while walking.

Generates a PDF report with:
  - Decay time polar plot (mean +/- std)
  - Peak deviation polar plot
  - Success rate polar plot
  - Summary table

Raw data is saved as JSON for offline re-plotting.

Usage:
    python scripts/rsl_rl/eval/impulse_360_eval.py --task Go2-LowLevel-v0 \\
        --checkpoint logs/rsl_rl/.../model_XXXX.pt \\
        --impulse_magnitude 15 --num_trials 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Impulse 360-degree evaluation while walking.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--impulse_magnitude", type=float, default=15.0)
parser.add_argument("--impulse_duration_s", type=float, default=0.5)
parser.add_argument("--recovery_s", type=float, default=4.0)
parser.add_argument("--walk_speed", type=float, default=0.5)
parser.add_argument("--warmup_s", type=float, default=3.0)
parser.add_argument("--num_trials", type=int, default=50)
parser.add_argument("--compliance_k", type=float, default=0.06)
parser.add_argument("--ema_alpha", type=float, default=0.1)
parser.add_argument("--stage1_checkpoint", type=str, default=None)
parser.add_argument("--estimator_checkpoint", type=str, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401

from eval.eval_utils import (
    apply_force_xy,
    clear_force,
    compute_decay_time,
    compute_peak_displacement,
    create_arrow_markers,
    create_eval_output_dir,
    create_runner,
    create_trajectory_markers,
    disable_force_events,
    get_asset_and_base,
    make_summary_table,
    polar_plot,
    reset_env,
    resolve_checkpoint,
    save_config,
    save_raw_data,
    save_report_pdf,
    set_flat_terrain,
    set_long_episode,
    set_walking_commands,
    setup_runner_for_eval,
    step_policy,
    update_force_arrow,
    update_trajectory_vis,
)

NUM_DIRECTIONS = 10
DIRECTIONS_DEG = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)
DIRECTIONS_RAD = np.deg2rad(DIRECTIONS_DEG)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    set_walking_commands(env_cfg, args_cli.walk_speed)
    set_flat_terrain(env_cfg)
    set_long_episode(env_cfg)
    disable_force_events(env_cfg, agent_cfg)

    resume_path = resolve_checkpoint(agent_cfg, args_cli)
    print(f"[impulse_360] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner, runner_class_name, is_stage2 = create_runner(env, agent_cfg, args_cli)
    runner.load(resume_path)
    runner.eval_mode()

    ctx, env = setup_runner_for_eval(runner, env, runner_class_name, is_stage2,
                                     env.unwrapped.device, args_cli.num_envs)

    isaac_env = env.unwrapped
    device = isaac_env.device
    n = args_cli.num_envs
    dt = isaac_env.step_dt
    asset, base_idx = get_asset_and_base(isaac_env)

    warmup_steps = int(args_cli.warmup_s / dt)
    impulse_steps = int(args_cli.impulse_duration_s / dt)
    recovery_steps = int(args_cli.recovery_s / dt)

    impulse_mag = args_cli.impulse_magnitude
    num_trials = args_cli.num_trials

    print(f"\n{'=' * 70}")
    print(f"  Impulse 360 Evaluation")
    print(f"  Directions: {NUM_DIRECTIONS} (every 36 deg)")
    print(f"  Impulse: {impulse_mag:.1f}N for {args_cli.impulse_duration_s}s")
    print(f"  Walk speed: {args_cli.walk_speed} m/s")
    print(f"  Trials/direction: {num_trials}")
    print(f"  Recovery window: {args_cli.recovery_s}s")
    print(f"  Runner: {runner_class_name}")
    print(f"{'=' * 70}\n")

    # ── Visualization ────────────────────────────────────────────────────
    force_arrow = create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))
    NUM_TRAJ_PTS = 100
    traj_markers = create_trajectory_markers("/World/Visuals/Trajectory", NUM_TRAJ_PTS, device)

    # results[mag_str][deg_str] = [list of trial dicts]
    mag_str = str(float(impulse_mag))
    results = {mag_str: {}}

    for deg, rad in zip(DIRECTIONS_DEG, DIRECTIONS_RAD):
        deg_str = str(float(deg))
        results[mag_str][deg_str] = []
        fx = impulse_mag * math.cos(rad)
        fy = impulse_mag * math.sin(rad)

        for trial in range(num_trials):
            print(f"  Dir={deg:5.1f} deg  Trial {trial + 1}/{num_trials}  ", end="", flush=True)

            # Reset robot to default pose
            obs = reset_env(env, ctx, isaac_env, runner, runner_class_name,
                            is_stage2, device, n)

            # ── Warmup ───────────────────────────────────────────────
            for _ in range(warmup_steps):
                obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                         runner_class_name, args_cli.compliance_k,
                                         args_cli.ema_alpha)

            pos_pre = asset.data.root_pos_w[0, :2].cpu().numpy().copy()
            # Draw desired trajectory line ahead
            qw_t, qx_t, qy_t, qz_t = asset.data.root_quat_w[0].cpu().numpy()
            yaw_t = math.atan2(2 * (qw_t * qz_t + qx_t * qy_t), 1 - 2 * (qy_t * qy_t + qz_t * qz_t))
            traj_pts = [(pos_pre[0] + i * 0.3 * math.cos(yaw_t),
                         pos_pre[1] + i * 0.3 * math.sin(yaw_t)) for i in range(NUM_TRAJ_PTS)]
            update_trajectory_vis(traj_markers, traj_pts, 0.01, device)
            # Walk direction for computing lateral deviation
            qw, qx, qy, qz = asset.data.root_quat_w[0].cpu().numpy()
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            walk_dir = np.array([math.cos(yaw), math.sin(yaw)])
            perp_dir = np.array([-walk_dir[1], walk_dir[0]])

            fell = False

            # ── Impulse phase ────────────────────────────────────────
            imp_vel, imp_pos, imp_forces, imp_time = [], [], [], []

            for step in range(impulse_steps):
                apply_force_xy(asset, base_idx, fx, fy, device, n)
                obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                         runner_class_name, args_cli.compliance_k,
                                         args_cli.ema_alpha)

                if step % 5 == 0:
                    update_force_arrow(force_arrow, asset, base_idx, fx, fy, device, n)

                if dones[0] > 0:
                    fell = True
                    break

                v = asset.data.root_lin_vel_b[0, :2].cpu().numpy()
                p = asset.data.root_pos_w[0, :2].cpu().numpy()
                imp_vel.append(v.copy())
                imp_pos.append(p.copy())
                imp_forces.append([fx, fy])
                imp_time.append(step * dt)

                if args_cli.real_time:
                    time.sleep(dt)

            clear_force(asset, base_idx, device, n)
            update_force_arrow(force_arrow, asset, base_idx, 0.0, 0.0, device, n)

            # ── Recovery phase ───────────────────────────────────────
            rec_vel, rec_pos, rec_time = [], [], []
            rec_torques, rec_joint_vel = [], []

            if not fell:
                for step in range(recovery_steps):
                    obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                             runner_class_name, args_cli.compliance_k,
                                             args_cli.ema_alpha)
                    if dones[0] > 0:
                        fell = True
                        break

                    v = asset.data.root_lin_vel_b[0, :2].cpu().numpy()
                    p = asset.data.root_pos_w[0, :2].cpu().numpy()
                    rec_vel.append(v.copy())
                    rec_pos.append(p.copy())
                    rec_time.append(step * dt)

                    # Power tracking
                    torques = asset.data.applied_torque[0].cpu().numpy()
                    dq = asset.data.joint_vel[0].cpu().numpy()
                    rec_torques.append(np.sum(np.abs(torques * dq)))

                    if args_cli.real_time:
                        time.sleep(dt)

            # ── Compute metrics ──────────────────────────────────────
            imp_vel_arr = np.array(imp_vel) if imp_vel else np.zeros((0, 2))
            imp_pos_arr = np.array(imp_pos) if imp_pos else np.zeros((0, 2))
            imp_forces_arr = np.array(imp_forces) if imp_forces else np.zeros((0, 2))
            rec_vel_arr = np.array(rec_vel) if rec_vel else np.zeros((0, 2))
            rec_pos_arr = np.array(rec_pos) if rec_pos else np.zeros((0, 2))

            if fell or len(rec_vel_arr) == 0:
                trial_data = {
                    "trial": trial, "success": False,
                    "decay_time": args_cli.recovery_s,
                    "peak_lateral_dev": 0.0, "mean_power": 0.0,
                    "force_on": {
                        "vel_xy": imp_vel_arr, "pos_xy": imp_pos_arr,
                        "forces_xy": imp_forces_arr, "time_s": np.array(imp_time),
                    },
                    "recovery": {
                        "vel_xy": rec_vel_arr, "pos_xy": rec_pos_arr,
                        "time_s": np.array(rec_time),
                    },
                }
                results[mag_str][deg_str].append(trial_data)
                print("FELL")
                continue

            # Velocity magnitude during recovery
            rec_vel_mag = np.linalg.norm(rec_vel_arr, axis=1)
            decay_t = compute_decay_time(rec_vel_mag, dt)

            # Peak lateral deviation from the walking line
            all_pos = np.vstack([imp_pos_arr, rec_pos_arr]) if len(imp_pos_arr) > 0 else rec_pos_arr
            deltas = all_pos - pos_pre
            lat_devs = np.abs(deltas @ perp_dir)
            peak_lat_dev = float(np.max(lat_devs))

            # Mean power during recovery
            mean_power = float(np.mean(rec_torques)) if rec_torques else 0.0

            trial_data = {
                "trial": trial, "success": True,
                "decay_time": decay_t, "peak_lateral_dev": peak_lat_dev,
                "mean_power": mean_power,
                "force_on": {
                    "vel_xy": imp_vel_arr, "pos_xy": imp_pos_arr,
                    "forces_xy": imp_forces_arr, "time_s": np.array(imp_time),
                },
                "recovery": {
                    "vel_xy": rec_vel_arr, "pos_xy": rec_pos_arr,
                    "time_s": np.array(rec_time),
                },
            }
            results[mag_str][deg_str].append(trial_data)
            print(f"decay={decay_t:.2f}s  lat_dev={peak_lat_dev:.3f}m  power={mean_power:.1f}W")

    env.close()

    # ── Save raw data ────────────────────────────────────────────────────
    output_dir = create_eval_output_dir(agent_cfg.experiment_name, "impulse_360")
    save_config(output_dir, args_cli, agent_cfg, dt)
    save_raw_data(output_dir, results)

    # ── Generate plots ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []

    # ── Page 1: Decay time polar (DMC Fig. 4b) ──────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.suptitle(f"Impulse Decay Time While Walking ({impulse_mag:.0f}N, {args_cli.impulse_duration_s}s)",
                 fontsize=14, fontweight="bold")
    decay_vals = {d: [t["decay_time"] for t in results[mag_str][str(float(d))] if t["success"]]
                  for d in DIRECTIONS_DEG}
    polar_plot(ax, DIRECTIONS_DEG, decay_vals, "Decay Time", "seconds", "tab:purple")
    plt.tight_layout()
    figures.append(fig)

    # ── Page 2: Peak deviation polar ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.suptitle(f"Peak Lateral Deviation ({impulse_mag:.0f}N impulse)",
                 fontsize=14, fontweight="bold")
    dev_vals = {d: [t["peak_lateral_dev"] for t in results[mag_str][str(float(d))] if t["success"]]
                for d in DIRECTIONS_DEG}
    polar_plot(ax, DIRECTIONS_DEG, dev_vals, "Peak Lateral Deviation", "meters", "tab:blue")
    plt.tight_layout()
    figures.append(fig)

    # ── Page 3: Success rate polar ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.suptitle(f"Success Rate ({impulse_mag:.0f}N impulse)",
                 fontsize=14, fontweight="bold")
    succ_vals = {d: [np.mean([t["success"] for t in results[mag_str][str(float(d))]]) * 100]
                 for d in DIRECTIONS_DEG}
    # For success rate use a single-value list, but polar_plot expects multiple values
    # Wrap as repeated single value for std=0
    succ_per_dir = {d: [t["success"] * 100.0 for t in results[mag_str][str(float(d))]]
                    for d in DIRECTIONS_DEG}
    polar_plot(ax, DIRECTIONS_DEG, succ_per_dir, "Success Rate", "%", "tab:green")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 4: Summary table ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    headers = ["Direction", "Success %", "Decay Time (s)", "Peak Dev (m)", "Power (W)"]
    rows = []
    for deg in DIRECTIONS_DEG:
        deg_str = str(float(deg))
        trials = results[mag_str][deg_str]
        succ = [t for t in trials if t["success"]]
        rows.append([
            f"{deg:.0f} deg",
            f"{np.mean([t['success'] for t in trials]) * 100:.0f}%",
            f"{np.mean([t['decay_time'] for t in succ]):.2f} +/- {np.std([t['decay_time'] for t in succ]):.2f}" if succ else "N/A",
            f"{np.mean([t['peak_lateral_dev'] for t in succ]):.3f} +/- {np.std([t['peak_lateral_dev'] for t in succ]):.3f}" if succ else "N/A",
            f"{np.mean([t['mean_power'] for t in succ]):.1f} +/- {np.std([t['mean_power'] for t in succ]):.1f}" if succ else "N/A",
        ])
    make_summary_table(ax, headers, rows, f"Impulse 360 Summary ({impulse_mag:.0f}N)")
    plt.tight_layout()
    figures.append(fig)

    save_report_pdf(output_dir, figures)
    for fig in figures:
        plt.close(fig)

    print(f"\n[impulse_360] All results saved to {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
