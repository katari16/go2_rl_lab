"""Static 360-degree force sweep evaluation with elevation + torque support.

Robot stands still while forces are applied from 10 azimuth directions at
multiple magnitudes. For 3D+ estimators, forces are also swept across
elevation angles. For 4D/6D estimators, a separate torque-only sweep follows.

All (magnitude, direction, elevation, trial) combos run in parallel across envs.
Records raw time-series data; metrics computed post-hoc by analyze_static_360.py.

Usage:
    python scripts/rsl_rl/eval/static_360_eval.py --task Go2-LowLevel-v0 \
        --checkpoint logs/rsl_rl/.../model_XXXX.pt \
        --force_magnitudes 5 10 15 20 25 --num_trials 20
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Static 360-degree force sweep evaluation.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--force_magnitudes", type=float, nargs="+", default=[5, 10, 15, 20, 25])
parser.add_argument("--pull", action="store_true", default=False, help="Flip force direction.")
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--force_hold_s", type=float, default=4.0)
parser.add_argument("--warmup_s", type=float, default=3.0)
parser.add_argument("--elevation_angles", type=float, nargs="+", default=None,
                    help="Elevation angles in degrees (default: auto from force_dim).")
parser.add_argument("--torque_magnitudes", type=float, nargs="+", default=[1, 2, 3, 5])
parser.add_argument("--skip_torque_sweep", action="store_true", default=False)
parser.add_argument("--linear_modulation", action="store_true", default=False,
                    help="Enable force estimate → velocity command modulation.")
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

import json
import math
import os
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
    clear_force,
    create_arrow_markers,
    create_eval_output_dir,
    create_runner,
    disable_force_events,
    update_force_arrows_per_env,
    get_asset_and_base,
    reset_env,
    resolve_checkpoint,
    save_config,
    set_flat_terrain,
    set_long_episode,
    set_standing_commands,
    setup_runner_for_eval,
    step_policy,
)
from eval.eval_utils import _NumpyEncoder

NUM_DIRECTIONS = 10
DIRECTIONS_DEG = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)
DIRECTIONS_RAD = np.deg2rad(DIRECTIONS_DEG)

TORQUE_AXIS_MAP = {"roll": 0, "pitch": 1, "yaw": 2}


def _run_phase(phase_name, n_active, force_tensor, torque_tensor, asset, base_idx,
               obs, ctx, env, runner, isaac_env, runner_class_name, device, n_total,
               force_steps, dt, compliance_k, ema_alpha, force_dim, has_estimator,
               force_arrow, fx_per_env, fy_per_env, real_time):
    """Run one simulation phase (force sweep or torque sweep).

    Returns:
        (all_vel, all_pos, all_force_est, fell, fell_step)
    """
    all_vel = np.zeros((force_steps, n_active, 2))
    all_pos = np.zeros((force_steps, n_active, 2))
    all_force_est = np.zeros((force_steps, n_active, force_dim)) if has_estimator else None
    fell = torch.zeros(n_active, dtype=torch.bool, device=device)
    fell_step = torch.full((n_active,), force_steps, dtype=torch.long, device=device)

    print(f"  [{phase_name}] Applying for {force_steps * dt:.1f}s...", flush=True)
    for step in range(force_steps):
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force_tensor, torques=torque_tensor,
        )

        if fx_per_env is not None and fy_per_env is not None:
            update_force_arrows_per_env(force_arrow, asset, fx_per_env, fy_per_env, device, n_total)
        else:
            force_arrow.set_visibility(False)

        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n_total,
                                 runner_class_name, compliance_k, ema_alpha)

        newly_fell = (dones[:n_active] > 0) & ~fell
        fell_step[newly_fell] = step
        fell |= (dones[:n_active] > 0)

        all_vel[step] = asset.data.root_lin_vel_b[:n_active, :2].cpu().numpy()
        all_pos[step] = asset.data.root_pos_w[:n_active, :2].cpu().numpy()
        if has_estimator:
            all_force_est[step] = isaac_env._force_estimate_xy[:n_active].cpu().numpy()

        if real_time:
            time.sleep(dt)

    clear_force(asset, base_idx, device, n_total)
    force_arrow.set_visibility(False)
    print(f"  [{phase_name}] Done. Fell: {fell.sum().item()}/{n_active} envs")

    return all_vel, all_pos, all_force_est, fell, fell_step


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    magnitudes = args_cli.force_magnitudes
    num_trials = args_cli.num_trials
    sign = -1.0 if args_cli.pull else 1.0
    torque_mags = args_cli.torque_magnitudes

    # We need to create the env first to get force_dim from the runner,
    # but env count depends on force_dim. Use a temporary small env to probe.
    # Actually, we can infer force_dim from the agent_cfg before env creation.
    est_cfg = agent_cfg.to_dict().get("estimator", {})
    force_dim = est_cfg.get("force_dim", 3)
    force_layout = est_cfg.get("force_layout", "auto")

    # ── Determine elevation angles ──────────────────────────────────────
    if args_cli.elevation_angles is not None:
        elevation_angles_deg = args_cli.elevation_angles
    elif force_dim >= 3 and force_layout != "xy_yaw":
        elevation_angles_deg = [0, 15, 30, 45, 75]
    else:
        elevation_angles_deg = [0]
    elevation_angles_rad = np.deg2rad(elevation_angles_deg)

    # ── Determine torque sweep ──────────────────────────────────────────
    if (force_dim >= 4 or force_layout == "xy_yaw") and not args_cli.skip_torque_sweep:
        do_torque = True
        if force_dim >= 6:
            torque_axes = ["roll", "pitch", "yaw"]
        else:
            torque_axes = ["yaw"]
    else:
        do_torque = False
        torque_axes = []

    # ── Compute env counts ──────────────────────────────────────────────
    n_elev = len(elevation_angles_deg)
    n_force = len(magnitudes) * NUM_DIRECTIONS * n_elev * num_trials
    n_torque = len(torque_mags) * 2 * len(torque_axes) * num_trials if do_torque else 0
    n = max(n_force, n_torque)
    env_cfg.scene.num_envs = n

    set_standing_commands(env_cfg)
    set_flat_terrain(env_cfg)
    set_long_episode(env_cfg)
    disable_force_events(env_cfg, agent_cfg)

    resume_path = resolve_checkpoint(agent_cfg, args_cli)
    print(f"[static_360] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner, runner_class_name, is_stage2 = create_runner(env, agent_cfg, args_cli)
    runner.load(resume_path)
    runner.eval_mode()

    ctx, env = setup_runner_for_eval(runner, env, runner_class_name, is_stage2,
                                     env.unwrapped.device, n)

    isaac_env = env.unwrapped
    device = isaac_env.device
    dt = isaac_env.step_dt
    asset, base_idx = get_asset_and_base(isaac_env)

    warmup_steps = int(args_cli.warmup_s / dt)
    force_steps = int(args_cli.force_hold_s / dt)
    compliance_k = args_cli.compliance_k if args_cli.linear_modulation else 0.0
    has_estimator = ctx.get("has_estimator", False)

    print(f"\n{'=' * 70}")
    print(f"  Static 360 Evaluation")
    print(f"  Force dim: {force_dim}  Layout: {force_layout}")
    print(f"  Directions: {NUM_DIRECTIONS} (every 36 deg)")
    print(f"  Elevations: {elevation_angles_deg} deg")
    print(f"  Magnitudes: {magnitudes} N")
    print(f"  Trials/config: {num_trials}")
    print(f"  Force envs: {n_force}  Torque envs: {n_torque}  Total: {n}")
    print(f"  Torque sweep: {do_torque}  Axes: {torque_axes}  Mags: {torque_mags if do_torque else 'n/a'}")
    print(f"  Force hold: {args_cli.force_hold_s}s  Warmup: {args_cli.warmup_s}s")
    print(f"  Mode: {'PULL' if args_cli.pull else 'PUSH'}")
    print(f"  Runner: {runner_class_name}")
    print(f"{'=' * 70}\n")

    force_arrow = create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Force sweep
    # ═══════════════════════════════════════════════════════════════════

    # ── Build per-env force assignments ─────────────────────────────────
    fx_per_env = torch.zeros(n, device=device)
    fy_per_env = torch.zeros(n, device=device)
    fz_per_env = torch.zeros(n, device=device)
    env_mag = []
    env_dir = []
    env_elev = []
    env_trial = []

    idx = 0
    for mag in magnitudes:
        for deg, azim_rad in zip(DIRECTIONS_DEG, DIRECTIONS_RAD):
            for elev_deg, elev_rad in zip(elevation_angles_deg, elevation_angles_rad):
                for trial in range(num_trials):
                    cos_elev = math.cos(elev_rad)
                    sin_elev = math.sin(elev_rad)
                    fx_per_env[idx] = sign * mag * cos_elev * math.cos(azim_rad)
                    fy_per_env[idx] = sign * mag * cos_elev * math.sin(azim_rad)
                    fz_per_env[idx] = sign * mag * sin_elev
                    env_mag.append(mag)
                    env_dir.append(deg)
                    env_elev.append(elev_deg)
                    env_trial.append(trial)
                    idx += 1

    # ── Reset + warmup ──────────────────────────────────────────────────
    obs = reset_env(env, ctx, isaac_env, runner, runner_class_name,
                    is_stage2, device, n)
    print(f"  [Force] Warmup ({args_cli.warmup_s}s)...", flush=True)
    for _ in range(warmup_steps):
        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, compliance_k, args_cli.ema_alpha)

    pos_start = asset.data.root_pos_w[:n_force, :2].cpu().numpy().copy()

    # ── Build force tensors ─────────────────────────────────────────────
    force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    force_tensor[:n_force, base_idx, 0] = fx_per_env[:n_force]
    force_tensor[:n_force, base_idx, 1] = fy_per_env[:n_force]
    force_tensor[:n_force, base_idx, 2] = fz_per_env[:n_force]

    # ── Run force phase ─────────────────────────────────────────────────
    all_vel, all_pos, all_force_est, fell, fell_step = _run_phase(
        "Force", n_force, force_tensor, torque_tensor, asset, base_idx,
        obs, ctx, env, runner, isaac_env, runner_class_name, device, n,
        force_steps, dt, compliance_k, args_cli.ema_alpha, force_dim,
        has_estimator, force_arrow, fx_per_env[:n], fy_per_env[:n],
        args_cli.real_time,
    )

    clear_force(asset, base_idx, device, n)

    # ── Pack force sweep results ────────────────────────────────────────
    force_sweep = {}
    fell_np = fell.cpu().numpy()
    fell_step_np = fell_step.cpu().numpy()

    for i in range(n_force):
        mag_str = str(float(env_mag[i]))
        deg_str = str(float(env_dir[i]))
        elev_str = str(float(env_elev[i]))
        force_sweep.setdefault(mag_str, {}).setdefault(deg_str, {}).setdefault(elev_str, [])

        valid_steps = fell_step_np[i]
        trial_data = {
            "trial": env_trial[i],
            "success": not fell_np[i],
            "pos_start": pos_start[i],
            "force_xyz": [fx_per_env[i].item(), fy_per_env[i].item(), fz_per_env[i].item()],
            "force_xy": [fx_per_env[i].item(), fy_per_env[i].item()],
            "vel_xy": all_vel[:valid_steps, i, :],
            "pos_xy": all_pos[:valid_steps, i, :],
            "dt": dt,
        }
        if has_estimator:
            trial_data["force_est"] = all_force_est[:valid_steps, i, :]
        force_sweep[mag_str][deg_str][elev_str].append(trial_data)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Torque sweep (if applicable)
    # ═══════════════════════════════════════════════════════════════════

    torque_sweep = None
    if do_torque and n_torque > 0:
        print(f"\n  [Torque] Starting torque sweep ({len(torque_axes)} axes, "
              f"{torque_mags} Nm)...", flush=True)

        # Build per-env torque assignments
        tq_axis_per_env = []
        tq_mag_per_env = []
        tq_sign_per_env = []
        tq_trial_per_env = []
        tq_vec_per_env = torch.zeros(n_torque, 3, device=device)

        idx = 0
        for tq_mag in torque_mags:
            for tq_sign_val, tq_sign_label in [(1.0, "+"), (-1.0, "-")]:
                for axis_name in torque_axes:
                    axis_idx = TORQUE_AXIS_MAP[axis_name]
                    for trial in range(num_trials):
                        tq_axis_per_env.append(axis_name)
                        tq_mag_per_env.append(tq_mag)
                        tq_sign_per_env.append(tq_sign_label)
                        tq_trial_per_env.append(trial)
                        tq_vec_per_env[idx, axis_idx] = tq_sign_val * tq_mag
                        idx += 1

        # Reset + warmup
        obs = reset_env(env, ctx, isaac_env, runner, runner_class_name,
                        is_stage2, device, n)
        print(f"  [Torque] Warmup ({args_cli.warmup_s}s)...", flush=True)
        for _ in range(warmup_steps):
            obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                     runner_class_name, compliance_k, args_cli.ema_alpha)

        tq_pos_start = asset.data.root_pos_w[:n_torque, :2].cpu().numpy().copy()

        # Build torque-only tensors
        force_tensor_tq = torch.zeros(n, asset.num_bodies, 3, device=device)
        torque_tensor_tq = torch.zeros(n, asset.num_bodies, 3, device=device)
        for i in range(n_torque):
            torque_tensor_tq[i, base_idx, :] = tq_vec_per_env[i]

        tq_vel, tq_pos, tq_est, tq_fell, tq_fell_step = _run_phase(
            "Torque", n_torque, force_tensor_tq, torque_tensor_tq, asset, base_idx,
            obs, ctx, env, runner, isaac_env, runner_class_name, device, n,
            force_steps, dt, compliance_k, args_cli.ema_alpha, force_dim,
            has_estimator, force_arrow, None, None, args_cli.real_time,
        )

        # Pack torque sweep results
        torque_sweep = {}
        tq_fell_np = tq_fell.cpu().numpy()
        tq_fell_step_np = tq_fell_step.cpu().numpy()

        for i in range(n_torque):
            axis = tq_axis_per_env[i]
            mag_str = str(float(tq_mag_per_env[i]))
            sign_str = tq_sign_per_env[i]
            torque_sweep.setdefault(axis, {}).setdefault(mag_str, {}).setdefault(sign_str, [])

            valid = tq_fell_step_np[i]
            trial_data = {
                "trial": tq_trial_per_env[i],
                "success": not tq_fell_np[i],
                "pos_start": tq_pos_start[i],
                "torque_axis": axis,
                "torque_mag": tq_mag_per_env[i],
                "torque_sign": sign_str,
                "torque_vec": tq_vec_per_env[i].cpu().numpy(),
                "vel_xy": tq_vel[:valid, i, :],
                "pos_xy": tq_pos[:valid, i, :],
                "dt": dt,
            }
            if has_estimator:
                trial_data["force_est"] = tq_est[:valid, i, :]
            torque_sweep[axis][mag_str][sign_str].append(trial_data)

    env.close()

    # ── Save data (v2 format) ───────────────────────────────────────────
    metadata = {
        "version": 2,
        "force_dim": force_dim,
        "force_layout": force_layout,
        "elevation_angles_deg": elevation_angles_deg,
        "torque_axes": torque_axes,
        "torque_magnitudes": torque_mags if do_torque else [],
        "task": args_cli.task,
    }

    task_short = args_cli.task.replace("Go2-Ablation-", "").replace("-v0", "")
    modulation_tag = "mapping" if args_cli.linear_modulation else "nomapping"
    max_force = int(max(magnitudes))
    dir_suffix = f"{task_short}_{modulation_tag}_{max_force}N"
    output_dir = create_eval_output_dir(agent_cfg.experiment_name, "static_360", suffix=dir_suffix)
    save_config(output_dir, args_cli, agent_cfg, dt)

    out_data = {
        "metadata": metadata,
        "force_sweep": force_sweep,
        "torque_sweep": torque_sweep,
    }
    path = os.path.join(output_dir, "raw_data.json")
    with open(path, "w") as f:
        json.dump(out_data, f, indent=1, cls=_NumpyEncoder)
    print(f"[static_360] Raw data saved to {path}")

    print(f"\n[static_360] Output: {output_dir}")
    print(f"  Run analysis with:")
    print(f"    python scripts/rsl_rl/eval/analyze_static_360.py {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
