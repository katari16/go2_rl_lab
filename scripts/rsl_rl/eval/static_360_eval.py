"""Static 360-degree force sweep evaluation.

Robot stands still while forces are applied from 10 directions at multiple
magnitudes. All (magnitude, direction, trial) combos run in parallel across
envs. Records raw time-series data; metrics computed post-hoc.

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
    save_raw_data,
    set_flat_terrain,
    set_long_episode,
    set_standing_commands,
    setup_runner_for_eval,
    step_policy,
    update_force_arrow,
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
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    magnitudes = args_cli.force_magnitudes
    num_trials = args_cli.num_trials
    sign = -1.0 if args_cli.pull else 1.0

    n = len(magnitudes) * NUM_DIRECTIONS * num_trials
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

    # ── Build per-env force assignments ──────────────────────────────────
    # Layout: env_idx = mag_idx * (NUM_DIRECTIONS * num_trials) + dir_idx * num_trials + trial_idx
    fx_per_env = torch.zeros(n, device=device)
    fy_per_env = torch.zeros(n, device=device)
    env_mag = []   # magnitude for each env
    env_dir = []   # direction (deg) for each env
    env_trial = [] # trial index for each env

    idx = 0
    for mag in magnitudes:
        for deg, rad in zip(DIRECTIONS_DEG, DIRECTIONS_RAD):
            for trial in range(num_trials):
                fx_per_env[idx] = sign * mag * math.cos(rad)
                fy_per_env[idx] = sign * mag * math.sin(rad)
                env_mag.append(mag)
                env_dir.append(deg)
                env_trial.append(trial)
                idx += 1

    print(f"\n{'=' * 70}")
    print(f"  Static 360 Evaluation — PARALLEL")
    print(f"  Directions: {NUM_DIRECTIONS} (every 36 deg)")
    print(f"  Magnitudes: {magnitudes} N")
    print(f"  Trials/config: {num_trials}")
    print(f"  Total envs: {n}")
    print(f"  Force hold: {args_cli.force_hold_s}s  Warmup: {args_cli.warmup_s}s")
    print(f"  Mode: {'PULL' if args_cli.pull else 'PUSH'}")
    print(f"  Linear modulation: {args_cli.linear_modulation} (k={compliance_k})")
    print(f"  Runner: {runner_class_name}")
    print(f"{'=' * 70}\n")

    force_arrow = create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))

    # ── Reset all envs ───────────────────────────────────────────────────
    obs = reset_env(env, ctx, isaac_env, runner, runner_class_name,
                    is_stage2, device, n)

    # ── Warmup: let all robots settle ────────────────────────────────────
    print(f"  Warmup ({args_cli.warmup_s}s)...", flush=True)
    for _ in range(warmup_steps):
        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, compliance_k,
                                 args_cli.ema_alpha)

    # Record start positions for all envs
    pos_start = asset.data.root_pos_w[:n, :2].cpu().numpy().copy()  # [n, 2]

    # ── Apply forces and record ──────────────────────────────────────────
    # Pre-allocate storage: [force_steps, n, ...]
    all_vel = np.zeros((force_steps, n, 2))
    all_pos = np.zeros((force_steps, n, 2))
    force_dim = ctx.get("force_dim", 3)
    has_estimator = ctx.get("has_estimator", False)
    all_force_est = np.zeros((force_steps, n, force_dim)) if has_estimator else None
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    fell_step = torch.full((n,), force_steps, dtype=torch.long, device=device)

    # Build force tensor for all envs
    force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    force_tensor[:, base_idx, 0] = fx_per_env
    force_tensor[:, base_idx, 1] = fy_per_env

    print(f"  Applying forces ({args_cli.force_hold_s}s)...", flush=True)
    for step in range(force_steps):
        # Apply per-env forces
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force_tensor, torques=torque_tensor,
        )

        # Visualize force arrows on all envs
        update_force_arrows_per_env(force_arrow, asset, fx_per_env, fy_per_env, device, n)

        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, compliance_k,
                                 args_cli.ema_alpha)

        # Track falls
        newly_fell = (dones > 0) & ~fell
        fell_step[newly_fell] = step
        fell |= (dones > 0)

        # Record data
        all_vel[step] = asset.data.root_lin_vel_b[:n, :2].cpu().numpy()
        all_pos[step] = asset.data.root_pos_w[:n, :2].cpu().numpy()
        if has_estimator:
            all_force_est[step] = isaac_env._force_estimate_xy[:n].cpu().numpy()

        if args_cli.real_time:
            time.sleep(dt)

    clear_force(asset, base_idx, device, n)
    force_arrow.set_visibility(False)
    print(f"  Done. Fell: {fell.sum().item()}/{n} envs")

    env.close()

    # ── Pack results into per-(mag, dir, trial) structure ────────────────
    results = {}
    fell_np = fell.cpu().numpy()
    fell_step_np = fell_step.cpu().numpy()

    for i in range(n):
        mag_str = str(float(env_mag[i]))
        deg_str = str(float(env_dir[i]))
        if mag_str not in results:
            results[mag_str] = {}
        if deg_str not in results[mag_str]:
            results[mag_str][deg_str] = []

        # Trim data to before fall
        valid_steps = fell_step_np[i]
        trial_data = {
            "trial": env_trial[i],
            "success": not fell_np[i],
            "pos_start": pos_start[i],
            "force_xy": [fx_per_env[i].item(), fy_per_env[i].item()],
            "vel_xy": all_vel[:valid_steps, i, :],
            "pos_xy": all_pos[:valid_steps, i, :],
            "dt": dt,
        }
        if has_estimator:
            trial_data["force_est"] = all_force_est[:valid_steps, i, :]
        results[mag_str][deg_str].append(trial_data)

    # ── Save raw data ────────────────────────────────────────────────────
    task_short = args_cli.task.replace("Go2-Ablation-", "").replace("-v0", "")
    modulation_tag = "linear" if args_cli.linear_modulation else "nonlinear"
    max_force = int(max(magnitudes))
    dir_suffix = f"{task_short}_{modulation_tag}_{max_force}N"
    output_dir = create_eval_output_dir(agent_cfg.experiment_name, "static_360", suffix=dir_suffix)
    save_config(output_dir, args_cli, agent_cfg, dt)
    save_raw_data(output_dir, results)

    print(f"\n[static_360] Raw data saved to {output_dir}")
    print(f"  Run analysis with:")
    print(f"    python scripts/rsl_rl/eval/analyze_static_360.py {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
