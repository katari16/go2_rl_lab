"""Dynamic compliance evaluation — DMC-style alternating perpendicular force.

Robot walks forward at a fixed speed, tracking a straight line via PI controller.
Alternating left/right perpendicular forces are applied after the robot is
confirmed to be tracking the line. All (magnitude, trial) combos run in parallel.

Records raw time-series data; metrics computed in analyze_dynamic_compliance.py.

Usage:
    python scripts/rsl_rl/eval/dynamic_compliance_eval.py --task Go2-LowLevel-v0 \
        --checkpoint logs/rsl_rl/.../model_XXXX.pt \
        --force_magnitudes 5 10 15 20 --num_trials 20 --num_alternations 3
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Dynamic compliance evaluation (DMC-style).")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--force_magnitudes", type=float, nargs="+", default=[5, 10, 15, 20])
parser.add_argument("--walk_speed", type=float, default=0.5)
parser.add_argument("--force_duration_s", type=float, default=2.0)
parser.add_argument("--recovery_timeout_s", type=float, default=10.0)
parser.add_argument("--num_alternations", type=int, default=3,
                    help="Number of left+right pairs (total cycles = 2 * num_alternations).")
parser.add_argument("--num_trials", type=int, default=20)
parser.add_argument("--pid_kp", type=float, default=2.0, help="Lateral PI proportional gain.")
parser.add_argument("--pid_ki", type=float, default=0.5, help="Lateral PI integral gain.")
parser.add_argument("--pid_vmax", type=float, default=0.3, help="Max lateral correction vel.")
parser.add_argument("--on_line_threshold", type=float, default=0.03,
                    help="Lateral threshold (m) for 'on line' condition.")
parser.add_argument("--on_line_sustain_s", type=float, default=1.0,
                    help="Seconds the robot must stay on line before force is applied.")
parser.add_argument("--on_line_timeout_s", type=float, default=15.0)
parser.add_argument("--warmup_s", type=float, default=5.0)
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
    create_trajectory_markers,
    disable_force_events,
    get_asset_and_base,
    reset_env,
    resolve_checkpoint,
    save_config,
    save_raw_data,
    set_flat_terrain,
    set_long_episode,
    set_walking_commands,
    setup_runner_for_eval,
    step_policy,
    update_force_arrow,
    update_trajectory_vis,
)


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
    num_cycles = args_cli.num_alternations * 2  # L, R, L, R, ...

    # Total envs = magnitudes × trials — all run in parallel
    n = len(magnitudes) * num_trials
    env_cfg.scene.num_envs = n

    set_walking_commands(env_cfg, args_cli.walk_speed)
    set_flat_terrain(env_cfg)
    set_long_episode(env_cfg)
    disable_force_events(env_cfg, agent_cfg)

    resume_path = resolve_checkpoint(agent_cfg, args_cli)
    print(f"[dynamic_compliance] Checkpoint: {resume_path}")

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
    force_steps = int(args_cli.force_duration_s / dt)
    recovery_timeout_steps = int(args_cli.recovery_timeout_s / dt)
    on_line_sustain_steps = int(args_cli.on_line_sustain_s / dt)
    on_line_timeout_steps = int(args_cli.on_line_timeout_s / dt)
    on_line_thresh = args_cli.on_line_threshold

    has_estimator = ctx.get("has_estimator", False)
    force_dim = ctx.get("force_dim", 3)

    # ── Per-env assignments ──────────────────────────────────────────────
    mag_per_env = torch.zeros(n, device=device)
    env_mag = []    # magnitude for each env
    env_trial = []  # trial index for each env

    idx = 0
    for mag in magnitudes:
        for trial in range(num_trials):
            mag_per_env[idx] = mag
            env_mag.append(mag)
            env_trial.append(trial)
            idx += 1

    print(f"\n{'=' * 70}")
    print(f"  Dynamic Compliance Evaluation — PARALLEL")
    print(f"  Walk speed: {args_cli.walk_speed} m/s")
    print(f"  Magnitudes: {magnitudes} N")
    print(f"  Cycles: {num_cycles} (L/R alternating)")
    print(f"  Trials/magnitude: {num_trials}")
    print(f"  Total envs: {n}")
    print(f"  Force: {args_cli.force_duration_s}s, Recovery timeout: {args_cli.recovery_timeout_s}s")
    print(f"  PI: Kp={args_cli.pid_kp}, Ki={args_cli.pid_ki}, Vmax={args_cli.pid_vmax}")
    print(f"  On-line: {on_line_thresh}m for {args_cli.on_line_sustain_s}s")
    print(f"  Runner: {runner_class_name}")
    print(f"{'=' * 70}\n")

    force_arrow = create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))
    NUM_TRAJ_PTS = 100
    traj_markers = create_trajectory_markers("/World/Visuals/Trajectory", NUM_TRAJ_PTS, device)

    # ── PI controller state ──────────────────────────────────────────────
    y_integral = torch.zeros(n, device=device)

    def step_with_pi(obs, y_start_t):
        nonlocal y_integral
        y_current = asset.data.root_pos_w[:n, 1]
        y_error = y_start_t - y_current
        y_integral += y_error * dt
        vy_corr = args_cli.pid_kp * y_error + args_cli.pid_ki * y_integral
        vy_corr = torch.clamp(vy_corr, -args_cli.pid_vmax, args_cli.pid_vmax)
        obs = {k: v.clone() for k, v in obs.items()}
        obs["policy"][:, 7] = obs["policy"][:, 7] + vy_corr
        return step_policy(obs, ctx, env, runner, isaac_env, n,
                           runner_class_name, args_cli.compliance_k,
                           args_cli.ema_alpha)

    # ── Reset all envs ───────────────────────────────────────────────────
    obs = reset_env(env, ctx, isaac_env, runner, runner_class_name,
                    is_stage2, device, n)

    # ── Warmup ───────────────────────────────────────────────────────────
    print(f"  Warmup ({args_cli.warmup_s}s)...", flush=True)
    for _ in range(warmup_steps):
        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, args_cli.compliance_k,
                                 args_cli.ema_alpha)

    # Record starting Y position (desired path) for each env
    y_start = asset.data.root_pos_w[:n, 1].clone()  # [n]
    x_start = asset.data.root_pos_w[0, 0].item()
    y_start_0 = y_start[0].item()

    # Draw desired trajectory line (env 0): straight ahead from current position
    traj_pts = [(x_start + i * 0.3, y_start_0) for i in range(NUM_TRAJ_PTS)]
    update_trajectory_vis(traj_markers, traj_pts, 0.15, device)

    # Reset integral after warmup
    y_integral.zero_()

    # Track falls
    fell = torch.zeros(n, dtype=torch.bool, device=device)

    # ── Data storage per cycle ───────────────────────────────────────────
    # Each cycle stores: wait, force_on, recovery phases
    all_cycles = []  # list of cycle dicts

    for cycle_idx in range(num_cycles):
        direction = "left" if cycle_idx % 2 == 0 else "right"
        force_sign = 1.0 if direction == "left" else -1.0
        fy_per_env = force_sign * mag_per_env  # [n]

        print(f"  Cycle {cycle_idx + 1}/{num_cycles} ({direction})", flush=True)

        # ── Wait until all envs are on line ──────────────────────────
        on_line_counter = torch.zeros(n, dtype=torch.long, device=device)
        wait_vel = []
        wait_pos = []
        wait_steps = 0

        print(f"    Waiting for line tracking...", end="", flush=True)
        for step in range(on_line_timeout_steps):
            obs, dones = step_with_pi(obs, y_start)
            newly_fell = (dones > 0) & ~fell
            fell |= (dones > 0)

            y_current = asset.data.root_pos_w[:n, 1]
            on_line = (torch.abs(y_current - y_start) < on_line_thresh) & ~fell
            on_line_counter[on_line] += 1
            on_line_counter[~on_line] = 0

            wait_vel.append(asset.data.root_lin_vel_b[:n, :2].cpu().numpy().copy())
            wait_pos.append(asset.data.root_pos_w[:n, :2].cpu().numpy().copy())
            wait_steps = step + 1

            if args_cli.real_time:
                time.sleep(dt)

            # All non-fallen envs sustained on line
            all_ready = ((on_line_counter >= on_line_sustain_steps) | fell).all()
            if all_ready:
                break

        print(f" {wait_steps} steps ({wait_steps * dt:.1f}s)", flush=True)

        # ── Force-on phase ───────────────────────────────────────────
        force_on_vel = np.zeros((force_steps, n, 2))
        force_on_pos = np.zeros((force_steps, n, 2))
        force_on_est = np.zeros((force_steps, n, force_dim)) if has_estimator else None

        # Build force tensor
        force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        force_tensor[:, base_idx, 1] = fy_per_env  # lateral force only

        fell_during_force = torch.zeros(n, dtype=torch.bool, device=device)
        fell_step_force = torch.full((n,), force_steps, dtype=torch.long, device=device)

        print(f"    Force ({args_cli.force_duration_s}s)...", end="", flush=True)
        for step in range(force_steps):
            asset.permanent_wrench_composer.set_forces_and_torques(
                forces=force_tensor, torques=torque_tensor,
            )
            obs, dones = step_with_pi(obs, y_start)

            newly_fell = (dones > 0) & ~fell
            fell_step_force[newly_fell & ~fell_during_force] = step
            fell_during_force |= newly_fell
            fell |= (dones > 0)

            force_on_vel[step] = asset.data.root_lin_vel_b[:n, :2].cpu().numpy()
            force_on_pos[step] = asset.data.root_pos_w[:n, :2].cpu().numpy()
            if has_estimator:
                force_on_est[step] = isaac_env._force_estimate_xy[:n].cpu().numpy()

            if args_cli.real_time:
                time.sleep(dt)

        clear_force(asset, base_idx, device, n)
        update_force_arrow(force_arrow, asset, base_idx, 0.0, 0.0, device, n)
        print(f" fell: {fell_during_force.sum().item()}", flush=True)

        # ── Recovery phase ───────────────────────────────────────────
        recovered = torch.zeros(n, dtype=torch.bool, device=device)
        recovered_step = torch.full((n,), recovery_timeout_steps, dtype=torch.long, device=device)
        recovery_vel_list = []
        recovery_pos_list = []
        recovery_steps = 0

        print(f"    Recovery...", end="", flush=True)
        for step in range(recovery_timeout_steps):
            obs, dones = step_with_pi(obs, y_start)
            newly_fell = (dones > 0) & ~fell
            fell |= (dones > 0)

            y_current = asset.data.root_pos_w[:n, 1]
            just_recovered = (torch.abs(y_current - y_start) < on_line_thresh) & ~recovered & ~fell
            recovered_step[just_recovered] = step
            recovered |= just_recovered

            recovery_vel_list.append(asset.data.root_lin_vel_b[:n, :2].cpu().numpy().copy())
            recovery_pos_list.append(asset.data.root_pos_w[:n, :2].cpu().numpy().copy())
            recovery_steps = step + 1

            if args_cli.real_time:
                time.sleep(dt)

            # All non-fallen envs recovered
            if ((recovered | fell).all()):
                break

        print(f" {recovery_steps} steps ({recovery_steps * dt:.1f}s)", flush=True)

        # Pack recovery arrays
        recovery_vel = np.array(recovery_vel_list) if recovery_vel_list else np.zeros((0, n, 2))
        recovery_pos = np.array(recovery_pos_list) if recovery_pos_list else np.zeros((0, n, 2))

        # Store cycle data
        cycle_data = {
            "cycle_idx": cycle_idx,
            "direction": direction,
            "force_on_vel": force_on_vel,          # [force_steps, n, 2]
            "force_on_pos": force_on_pos,           # [force_steps, n, 2]
            "force_on_est": force_on_est,           # [force_steps, n, force_dim] or None
            "fell_step_force": fell_step_force.cpu().numpy(),  # [n]
            "recovery_vel": recovery_vel,            # [recovery_steps, n, 2]
            "recovery_pos": recovery_pos,            # [recovery_steps, n, 2]
            "recovered_step": recovered_step.cpu().numpy(),    # [n]
            "wait_steps": wait_steps,
        }
        all_cycles.append(cycle_data)

    total_fell = fell.sum().item()
    print(f"\n  Done. Total fell: {total_fell}/{n} envs")
    env.close()

    # ── Reshape into per-(mag, trial) structure ──────────────────────────
    fell_np = fell.cpu().numpy()
    results = {}

    for i in range(n):
        mag_str = str(float(env_mag[i]))
        if mag_str not in results:
            results[mag_str] = []

        trial_cycles = []
        for cyc in all_cycles:
            valid_force = int(cyc["fell_step_force"][i])
            valid_recovery = int(cyc["recovered_step"][i])
            # If not recovered, use all recovery steps
            if valid_recovery >= recovery_timeout_steps:
                valid_recovery = cyc["recovery_vel"].shape[0]

            tc = {
                "direction": cyc["direction"],
                "force_on": {
                    "vel_xy": cyc["force_on_vel"][:valid_force, i, :],
                    "pos_xy": cyc["force_on_pos"][:valid_force, i, :],
                },
                "recovery": {
                    "vel_xy": cyc["recovery_vel"][:valid_recovery, i, :] if cyc["recovery_vel"].shape[0] > 0 else np.zeros((0, 2)),
                    "pos_xy": cyc["recovery_pos"][:valid_recovery, i, :] if cyc["recovery_pos"].shape[0] > 0 else np.zeros((0, 2)),
                },
                "recovered_step": int(cyc["recovered_step"][i]),
                "wait_steps": cyc["wait_steps"],
            }
            if cyc["force_on_est"] is not None:
                tc["force_on"]["force_est"] = cyc["force_on_est"][:valid_force, i, :]
            trial_cycles.append(tc)

        trial_data = {
            "trial": env_trial[i],
            "success": not fell_np[i],
            "y_start": y_start[i].item(),
            "force_sign": 1.0,  # will be per-cycle
            "cycles": trial_cycles,
            "dt": dt,
        }
        results[mag_str].append(trial_data)

    # ── Save raw data ────────────────────────────────────────────────────
    output_dir = create_eval_output_dir(agent_cfg.experiment_name, "dynamic_compliance")
    save_config(output_dir, args_cli, agent_cfg, dt)
    save_raw_data(output_dir, results)

    print(f"\n[dynamic_compliance] Raw data saved to {output_dir}")
    print(f"  Run analysis with:")
    print(f"    python scripts/rsl_rl/eval/analyze_dynamic_compliance.py {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
