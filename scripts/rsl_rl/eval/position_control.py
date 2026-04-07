"""Debug script: single robot walking a straight line with PI lateral control.

No forces, no data collection. Just verify the PI controller tracks the line
and that the trajectory markers render.

Usage:
    python scripts/rsl_rl/eval/position_control.py --task Go2-LowLevel-v0 \
        --checkpoint logs/rsl_rl/go2_lowlevel/2026-03-31_20-57-20/model_6500.pt \
        --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Position control debug — walk a straight line.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--walk_speed", type=float, default=0.5)
parser.add_argument("--pid_kp", type=float, default=2.0)
parser.add_argument("--pid_ki", type=float, default=0.5)
parser.add_argument("--pid_vmax", type=float, default=0.3)
parser.add_argument("--compliance_k", type=float, default=0.06)
parser.add_argument("--ema_alpha", type=float, default=0.1)
parser.add_argument("--warmup_s", type=float, default=3.0)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument("--stage1_checkpoint", type=str, default=None)
parser.add_argument("--estimator_checkpoint", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401

from eval.eval_utils import (
    create_runner,
    create_trajectory_markers,
    disable_force_events,
    get_asset_and_base,
    resolve_checkpoint,
    set_flat_terrain,
    set_long_episode,
    set_walking_commands,
    setup_runner_for_eval,
    step_policy,
    update_trajectory_vis,
)


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
    print(f"[position_control] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner, runner_class_name, is_stage2 = create_runner(env, agent_cfg, args_cli)
    runner.load(resume_path)
    runner.eval_mode()

    n = args_cli.num_envs
    ctx, env = setup_runner_for_eval(runner, env, runner_class_name, is_stage2,
                                     env.unwrapped.device, n)

    isaac_env = env.unwrapped
    device = isaac_env.device
    dt = isaac_env.step_dt
    asset, base_idx = get_asset_and_base(isaac_env)

    warmup_steps = int(args_cli.warmup_s / dt)
    run_steps = int(args_cli.duration_s / dt)

    print(f"\n{'=' * 70}")
    print(f"  Position Control Debug")
    print(f"  Walk speed: {args_cli.walk_speed} m/s")
    print(f"  PI: Kp={args_cli.pid_kp}, Ki={args_cli.pid_ki}, Vmax={args_cli.pid_vmax}")
    print(f"  Duration: {args_cli.duration_s}s")
    print(f"{'=' * 70}\n")

    # ── Trajectory markers ───────────────────────────────────────────────
    NUM_TRAJ_PTS = 200
    traj_markers = create_trajectory_markers("/World/Visuals/Trajectory", NUM_TRAJ_PTS, device)

    # ── Get initial observations ─────────────────────────────────────────
    obs = env.get_observations()

    # ── Warmup ───────────────────────────────────────────────────────────
    print(f"  Warmup ({args_cli.warmup_s}s)...", flush=True)
    for _ in range(warmup_steps):
        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, args_cli.compliance_k,
                                 args_cli.ema_alpha)

    # Record desired path Y
    y_start = asset.data.root_pos_w[0, 1].item()
    x_start = asset.data.root_pos_w[0, 0].item()
    heading = asset.data.heading_w[0].item()

    print(f"  Start position: x={x_start:.3f}, y={y_start:.3f}")
    print(f"  Heading: {heading:.3f} rad ({heading * 180 / 3.14159:.1f} deg)")
    print(f"  Obs[6:9] (vel cmd): {obs['policy'][0, 6:9].cpu().numpy()}")

    # Draw trajectory line along robot's forward direction (heading), not world X
    import math
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    traj_pts = [(x_start + i * 0.2 * cos_h, y_start + i * 0.2 * sin_h) for i in range(NUM_TRAJ_PTS)]
    update_trajectory_vis(traj_markers, traj_pts, 0.15, device)
    print(f"  Trajectory markers placed: {NUM_TRAJ_PTS} spheres along heading")

    # ── PI controller state ──────────────────────────────────────────────
    y_integral = 0.0

    print(f"\n  Running for {args_cli.duration_s}s with PI control...")
    print(f"  {'Step':>6}  {'Time':>6}  {'X':>8}  {'Y':>8}  {'Y_err':>8}  {'VY_cmd':>8}  {'VY_corr':>8}  {'Obs[6]':>8}  {'Obs[7]':>8}")
    print(f"  {'-' * 78}")

    import math as _math

    for step in range(run_steps):
        # Read current state
        pos = asset.data.root_pos_w[0, :2].cpu().numpy()
        hdg = asset.data.heading_w[0].item()

        # Compute lateral error: perpendicular distance from the desired line
        # The desired line passes through (x_start, y_start) in direction (cos_h0, sin_h0)
        # where h0 is the initial heading.
        # Vector from start to current position:
        dx = pos[0] - x_start
        dy = pos[1] - y_start
        # Cross product gives signed perpendicular distance (positive = left of path)
        lateral_error = -(-_math.sin(heading) * dx + _math.cos(heading) * dy)
        # lateral_error > 0 means robot is to the right, needs negative vy to go left

        # PI control
        y_integral += lateral_error * dt
        vy_correction = args_cli.pid_kp * lateral_error + args_cli.pid_ki * y_integral
        vy_correction = max(-args_cli.pid_vmax, min(args_cli.pid_vmax, vy_correction))

        # Read what the env gives as obs before modification
        obs_vy_before = obs["policy"][0, 7].item()
        obs_vx_before = obs["policy"][0, 6].item()

        # Inject correction into vy command (body frame)
        obs = {k: v.clone() for k, v in obs.items()}
        obs["policy"][:, 7] = obs["policy"][:, 7] + vy_correction

        obs_vy_after = obs["policy"][0, 7].item()

        # Step
        obs, dones = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, args_cli.compliance_k,
                                 args_cli.ema_alpha)

        # Print diagnostics every 50 steps (1s)
        if step % 50 == 0:
            print(f"  {step:6d}  {step * dt:6.1f}  x={pos[0]:7.3f}  y={pos[1]:7.3f}  "
                  f"lat_err={lateral_error:8.4f}  vy_cmd={obs_vy_before:7.4f}  "
                  f"vy_corr={vy_correction:7.4f}  hdg={hdg:6.3f}")

        if dones[0] > 0:
            print(f"\n  Robot fell at step {step} ({step * dt:.1f}s)!")
            break

        time.sleep(dt)

    pos_final = asset.data.root_pos_w[0, :2].cpu().numpy()
    dx = pos_final[0] - x_start
    dy = pos_final[1] - y_start
    final_lat = -(-_math.sin(heading) * dx + _math.cos(heading) * dy)
    print(f"\n  Final position: x={pos_final[0]:.3f}, y={pos_final[1]:.3f}")
    print(f"  Final lateral error: {final_lat:.4f}m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
