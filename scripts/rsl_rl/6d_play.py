"""Play a 6Dctrl checkpoint with overridable pose commands and arrow visualization.

Modes (the env's UniformVelocityPoseCommand still does the sampling — these
modes only zero out the channels you don't want to see):
  - all      no override; random commands like training (default)
  - static   hold (vx, vy, wz, roll, pitch, height) at the CLI values
  - roll     vx=vy=wz=0, pitch=0, height=nominal; roll is sampled by the env
  - pitch    vx=vy=wz=0, roll=0, height=nominal; pitch is sampled by the env
  - height   vx=vy=wz=0, roll=pitch=0; height is sampled by the env

Visualization (all envs):
  - red arrow above the robot, pointing in the commanded tilt direction
    (length proportional to sqrt(roll^2 + pitch^2))
  - green arrow above the robot, pointing in the current tilt direction
  - red sphere at (robot_xy, commanded_height)
  - green sphere at (robot_xy, current_height)

Usage:
  python scripts/rsl_rl/6d_play.py --task Go2-Ablation-6Dctrl-v0 \\
      --checkpoint <path> --num_envs 1 --mode roll
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a 6Dctrl checkpoint with pose-command overrides.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--estimator_checkpoint", type=str, default=None)

parser.add_argument("--mode", type=str, default="all",
                    choices=["all", "static", "roll", "pitch", "height"],
                    help="Command override mode (sampling stays the env's job for the active channel).")
parser.add_argument("--cmd_vx", type=float, default=0.0, help="Static-mode vx (m/s).")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Static-mode vy (m/s).")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Static-mode wz (rad/s).")
parser.add_argument("--cmd_roll", type=float, default=0.0, help="Static-mode roll cmd (rad).")
parser.add_argument("--cmd_pitch", type=float, default=0.0, help="Static-mode pitch cmd (rad).")
parser.add_argument("--cmd_height", type=float, default=0.34, help="Static-mode height cmd (m).")
parser.add_argument("--nominal_height", type=float, default=0.34,
                    help="Height value used for masked-out height in roll/pitch modes.")
parser.add_argument("--no_pose_arrows", action="store_true",
                    help="Disable the extra commanded/current pose arrows and spheres.")
parser.add_argument("--rough_terrain", action="store_true",
                    help="Use the training rough terrain (default is a flat plane).")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- after app launch ----------------------------------------------------------
import os
import time

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG,
)
from isaaclab.utils.assets import retrieve_file_path
import isaaclab.utils.math as math_utils

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg, RslRlVecEnvWrapper,
    export_policy_as_jit, export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


def install_command_overrides(cmd_term, mode: str, args) -> None:
    """Monkey-patch the command term's resample/update so the env's own
    pipeline emits our chosen commands instead of the training distribution.

    - mode="all": leave the original behavior alone.
    - mode="static": every resample writes the CLI values; _update_command is
      neutralized so heading control doesn't rewrite wz.
    - mode="roll"/"pitch"/"height": call the original resample (so the active
      channel uses the env's training-style sampling), then zero out the
      inactive channels and disable standing/heading overrides.
    """
    if mode == "all":
        return

    has_pose = hasattr(cmd_term, "pose_command")
    nominal_h = args.nominal_height
    cls = type(cmd_term)
    original_resample = cls._resample_command

    if mode == "static":
        def _resample(env_ids):
            cmd_term.vel_command_b[env_ids, 0] = args.cmd_vx
            cmd_term.vel_command_b[env_ids, 1] = args.cmd_vy
            cmd_term.vel_command_b[env_ids, 2] = args.cmd_wz
            cmd_term.is_standing_env[env_ids] = False
            cmd_term.is_heading_env[env_ids] = False
            if has_pose:
                cmd_term.pose_command[env_ids, 0] = args.cmd_roll
                cmd_term.pose_command[env_ids, 1] = args.cmd_pitch
                cmd_term.pose_command[env_ids, 2] = args.cmd_height
    else:
        def _resample(env_ids):
            original_resample(cmd_term, env_ids)
            cmd_term.vel_command_b[env_ids, 0] = 0.0
            cmd_term.vel_command_b[env_ids, 1] = 0.0
            cmd_term.vel_command_b[env_ids, 2] = 0.0
            cmd_term.is_standing_env[env_ids] = False
            cmd_term.is_heading_env[env_ids] = False
            if has_pose:
                if mode == "roll":
                    cmd_term.pose_command[env_ids, 1] = 0.0
                    cmd_term.pose_command[env_ids, 2] = nominal_h
                elif mode == "pitch":
                    cmd_term.pose_command[env_ids, 0] = 0.0
                    cmd_term.pose_command[env_ids, 2] = nominal_h
                elif mode == "height":
                    cmd_term.pose_command[env_ids, 0] = 0.0
                    cmd_term.pose_command[env_ids, 1] = 0.0

    cmd_term._resample_command = _resample
    cmd_term._update_command = lambda: None  # kill heading/standing override

    # Re-apply to all envs immediately so the very first obs is consistent.
    all_ids = torch.arange(cmd_term.num_envs, device=cmd_term.device)
    cmd_term._resample_command(all_ids)


def _tilt_arrow(roll_t, pitch_t, base_pos_w, base_quat_w, scale_K: float = 1.5):
    """Build (positions, quats, scales) for an arrow above each robot pointing
    in the body-frame downhill direction. Length proportional to tilt magnitude.

    Body-frame downhill direction: x_b = sin(pitch), y_b = sin(roll).
    """
    n = roll_t.shape[0]
    device = roll_t.device
    pos = base_pos_w.clone()
    pos[:, 2] += 0.5
    # heading angle of (sin(pitch), sin(roll)) in body frame
    heading = torch.atan2(torch.sin(roll_t), torch.sin(pitch_t))
    zeros = torch.zeros_like(heading)
    arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
    arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
    mag = torch.sqrt(torch.sin(roll_t) ** 2 + torch.sin(pitch_t) ** 2)
    arrow_scale = torch.ones((n, 3), device=device) * 0.05  # small head/tail
    arrow_scale[:, 0] = 0.05 + scale_K * mag  # length along x
    return pos, arrow_quat, arrow_scale


def _height_sphere(z_t, base_pos_w):
    """Build (positions, quats, scales) for a sphere at (robot_xy, z_t)."""
    n = z_t.shape[0]
    device = z_t.device
    pos = base_pos_w.clone()
    pos[:, 2] = z_t
    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    scale = torch.ones((n, 3), device=device) * 0.04
    return pos, quat, scale


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Default to a flat plane so visual posture tracking is easy to read.
    if not args_cli.rough_terrain:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    if args_cli.estimator_checkpoint and hasattr(env_cfg, "force_estimator_checkpoint"):
        env_cfg.force_estimator_checkpoint = args_cli.estimator_checkpoint

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Runner ────────────────────────────────────────────────────────────────
    if agent_cfg.class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.estimator_checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.estimator_checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"6d_play.py expects CompliantOnPolicyRunner, got: {agent_cfg.class_name}")
    runner.load(resume_path)

    if hasattr(runner, "_wrapped_env"):
        env = runner._wrapped_env

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)

    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")

    # ── Command term + visualization markers ─────────────────────────────────
    isaac_env = env.unwrapped
    cmd_term = isaac_env.command_manager._terms["base_velocity"]
    has_pose = hasattr(cmd_term, "pose_command")
    if not has_pose and args_cli.mode != "all":
        print("[WARN] Task is not 6Dctrl — no pose_command on command term. "
              "Falling back to mode=all.")
        args_cli.mode = "all"

    # Silence the env's built-in velocity arrow — we draw our own pose markers.
    try:
        cmd_term.set_debug_vis(False)
    except Exception:
        pass

    # Install command-mode overrides BEFORE the first step so the initial obs
    # already reflects the desired commands.
    install_command_overrides(cmd_term, args_cli.mode, args_cli)
    obs = env.get_observations()

    if not args_cli.no_pose_arrows and has_pose:
        cmd_tilt_marker = VisualizationMarkers(RED_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/cmd_tilt"))
        cur_tilt_marker = VisualizationMarkers(GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/cur_tilt"))
        cmd_h_marker = VisualizationMarkers(SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/cmd_height"))
        cur_h_marker = VisualizationMarkers(SPHERE_MARKER_CFG.replace(
            prim_path="/Visuals/cur_height"))
    else:
        cmd_tilt_marker = cur_tilt_marker = cmd_h_marker = cur_h_marker = None

    print(f"[INFO] mode={args_cli.mode}  "
          f"static=({args_cli.cmd_vx}, {args_cli.cmd_vy}, {args_cli.cmd_wz} | "
          f"{args_cli.cmd_roll}, {args_cli.cmd_pitch}, {args_cli.cmd_height})  "
          f"nominal_height={args_cli.nominal_height}")

    dt = isaac_env.step_dt
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            # update markers
            if cmd_tilt_marker is not None:
                asset = isaac_env.scene["robot"]
                base_pos = asset.data.root_pos_w
                base_quat = asset.data.root_quat_w
                roll_now, pitch_now, _ = math_utils.euler_xyz_from_quat(base_quat)

                # commanded tilt arrow (red) — uses pose_command
                cmd_pos, cmd_quat, cmd_scale = _tilt_arrow(
                    cmd_term.pose_command[:, 0], cmd_term.pose_command[:, 1],
                    base_pos, base_quat,
                )
                cmd_tilt_marker.visualize(cmd_pos, cmd_quat, cmd_scale)

                # current tilt arrow (green)
                cur_pos, cur_quat, cur_scale = _tilt_arrow(
                    roll_now, pitch_now, base_pos, base_quat,
                )
                cur_tilt_marker.visualize(cur_pos, cur_quat, cur_scale)

                # height markers
                cmd_hp, cmd_hq, cmd_hs = _height_sphere(cmd_term.pose_command[:, 2], base_pos)
                cmd_h_marker.visualize(cmd_hp, cmd_hq, cmd_hs)
                cur_hp, cur_hq, cur_hs = _height_sphere(base_pos[:, 2], base_pos)
                cur_h_marker.visualize(cur_hp, cur_hq, cur_hs)

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if args_cli.real_time:
            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
