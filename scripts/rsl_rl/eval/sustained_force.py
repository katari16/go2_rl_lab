"""Sustained Force Evaluation — measure velocity deviation under constant forces.

Robot walks forward at a fixed speed while a sustained force is applied from
8 directions. Measures how much the robot deviates from the commanded velocity
in each direction — higher deviation = more compliant.

Generates a polar plot of velocity deviation magnitude per force direction,
similar to Fig. 5 in the Deep Compliant Control paper.

Supports both stage-1 (CompliantOnPolicyRunner) and stage-2 (ComplianceOnPolicyRunner).

Usage:
    # Stage 1:
    python scripts/rsl_rl/eval/sustained_force.py --task Go2-LowLevel-v0 --num_envs 1 \
        --checkpoint logs/rsl_rl/.../model_XXXX.pt --force_magnitude 15

    # Stage 2:
    python scripts/rsl_rl/eval/sustained_force.py --task Go2-HighLevel-NonLinear-v0 --num_envs 1 \
        --stage1_checkpoint /path/to/stage1.pt \
        --checkpoint /path/to/stage2.pt --force_magnitude 15
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Sustained force compliance evaluation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--force_magnitude", type=float, default=15.0, help="Sustained force magnitude (N).")
parser.add_argument("--walk_speed", type=float, default=0.5, help="Forward walking speed (m/s).")
parser.add_argument("--num_trials", type=int, default=3, help="Trials per direction.")
parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Stage-1 checkpoint (for stage 2).")
parser.add_argument("--estimator_checkpoint", type=str, default=None, help="Estimator checkpoint.")
parser.add_argument("--compliance_k", type=float, default=0.06, help="Linear mapping k (for stage 1).")
parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA alpha for force estimate.")
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

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


NUM_DIRECTIONS = 8
DIRECTIONS_DEG = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)
DIRECTIONS_RAD = np.deg2rad(DIRECTIONS_DEG)

WARMUP_S = 3.0      # walk before force
FORCE_DURATION_S = 5.0  # how long force is applied
MEASURE_START_S = 2.0   # start measuring after force settles (skip transient)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Walking forward at fixed speed
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.walk_speed, args_cli.walk_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Disable automatic force events — we apply forces manually
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
    print(f"[sustained_force] Checkpoint: {resume_path}")

    # Create env + runner
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    print(f"[sustained_force] Runner: {runner_class_name}")

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
    if is_stage2:
        wrapper = runner._compliance_wrapper
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
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)
    dt = isaac_env.step_dt
    n = args_cli.num_envs

    # Init force estimate for stage 1
    if runner_class_name == "CompliantOnPolicyRunner":
        force_dim = getattr(runner, "_force_dim", 3)
        isaac_env._force_estimate_xy = torch.zeros(n, force_dim, device=device)
        compliant_raw_obs_dim = runner._num_one_step_obs
        force_ema = torch.zeros(n, 2, device=device)

    compliance_k = args_cli.compliance_k
    ema_alpha = args_cli.ema_alpha
    force_mag = args_cli.force_magnitude
    num_trials = args_cli.num_trials

    warmup_steps = int(WARMUP_S / dt)
    force_steps = int(FORCE_DURATION_S / dt)
    measure_start = int(MEASURE_START_S / dt)

    print(f"\n{'=' * 70}")
    print(f"  Sustained Force Evaluation")
    print(f"  Force: {force_mag:.0f}N from {NUM_DIRECTIONS} directions")
    print(f"  Walk speed: {args_cli.walk_speed:.1f} m/s")
    print(f"  Trials per direction: {num_trials}")
    print(f"  Force duration: {FORCE_DURATION_S:.0f}s (measure after {MEASURE_START_S:.0f}s)")
    if not is_stage2:
        print(f"  Compliance k={compliance_k:.4f} (linear mapping)")
    else:
        print(f"  Using learned high-level policy")
    print(f"{'=' * 70}\n")

    # Results
    all_vel_deviation = {d: [] for d in DIRECTIONS_DEG}
    all_vel_deviation_xy = {d: [] for d in DIRECTIONS_DEG}  # separate x/y
    all_peak_torque = {d: [] for d in DIRECTIONS_DEG}
    all_mean_power = {d: [] for d in DIRECTIONS_DEG}
    all_success = {d: [] for d in DIRECTIONS_DEG}

    for trial in range(num_trials):
        for d_idx, (deg, rad) in enumerate(zip(DIRECTIONS_DEG, DIRECTIONS_RAD)):
            fx = force_mag * math.cos(rad)
            fy = force_mag * math.sin(rad)

            print(f"  Trial {trial+1}/{num_trials}  Dir {deg:.0f}°  "
                  f"F=[{fx:+.1f},{fy:+.1f}]N  ", end="", flush=True)

            # Reset
            if is_stage2:
                obs = wrapper.get_observations()
            else:
                obs = env.get_observations()

            if runner_class_name == "CompliantOnPolicyRunner":
                force_ema = torch.zeros(n, 2, device=device)

            # Phase 1: warmup
            for step in range(warmup_steps):
                with torch.inference_mode():
                    if runner_class_name == "CompliantOnPolicyRunner":
                        raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
                        runner._history_buffer.insert(raw_obs)
                        fhat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
                        isaac_env._force_estimate_xy = fhat
                        force_ema = ema_alpha * fhat[:, :2] + (1.0 - ema_alpha) * force_ema
                        if compliance_k > 0:
                            obs["policy"][:, 6] += compliance_k * force_ema[:, 0]
                            obs["policy"][:, 7] += compliance_k * force_ema[:, 1]
                    if is_stage2:
                        actions = policy(obs)
                        obs, _, dones, _ = wrapper.step(actions)
                    else:
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                    policy_nn.reset(dones)

            # Phase 2: apply sustained force and measure
            vel_devs = []
            torque_log = []
            power_log = []
            fell = False

            # Set the force on the wrench composer
            force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
            force_tensor[:, base_idx, 0] = fx
            force_tensor[:, base_idx, 1] = fy
            torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)

            for step in range(force_steps):
                # Apply force every step
                asset.set_external_force_and_torque(force_tensor, torque_tensor)

                with torch.inference_mode():
                    if runner_class_name == "CompliantOnPolicyRunner":
                        raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
                        runner._history_buffer.insert(raw_obs)
                        fhat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
                        isaac_env._force_estimate_xy = fhat
                        force_ema = ema_alpha * fhat[:, :2] + (1.0 - ema_alpha) * force_ema
                        if compliance_k > 0:
                            obs["policy"][:, 6] += compliance_k * force_ema[:, 0]
                            obs["policy"][:, 7] += compliance_k * force_ema[:, 1]
                    if is_stage2:
                        actions = policy(obs)
                        obs, _, dones, _ = wrapper.step(actions)
                    else:
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                    policy_nn.reset(dones)

                if dones[0] > 0:
                    fell = True
                    break

                # Record after transient settles
                if step >= measure_start:
                    v_body = asset.data.root_lin_vel_b[0, :2].cpu().numpy()
                    cmd = np.array([args_cli.walk_speed, 0.0])
                    deviation = v_body - cmd
                    vel_devs.append(deviation)

                    torques = asset.data.applied_torque[0].cpu().numpy()
                    torque_log.append(np.max(np.abs(torques)))

                    dq = asset.data.joint_vel[0].cpu().numpy()
                    power_log.append(np.sum(np.abs(torques * dq)))

            # Clear force
            force_tensor.zero_()
            asset.set_external_force_and_torque(force_tensor, torque_tensor)

            if fell or len(vel_devs) == 0:
                all_vel_deviation[deg].append(0.0)
                all_vel_deviation_xy[deg].append(np.array([0.0, 0.0]))
                all_peak_torque[deg].append(0.0)
                all_mean_power[deg].append(0.0)
                all_success[deg].append(False)
                print("FELL")
                continue

            vel_devs = np.array(vel_devs)
            mean_dev = np.mean(vel_devs, axis=0)
            dev_mag = np.linalg.norm(mean_dev)

            all_vel_deviation[deg].append(dev_mag)
            all_vel_deviation_xy[deg].append(mean_dev)
            all_peak_torque[deg].append(np.max(torque_log))
            all_mean_power[deg].append(np.mean(power_log))
            all_success[deg].append(True)

            print(f"dev={dev_mag:.3f} m/s  [{mean_dev[0]:+.3f},{mean_dev[1]:+.3f}]  "
                  f"peak_τ={np.max(torque_log):.1f}  power={np.mean(power_log):.1f}W")

    env.close()

    # ── Generate plots ────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    eval_dir = os.path.join(os.path.dirname(resume_path), "eval")
    os.makedirs(eval_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_path = os.path.join(eval_dir, f"sustained_force_{force_mag:.0f}N_{timestamp}.pdf")

    angles = np.deg2rad(DIRECTIONS_DEG)
    angles_closed = np.append(angles, angles[0])

    def polar_plot(ax, values_dict, title, unit, color="tab:blue"):
        means = [np.mean(values_dict[d]) for d in DIRECTIONS_DEG]
        stds = [np.std(values_dict[d]) for d in DIRECTIONS_DEG]
        means_closed = means + [means[0]]
        stds_closed = stds + [stds[0]]
        means_arr = np.array(means_closed)
        stds_arr = np.array(stds_closed)
        ax.plot(angles_closed, means_arr, color=color, linewidth=2)
        ax.fill_between(angles_closed, np.clip(means_arr - stds_arr, 0, None),
                        means_arr + stds_arr, alpha=0.2, color=color)
        ax.set_title(f"{title}\n({unit})", fontsize=11, pad=15)
        ax.set_thetagrids(DIRECTIONS_DEG)

    with PdfPages(pdf_path) as pdf:
        # Page 1: polar plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={"projection": "polar"})
        fig.suptitle(
            f"Sustained Force — {force_mag:.0f}N, walk={args_cli.walk_speed:.1f} m/s, "
            f"{num_trials} trials/dir",
            fontsize=14, fontweight="bold",
        )

        polar_plot(axes[0, 0], all_vel_deviation, "Velocity Deviation", "m/s", "tab:blue")
        polar_plot(axes[0, 1], all_peak_torque, "Peak Joint Torque", "N·m", "tab:red")
        polar_plot(axes[1, 0], all_mean_power, "Mean Power", "Watts", "tab:orange")

        ax = axes[1, 1]
        success_rates = [np.mean(all_success[d]) * 100 for d in DIRECTIONS_DEG]
        success_closed = success_rates + [success_rates[0]]
        ax.plot(angles_closed, success_closed, color="tab:green", linewidth=2)
        ax.fill(angles_closed, success_closed, alpha=0.2, color="tab:green")
        ax.set_title("Success Rate\n(%)", fontsize=11, pad=15)
        ax.set_thetagrids(DIRECTIONS_DEG)
        ax.set_ylim(0, 110)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: velocity deviation vectors
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_title(f"Mean Velocity Deviation Vectors\n(Force={force_mag:.0f}N, Walk={args_cli.walk_speed:.1f} m/s)",
                     fontsize=13)
        ax.set_xlabel("Δvx (m/s)")
        ax.set_ylabel("Δvy (m/s)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)

        for d_idx, deg in enumerate(DIRECTIONS_DEG):
            rad = DIRECTIONS_RAD[d_idx]
            devs = all_vel_deviation_xy[deg]
            if len(devs) == 0:
                continue
            mean_dev = np.mean(devs, axis=0)

            # Force direction arrow (thin, gray)
            f_scale = 0.01  # N to m/s visual scale
            ax.annotate("", xy=(f_scale * force_mag * math.cos(rad), f_scale * force_mag * math.sin(rad)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1, alpha=0.5))

            # Deviation arrow (thick, colored)
            ax.annotate("", xy=(mean_dev[0], mean_dev[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
            ax.annotate(f"{deg:.0f}°", xy=(mean_dev[0], mean_dev[1]), fontsize=8, ha="center")

        max_dev = max([np.linalg.norm(np.mean(all_vel_deviation_xy[d], axis=0))
                       for d in DIRECTIONS_DEG if len(all_vel_deviation_xy[d]) > 0] + [0.1])
        ax.set_xlim(-max_dev * 1.3, max_dev * 1.3)
        ax.set_ylim(-max_dev * 1.3, max_dev * 1.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: summary table
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")
        headers = ["Direction", "Vel Dev (m/s)", "Δvx", "Δvy", "Peak τ (N·m)", "Power (W)", "Success"]
        rows = []
        for d in DIRECTIONS_DEG:
            devs = all_vel_deviation_xy[d]
            mean_dev = np.mean(devs, axis=0) if len(devs) > 0 else np.array([0, 0])
            rows.append([
                f"{d:.0f}°",
                f"{np.mean(all_vel_deviation[d]):.3f} ± {np.std(all_vel_deviation[d]):.3f}",
                f"{mean_dev[0]:+.3f}",
                f"{mean_dev[1]:+.3f}",
                f"{np.mean(all_peak_torque[d]):.1f}",
                f"{np.mean(all_mean_power[d]):.1f}",
                f"{np.mean(all_success[d])*100:.0f}%",
            ])
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.5)
        ax.set_title("Sustained Force Summary", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n[sustained_force] Results saved to {pdf_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
