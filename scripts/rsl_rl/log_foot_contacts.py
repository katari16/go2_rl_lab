"""Log foot contact force norms for all 4 legs and save a time-series plot.

Runs a trained policy and records the net contact force norm for each foot
at every step. Saves a PDF plot in scripts/rsl_rl/.

Usage:
    python scripts/rsl_rl/log_foot_contacts.py --task Go2-Compliant-v0 --num_envs 1 \
        --checkpoint logs/rsl_rl/go2_compliant/2026-XX-XX/model_XXXX.pt --duration 10
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Log foot contact force norms.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
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

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Resolve checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[log_foot_contacts] Checkpoint: {resume_path}")

    # Create env + runner
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    print(f"[log_foot_contacts] Runner class: {runner_class_name}")

    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "ForceOnPolicyRunner":
        from go2_rl_lab.estimator.force_runner import ForceOnPolicyRunner
        runner = ForceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "OnPolicyRunner":
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {runner_class_name}")

    runner.load(resume_path)
    runner.eval_mode()

    if hasattr(runner, "_wrapped_env"):
        env = runner._wrapped_env

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Access contact sensor — resolve foot body ids through SceneEntityCfg
    # (same way the obs term does it)
    isaac_env = env.unwrapped
    device = isaac_env.device
    contact_sensor = isaac_env.scene.sensors["contact_forces"]
    foot_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
    foot_cfg.resolve(isaac_env.scene)
    foot_body_ids = foot_cfg.body_ids
    # Get foot names from the asset for labeling
    asset = isaac_env.scene["robot"]
    _, foot_names = asset.find_bodies(".*_foot")
    print(f"[log_foot_contacts] Foot bodies: {foot_names} (sensor ids: {foot_body_ids})")

    # Run loop
    dt = isaac_env.step_dt
    n = args_cli.num_envs
    max_steps = int(args_cli.duration / dt)

    # Initialize for CompliantOnPolicyRunner
    if runner_class_name == "CompliantOnPolicyRunner":
        isaac_env._force_estimate_xy = torch.zeros(n, 2, device=device)
        compliant_raw_obs_dim = runner._num_one_step_obs

    obs = env.get_observations()

    log_data = {"time_s": []}
    for name in foot_names:
        log_data[name + "_raw"] = []
        log_data[name + "_scaled"] = []

    print(f"[log_foot_contacts] Recording {args_cli.duration:.0f}s ({max_steps} steps)...")

    try:
        for step in range(max_steps):
            if not simulation_app.is_running():
                break
            start_time = time.time()

            with torch.inference_mode():
                if runner_class_name == "CompliantOnPolicyRunner":
                    raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
                    runner._history_buffer.insert(raw_obs)
                    force_hat_pre, _ = runner.estimator.get_latent(
                        runner._history_buffer.get_flattened()
                    )
                    isaac_env._force_estimate_xy = force_hat_pre

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                if runner_class_name == "CompliantOnPolicyRunner":
                    done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        runner._history_buffer.reset(done_ids)

                # Read contact forces for env 0
                net_forces = contact_sensor.data.net_forces_w[0, foot_body_ids, :]  # [4, 3]
                force_norms_raw = torch.norm(net_forces, dim=-1)  # [4] in Newtons
                force_norms_scaled = force_norms_raw * 0.01  # what the policy sees

                log_data["time_s"].append(step * dt)
                for i, name in enumerate(foot_names):
                    log_data[name + "_raw"].append(force_norms_raw[i].item())
                    log_data[name + "_scaled"].append(force_norms_scaled[i].item())

                if step % 50 == 0:
                    elapsed = step * dt
                    vals = " ".join(f"{name}:{force_norms_scaled[i]:.2f}" for i, name in enumerate(foot_names))
                    print(f"\r  {elapsed:.1f}/{args_cli.duration:.0f}s  (scaled) {vals}", end="", flush=True)

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n[log_foot_contacts] Stopped ({type(exc).__name__}).")

    print()
    env.close()

    if len(log_data["time_s"]) < 10:
        print("[log_foot_contacts] Too few steps, skipping plot.")
        return

    import matplotlib.pyplot as plt

    t = np.array(log_data["time_s"])
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

    # 4 rows x 2 cols: left = raw (N), right = scaled (x0.01, what policy sees)
    fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Foot Contact Force Norms — Raw (N) vs Scaled (x0.01, policy obs)", fontsize=14, fontweight="bold")

    for i, name in enumerate(foot_names):
        # Raw (Newtons)
        ax = axes[i, 0]
        vals_raw = np.array(log_data[name + "_raw"])
        ax.plot(t, vals_raw, color=colors[i], linewidth=0.8, alpha=0.9)
        ax.set_ylabel("Force (N)")
        if i == 0:
            ax.set_title("Raw (Newtons)", fontsize=11)
        ax.text(0.02, 0.95, name, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
        ax.text(
            0.98, 0.95,
            f"mean={np.mean(vals_raw):.1f}  max={np.max(vals_raw):.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.grid(True, alpha=0.3)

        # Scaled (what policy sees)
        ax = axes[i, 1]
        vals_scaled = np.array(log_data[name + "_scaled"])
        ax.plot(t, vals_scaled, color=colors[i], linewidth=0.8, alpha=0.9)
        ax.set_ylabel("Scaled (x0.01)")
        if i == 0:
            ax.set_title("Scaled (policy obs, compare with real robot)", fontsize=11)
        ax.text(0.02, 0.95, name, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
        ax.text(
            0.98, 0.95,
            f"mean={np.mean(vals_scaled):.2f}  max={np.max(vals_scaled):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "foot_contact_forces.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[log_foot_contacts] Plot saved: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
