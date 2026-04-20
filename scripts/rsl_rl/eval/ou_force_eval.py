"""OU Force Eval — applies Ornstein-Uhlenbeck forces to a standing robot and compares GT vs estimator.

Forces are generated per-axis via the OU process, clipped to the training distribution range.
Robot stands on flat ground with zero velocity command throughout.

Usage:
    python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-J5-v0 \\
        --checkpoint logs/rsl_rl/ablation_J5_.../model_XXXX.pt \\
        --duration 60 --show_gt --show_est --real-time
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="OU force eval for CompliantOnPolicyRunner ablation runs.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration", type=float, default=60.0, help="Total eval duration in seconds.")
parser.add_argument("--ou_theta", type=float, default=0.15, help="OU mean-reversion rate (lower=slower changes).")
parser.add_argument("--ou_sigma_force", type=float, default=None, help="OU noise sigma for force axes (default: 0.3 * max_force).")
parser.add_argument("--ou_sigma_torque", type=float, default=None, help="OU noise sigma for torque axis (default: 0.3 * max_torque).")
parser.add_argument("--max_force_override", type=float, default=None, help="Override training max_force (N).")
parser.add_argument("--max_torque_override", type=float, default=None, help="Override training max_torque (Nm).")
parser.add_argument("--fz_scale_override", type=float, default=None, help="Override fz_scale (default: read from event params).")
parser.add_argument("--show_gt", action="store_true", default=False, help="Show applied force arrow (red).")
parser.add_argument("--show_est", action="store_true", default=False, help="Show estimator output arrow (blue).")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401


# ── Arrow helpers ──────────────────────────────────────────────────────────────

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


def _force_to_quat(force: torch.Tensor, device: torch.device) -> torch.Tensor:
    n = force.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    for i in range(n):
        mag = force[i].norm()
        if mag < 0.1:
            continue
        x_ax = force[i] / mag
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


# ── OU process ────────────────────────────────────────────────────────────────

class OUProcess:
    """Per-axis Ornstein-Uhlenbeck process, clipped to [-clip, clip]."""

    def __init__(self, shape: tuple, theta: float, sigma: float, clip: float, dt: float, device):
        self.theta = theta
        self.sigma = sigma
        self.clip = clip
        self.dt = dt
        self.device = device
        self.state = torch.zeros(shape, device=device)

    def step(self) -> torch.Tensor:
        noise = torch.randn_like(self.state)
        self.state = (
            self.state * (1.0 - self.theta * self.dt)
            + self.sigma * (self.dt ** 0.5) * noise
        )
        self.state.clamp_(-self.clip, self.clip)
        return self.state.clone()

    def reset(self, idx: torch.Tensor | None = None):
        if idx is None:
            self.state.zero_()
        else:
            self.state[idx] = 0.0


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_plots(log: dict, args, checkpoint_path: str, force_dim: int, max_force: float, max_torque: float, fz_scale: float) -> None:
    import matplotlib.pyplot as plt
    from datetime import datetime

    t = np.array(log["time_s"])
    has_torque_yaw = force_dim >= 4
    has_torque_rp  = force_dim >= 6

    specs = [
        ("Fx (N)",       "gt_fx",     "est_fx",     "tab:red",    max_force),
        ("Fy (N)",       "gt_fy",     "est_fy",     "darkred",    max_force),
        ("Fz (N)",       "gt_fz",     "est_fz",     "tab:purple", max_force * fz_scale),
    ]
    if has_torque_rp:
        specs.append(("τ_roll (Nm)",  "gt_troll",  "est_troll",  "tab:orange",  max_torque))
        specs.append(("τ_pitch (Nm)", "gt_tpitch", "est_tpitch", "darkorange",  max_torque))
    if has_torque_yaw:
        specs.append(("τ_yaw (Nm)",   "gt_tyaw",   "est_tyaw",   "tab:brown",   max_torque))

    n_panels = len(specs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"OU Force Eval — {args.task} — θ={args.ou_theta:.2f} — "
        f"max_force={max_force:.0f}N fz_scale={fz_scale:.1f} max_torque={max_torque:.0f}Nm",
        fontsize=13, fontweight="bold",
    )

    for ax, (ylabel, gt_key, est_key, color, ylim) in zip(axes, specs):
        ax.plot(t, log[gt_key],  color=color, lw=1.2, label="GT (applied)")
        ax.plot(t, log[est_key], color=color, lw=0.9, ls="--", alpha=0.7, label="Est")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_ylim(-ylim * 1.15, ylim * 1.15)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{ylabel.split()[0]}: GT (solid) vs Estimator (dashed)", fontsize=11)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "ou_force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fig_path = os.path.join(eval_dir, f"ou_force_{args.duration:.0f}s_theta{args.ou_theta:.2f}_{ts}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ou_force_eval] Plot saved: {fig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Flat terrain
    env_cfg.scene.terrain.terrain_type = "plane"

    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None

    # Zero velocity command throughout
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Disable the training force event — we apply forces manually via OU
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    force_event = getattr(env_cfg.events, force_event_name, None)
    if force_event is not None:
        force_event.params["force_range"] = (0.0, 0.0)
        if "torque_range" in force_event.params:
            force_event.params["torque_range"] = (0.0, 0.0)

    env_cfg.episode_length_s = args_cli.duration + 5.0

    # Checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    print(f"[ou_force_eval] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    if runner_class_name != "CompliantOnPolicyRunner":
        raise ValueError(f"ou_force_eval only supports CompliantOnPolicyRunner, got: {runner_class_name}")

    from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
    train_cfg = agent_cfg.to_dict()
    if args_cli.checkpoint:
        train_cfg["estimator_checkpoint"] = args_cli.checkpoint
    runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    runner.eval_mode()

    force_dim = getattr(runner, "_force_dim", 2)
    fd3 = min(force_dim, 3)
    raw_obs_dim = runner._num_one_step_obs
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
    _body_ids = base_body_ids if isinstance(base_body_ids, list) else [int(base_body_ids)]
    n = args_cli.num_envs
    _all_env_ids = torch.arange(n, device=device)

    # Force range from training config
    max_force = args_cli.max_force_override or agent_cfg.to_dict().get("max_force", 30.0)
    max_torque = args_cli.max_torque_override or agent_cfg.to_dict().get("max_torque", 10.0)

    # fz_scale from event params
    fz_scale = 1.0
    if force_event is not None and "fz_scale" in force_event.params:
        fz_scale = force_event.params["fz_scale"]
    if args_cli.fz_scale_override is not None:
        fz_scale = args_cli.fz_scale_override

    dt = isaac_env.step_dt
    theta = args_cli.ou_theta
    sigma_f = args_cli.ou_sigma_force  or 0.3 * max_force
    sigma_t = args_cli.ou_sigma_torque or 0.3 * max_torque

    # OU processes: one per force axis, one per torque axis
    ou_fx    = OUProcess((n,), theta, sigma_f, max_force,            dt, device)
    ou_fy    = OUProcess((n,), theta, sigma_f, max_force,            dt, device)
    ou_fz    = OUProcess((n,), theta, sigma_f, max_force * fz_scale, dt, device)
    ou_roll  = OUProcess((n,), theta, sigma_t, max_torque,           dt, device) if force_dim >= 6 else None
    ou_pitch = OUProcess((n,), theta, sigma_t, max_torque,           dt, device) if force_dim >= 6 else None
    ou_yaw   = OUProcess((n,), theta, sigma_t, max_torque,           dt, device) if force_dim >= 4 else None

    yaw_idx = {4: 3, 6: 5}.get(force_dim, None)

    # Arrow markers
    gt_markers  = _create_arrow_markers("/World/Visuals/OUGTForce",  (1.0, 0.0, 0.0)) if args_cli.show_gt  else None
    est_markers = _create_arrow_markers("/World/Visuals/OUEstForce", (0.2, 0.4, 1.0)) if args_cli.show_est else None

    total_steps = int(args_cli.duration / dt)

    isaac_env._force_estimate_xy = torch.zeros(n, force_dim, device=device)
    obs = env.get_observations()

    log: dict = {
        "time_s": [],
        "gt_fx": [], "gt_fy": [], "gt_fz": [],
        "est_fx": [], "est_fy": [], "est_fz": [],
    }
    if force_dim >= 6:
        log["gt_troll"]  = []; log["est_troll"]  = []
        log["gt_tpitch"] = []; log["est_tpitch"] = []
    if force_dim >= 4:
        log["gt_tyaw"]  = []
        log["est_tyaw"] = []

    force_scale = 0.05

    print(f"\n{'=' * 70}")
    print(f"  Task        : {args_cli.task}")
    print(f"  Force dim   : {force_dim}D")
    print(f"  Max force   : {max_force:.0f}N (fz_scale={fz_scale:.1f})")
    print(f"  Max torque  : {max_torque:.0f}Nm" if force_dim >= 4 else "  Torque      : none")
    print(f"  OU theta    : {theta:.3f}  σ_force={sigma_f:.2f}  σ_torque={sigma_t:.2f}")
    print(f"  Duration    : {args_cli.duration:.0f}s")
    print(f"{'=' * 70}\n")

    step_count = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # Update OU processes
                fx_ou    = ou_fx.step()
                fy_ou    = ou_fy.step()
                fz_ou    = ou_fz.step()
                roll_ou  = ou_roll.step()  if ou_roll  is not None else torch.zeros(n, device=device)
                pitch_ou = ou_pitch.step() if ou_pitch is not None else torch.zeros(n, device=device)
                yaw_ou   = ou_yaw.step()   if ou_yaw   is not None else torch.zeros(n, device=device)

                base_quat = asset.data.root_quat_w[:n]

                # Build body-frame force/torque tensors [n, 1, 3]
                f_body_w = torch.zeros(n, 1, 3, device=device)
                f_body_w[:, 0, 0] = fx_ou
                f_body_w[:, 0, 1] = fy_ou
                f_body_w[:, 0, 2] = fz_ou
                t_body_w = torch.zeros(n, 1, 3, device=device)
                t_body_w[:, 0, 0] = roll_ou
                t_body_w[:, 0, 1] = pitch_ou
                t_body_w[:, 0, 2] = yaw_ou

                # Apply via permanent wrench composer (same as training)
                asset.permanent_wrench_composer.set_forces_and_torques(
                    forces=f_body_w,
                    torques=t_body_w,
                    body_ids=_body_ids,
                    env_ids=_all_env_ids,
                )

                # GT from wrench composer (what's actually applied)
                gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :3]
                gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, :3]

                # World-frame force for arrow rendering only
                f_world = quat_apply(base_quat, gt_f)

                # Estimator
                raw_obs = obs["policy"][:, :raw_obs_dim]
                runner._history_buffer.insert(raw_obs)
                force_hat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
                force_hat = force_hat[:n]
                isaac_env._force_estimate_xy = force_hat

                # Policy step (zero velocity cmd injected via env config)
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

                done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if len(done_ids) > 0:
                    runner._history_buffer.reset(done_ids)
                    ou_fx.reset(done_ids); ou_fy.reset(done_ids); ou_fz.reset(done_ids)
                    if ou_roll  is not None: ou_roll.reset(done_ids)
                    if ou_pitch is not None: ou_pitch.reset(done_ids)
                    if ou_yaw   is not None: ou_yaw.reset(done_ids)

                # Arrow visualization
                base_pos = asset.data.root_pos_w[:n]
                if gt_markers is not None:
                    gt_pos = base_pos.clone(); gt_pos[:, 2] += 0.55
                    gt_scales = torch.full((n, 3), 0.3, device=device)
                    gt_scales[:, 0:1] = (f_world.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                    gt_markers.visualize(gt_pos, _force_to_quat(f_world, device), gt_scales)

                if est_markers is not None:
                    est_3d = torch.zeros(n, 3, device=device)
                    est_3d[:, :fd3] = force_hat[:, :fd3]
                    est_world = quat_apply(base_quat, est_3d)
                    est_pos = base_pos.clone(); est_pos[:, 2] += 0.42
                    est_scales = torch.full((n, 3), 0.3, device=device)
                    est_scales[:, 0:1] = (est_world.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                    est_markers.visualize(est_pos, _force_to_quat(est_world, device), est_scales)

                # Log (env 0) — GT from wrench composer
                log["time_s"].append(step_count * dt)
                log["gt_fx"].append(gt_f[0, 0].item())
                log["gt_fy"].append(gt_f[0, 1].item())
                log["gt_fz"].append(gt_f[0, 2].item())
                log["est_fx"].append(force_hat[0, 0].item())
                log["est_fy"].append(force_hat[0, 1].item())
                log["est_fz"].append(force_hat[0, 2].item() if force_dim >= 3 else 0.0)
                if force_dim >= 6:
                    log["gt_troll"].append(gt_t[0, 0].item())
                    log["gt_tpitch"].append(gt_t[0, 1].item())
                    log["est_troll"].append(force_hat[0, 3].item())
                    log["est_tpitch"].append(force_hat[0, 4].item())
                if force_dim >= 4:
                    log["gt_tyaw"].append(gt_t[0, 2].item())
                    log["est_tyaw"].append(force_hat[0, yaw_idx].item())

                step_count += 1
                if step_count % 25 == 0:
                    pct = step_count / total_steps
                    bar = "\u2588" * int(20 * pct) + "\u2591" * (20 - int(20 * pct))
                    e0 = force_hat[0]
                    print(
                        f"\r  [{bar}] {step_count * dt:.1f}/{args_cli.duration:.0f}s  "
                        f"GT:[{gt_f[0,0]:+5.1f},{gt_f[0,1]:+5.1f},{gt_f[0,2]:+5.1f}]N  "
                        f"Est:[{e0[0]:+5.1f},{e0[1]:+5.1f},{e0[2]:+5.1f}]N",
                        end="", flush=True,
                    )

                if step_count >= total_steps:
                    break

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n\n[ou_force_eval] Stopped ({type(exc).__name__}).")

    print()
    env.close()

    from time import sleep
    sleep(1)

    if len(log["time_s"]) > 10:
        generate_plots(log, args_cli, resume_path, force_dim, max_force, max_torque, fz_scale)
    else:
        print("[ou_force_eval] Too few steps, skipping plots.")


if __name__ == "__main__":
    main()
    simulation_app.close()
