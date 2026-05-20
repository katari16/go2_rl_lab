"""Transient force tracking evaluation.

Applies a reproducible sequence of step-change forces to a standing robot
and measures how quickly the force estimator converges after each change.

The force sequence can be saved and reloaded so different checkpoints are
compared on the exact same perturbation pattern.

Usage:
    # Generate a new sequence and evaluate:
    python scripts/rsl_rl/transient_eval.py --task Go2-Est-Deploy-v0 --checkpoint <path> --num_changes 20 --hold_s 3.0 --force_magnitudes 10 20 30 40 --save_sequence data/eval/force_seq.json

    # Reuse the same sequence on a different checkpoint:
    python scripts/rsl_rl/transient_eval.py --task Go2-Est-Deploy-v0 --checkpoint <path2> --load_sequence data/eval/force_seq.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Transient force tracking evaluation.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_changes", type=int, default=20, help="Number of force direction changes.")
parser.add_argument("--hold_s", type=float, default=3.0, help="Seconds per force episode.")
parser.add_argument("--warmup_s", type=float, default=3.0, help="Warmup seconds before first force.")
parser.add_argument("--force_magnitudes", type=float, nargs="+", default=[10, 20, 30, 40])
parser.add_argument("--directions_deg", type=float, nargs="+", default=[0, 90, 180, 270])
parser.add_argument("--include_zero", action="store_true", default=False, help="Insert zero-force rest between changes.")
parser.add_argument("--zero_hold_s", type=float, default=2.0, help="Duration of zero-force rest periods.")
parser.add_argument("--seed_sequence", type=int, default=42, help="RNG seed for force sequence generation.")
parser.add_argument("--save_sequence", type=str, default=None, help="Path to save generated force sequence JSON.")
parser.add_argument("--load_sequence", type=str, default=None, help="Path to load a previously saved force sequence.")
parser.add_argument("--convergence_frac", type=float, default=0.15, help="Convergence threshold as fraction of magnitude.")
parser.add_argument("--ema_alpha", type=float, default=0.1)
parser.add_argument("--compliance_k", type=float, default=0.0)
parser.add_argument("--show_est", action="store_true", default=False, help="Show estimated force arrow (blue).")
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


# ── Force sequence generation / loading ──────────────────────────────────────


def generate_sequence(args_cli):
    rng = np.random.RandomState(args_cli.seed_sequence)
    episodes = []
    for _ in range(args_cli.num_changes):
        if args_cli.include_zero:
            episodes.append({"direction_deg": 0.0, "magnitude": 0.0, "hold_s": args_cli.zero_hold_s})
        deg = float(rng.choice(args_cli.directions_deg))
        mag = float(rng.choice(args_cli.force_magnitudes))
        episodes.append({"direction_deg": deg, "magnitude": mag, "hold_s": args_cli.hold_s})
    seq = {"seed": args_cli.seed_sequence, "episodes": episodes}
    if args_cli.save_sequence:
        os.makedirs(os.path.dirname(args_cli.save_sequence) or ".", exist_ok=True)
        with open(args_cli.save_sequence, "w") as f:
            json.dump(seq, f, indent=2)
        print(f"[transient] Sequence saved: {args_cli.save_sequence}")
    return seq


def load_sequence(path):
    with open(path) as f:
        seq = json.load(f)
    print(f"[transient] Sequence loaded: {path} ({len(seq['episodes'])} episodes)")
    return seq


# ── Arrow helpers ────────────────────────────────────────────────────────────


def _create_arrow_markers(prim_path, color):
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


def _force_to_quat(force_world, device):
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        up = torch.tensor([0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0], device=device)
        z_ax = torch.linalg.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.linalg.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)
        quats[i] = quat_from_matrix(rot_mat)
    return quats


def _render_arrow(markers, asset, force_body_3d, device, n, z_offset, scale_factor):
    if markers is None:
        return
    base_quat = asset.data.root_quat_w[:n]
    force_world = quat_apply(base_quat, force_body_3d)
    pos = asset.data.root_pos_w[:n].clone()
    pos[:, 2] += z_offset
    quats = _force_to_quat(force_world, device)
    mag = force_world.norm(dim=-1, keepdim=True)
    scales = torch.full((n, 3), 0.3, device=device)
    scales[:, 0:1] = (mag * scale_factor).clamp(min=0.05)
    markers.visualize(pos, quats, scales)


# ── Convergence metric ───────────────────────────────────────────────────────


def compute_convergence_times(gt_series, est_series, transitions, dt, threshold_frac, consecutive=10):
    """Compute convergence time after each force transition.

    Args:
        gt_series: [T, force_dim] GT force at each step.
        est_series: [T, force_dim] estimated force at each step.
        transitions: list of (step_idx, new_magnitude) at each force change.
        dt: timestep.
        threshold_frac: fraction of magnitude for convergence.
        consecutive: number of consecutive steps below threshold.

    Returns:
        list of dicts with convergence_time_s, steady_state_err, etc.
    """
    results = []
    for i, (t_step, mag) in enumerate(transitions):
        end_step = transitions[i + 1][0] if i + 1 < len(transitions) else len(gt_series)
        if end_step <= t_step:
            results.append({"convergence_s": float("nan"), "steady_state_err": float("nan")})
            continue

        gt_new = gt_series[t_step]
        threshold = max(threshold_frac * mag, 1.0)

        seg_est = est_series[t_step:end_step, :2]
        seg_gt = gt_new[:2]
        err = np.linalg.norm(seg_est - seg_gt, axis=1)

        conv_step = None
        count = 0
        for j, e in enumerate(err):
            if e < threshold:
                count += 1
                if count >= consecutive:
                    conv_step = j - consecutive + 1
                    break
            else:
                count = 0

        conv_s = conv_step * dt if conv_step is not None else (end_step - t_step) * dt

        last_quarter = err[max(0, int(len(err) * 0.75)):]
        ss_err = float(np.mean(last_quarter)) if len(last_quarter) > 0 else float("nan")

        results.append({"convergence_s": conv_s, "steady_state_err": ss_err})
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────


def generate_plots(log, episodes, conv_results, force_dim, checkpoint_path, args_cli):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime

    t = np.array(log["time_s"])
    gt = np.array(log["gt_force"])
    est = np.array(log["est_force"])
    transitions = log["transitions"]

    fd3 = min(force_dim, 3)
    yaw_idx = {4: 3, 6: 5}.get(force_dim, None)

    figures = []

    # ── Page 1: Full time series Fx, Fy ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Transient Force Tracking — Fx / Fy", fontsize=14, fontweight="bold")

    for t_step, _ in transitions:
        if t_step < len(t):
            for ax in axes:
                ax.axvline(t[t_step], color="gray", alpha=0.4, linewidth=0.8, linestyle="--")

    axes[0].plot(t, gt[:, 0], "tab:red", linewidth=1.5, label="GT Fx")
    axes[0].plot(t, est[:, 0], "tab:blue", linewidth=1.0, alpha=0.7, linestyle="--", label="Est Fx")
    axes[0].set_ylabel("Fx (N)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, gt[:, 1], "tab:red", linewidth=1.5, label="GT Fy")
    axes[1].plot(t, est[:, 1], "tab:blue", linewidth=1.0, alpha=0.7, linestyle="--", label="Est Fy")
    axes[1].set_ylabel("Fy (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 2: Fz (and torque yaw if applicable) ────────────────────────
    if fd3 >= 3:
        n_panels = 1 + (1 if yaw_idx is not None else 0)
        fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        fig.suptitle("Transient Force Tracking — Fz / Torque", fontsize=14, fontweight="bold")
        for t_step, _ in transitions:
            if t_step < len(t):
                for ax in axes:
                    ax.axvline(t[t_step], color="gray", alpha=0.4, linewidth=0.8, linestyle="--")

        axes[0].plot(t, gt[:, 2], "tab:red", linewidth=1.5, label="GT Fz")
        axes[0].plot(t, est[:, 2], "tab:blue", linewidth=1.0, alpha=0.7, linestyle="--", label="Est Fz")
        axes[0].set_ylabel("Fz (N)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        if yaw_idx is not None:
            axes[1].plot(t, gt[:, yaw_idx], "tab:red", linewidth=1.5, label="GT τ_yaw")
            axes[1].plot(t, est[:, yaw_idx], "tab:blue", linewidth=1.0, alpha=0.7, linestyle="--", label="Est τ_yaw")
            axes[1].set_ylabel("τ_yaw (Nm)")
            axes[1].set_xlabel("Time (s)")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    # ── Page 3: XY error magnitude over time ─────────────────────────────
    err_xy = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, err_xy, "tab:purple", linewidth=1.0)
    for t_step, _ in transitions:
        if t_step < len(t):
            ax.axvline(t[t_step], color="gray", alpha=0.4, linewidth=0.8, linestyle="--")
    ax.set_ylabel("XY Error (N)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Force Estimation Error (XY L2 norm)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 4: Convergence time bar chart ───────────────────────────────
    if conv_results:
        conv_times = [r["convergence_s"] for r in conv_results]
        ss_errs = [r["steady_state_err"] for r in conv_results]
        mags = [mag for _, mag in transitions]

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        fig.suptitle("Per-Transition Metrics", fontsize=14, fontweight="bold")

        x = np.arange(len(conv_times))
        colors = ["tab:green" if ct < args_cli.hold_s * 0.5 else "tab:orange" if ct < args_cli.hold_s else "tab:red"
                  for ct in conv_times]
        axes[0].bar(x, conv_times, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_ylabel("Convergence Time (s)")
        axes[0].set_title(f"Convergence Time (threshold: {args_cli.convergence_frac * 100:.0f}% of magnitude)", fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{m:.0f}N" for m in mags], rotation=45, fontsize=7)
        axes[0].axhline(np.nanmean(conv_times), color="black", linestyle="--", alpha=0.5,
                        label=f"mean={np.nanmean(conv_times):.2f}s")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(x, ss_errs, color="tab:blue", edgecolor="black", linewidth=0.5)
        axes[1].set_ylabel("Steady-State Error (N)")
        axes[1].set_title("Steady-State XY Error (last 25% of episode)", fontsize=11)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{m:.0f}N" for m in mags], rotation=45, fontsize=7)
        axes[1].axhline(np.nanmean(ss_errs), color="black", linestyle="--", alpha=0.5,
                        label=f"mean={np.nanmean(ss_errs):.2f}N")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    # ── Page 5: Summary table ────────────────────────────────────────────
    if conv_results:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        conv_arr = np.array([r["convergence_s"] for r in conv_results])
        ss_arr = np.array([r["steady_state_err"] for r in conv_results])
        headers = ["Metric", "Mean", "Median", "Std", "Min", "Max"]
        rows = [
            ["Conv. time (s)", f"{np.nanmean(conv_arr):.3f}", f"{np.nanmedian(conv_arr):.3f}",
             f"{np.nanstd(conv_arr):.3f}", f"{np.nanmin(conv_arr):.3f}", f"{np.nanmax(conv_arr):.3f}"],
            ["SS error (N)", f"{np.nanmean(ss_arr):.2f}", f"{np.nanmedian(ss_arr):.2f}",
             f"{np.nanstd(ss_arr):.2f}", f"{np.nanmin(ss_arr):.2f}", f"{np.nanmax(ss_arr):.2f}"],
        ]
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.5)
        fig.suptitle("Transient Tracking Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        figures.append(fig)

    # ── Save PDF ─────────────────────────────────────────────────────────
    eval_dir = os.path.join(os.path.dirname(checkpoint_path), "force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dim_str = {2: "xy", 3: "xyz", 4: "4d", 6: "6d"}.get(force_dim, f"{force_dim}d")
    pdf_path = os.path.join(eval_dir, f"transient_{dim_str}_{timestamp}.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[transient] Report saved: {pdf_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Standing robot
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.debug_vis = False

    # Disable automatic force events
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    try:
        force_event = getattr(env_cfg.events, force_event_name)
        force_event.params["force_range"] = (0.0, 0.0)
        if "torque_range" in force_event.params:
            force_event.params["torque_range"] = (0.0, 0.0)
    except AttributeError:
        pass

    # Long episode
    env_cfg.episode_length_s = 600.0

    # ── Force sequence ───────────────────────────────────────────────────
    if args_cli.load_sequence:
        seq = load_sequence(args_cli.load_sequence)
    else:
        seq = generate_sequence(args_cli)
    episodes = seq["episodes"]

    # ── Checkpoint ───────────────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[transient] Checkpoint: {resume_path}")

    # ── Create env + runner ──────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_class_name = agent_cfg.class_name
    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner
        train_cfg = agent_cfg.to_dict()
        if args_cli.checkpoint:
            train_cfg["estimator_checkpoint"] = args_cli.checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "ForceOnPolicyRunner":
        from go2_rl_lab.estimator.force_runner import ForceOnPolicyRunner
        runner = ForceOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {runner_class_name}")

    runner.load(resume_path)
    runner.eval_mode()

    if hasattr(runner, "_wrapped_env"):
        env = runner._wrapped_env
    if hasattr(runner, "_mapping_active"):
        runner._mapping_active = True
        isaac_env_raw = env.unwrapped if hasattr(env, "unwrapped") else env
        isaac_env_raw._mapping_active = True
        isaac_env_raw._compliance_alpha = runner._compliance_alpha
        isaac_env_raw._compliance_beta = runner._compliance_beta

    compliant_raw_obs_dim = getattr(runner, "_num_one_step_obs", None)
    force_dim = getattr(runner, "_force_dim", 2)
    fd3 = min(force_dim, 3)
    yaw_idx = {4: 3, 6: 5}.get(force_dim, None)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    isaac_env = env.unwrapped
    device = isaac_env.device
    dt = isaac_env.step_dt
    n = args_cli.num_envs
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)

    if compliant_raw_obs_dim is not None:
        isaac_env._force_estimate_xy = torch.zeros(n, force_dim, device=device)

    force_ema = torch.zeros(n, force_dim, device=device)
    compliance_k = args_cli.compliance_k
    ema_alpha = args_cli.ema_alpha

    # Arrows
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0))
    est_markers = _create_arrow_markers("/World/Visuals/EstForceArrow", (0.2, 0.4, 1.0)) if args_cli.show_est else None

    # ── Compute total steps ──────────────────────────────────────────────
    warmup_steps = int(args_cli.warmup_s / dt)
    total_episode_s = sum(ep["hold_s"] for ep in episodes)
    total_steps = warmup_steps + int(total_episode_s / dt)

    dim_labels = {2: "XY (2D)", 3: "XYZ (3D)", 4: "4D wrench", 6: "6D wrench"}
    print(f"\n{'=' * 70}")
    print(f"  Transient Tracking Evaluation")
    print(f"  Runner      : {runner_class_name}")
    print(f"  Force dims  : {dim_labels.get(force_dim, f'{force_dim}D')}")
    print(f"  Episodes    : {len(episodes)} changes, {total_episode_s:.0f}s total")
    print(f"  Warmup      : {args_cli.warmup_s}s")
    print(f"  Magnitudes  : {sorted(set(ep['magnitude'] for ep in episodes))} N")
    print(f"  Directions  : {sorted(set(ep['direction_deg'] for ep in episodes))}°")
    arrows = ["RED=GT"]
    if args_cli.show_est:
        arrows.append("BLUE=est")
    print(f"  Arrows      : {', '.join(arrows)}")
    print(f"{'=' * 70}\n")

    # ── Data recording ───────────────────────────────────────────────────
    log_gt = []
    log_est = []
    log_time = []
    transitions = []

    obs = env.get_observations()
    step_count = 0

    def _read_gt():
        if force_dim <= 3:
            gt = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :force_dim]
        elif force_dim == 4:
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :3]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, 2:3]
            gt = torch.cat([gt_f, gt_t], dim=-1)
        else:
            gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :3]
            gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, :3]
            gt = torch.cat([gt_f, gt_t], dim=-1)
        if gt.dim() == 3:
            gt = gt.squeeze(1)
        return gt

    def _step():
        nonlocal obs, step_count, force_ema
        with torch.inference_mode():
            if compliant_raw_obs_dim is not None:
                raw_obs = obs["policy"][:, :compliant_raw_obs_dim]
                runner._history_buffer.insert(raw_obs)
                fhat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
                isaac_env._force_estimate_xy = fhat
                force_ema = ema_alpha * fhat + (1.0 - ema_alpha) * force_ema

            if compliance_k > 0.0:
                obs["policy"][:, 6] = obs["policy"][:, 6] + compliance_k * force_ema[:, 0]
                obs["policy"][:, 7] = obs["policy"][:, 7] + compliance_k * force_ema[:, 1]
                if yaw_idx is not None:
                    obs["policy"][:, 8] = obs["policy"][:, 8] + compliance_k * force_ema[:, yaw_idx]

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            if compliant_raw_obs_dim is not None:
                done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if len(done_ids) > 0:
                    runner._history_buffer.reset(done_ids)

            # Record
            gt_force = _read_gt()
            fhat_now, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
            log_gt.append(gt_force[0].cpu().numpy())
            log_est.append(fhat_now[0].cpu().numpy())
            log_time.append(step_count * dt)

            # Arrows
            gt_3d = torch.zeros(n, 3, device=device)
            gt_3d[:, :fd3] = gt_force[:, :fd3]
            _render_arrow(gt_markers, asset, gt_3d, device, n, 0.55, 0.05)
            if est_markers is not None:
                est_3d = torch.zeros(n, 3, device=device)
                est_3d[:, :fd3] = fhat_now[:n, :fd3]
                _render_arrow(est_markers, asset, est_3d, device, n, 0.45, 0.05)

            step_count += 1

    def _apply_force(fx, fy):
        force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        force_tensor[:, base_idx, 0] = fx
        force_tensor[:, base_idx, 1] = fy
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force_tensor, torques=torque_tensor,
        )

    # ── Run ──────────────────────────────────────────────────────────────
    try:
        # Warmup (zero force)
        print(f"  Warmup ({args_cli.warmup_s}s)...", flush=True)
        _apply_force(0.0, 0.0)
        for _ in range(warmup_steps):
            _step()
            if args_cli.real_time:
                time.sleep(dt)

        # Force episodes
        for ep_idx, ep in enumerate(episodes):
            deg = ep["direction_deg"]
            mag = ep["magnitude"]
            hold_steps = int(ep["hold_s"] / dt)
            rad = math.radians(deg)
            fx = mag * math.cos(rad)
            fy = mag * math.sin(rad)

            transitions.append((step_count, mag))
            _apply_force(fx, fy)

            gt_parts = f"Fx={fx:+6.1f}, Fy={fy:+6.1f}"
            print(f"  [{ep_idx + 1}/{len(episodes)}] {deg:+6.0f}° {mag:5.1f}N ({gt_parts}) for {ep['hold_s']:.1f}s", flush=True)

            for _ in range(hold_steps):
                start_t = time.time()
                _step()
                if args_cli.real_time:
                    sleep_t = dt - (time.time() - start_t)
                    if sleep_t > 0:
                        time.sleep(sleep_t)

        _apply_force(0.0, 0.0)

    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        print(f"\n[transient] Stopping ({type(exc).__name__}).")

    print()
    env.close()

    # ── Post-hoc metrics ─────────────────────────────────────────────────
    gt_arr = np.array(log_gt)
    est_arr = np.array(log_est)

    conv_results = compute_convergence_times(
        gt_arr, est_arr, transitions, dt,
        args_cli.convergence_frac,
    )

    # Terminal summary
    conv_times = [r["convergence_s"] for r in conv_results]
    ss_errs = [r["steady_state_err"] for r in conv_results]
    print(f"  Convergence time: mean={np.nanmean(conv_times):.3f}s  median={np.nanmedian(conv_times):.3f}s  std={np.nanstd(conv_times):.3f}s")
    print(f"  Steady-state err: mean={np.nanmean(ss_errs):.2f}N  median={np.nanmedian(ss_errs):.2f}N")

    # ── Save raw data ────────────────────────────────────────────────────
    eval_dir = os.path.join(os.path.dirname(resume_path), "force_eval")
    os.makedirs(eval_dir, exist_ok=True)
    raw_data = {
        "sequence": seq,
        "force_dim": force_dim,
        "dt": dt,
        "transitions": transitions,
        "convergence": conv_results,
        "gt_force": gt_arr.tolist(),
        "est_force": est_arr.tolist(),
        "time_s": log_time,
    }
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    raw_path = os.path.join(eval_dir, f"transient_raw_{timestamp}.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f)
    print(f"[transient] Raw data saved: {raw_path}")

    # ── Plots ────────────────────────────────────────────────────────────
    from time import sleep
    sleep(1)

    log_dict = {"time_s": log_time, "gt_force": gt_arr, "est_force": est_arr, "transitions": transitions}
    generate_plots(log_dict, episodes, conv_results, force_dim, resume_path, args_cli)


if __name__ == "__main__":
    main()
    simulation_app.close()
