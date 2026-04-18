"""Rollout-based force estimator evaluation with force baskets.

Runs many envs with manually applied random forces at specific magnitude
ranges (baskets). Forces are re-randomized every 1-3s, mimicking the
training force events. Metrics are computed per basket.

The 5 sample envs (0-4) use a deterministic force schedule from a fixed
seed, so their GT profiles are identical across runs — enabling direct
comparison of different estimators on the same force inputs.

Outputs per basket:
  - Console summary: MAE, per-axis MAE, relative error, angular error
  - PDF: 5 sample env force + torque time series, metrics table
  - sample_envs_<B>N.json: GT + est time series for envs 0-4
  - metrics.json: all aggregate numbers across baskets

Usage:
    python scripts/rsl_rl/eval/rollout_estimator_eval.py \
        --task Go2-Ablation-H15-50N-v0 \
        --checkpoint logs/rsl_rl/.../model_9500.pt \
        --force_baskets 20 40 50 --num_envs 4096 --duration 20 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Rollout-based force estimator evaluation.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--duration", type=float, default=20.0)
parser.add_argument("--warmup_s", type=float, default=3.0)
parser.add_argument("--force_baskets", type=float, nargs="+", required=True,
                    help="Force magnitude baskets in N. Each basket B runs forces in [0.8*B, B] per axis.")
parser.add_argument("--basket_band", type=float, default=0.8,
                    help="Lower fraction of basket range (default: 0.8 → [0.8*B, B]).")
parser.add_argument("--ema_alpha", type=float, default=0.1)
parser.add_argument("--compliance_k", type=float, default=0.0)
parser.add_argument("--no_cmd_vis", action="store_true", default=False,
                    help="Disable built-in velocity command and base velocity debug arrows.")
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
import os
import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply, quat_from_matrix
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_rl_lab.tasks  # noqa: F401

from eval.eval_utils import (
    create_eval_output_dir,
    create_runner,
    disable_force_events,
    get_asset_and_base,
    read_gt_force,
    reset_env,
    resolve_checkpoint,
    save_config,
    set_long_episode,
    setup_runner_for_eval,
    step_policy,
)
from eval.eval_utils import _NumpyEncoder


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


def _force_to_quat(force_world: torch.Tensor, device) -> torch.Tensor:
    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        up = torch.tensor(
            [0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0],
            device=device,
        )
        z_ax = torch.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)
        quats[i] = quat_from_matrix(rot_mat)
    return quats


DIM_LABELS = {
    2: ["Fx", "Fy"],
    3: ["Fx", "Fy", "Fz"],
    4: ["Fx", "Fy", "Fz", "τ_yaw"],
    6: ["Fx", "Fy", "Fz", "τ_roll", "τ_pitch", "τ_yaw"],
}

SAMPLE_ENVS = [0, 1, 2, 3, 4]


def _generate_force_schedule(rng, max_steps, n, f_min, f_max, fz_scale,
                             is_wrench, torque_range, dt):
    """Pre-generate deterministic force+torque schedule for all envs.

    Returns:
        forces: [max_steps, n, 3]
        torques: [max_steps, n, 3]
    """
    forces = np.zeros((max_steps, n, 3), dtype=np.float32)
    torques = np.zeros((max_steps, n, 3), dtype=np.float32)

    for env_i in range(n):
        step = 0
        while step < max_steps:
            dur = max(1, int(rng.uniform(1.0, 3.0) / dt))
            end = min(step + dur, max_steps)

            f_mag = rng.uniform(f_min, f_max, size=3).astype(np.float32)
            f_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=3)
            f = f_mag * f_sign
            f[2] *= fz_scale
            forces[step:end, env_i, :] = f

            if is_wrench:
                t_mag = rng.uniform(torque_range[0], torque_range[1], size=3).astype(np.float32)
                t_sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=3)
                torques[step:end, env_i, :] = t_mag * t_sign

            step = end

    return forces, torques


def _compute_metrics(all_gt_np, all_est_np, force_dim, dim_labels):
    """Compute aggregate estimation metrics.

    Returns:
        dict of metrics
    """
    err = all_est_np - all_gt_np
    abs_err = np.abs(err)                          # [steps, envs, force_dim]
    per_sample_mae = abs_err.mean(axis=2)          # [steps, envs] — matches training mae_total
    norm_err = np.linalg.norm(err, axis=2)         # L2 norm — used only for relative error
    gt_mag = np.linalg.norm(all_gt_np, axis=2)

    active_mask = gt_mag > 1.0
    has_active = active_mask.any()

    # Per-component mean AE — directly comparable to training Estimator/mae_total
    mae = float(np.mean(per_sample_mae[active_mask])) if has_active else 0.0
    median_ae = float(np.median(per_sample_mae[active_mask])) if has_active else 0.0

    rel_err = 0.0
    if has_active:
        rel_per_step = norm_err[active_mask] / gt_mag[active_mask] * 100
        rel_err = float(np.mean(rel_per_step))

    per_axis_mae = {}
    for d in range(force_dim):
        axis_ae = np.abs(err[:, :, d])
        per_axis_mae[dim_labels[d]] = float(np.mean(axis_ae[active_mask])) if has_active else 0.0

    gt_angle = np.arctan2(all_gt_np[:, :, 1], all_gt_np[:, :, 0])
    est_angle = np.arctan2(all_est_np[:, :, 1], all_est_np[:, :, 0])
    angle_diff = np.arctan2(np.sin(est_angle - gt_angle), np.cos(est_angle - gt_angle))
    angle_err_deg = np.abs(angle_diff) * 180.0 / np.pi
    xy_mask = np.linalg.norm(all_gt_np[:, :, :2], axis=2) > 1.0
    has_xy = xy_mask.any()
    angular_mean = float(np.mean(angle_err_deg[xy_mask])) if has_xy else 0.0
    angular_median = float(np.median(angle_err_deg[xy_mask])) if has_xy else 0.0

    force_indices = list(range(min(force_dim, 3)))
    torque_indices = list(range(3, force_dim))

    # Force magnitude error: | ||F_hat_xyz|| - ||F_gt_xyz|| |
    gt_force_mag = np.linalg.norm(all_gt_np[:, :, :min(force_dim, 3)], axis=2)
    est_force_mag = np.linalg.norm(all_est_np[:, :, :min(force_dim, 3)], axis=2)
    force_mag_err = np.abs(est_force_mag - gt_force_mag)
    force_mag_mask = gt_force_mag > 1.0
    force_mag_mae = float(np.mean(force_mag_err[force_mag_mask])) if force_mag_mask.any() else 0.0

    return {
        "mae": mae,
        "median_ae": median_ae,
        "relative_err_pct": rel_err,
        "angular_err_xy_deg_mean": angular_mean,
        "angular_err_xy_deg_median": angular_median,
        "per_axis_mae": per_axis_mae,
        "force_mae": float(np.mean([per_axis_mae[dim_labels[d]] for d in force_indices])),
        "torque_mae": float(np.mean([per_axis_mae[dim_labels[d]] for d in torque_indices])) if torque_indices else None,
        "force_mag_mae": force_mag_mae,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    n = args_cli.num_envs
    env_cfg.scene.num_envs = n
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xy_force")
    is_wrench = "wrench" in force_event_name

    # Read fz_scale and torque_range from env config before disabling events
    try:
        force_event = getattr(env_cfg.events, force_event_name)
        fz_scale = force_event.params.get("fz_scale", 0.6)
        torque_range = force_event.params.get("torque_range", (0.0, 5.0))
    except AttributeError:
        fz_scale = 0.6
        torque_range = (0.0, 5.0)

    # Disable automatic force events — we apply forces manually
    disable_force_events(env_cfg, agent_cfg)
    set_long_episode(env_cfg)

    if args_cli.no_cmd_vis:
        env_cfg.commands.base_velocity.debug_vis = False

    resume_path = resolve_checkpoint(agent_cfg, args_cli)
    print(f"[rollout_eval] Checkpoint: {resume_path}")

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

    force_dim = ctx["force_dim"]
    has_estimator = ctx["has_estimator"]

    # Arrow markers — only when not headless
    _headless = getattr(args_cli, "headless", True)
    gt_markers = _create_arrow_markers("/World/Visuals/GTForceArrow", (1.0, 0.0, 0.0)) if not _headless else None
    est_markers = _create_arrow_markers("/World/Visuals/EstForceArrow", (0.2, 0.4, 1.0)) if not _headless else None

    if not has_estimator:
        print("[rollout_eval] ERROR: No estimator found. This eval requires an estimator.")
        env.close()
        simulation_app.close()
        return

    max_steps = int(args_cli.duration / dt)
    warmup_steps = int(args_cli.warmup_s / dt)
    compliance_k = args_cli.compliance_k
    dim_labels = DIM_LABELS.get(force_dim, [f"d{i}" for i in range(force_dim)])
    band = args_cli.basket_band

    print(f"\n{'=' * 70}")
    print(f"  Rollout Estimator Evaluation")
    print(f"  Task: {args_cli.task}")
    print(f"  Runner: {runner_class_name}")
    print(f"  Force dim: {force_dim}  ({', '.join(dim_labels)})")
    print(f"  Envs: {n}   Duration: {args_cli.duration}s   Steps: {max_steps}")
    print(f"  Force baskets: {args_cli.force_baskets} N  (band: [{band:.0%}, 100%])")
    print(f"  Wrench mode: {is_wrench}  fz_scale: {fz_scale}  torque_range: {torque_range}")
    print(f"  Sample envs: {SAMPLE_ENVS} (deterministic from seed 42)")
    print(f"{'=' * 70}\n")

    # ═══════════════════════════════════════════════════════════════════
    # Run each force basket
    # ═══════════════════════════════════════════════════════════════════

    all_basket_metrics = {}
    all_sample_data = {}

    for basket_idx, basket in enumerate(args_cli.force_baskets):
        f_min = band * basket
        f_max = basket
        print(f"\n{'─' * 60}")
        print(f"  Basket {basket:.0f}N  (force per axis: [{f_min:.0f}, {f_max:.0f}] N)")
        print(f"{'─' * 60}")

        # Deterministic force schedule: seed = 42 + basket_idx
        # Same seed + same basket → same GT forces for envs 0-4 across runs
        rng = np.random.default_rng(42 + basket_idx)
        force_sched, torque_sched = _generate_force_schedule(
            rng, max_steps, n, f_min, f_max, fz_scale,
            is_wrench, torque_range, dt,
        )
        print(f"  Force schedule generated ({force_sched.shape})")

        # Convert to torch (on GPU)
        force_sched_t = torch.from_numpy(force_sched).to(device)
        torque_sched_t = torch.from_numpy(torque_sched).to(device)

        # Pre-allocate tensors for direct PhysX force application
        force_buf = torch.zeros(n, asset.num_bodies, 3, device=device)
        torque_buf = torch.zeros(n, asset.num_bodies, 3, device=device)
        all_indices = torch.arange(n, dtype=torch.long, device=device)
        # [OLD — permanent_wrench_composer]:
        # force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        # torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)

        # Reset + warmup
        obs = reset_env(env, ctx, isaac_env, runner, runner_class_name,
                        is_stage2, device, n)
        print(f"  Warmup ({args_cli.warmup_s}s)...", flush=True)
        for _ in range(warmup_steps):
            obs, _ = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, compliance_k, args_cli.ema_alpha)

        # ── Collect data ────────────────────────────────────────────────
        all_gt = torch.zeros(max_steps, n, force_dim, device=device)
        all_est = torch.zeros(max_steps, n, force_dim, device=device)

        for step in range(max_steps):
            # Apply scheduled forces directly via PhysX (bypasses WrenchComposer buffers)
            force_buf[:, base_idx, :] = force_sched_t[step]
            torque_buf[:, base_idx, :] = torque_sched_t[step]
            asset.root_physx_view.apply_forces_and_torques_at_position(
                force_data=force_buf.view(-1, 3),
                torque_data=torque_buf.view(-1, 3),
                position_data=None,
                indices=all_indices,
                is_global=False,
            )
            # [OLD — permanent_wrench_composer]:
            # force_tensor[:, base_idx, :] = force_sched_t[step]
            # torque_tensor[:, base_idx, :] = torque_sched_t[step]
            # asset.permanent_wrench_composer.set_forces_and_torques(
            #     forces=force_tensor, torques=torque_tensor,
            # )

            obs, _ = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, compliance_k, args_cli.ema_alpha)

            gt = read_gt_force(asset, base_idx, force_dim, n)
            est = isaac_env._force_estimate_xy[:n]
            all_gt[step] = gt
            all_est[step] = est

            # ── Force arrows (only when not headless) ───────────────
            if gt_markers is not None:
                force_scale = 0.05
                base_pos = asset.data.root_pos_w[:n]
                base_quat = asset.data.root_quat_w[:n]
                fd3 = min(force_dim, 3)

                gt_3d = torch.zeros(n, 3, device=device)
                gt_3d[:, :fd3] = gt[:, :fd3]
                gt_world = quat_apply(base_quat, gt_3d)
                gt_pos = base_pos.clone(); gt_pos[:, 2] += 0.55
                gt_scales = torch.full((n, 3), 0.3, device=device)
                gt_scales[:, 0:1] = (gt_world.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                gt_markers.visualize(gt_pos, _force_to_quat(gt_world, device), gt_scales)

                est_3d = torch.zeros(n, 3, device=device)
                est_3d[:, :fd3] = est[:, :fd3]
                est_world = quat_apply(base_quat, est_3d)
                est_pos = base_pos.clone(); est_pos[:, 2] += 0.45
                est_scales = torch.full((n, 3), 0.3, device=device)
                est_scales[:, 0:1] = (est_world.norm(dim=-1, keepdim=True) * force_scale).clamp(min=0.05)
                est_markers.visualize(est_pos, _force_to_quat(est_world, device), est_scales)

            if args_cli.real_time:
                time.sleep(dt)

            if (step + 1) % 100 == 0:
                pct = (step + 1) / max_steps * 100
                print(f"\r  [{pct:5.1f}%] {(step + 1) * dt:.1f}/{args_cli.duration:.0f}s",
                      end="", flush=True)

        print(f"\n  Basket {basket:.0f}N done.")

        # Clear forces
        force_buf.zero_()
        torque_buf.zero_()
        asset.root_physx_view.apply_forces_and_torques_at_position(
            force_data=force_buf.view(-1, 3),
            torque_data=torque_buf.view(-1, 3),
            position_data=None,
            indices=all_indices,
            is_global=False,
        )
        # [OLD — permanent_wrench_composer]:
        # force_tensor.zero_()
        # torque_tensor.zero_()
        # asset.permanent_wrench_composer.set_forces_and_torques(
        #     forces=force_tensor, torques=torque_tensor,
        # )

        # Free schedule tensors
        del force_sched_t, torque_sched_t

        # ── Compute metrics ─────────────────────────────────────────────
        gt_np = all_gt.cpu().numpy()
        est_np = all_est.cpu().numpy()

        metrics = _compute_metrics(gt_np, est_np, force_dim, dim_labels)
        metrics["basket_N"] = basket
        metrics["force_range"] = [f_min, f_max]
        all_basket_metrics[f"{basket:.0f}N"] = metrics

        # ── Print basket summary ────────────────────────────────────────
        print(f"  MAE total: {metrics['mae']:.3f}   Force MAE: {metrics['force_mae']:.3f} N"
              + (f"   Torque MAE: {metrics['torque_mae']:.3f} Nm" if metrics['torque_mae'] is not None else "")
              + f"   Force mag MAE: {metrics['force_mag_mae']:.3f} N")
        print(f"  Median AE: {metrics['median_ae']:.3f}   Rel Err: {metrics['relative_err_pct']:.1f}%"
              f"   Ang Err: {metrics['angular_err_xy_deg_mean']:.1f}°")
        for d in range(force_dim):
            unit = "Nm" if d >= 3 else "N"
            print(f"    {dim_labels[d]:>8s}: {metrics['per_axis_mae'][dim_labels[d]]:.3f} {unit}")

        # ── Save sample env data ────────────────────────────────────────
        sample_data = {}
        for env_i in SAMPLE_ENVS:
            if env_i >= n:
                continue
            sample_data[str(env_i)] = {
                "gt": gt_np[:, env_i, :].tolist(),
                "est": est_np[:, env_i, :].tolist(),
                "gt_force_schedule": force_sched[:, env_i, :].tolist(),
                "gt_torque_schedule": torque_sched[:, env_i, :].tolist(),
            }
        all_sample_data[f"{basket:.0f}N"] = sample_data

        del all_gt, all_est, gt_np, est_np

    env.close()

    # ═══════════════════════════════════════════════════════════════════
    # Generate plots
    # ═══════════════════════════════════════════════════════════════════

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    figures = []
    time_s = np.arange(max_steps) * dt
    n_samples = min(len(SAMPLE_ENVS), n)
    force_colors = ["tab:blue", "tab:red", "tab:green"]
    torque_colors = ["tab:purple", "tab:brown", "tab:orange"]
    n_force_axes = min(force_dim, 3)
    torque_indices = list(range(3, force_dim))

    for basket_key in all_basket_metrics:
        sd = all_sample_data[basket_key]

        # ── Force time series: 5 envs × force axes ─────────────────────
        fig, axes_grid = plt.subplots(n_samples, n_force_axes,
                                      figsize=(6 * n_force_axes, 3 * n_samples))
        if n_samples == 1:
            axes_grid = np.atleast_2d(axes_grid)
        if n_force_axes == 1:
            axes_grid = axes_grid.reshape(-1, 1)
        fig.suptitle(f"Force Estimate vs GT — {basket_key}\n{args_cli.task}",
                     fontsize=14, fontweight="bold")
        for row, env_i in enumerate(SAMPLE_ENVS[:n_samples]):
            env_data = sd[str(env_i)]
            gt = np.array(env_data["gt"])
            est = np.array(env_data["est"])
            for col in range(n_force_axes):
                ax = axes_grid[row, col]
                ax.plot(time_s, gt[:, col], color=force_colors[col],
                        linewidth=1.0, alpha=0.9, label="GT")
                ax.plot(time_s, est[:, col], color=force_colors[col],
                        linewidth=0.8, alpha=0.5, linestyle="--", label="Est")
                if row == 0:
                    ax.set_title(dim_labels[col], fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"Env {env_i}\nForce (N)", fontsize=9)
                if row == n_samples - 1:
                    ax.set_xlabel("Time (s)", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        plt.tight_layout()
        figures.append(fig)

        # ── Torque time series: 5 envs × torque axes ───────────────────
        if torque_indices:
            n_torque_axes = len(torque_indices)
            torque_labels = [dim_labels[d] for d in torque_indices]
            fig, axes_grid = plt.subplots(n_samples, n_torque_axes,
                                          figsize=(6 * n_torque_axes, 3 * n_samples))
            if n_samples == 1:
                axes_grid = np.atleast_2d(axes_grid)
            if n_torque_axes == 1:
                axes_grid = axes_grid.reshape(-1, 1)
            fig.suptitle(f"Torque Estimate vs GT — {basket_key}\n{args_cli.task}",
                         fontsize=14, fontweight="bold")
            for row, env_i in enumerate(SAMPLE_ENVS[:n_samples]):
                env_data = sd[str(env_i)]
                gt = np.array(env_data["gt"])
                est = np.array(env_data["est"])
                for col_i, d in enumerate(torque_indices):
                    ax = axes_grid[row, col_i]
                    ax.plot(time_s, gt[:, d], color=torque_colors[col_i % 3],
                            linewidth=1.0, alpha=0.9, label="GT")
                    ax.plot(time_s, est[:, d], color=torque_colors[col_i % 3],
                            linewidth=0.8, alpha=0.5, linestyle="--", label="Est")
                    if row == 0:
                        ax.set_title(torque_labels[col_i], fontsize=11)
                    if col_i == 0:
                        ax.set_ylabel(f"Env {env_i}\nTorque (Nm)", fontsize=9)
                    if row == n_samples - 1:
                        ax.set_xlabel("Time (s)", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=7)
                    if row == 0 and col_i == 0:
                        ax.legend(fontsize=7)
            plt.tight_layout()
            figures.append(fig)

    # ── Combined metrics summary table ──────────────────────────────────
    baskets = list(all_basket_metrics.keys())
    headers = ["Metric"] + baskets
    row_keys = [
        ("MAE total (per-dim)", "mae"),
        ("MAE force (N)", "force_mae"),
        ("MAE torque (Nm)", "torque_mae"),
        ("Median AE per-dim", "median_ae"),
        ("Rel Err (%)", "relative_err_pct"),
        ("Ang Err XY (°)", "angular_err_xy_deg_mean"),
        ("Force mag MAE (N)", "force_mag_mae"),
    ]
    for d in range(force_dim):
        unit = "Nm" if d >= 3 else "N"
        row_keys.append((f"MAE {dim_labels[d]} ({unit})", None))

    table_rows = []
    for label, key in row_keys:
        row = [label]
        for bk in baskets:
            m = all_basket_metrics[bk]
            if key is not None:
                v = m[key]
                row.append(f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v)))
            else:
                d_label = label.split("MAE ")[1].split(" (")[0]
                row.append(f"{m['per_axis_mae'][d_label]:.3f}")
        table_rows.append(row)

    fig, ax = plt.subplots(figsize=(4 + 3 * len(baskets), 2 + len(table_rows) * 0.5))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.suptitle(f"Rollout Estimator Metrics — {args_cli.task}\n{n} envs × {args_cli.duration}s",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    figures.append(fig)

    # ── Save everything ─────────────────────────────────────────────────
    task_short = args_cli.task.replace("Go2-Ablation-", "").replace("-v0", "")
    baskets_tag = "_".join(f"{b:.0f}N" for b in args_cli.force_baskets)
    output_dir = create_eval_output_dir(agent_cfg.experiment_name, "rollout_estimator",
                                        suffix=f"{task_short}_{baskets_tag}")
    save_config(output_dir, args_cli, agent_cfg, dt)

    # Metrics
    combined = {
        "task": args_cli.task,
        "force_dim": force_dim,
        "n_envs": n,
        "duration_s": args_cli.duration,
        "dt": dt,
        "baskets": all_basket_metrics,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(combined, f, indent=2, cls=_NumpyEncoder)

    # Sample env data per basket
    for basket_key, sd in all_sample_data.items():
        path = os.path.join(output_dir, f"sample_envs_{basket_key}.json")
        with open(path, "w") as f:
            json.dump(sd, f, indent=1, cls=_NumpyEncoder)

    # PDF
    pdf_path = os.path.join(output_dir, "analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n[rollout_eval] Output: {output_dir}")
    print(f"[rollout_eval] Metrics: {os.path.join(output_dir, 'metrics.json')}")
    print(f"[rollout_eval] PDF: {pdf_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
