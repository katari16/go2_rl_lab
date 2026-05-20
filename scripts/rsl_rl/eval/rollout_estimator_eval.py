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
        --task Go2-Est-Deploy-v0 \
        --checkpoint logs/rsl_rl/.../model_9500.pt \
        --num_envs 4096 --duration 20 --training_regime --headless
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
parser.add_argument("--force_baskets", type=float, nargs="+", default=None,
                    help="Force magnitude baskets in N. Each basket B runs forces in [0.8*B, B] per axis.")
parser.add_argument("--basket_band", type=float, default=0.8,
                    help="Lower fraction of basket range (default: 0.8 → [0.8*B, B]).")
parser.add_argument("--training_regime", action="store_true", default=False,
                    help="Match training force distribution exactly: per-axis uniform(0, max_force), "
                         "fz_scale, 3-5s resample, force_free_fraction. Reads max_force from runner cfg.")
parser.add_argument("--no_active_mask", action="store_true", default=False,
                    help="Disable |F_gt|>1N filter on metrics — average over ALL samples "
                         "including near-zero GT, matching training tensorboard computation.")
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
    """Pre-generate deterministic force+torque schedule (basket mode).

    All 3 axes get the same magnitude range [f_min, f_max] simultaneously.
    Resample interval: 1-3s.

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


def _generate_training_regime_schedule(rng, max_steps, n, max_force, fz_scale,
                                       is_wrench, torque_range,
                                       force_free_fraction, dt):
    """Pre-generate force schedule matching training distribution exactly.

    Per-axis independent: each of fx, fy sampled from U(0, max_force) with
    random sign. fz = U(0, max_force * fz_scale) with random sign.
    Resample interval: 3-5s (same as training event).
    force_free_fraction: probability of zero wrench per interval.

    Returns:
        forces: [max_steps, n, 3]
        torques: [max_steps, n, 3]
    """
    forces = np.zeros((max_steps, n, 3), dtype=np.float32)
    torques = np.zeros((max_steps, n, 3), dtype=np.float32)

    for env_i in range(n):
        step = 0
        while step < max_steps:
            dur = max(1, int(rng.uniform(3.0, 5.0) / dt))
            end = min(step + dur, max_steps)

            if rng.random() < force_free_fraction:
                # zero wrench this interval
                step = end
                continue

            fx = rng.uniform(0, max_force) * rng.choice([-1.0, 1.0])
            fy = rng.uniform(0, max_force) * rng.choice([-1.0, 1.0])
            fz = rng.uniform(0, max_force * fz_scale) * rng.choice([-1.0, 1.0])
            forces[step:end, env_i, :] = [fx, fy, fz]

            if is_wrench:
                t_lo, t_hi = torque_range
                tx = rng.uniform(t_lo, t_hi) * rng.choice([-1.0, 1.0])
                ty = rng.uniform(t_lo, t_hi) * rng.choice([-1.0, 1.0])
                tz = rng.uniform(t_lo, t_hi) * rng.choice([-1.0, 1.0])
                torques[step:end, env_i, :] = [tx, ty, tz]

            step = end

    return forces, torques


def _compute_metrics(all_gt_np, all_est_np, force_dim, dim_labels, no_active_mask=False):
    """Compute aggregate estimation metrics.

    Args:
        no_active_mask: If True, average over ALL samples (matching training
            tensorboard). If False, only average where |F_gt| > 1N.

    Returns:
        dict of metrics
    """
    err = all_est_np - all_gt_np
    abs_err = np.abs(err)                          # [steps, envs, force_dim]
    per_sample_mae = abs_err.mean(axis=2)          # [steps, envs] — matches training mae_total
    norm_err = np.linalg.norm(err, axis=2)         # L2 norm — used only for relative error
    gt_mag = np.linalg.norm(all_gt_np, axis=2)

    if no_active_mask:
        active_mask = np.ones_like(gt_mag, dtype=bool)
    else:
        active_mask = gt_mag > 1.0
    has_active = active_mask.any()

    mae = float(np.mean(per_sample_mae[active_mask])) if has_active else 0.0
    median_ae = float(np.median(per_sample_mae[active_mask])) if has_active else 0.0

    # Per-env MAE → std dev across envs
    n_envs = all_gt_np.shape[1]
    per_env_mae = np.zeros(n_envs)
    for e in range(n_envs):
        m = active_mask[:, e]
        per_env_mae[e] = float(np.mean(per_sample_mae[:, e][m])) if m.any() else 0.0
    mae_std = float(np.std(per_env_mae))

    rel_err = 0.0
    rel_mask = gt_mag > 1.0
    if rel_mask.any():
        rel_per_step = norm_err[rel_mask] / gt_mag[rel_mask] * 100
        rel_err = float(np.mean(rel_per_step))

    # Relative error without the 1N mask (exclude only literal zeros to avoid div/0)
    nonzero_mask = gt_mag > 0.01
    rel_err_no_mask = 0.0
    if nonzero_mask.any():
        rel_err_no_mask = float(np.mean(norm_err[nonzero_mask] / gt_mag[nonzero_mask] * 100))

    per_axis_mae = {}
    per_axis_mae_std = {}
    for d in range(force_dim):
        axis_ae = np.abs(err[:, :, d])
        per_axis_mae[dim_labels[d]] = float(np.mean(axis_ae[active_mask])) if has_active else 0.0
        per_env_axis = np.zeros(n_envs)
        for e in range(n_envs):
            m = active_mask[:, e]
            per_env_axis[e] = float(np.mean(axis_ae[:, e][m])) if m.any() else 0.0
        per_axis_mae_std[dim_labels[d]] = float(np.std(per_env_axis))

    # Angular error — always uses xy > 1N mask (angle is meaningless near zero)
    gt_angle = np.arctan2(all_gt_np[:, :, 1], all_gt_np[:, :, 0])
    est_angle = np.arctan2(all_est_np[:, :, 1], all_est_np[:, :, 0])
    angle_diff = np.arctan2(np.sin(est_angle - gt_angle), np.cos(est_angle - gt_angle))
    angle_err_deg = np.abs(angle_diff) * 180.0 / np.pi
    xy_mask = np.linalg.norm(all_gt_np[:, :, :2], axis=2) > 1.0
    has_xy = xy_mask.any()
    angular_mean = float(np.mean(angle_err_deg[xy_mask])) if has_xy else 0.0
    angular_median = float(np.median(angle_err_deg[xy_mask])) if has_xy else 0.0
    per_env_ang_median = np.zeros(n_envs)
    per_env_ang_mean   = np.zeros(n_envs)
    for e in range(n_envs):
        m = xy_mask[:, e]
        per_env_ang_median[e] = float(np.median(angle_err_deg[:, e][m])) if m.any() else 0.0
        per_env_ang_mean[e]   = float(np.mean(angle_err_deg[:, e][m]))   if m.any() else 0.0
    angular_median_std = float(np.std(per_env_ang_median))
    angular_mean_std   = float(np.std(per_env_ang_mean))

    force_indices = [d for d in range(force_dim) if "τ" not in dim_labels[d]]
    torque_indices = [d for d in range(force_dim) if "τ" in dim_labels[d]]

    gt_force_mag = np.linalg.norm(all_gt_np[:, :, force_indices], axis=2)
    est_force_mag = np.linalg.norm(all_est_np[:, :, force_indices], axis=2)
    force_mag_err = np.abs(est_force_mag - gt_force_mag)
    force_mag_mask = gt_force_mag > 1.0
    force_mag_mae = float(np.mean(force_mag_err[force_mag_mask])) if force_mag_mask.any() else 0.0

    torque_mae = float(np.mean([per_axis_mae[dim_labels[d]] for d in torque_indices])) if torque_indices else None
    torque_mae_std = float(np.mean([per_axis_mae_std[dim_labels[d]] for d in torque_indices])) if torque_indices else None

    return {
        "mae": mae,
        "mae_std": mae_std,
        "median_ae": median_ae,
        "relative_err_pct": rel_err,
        "relative_err_no_mask_pct": rel_err_no_mask,
        "angular_err_xy_deg_mean": angular_mean,
        "angular_err_xy_deg_mean_std": angular_mean_std,
        "angular_err_xy_deg_median": angular_median,
        "angular_err_xy_deg_median_std": angular_median_std,
        "per_axis_mae": per_axis_mae,
        "per_axis_mae_std": per_axis_mae_std,
        "force_mae": float(np.mean([per_axis_mae[dim_labels[d]] for d in force_indices])),
        "force_mae_std": float(np.mean([per_axis_mae_std[dim_labels[d]] for d in force_indices])),
        "torque_mae": torque_mae,
        "torque_mae_std": torque_mae_std,
        "force_mag_mae": force_mag_mae,
        "active_mask_used": not no_active_mask,
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

    agent_dict = agent_cfg.to_dict()
    force_event_name = agent_dict.get("force_event_term_name", "persistent_xy_force")
    is_wrench = "wrench" in force_event_name

    # Read fz_scale, torque_range, force_free_fraction from env config before disabling events
    try:
        force_event = getattr(env_cfg.events, force_event_name)
        fz_scale = force_event.params.get("fz_scale", 0.6)
        torque_range = force_event.params.get("torque_range", (0.0, 5.0))
        force_free_fraction = force_event.params.get("force_free_fraction", 0.0)
    except AttributeError:
        fz_scale = 0.6
        torque_range = (0.0, 5.0)
        force_free_fraction = 0.0

    training_regime = args_cli.training_regime
    max_force_cfg = agent_dict.get("max_force", 20.0)
    max_torque_cfg = agent_dict.get("max_torque", 5.0)

    if training_regime:
        if args_cli.force_baskets:
            print("[rollout_eval] WARNING: --force_baskets ignored in --training_regime mode")
        baskets_list = [("training", 0.0, max_force_cfg)]
    else:
        if not args_cli.force_baskets:
            print("[rollout_eval] ERROR: --force_baskets required (or use --training_regime)")
            return
        band = args_cli.basket_band
        baskets_list = [(f"{b:.0f}N", band * b, b) for b in args_cli.force_baskets]

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
    force_layout = ctx.get("force_layout", "auto")
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
    if force_layout == "xy_yaw" and force_dim == 3:
        dim_labels = ["Fx", "Fy", "τ_yaw"]
    else:
        dim_labels = DIM_LABELS.get(force_dim, [f"d{i}" for i in range(force_dim)])

    print(f"\n{'=' * 70}")
    print(f"  Rollout Estimator Evaluation")
    print(f"  Task: {args_cli.task}")
    print(f"  Runner: {runner_class_name}")
    print(f"  Force dim: {force_dim}  ({', '.join(dim_labels)})")
    print(f"  Envs: {n}   Duration: {args_cli.duration}s   Steps: {max_steps}")
    if training_regime:
        print(f"  Mode: TRAINING REGIME  max_force: {max_force_cfg}N  fz_scale: {fz_scale}")
        print(f"  Torque range: {torque_range}  force_free_frac: {force_free_fraction}")
        print(f"  Resample interval: 3-5s (per-axis independent, matching training)")
    else:
        band = args_cli.basket_band
        print(f"  Mode: BASKETS  {args_cli.force_baskets} N  (band: [{band:.0%}, 100%])")
    print(f"  Wrench mode: {is_wrench}  fz_scale: {fz_scale}  torque_range: {torque_range}")
    print(f"  Sample envs: {SAMPLE_ENVS} (deterministic from seed 42)")
    print(f"{'=' * 70}\n")

    # ═══════════════════════════════════════════════════════════════════
    # Run each force basket (or single training-regime run)
    # ═══════════════════════════════════════════════════════════════════

    all_basket_metrics = {}
    all_sample_data = {}

    for basket_idx, (basket_key, f_min, f_max) in enumerate(baskets_list):
        print(f"\n{'─' * 60}")
        if training_regime:
            print(f"  Training regime: per-axis U(0, {f_max:.0f}N), fz_scale={fz_scale}, "
                  f"resample 3-5s, zero_prob={force_free_fraction:.0%}")
        else:
            print(f"  Basket {basket_key}  (force per axis: [{f_min:.0f}, {f_max:.0f}] N)")
        print(f"{'─' * 60}")

        rng = np.random.default_rng(42 + basket_idx)
        if training_regime:
            force_sched, torque_sched = _generate_training_regime_schedule(
                rng, max_steps, n, f_max, fz_scale,
                is_wrench, (0.0, max_torque_cfg),
                force_free_fraction, dt,
            )
        else:
            force_sched, torque_sched = _generate_force_schedule(
                rng, max_steps, n, f_min, f_max, fz_scale,
                is_wrench, torque_range, dt,
            )
        print(f"  Force schedule generated ({force_sched.shape})")

        # Convert to torch (on GPU)
        force_sched_t = torch.from_numpy(force_sched).to(device)
        torque_sched_t = torch.from_numpy(torque_sched).to(device)

        force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)

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
            force_tensor[:, base_idx, :] = force_sched_t[step]
            torque_tensor[:, base_idx, :] = torque_sched_t[step]
            asset.permanent_wrench_composer.set_forces_and_torques(
                forces=force_tensor, torques=torque_tensor,
            )

            obs, _ = step_policy(obs, ctx, env, runner, isaac_env, n,
                                 runner_class_name, compliance_k, args_cli.ema_alpha)

            gt = read_gt_force(asset, base_idx, force_dim, n, force_layout=force_layout)
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

        print(f"\n  {basket_key} done.")

        # Clear forces
        force_tensor.zero_()
        torque_tensor.zero_()
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force_tensor, torques=torque_tensor,
        )

        # Free schedule tensors
        del force_sched_t, torque_sched_t

        # ── Compute metrics ─────────────────────────────────────────────
        gt_np = all_gt.cpu().numpy()
        est_np = all_est.cpu().numpy()

        metrics = _compute_metrics(gt_np, est_np, force_dim, dim_labels, args_cli.no_active_mask)
        metrics["basket_key"] = basket_key
        metrics["force_range"] = [f_min, f_max]
        all_basket_metrics[basket_key] = metrics

        # ── Print basket summary ────────────────────────────────────────
        print(f"  MAE total: {metrics['mae']:.3f}   Force MAE: {metrics['force_mae']:.3f} N"
              + (f"   Torque MAE: {metrics['torque_mae']:.3f} Nm" if metrics['torque_mae'] is not None else "")
              + f"   Force mag MAE: {metrics['force_mag_mae']:.3f} N")
        print(f"  Median AE: {metrics['median_ae']:.3f}   Rel Err: {metrics['relative_err_pct']:.1f}%"
              f"   Ang Err mean: {metrics['angular_err_xy_deg_mean']:.1f}°"
              f"   Ang Err median: {metrics['angular_err_xy_deg_median']:.1f}°")
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
        all_sample_data[basket_key] = sample_data

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

    # ── Structured metrics summary page ──────────────────────────────────
    baskets = list(all_basket_metrics.keys())

    if training_regime:
        force_range_str = f"Force range: 0–{max_force_cfg:.0f} N"
    else:
        force_range_str = "Force ranges: " + ", ".join(f"{b:.0f} N" for b in args_cli.force_baskets)

    force_axes = [(d, dim_labels[d]) for d in range(force_dim) if "τ" not in dim_labels[d]]
    torque_axes = [(d, dim_labels[d]) for d in range(force_dim) if "τ" in dim_labels[d]]
    has_torque = len(torque_axes) > 0

    fig, ax = plt.subplots(figsize=(5 + 2.5 * len(baskets), 11))
    ax.axis("off")
    fig.suptitle(
        f"Rollout Estimator Metrics — {args_cli.task}\n"
        f"{n} envs × {args_cli.duration}s   |   {force_range_str}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    col_x = [0.45 + i * (0.5 / max(len(baskets), 1)) for i in range(len(baskets))]
    label_x = 0.03
    lh = 0.048   # line height
    sh = 0.028   # sub-line height (formula / label indent)
    gap = 0.022  # section gap

    def txt(x, y, s, **kw):
        ax.text(x, y, s, transform=ax.transAxes, va="top", clip_on=False, **kw)

    y = 0.91

    # basket column headers
    for i, bk in enumerate(baskets):
        txt(col_x[i], y, bk, ha="center", fontsize=11, fontweight="bold", color="steelblue")
    y -= lh * 1.3

    # axis-specific max ranges for normalization
    def _axis_range(lbl):
        if "τ" in lbl:
            return float(max_torque_cfg), "Nm"
        if "Fz" in lbl:
            return float(max_force_cfg) * float(fz_scale), "N"
        return float(max_force_cfg), "N"

    # ── 1. Per-axis MAE ──────────────────────────────────────────────
    txt(label_x, y, "Per-axis Mean Absolute Error", ha="left", fontsize=11, fontweight="bold")
    y -= lh * 0.85

    for d, lbl in force_axes:
        ax_max, unit = _axis_range(lbl)
        txt(label_x + 0.02, y, f"{lbl}  (max: {ax_max:.0f} {unit})",
            ha="left", fontsize=10, color="dimgray")
        for i, bk in enumerate(baskets):
            v = all_basket_metrics[bk]["per_axis_mae"][lbl]
            norm_pct = v / ax_max * 100 if ax_max > 0 else 0.0
            txt(col_x[i], y, f"{v:.3f} {unit}  ({norm_pct:.1f}%)", ha="center", fontsize=10)
        y -= lh * 0.85

    if has_torque:
        for d, lbl in torque_axes:
            ax_max, unit = _axis_range(lbl)
            txt(label_x + 0.02, y, f"{lbl}  (max: {ax_max:.1f} {unit})",
                ha="left", fontsize=10, color="dimgray")
            for i, bk in enumerate(baskets):
                v = all_basket_metrics[bk]["per_axis_mae"][lbl]
                norm_pct = v / ax_max * 100 if ax_max > 0 else 0.0
                txt(col_x[i], y, f"{v:.3f} {unit}  ({norm_pct:.1f}%)", ha="center", fontsize=10)
            y -= lh * 0.85

    y -= gap

    # ── 2. Directional error ─────────────────────────────────────────
    txt(label_x, y, "Directional Error in XY Plane", ha="left", fontsize=11, fontweight="bold")
    y -= sh
    txt(label_x + 0.02, y, "Median Angular Error", ha="left", fontsize=9,
        style="italic", color="dimgray")
    y -= sh * 0.9
    for i, bk in enumerate(baskets):
        v = all_basket_metrics[bk]["angular_err_xy_deg_median"]
        txt(col_x[i], y, f"{v:.2f}°", ha="center", fontsize=10)
    y -= lh * 0.8
    y -= gap

    # ── 3. Relative error ────────────────────────────────────────────
    txt(label_x, y, "Relative Error", ha="left", fontsize=11, fontweight="bold")
    y -= sh

    txt(label_x + 0.02, y, "mean( ‖est − gt‖₂ / ‖gt‖₂ ) × 100   [‖gt‖ > 1 N]",
        ha="left", fontsize=8, style="italic", color="dimgray")
    y -= sh * 0.85
    for i, bk in enumerate(baskets):
        v = all_basket_metrics[bk]["relative_err_pct"]
        txt(col_x[i], y, f"{v:.1f} %", ha="center", fontsize=10)
    y -= lh * 0.75

    txt(label_x + 0.02, y, "mean( ‖est − gt‖₂ / ‖gt‖₂ ) × 100   [‖gt‖ > 0.01 N, no mask]",
        ha="left", fontsize=8, style="italic", color="dimgray")
    y -= sh * 0.85
    for i, bk in enumerate(baskets):
        v = all_basket_metrics[bk]["relative_err_no_mask_pct"]
        txt(col_x[i], y, f"{v:.1f} %", ha="center", fontsize=10)
    y -= lh * 0.75

    txt(label_x + 0.02, y, "force_mae / max_force × 100   [range-normalized]",
        ha="left", fontsize=8, style="italic", color="dimgray")
    y -= sh * 0.85
    for i, bk in enumerate(baskets):
        v = all_basket_metrics[bk]["force_mae"] / max_force_cfg * 100 if max_force_cfg > 0 else 0.0
        txt(col_x[i], y, f"{v:.1f} %", ha="center", fontsize=10)
    y -= lh * 0.8
    y -= gap

    # ── 4. MAE Force ─────────────────────────────────────────────────
    txt(label_x, y, "MAE Force Total", ha="left", fontsize=11, fontweight="bold")
    y -= sh
    txt(label_x + 0.02, y, "mean over (Fx, Fy, Fz) of  mean(|est_d − gt_d|)",
        ha="left", fontsize=8, style="italic", color="dimgray")
    y -= sh * 0.9
    for i, bk in enumerate(baskets):
        v = all_basket_metrics[bk]["force_mae"]
        txt(col_x[i], y, f"{v:.3f} N", ha="center", fontsize=10)
    y -= lh * 0.8

    if has_torque:
        y -= gap
        txt(label_x, y, "MAE Torque Total", ha="left", fontsize=11, fontweight="bold")
        y -= sh
        txt(label_x + 0.02, y, "mean over torque axes of  mean(|est_d − gt_d|)",
            ha="left", fontsize=8, style="italic", color="dimgray")
        y -= sh * 0.9
        for i, bk in enumerate(baskets):
            v = all_basket_metrics[bk]["torque_mae"]
            txt(col_x[i], y, f"{v:.3f} Nm" if v is not None else "—", ha="center", fontsize=10)
        y -= lh * 0.8

    y -= gap

    # ── 5. MAE Total ─────────────────────────────────────────────────
    txt(label_x, y, "MAE Total (all dims)", ha="left", fontsize=12, fontweight="bold")
    y -= sh
    txt(label_x + 0.02, y, "mean over all dims of  mean(|est_d − gt_d|)",
        ha="left", fontsize=8, style="italic", color="dimgray")
    y -= sh * 0.9
    for i, bk in enumerate(baskets):
        v = all_basket_metrics[bk]["mae"]
        txt(col_x[i], y, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figures.append(fig)

    # ── Save everything ─────────────────────────────────────────────────
    task_short = args_cli.task.replace("Go2-Est-", "").replace("Go2-Ablation-", "").replace("-v0", "")
    if training_regime:
        suffix_tag = f"{task_short}_training_regime"
    else:
        suffix_tag = f"{task_short}_{'_'.join(f'{b:.0f}N' for b in args_cli.force_baskets)}"
    output_dir = create_eval_output_dir(agent_cfg.experiment_name, "rollout_estimator",
                                        suffix=suffix_tag)
    save_config(output_dir, args_cli, agent_cfg, dt)

    # Metrics
    combined = {
        "task": args_cli.task,
        "force_dim": force_dim,
        "n_envs": n,
        "duration_s": args_cli.duration,
        "dt": dt,
        "mode": "training_regime" if training_regime else "baskets",
        "baskets": all_basket_metrics,
    }
    if training_regime:
        combined["training_regime"] = {
            "max_force": max_force_cfg,
            "max_torque": max_torque_cfg,
            "fz_scale": fz_scale,
            "force_free_fraction": force_free_fraction,
            "resample_interval_s": [3.0, 5.0],
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
