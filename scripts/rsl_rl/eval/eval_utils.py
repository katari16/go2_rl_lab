"""Shared utilities for evaluation scripts.

Must be imported AFTER AppLauncher is created (imports Isaac Lab modules).

Provides:
- Runner creation and dispatch
- Policy stepping with estimator handling
- Force application / clearing
- Env config helpers
- Metric computation (decay time, effective compliance, settling time, etc.)
- Plotting helpers (polar plots, heatmaps, summary tables)
- Data saving (raw JSON + config + PDF)
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path


# ── Runner management ────────────────────────────────────────────────────────


def create_runner(env, agent_cfg, args_cli):
    """Dispatch on agent_cfg.class_name and return the loaded runner.

    Returns:
        (runner, runner_class_name, is_stage2)
    """
    runner_class_name = agent_cfg.class_name
    train_cfg = agent_cfg.to_dict()

    if runner_class_name == "CompliantOnPolicyRunner":
        from go2_rl_lab.estimator.compliant_on_policy_runner import CompliantOnPolicyRunner

        if getattr(args_cli, "estimator_checkpoint", None):
            train_cfg["estimator_checkpoint"] = args_cli.estimator_checkpoint
        runner = CompliantOnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
        is_stage2 = False

    else:
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        is_stage2 = False

    return runner, runner_class_name, is_stage2


def setup_runner_for_eval(runner, env, runner_class_name, is_stage2, device, n):
    """Post-load setup: extract policy, dims, initialize buffers.

    Returns:
        ctx dict with keys: policy, policy_nn, wrapper, compliant_raw_obs_dim,
                            force_dim, force_ema, has_estimator
    """
    ctx = {
        "wrapper": None,
        "compliant_raw_obs_dim": None,
        "policy_raw_dim": None,
        "has_privileged": False,
        "force_dim": 3,
        "force_layout": "auto",
        "force_ema": None,  # initialized after force_dim is known
        "has_estimator": False,
    }

    if is_stage2:
        ctx["wrapper"] = runner._compliance_wrapper
        ctx["policy"] = runner.get_inference_policy(device=device)
        ctx["policy_nn"] = runner.alg.policy
    else:
        if hasattr(runner, "_wrapped_env"):
            env = runner._wrapped_env
        ctx["policy"] = runner.get_inference_policy(device=device)
        try:
            ctx["policy_nn"] = runner.alg.policy
        except AttributeError:
            ctx["policy_nn"] = runner.alg.actor_critic

    _estimator_runners = ("CompliantOnPolicyRunner",)
    if runner_class_name in _estimator_runners:
        ctx["has_estimator"] = True
        ctx["force_dim"] = getattr(runner, "_force_dim", 3)
        ctx["force_layout"] = getattr(runner.estimator, "force_layout", "auto")
        ctx["compliant_raw_obs_dim"] = runner._num_one_step_obs
        ctx["policy_raw_dim"] = getattr(runner, "_policy_raw_dim", runner._num_one_step_obs)
        ctx["has_privileged"] = getattr(runner, "_priv_dim", 0) > 0
        isaac_env = env.unwrapped if hasattr(env, "unwrapped") else env
        isaac_env._force_estimate_xy = torch.zeros(n, ctx["force_dim"], device=device)

    ctx["force_ema"] = torch.zeros(n, ctx["force_dim"], device=device)

    return ctx, env


def step_policy(obs, ctx, env, runner, isaac_env, n, runner_class_name,
                compliance_k=0.0, ema_alpha=0.1):
    """One policy step with estimator + compliance injection.

    Returns:
        (obs_next, dones)
    """
    device = isaac_env.device
    with torch.inference_mode():
        if ctx["has_estimator"]:
            raw_obs = obs["policy"][:, :ctx["policy_raw_dim"]]
            if ctx["has_privileged"]:
                priv_obs = runner._get_privileged_obs()
                raw_obs = torch.cat([raw_obs, priv_obs], dim=-1)
            runner._history_buffer.insert(raw_obs)
            fhat, _ = runner.estimator.get_latent(runner._history_buffer.get_flattened())
            isaac_env._force_estimate_xy = fhat
            ctx["force_ema"] = ema_alpha * fhat + (1.0 - ema_alpha) * ctx["force_ema"]
            if compliance_k > 0:
                obs = {k: v.clone() for k, v in obs.items()}
                obs["policy"][:, 6] = obs["policy"][:, 6] + compliance_k * ctx["force_ema"][:, 0]
                obs["policy"][:, 7] = obs["policy"][:, 7] + compliance_k * ctx["force_ema"][:, 1]
                # Yaw torque compliance: τ_yaw → ωz command
                force_dim = ctx["force_dim"]
                if force_dim >= 4:
                    yaw_idx = 5 if force_dim >= 6 else 3
                    obs["policy"][:, 8] = obs["policy"][:, 8] + compliance_k * ctx["force_ema"][:, yaw_idx]

        if ctx["wrapper"] is not None:
            actions = ctx["policy"](obs)
            obs_next, _, dones, _ = ctx["wrapper"].step(actions)
        else:
            actions = ctx["policy"](obs)
            obs_next, _, dones, _ = env.step(actions)
        ctx["policy_nn"].reset(dones)

    return obs_next, dones


# ── Force application ────────────────────────────────────────────────────────


def apply_force_xy(asset, base_idx, fx, fy, device, n):
    """Apply horizontal force to base via permanent_wrench_composer.

    Returns:
        (force_tensor, torque_tensor) for reference.
    """
    force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    force_tensor[:, base_idx, 0] = fx
    force_tensor[:, base_idx, 1] = fy
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=force_tensor, torques=torque_tensor,
    )
    return force_tensor, torque_tensor


def clear_force(asset, base_idx, device, n):
    """Zero out external forces."""
    force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=force_tensor, torques=torque_tensor,
    )


def read_gt_force(asset, base_idx, force_dim, n, force_layout="auto"):
    """Read GT from permanent_wrench_composer.

    Returns:
        Tensor [n, force_dim].
    """
    if force_layout == "xy_yaw" and force_dim == 3:
        gt_f = asset.permanent_wrench_composer.composed_force_as_torch[:n, base_idx, :2]
        gt_t = asset.permanent_wrench_composer.composed_torque_as_torch[:n, base_idx, 2:3]
        gt = torch.cat([gt_f, gt_t], dim=-1)
    elif force_dim <= 3:
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


# ── Env config helpers ───────────────────────────────────────────────────────


def set_flat_terrain(env_cfg):
    """Override terrain to flat plane for evaluation."""
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    # Disable terrain curriculum (it references terrain_generator.size)
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None


def disable_force_events(env_cfg, agent_cfg):
    """Disable automatic force events so forces are applied manually."""
    force_event_name = agent_cfg.to_dict().get("force_event_term_name", "persistent_xyz_force")
    try:
        force_event = getattr(env_cfg.events, force_event_name)
        force_event.params["force_range"] = (0.0, 0.0)
        if "torque_range" in force_event.params:
            force_event.params["torque_range"] = (0.0, 0.0)
    except AttributeError:
        pass
    try:
        env_cfg.events.push_robot = None
    except AttributeError:
        pass


def set_standing_commands(env_cfg):
    """Set zero velocity commands for standing evaluation."""
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.debug_vis = False


def set_walking_commands(env_cfg, walk_speed):
    """Set forward walking velocity commands."""
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (walk_speed, walk_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False


def set_long_episode(env_cfg, duration_s: float = 600.0):
    """Set a very long episode to prevent auto-resets during evaluation."""
    env_cfg.episode_length_s = duration_s


def reset_env(env, ctx, isaac_env, runner, runner_class_name, is_stage2, device, n):
    """Force a full env reset between trials.

    Uses Isaac Lab's built-in _reset_idx (same path as timeout/termination resets)
    to avoid inference-tensor in-place update errors.
    """
    env_ids = torch.arange(n, device=device)

    with torch.inference_mode():
        # Use the env's own reset mechanism (handles default poses, randomization, etc.)
        isaac_env._reset_idx(env_ids)
        isaac_env.scene.write_data_to_sim()
        isaac_env.sim.step(render=isaac_env.sim.has_gui())
        isaac_env.scene.update(dt=isaac_env.physics_dt)

        # Clear any residual forces
        asset = isaac_env.scene["robot"]
        force_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        torque_tensor = torch.zeros(n, asset.num_bodies, 3, device=device)
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=force_tensor, torques=torque_tensor,
        )

        # Reset estimator history buffer if applicable
        _estimator_runners = ("CompliantOnPolicyRunner",)
        if runner_class_name in _estimator_runners and hasattr(runner, "_history_buffer"):
            runner._history_buffer.reset(env_ids)
            ctx["force_ema"] = torch.zeros(n, ctx["force_dim"], device=device)
            isaac_env._force_estimate_xy = torch.zeros(n, ctx["force_dim"], device=device)

        # Reset policy hidden states
        dones = torch.ones(n, device=device)
        ctx["policy_nn"].reset(dones)

    # Get fresh observations
    if is_stage2:
        obs = ctx["wrapper"].get_observations()
    else:
        obs = env.get_observations()

    return obs


def get_asset_and_base(isaac_env):
    """Get robot asset and base body index.

    Returns:
        (asset, base_idx)
    """
    asset = isaac_env.scene["robot"]
    base_body_ids, _ = asset.find_bodies("base")
    base_idx = base_body_ids[0] if isinstance(base_body_ids, (list, tuple)) else int(base_body_ids)
    return asset, base_idx


def resolve_checkpoint(agent_cfg, args_cli):
    """Resolve checkpoint path from CLI or agent config.

    Returns:
        resume_path (str)
    """
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if getattr(args_cli, "checkpoint", None):
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


# ── Metric computation ───────────────────────────────────────────────────────


def compute_decay_time(vel_magnitudes: np.ndarray, dt: float, threshold_frac: float = 0.1) -> float:
    """Time from peak velocity to velocity dropping below threshold_frac * peak.

    Args:
        vel_magnitudes: 1D array of velocity magnitudes (e.g. during recovery).
        dt: Timestep in seconds.
        threshold_frac: Fraction of peak below which we consider decayed.

    Returns:
        Decay time in seconds. Returns len(vel_magnitudes)*dt if never decays.
    """
    if len(vel_magnitudes) == 0:
        return 0.0
    peak = np.max(vel_magnitudes)
    if peak < 1e-6:
        return 0.0
    threshold = threshold_frac * peak
    peak_idx = np.argmax(vel_magnitudes)
    after_peak = vel_magnitudes[peak_idx:]
    below = np.where(after_peak < threshold)[0]
    if len(below) > 0:
        return below[0] * dt
    return len(after_peak) * dt


def compute_effective_compliance(vel_xy: np.ndarray, cmd_vel_xy: np.ndarray,
                                  force_xy: np.ndarray, dt: float) -> float:
    """SAC-Loco Eq.13: C = (1/|T_F|) * sum((v - v')^T F / (F^T F)).

    Args:
        vel_xy: [T, 2] body velocities.
        cmd_vel_xy: [T, 2] commanded velocities.
        force_xy: [T, 2] applied forces.
        dt: Timestep (unused, included for API consistency).

    Returns:
        Effective compliance in s/kg.
    """
    force_mag_sq = np.sum(force_xy ** 2, axis=1)  # [T]
    active = force_mag_sq > 1e-6
    if not np.any(active):
        return 0.0
    delta_v = vel_xy[active] - cmd_vel_xy[active]
    f = force_xy[active]
    f_sq = force_mag_sq[active]
    # dot product of delta_v with force, normalized by force magnitude squared
    dot = np.sum(delta_v * f, axis=1) / f_sq  # [T_F]
    return float(np.mean(dot))


def compute_settling_time(positions_xy: np.ndarray, start_pos: np.ndarray,
                           dt: float, threshold_m: float = 0.05) -> float:
    """Time to return within threshold of start position.

    Args:
        positions_xy: [T, 2] CoM XY positions during recovery.
        start_pos: [2] starting position.
        dt: Timestep.
        threshold_m: Distance threshold in meters.

    Returns:
        Settling time in seconds. Returns total time if never settles.
    """
    if len(positions_xy) == 0:
        return 0.0
    dists = np.linalg.norm(positions_xy - start_pos, axis=1)
    within = np.where(dists < threshold_m)[0]
    if len(within) > 0:
        return within[0] * dt
    return len(positions_xy) * dt


def compute_peak_displacement(positions_xy: np.ndarray, start_pos: np.ndarray) -> float:
    """Max CoM XY deviation from start position.

    Returns:
        Peak displacement in meters.
    """
    if len(positions_xy) == 0:
        return 0.0
    dists = np.linalg.norm(positions_xy - start_pos, axis=1)
    return float(np.max(dists))


# ── Plotting helpers ─────────────────────────────────────────────────────────


def polar_plot(ax, angles_deg: np.ndarray, values_per_dir: dict,
               title: str, unit: str, color: str = "tab:blue"):
    """Reusable polar plot: mean +/- std, closed loop.

    Args:
        ax: Matplotlib polar axis.
        angles_deg: Array of direction angles in degrees.
        values_per_dir: Dict mapping degree (float) → list of float values.
        title: Plot title.
        unit: Unit string for subtitle.
        color: Line color.
    """
    angles_rad = np.deg2rad(angles_deg)
    angles_closed = np.append(angles_rad, angles_rad[0])

    means = [np.mean(values_per_dir[d]) if len(values_per_dir[d]) > 0 else 0.0 for d in angles_deg]
    stds = [np.std(values_per_dir[d]) if len(values_per_dir[d]) > 0 else 0.0 for d in angles_deg]
    means_closed = np.array(means + [means[0]])
    stds_closed = np.array(stds + [stds[0]])

    ax.plot(angles_closed, means_closed, color=color, linewidth=2)
    ax.fill_between(angles_closed, np.clip(means_closed - stds_closed, 0, None),
                    means_closed + stds_closed, alpha=0.2, color=color)
    ax.set_title(f"{title}\n({unit})", fontsize=11, pad=15)
    ax.set_thetagrids(angles_deg)


def polar_heatmap(ax, angles_deg: np.ndarray, radii: np.ndarray,
                  values_2d: np.ndarray, title: str, cmap: str = "RdYlGn"):
    """Polar heatmap: radius = force magnitude, angle = direction, color = metric.

    Args:
        ax: Matplotlib polar axis.
        angles_deg: 1D array of angles in degrees.
        radii: 1D array of radii (e.g. force magnitudes).
        values_2d: [len(radii), len(angles_deg)] array of values.
        title: Plot title.
        cmap: Colormap name.
    """
    angles_rad = np.deg2rad(angles_deg)
    # Create mesh: need to close the angle loop
    angles_closed = np.append(angles_rad, angles_rad[0])
    # Add a zero radius for the center
    radii_extended = np.concatenate([[0], radii])
    # Extend values: close angle loop, add center row
    vals_closed = np.hstack([values_2d, values_2d[:, 0:1]])  # close angle
    center_vals = np.mean(vals_closed, axis=1, keepdims=True)
    # Not needed for pcolormesh — just use the grid directly

    theta, r = np.meshgrid(angles_closed, radii_extended)
    # Extend values to match mesh: [len(radii_extended), len(angles_closed)]
    # First row (r=0) gets average of surrounding values
    vals_full = np.vstack([np.mean(vals_closed, axis=0, keepdims=True), vals_closed])

    mesh = ax.pcolormesh(theta, r, vals_full, cmap=cmap, shading="auto", vmin=0, vmax=100)
    ax.set_thetagrids(angles_deg)
    ax.set_title(title, fontsize=11, pad=15)

    # Add 80% contour
    try:
        ax.contour(theta, r, vals_full, levels=[80], colors="black", linewidths=1.5, linestyles="--")
    except Exception:
        pass  # contour may fail if all values are above/below 80

    return mesh


def make_summary_table(ax, headers: list[str], rows: list[list[str]], title: str):
    """Create a summary table on a matplotlib axis."""
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)


# ── Visualization ────────────────────────────────────────────────────────────


def create_arrow_markers(prim_path: str, color: tuple) -> VisualizationMarkers:
    """Create coloured arrow markers for force visualization."""
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


def dir_to_quat(dx: float, dy: float, device) -> torch.Tensor:
    """Quaternion to orient arrow along (dx, dy, 0) direction. Returns [1, 4]."""
    quats = torch.zeros(1, 4, device=device)
    quats[0, 0] = 1.0  # identity
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 0.01:
        return quats
    angle = math.atan2(dy, dx)
    quats[0, 0] = math.cos(angle / 2)
    quats[0, 3] = math.sin(angle / 2)
    return quats


def _force_vec_to_quat(force_world: torch.Tensor, device) -> torch.Tensor:
    """Convert world-frame force vectors [N, 3] to arrow quaternions [N, 4].

    Builds a proper rotation matrix from the force direction.
    Identical to static_eval.py ``_force_to_quat``.
    """
    from isaaclab.utils.math import quat_from_matrix

    n = force_world.shape[0]
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0  # identity (w, x, y, z)

    for i in range(n):
        mag = force_world[i].norm()
        if mag < 0.1:
            continue
        x_ax = force_world[i] / mag
        up = torch.tensor(
            [0.0, 0.0, 1.0] if abs(x_ax[2].item()) < 0.9 else [1.0, 0.0, 0.0],
            device=device,
        )
        z_ax = torch.linalg.cross(x_ax, up)
        z_ax = z_ax / z_ax.norm()
        y_ax = torch.linalg.cross(z_ax, x_ax)
        rot_mat = torch.stack([x_ax, y_ax, z_ax], dim=1)  # [3, 3]
        quats[i] = quat_from_matrix(rot_mat)
    return quats


def create_trajectory_markers(prim_path: str, num_points: int, device) -> VisualizationMarkers:
    """Create sphere markers for trajectory visualization (green line of dots)."""
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def update_force_arrow(markers: VisualizationMarkers, asset, base_idx,
                       fx: float, fy: float, device, n: int):
    """Update the GT force arrow visualization above the robot.

    The force (fx, fy) is in the body frame. It is transformed to world frame
    using the robot's orientation before rendering, matching static_eval.py.

    Args:
        markers: Red arrow VisualizationMarkers instance.
        asset: Robot asset.
        base_idx: Base body index.
        fx, fy: Applied force components (body frame).
        device: Torch device.
        n: Number of envs.
    """
    from isaaclab.utils.math import quat_apply

    force_mag = math.sqrt(fx * fx + fy * fy)
    if force_mag < 0.1:
        markers.set_visibility(False)
        return

    markers.set_visibility(True)

    # Body-frame force → world-frame force
    base_quat = asset.data.root_quat_w[:n]
    force_body = torch.zeros(n, 3, device=device)
    force_body[:, 0] = fx
    force_body[:, 1] = fy
    force_world = quat_apply(base_quat, force_body)

    # Position above robot
    pos = asset.data.root_pos_w[:n].clone()
    pos[:, 2] += 0.55

    # Orientation from world-frame force direction
    quats = _force_vec_to_quat(force_world, device)

    # Scale: length proportional to force magnitude
    force_scale = 0.05  # 20N -> length 1.0
    gt_mag = force_world.norm(dim=-1, keepdim=True)
    scales = torch.full((n, 3), 0.3, device=device)
    scales[:, 0:1] = (gt_mag * force_scale).clamp(min=0.05)

    markers.visualize(pos, quats, scales)


def update_force_arrows_per_env(markers: VisualizationMarkers, asset,
                                fx_per_env: torch.Tensor, fy_per_env: torch.Tensor,
                                device, n: int):
    """Update per-env force arrow visualization.

    Like update_force_arrow but each env has its own (fx, fy) force vector.

    Args:
        markers: Arrow VisualizationMarkers instance.
        asset: Robot asset.
        fx_per_env: [n] tensor of per-env X force components (world frame).
        fy_per_env: [n] tensor of per-env Y force components (world frame).
        device: Torch device.
        n: Number of envs.
    """
    force_world = torch.zeros(n, 3, device=device)
    force_world[:, 0] = fx_per_env[:n]
    force_world[:, 1] = fy_per_env[:n]

    any_force = force_world.norm(dim=-1).max() > 0.1
    if not any_force:
        markers.set_visibility(False)
        return

    markers.set_visibility(True)

    pos = asset.data.root_pos_w[:n].clone()
    pos[:, 2] += 0.55

    quats = _force_vec_to_quat(force_world, device)

    force_scale = 0.05
    gt_mag = force_world.norm(dim=-1, keepdim=True)
    scales = torch.full((n, 3), 0.3, device=device)
    scales[:, 0:1] = (gt_mag * force_scale).clamp(min=0.05)

    markers.visualize(pos, quats, scales)


def update_trajectory_vis(markers: VisualizationMarkers, positions: list[tuple],
                          height: float, device):
    """Update trajectory visualization with a line of sphere markers.

    Args:
        markers: Sphere VisualizationMarkers instance.
        positions: List of (x, y) world-frame positions along the desired path.
        height: Z height for the markers.
        device: Torch device.
    """
    if not positions:
        return
    n = len(positions)
    pos = torch.zeros(n, 3, device=device)
    for i, (x, y) in enumerate(positions):
        pos[i, 0] = x
        pos[i, 1] = y
        pos[i, 2] = height
    # Identity quaternion for spheres
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 1.0, device=device)
    markers.visualize(pos, quats, scales)


# ── Data saving ──────────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to Python types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def create_eval_output_dir(experiment_name: str, eval_type: str, suffix: str = "") -> str:
    """Create go2_rl_lab/data/eval/<experiment_name>/<eval_type>_<timestamp>[_<suffix>]/.

    Args:
        experiment_name: Name of the experiment (e.g. "go2_lowlevel").
        eval_type: Type of evaluation (e.g. "static_360").
        suffix: Optional suffix appended after the timestamp (e.g. "_A1_nonlinear_25N").

    Returns:
        Path to the output directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = f"{eval_type}_{timestamp}"
    if suffix:
        dirname += f"_{suffix}"
    out_dir = os.path.join("data", "eval", experiment_name, dirname)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_config(output_dir: str, args_cli, agent_cfg, dt: float):
    """Save config.json with CLI args and env metadata."""
    config = {
        "task": getattr(args_cli, "task", None),
        "checkpoint": getattr(args_cli, "checkpoint", None),
        "runner_class": getattr(agent_cfg, "class_name", None),
        "experiment_name": getattr(agent_cfg, "experiment_name", None),
        "dt": dt,
        "timestamp": datetime.now().isoformat(),
    }
    # Add all CLI args
    for key, val in vars(args_cli).items():
        if key not in config:
            try:
                json.dumps(val)  # check serializable
                config[key] = val
            except (TypeError, ValueError):
                config[key] = str(val)

    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, cls=_NumpyEncoder)
    print(f"[eval] Config saved: {path}")


def save_raw_data(output_dir: str, results: dict):
    """Save raw_data.json. Converts numpy arrays to lists."""
    path = os.path.join(output_dir, "raw_data.json")
    with open(path, "w") as f:
        json.dump({"results": results}, f, indent=1, cls=_NumpyEncoder)
    print(f"[eval] Raw data saved: {path}")


def save_report_pdf(output_dir: str, figures: list):
    """Save report.pdf from list of matplotlib figures."""
    from matplotlib.backends.backend_pdf import PdfPages

    path = os.path.join(output_dir, "report.pdf")
    with PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
    print(f"[eval] Report saved: {path}")
