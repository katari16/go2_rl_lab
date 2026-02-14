"""Custom event functions for locomotion tasks."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity_with_return(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Push the asset by setting the root velocity and return the sampled velocity delta.

    Same as isaaclab.envs.mdp.events.push_by_setting_velocity but returns the sampled
    velocity delta for visualization purposes.

    Args:
        env: The environment instance.
        env_ids: The environment indices to apply the push to.
        velocity_range: Dictionary with velocity ranges for each axis.
            Keys: "x", "y", "z", "roll", "pitch", "yaw". Values: (min, max) tuples.
        asset_cfg: The asset configuration to apply the push to.

    Returns:
        The sampled velocity delta tensor of shape (num_env_ids, 6) containing
        [x, y, z, roll, pitch, yaw] velocities.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)

    # apply the velocity
    vel_w = vel_w + sampled_vel

    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    # return the sampled velocity delta
    return sampled_vel


def push_with_visualization(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset and visualize the push direction with arrows.

    This event is intended for play/evaluation mode to test compliant behavior.
    It applies a velocity push and shows red arrows indicating push direction.

    Args:
        env: The environment instance.
        env_ids: The environment indices to apply the push to.
        velocity_range: Dictionary with velocity ranges for each axis.
        asset_cfg: The asset configuration to apply the push to.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Initialize visualization markers on first call
    if not hasattr(env, "_push_visualizer"):
        marker_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Events/push_velocity")
        marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        env._push_visualizer = VisualizationMarkers(marker_cfg)
        env._push_vel_storage = torch.zeros(env.num_envs, 2, device=env.device)
        env._push_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._push_timer = torch.zeros(env.num_envs, device=env.device)

    # Apply push
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    sampled_vel = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w = vel_w + sampled_vel
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

    # Store push velocity for visualization (x, y components)
    env._push_vel_storage[env_ids, 0] = sampled_vel[:, 0]
    env._push_vel_storage[env_ids, 1] = sampled_vel[:, 1]
    env._push_active[env_ids] = True
    env._push_timer[env_ids] = 1.0  # Show arrow for 1 second

    # Update visualization for all active pushes
    _update_push_visualization(env, asset)


def _update_push_visualization(env: ManagerBasedEnv, asset: RigidObject | Articulation):
    """Update push visualization markers."""
    if not hasattr(env, "_push_visualizer"):
        return

    # Decay timer
    env._push_timer = (env._push_timer - env.step_dt).clamp(min=0)
    env._push_active = env._push_timer > 0

    active_envs = env._push_active.nonzero(as_tuple=True)[0]

    if len(active_envs) == 0:
        env._push_visualizer.set_visibility(False)
        return

    env._push_visualizer.set_visibility(True)

    # Get positions and orientations for active envs
    base_pos_w = asset.data.root_pos_w[active_envs].clone()
    base_pos_w[:, 2] += 0.5  # Offset above robot
    base_quat_w = asset.data.root_quat_w[active_envs]

    # Convert push velocity to arrow orientation
    push_vel_xy = env._push_vel_storage[active_envs]
    arrow_scale, arrow_quat = _velocity_to_arrow(push_vel_xy, base_quat_w, env.device)

    env._push_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)


def _velocity_to_arrow(
    xy_velocity: torch.Tensor, base_quat_w: torch.Tensor, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert XY velocity to arrow scale and quaternion."""
    num_envs = xy_velocity.shape[0]

    # Default arrow scale
    arrow_scale = torch.tensor([0.5, 0.5, 0.5], device=device).repeat(num_envs, 1)
    vel_magnitude = torch.linalg.norm(xy_velocity, dim=1)
    arrow_scale[:, 0] *= vel_magnitude * 3.0

    # Heading angle from velocity
    heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    zeros = torch.zeros_like(heading_angle)
    arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
    arrow_quat = quat_mul(base_quat_w, arrow_quat)

    return arrow_scale, arrow_quat