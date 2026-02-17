from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)



def base_pose_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, desired_height: float
) -> torch.Tensor:
    """Paper rpose: φ² + ψ² + 10·(y - ydes)²"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get roll, pitch, yaw from quaternion
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # Height error
    height_error = asset.data.root_pos_w[:, 2] - desired_height
    
    return roll**2 + pitch**2 + 10 * height_error**2

"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_clearence_dense(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)

    # this is reward is more oermisice on fast swings but we want smooth motion when we push the robot so palize velocity^2 rather than tanh(velocity)
    # foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    # reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)

    # lets penalize more strong movement with L2 norm
    foot_velocity_sq = torch.sum(footvel_in_body_frame[:, :, :2]**2, dim=2)  # ||v_xz||²
    reward = torch.sum(foot_z_target_error * foot_velocity_sq, dim=1)

    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return reward


def foot_height_sparse(env: ManagerBasedRLEnv,asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,target_height: float
) -> torch.Tensor:
    """Paper rh: Σ (ppeak / pdes - 1)² - Applied when landing"""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Initialize peak height tracker
    if not hasattr(env, 'foot_peak_heights'):   
        env.foot_peak_heights = torch.zeros(
            env.num_envs, len(asset_cfg.body_ids), device=env.device
        )
    
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    
    # Track peak during swing, reset when in contact
    peak_before_reset = env.foot_peak_heights.clone()

    env.foot_peak_heights = torch.where(
        is_contact,
        foot_heights,
        torch.maximum(env.foot_peak_heights, foot_heights)
    )
    
    # Penalty at touchdown: (ppeak / pdes - 1)²
    just_contacted = (contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] < env.step_dt) & is_contact
    penalty = ((peak_before_reset / target_height - 1)**2) * just_contacted.float()
    return torch.sum(penalty, dim=1)
    




def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def follow_force_direction(env: ManagerBasedRLEnv,std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")
) -> torch.Tensor:
    """Reward for moving in the direction of applied force (exponential kernel)."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get applied force in body frame
    external_force_b = asset._external_force_b[:, asset_cfg.body_ids, :].squeeze(1)

    external_force_b_xy = external_force_b.clone()
    external_force_b_xy[:, 2] = 0.0

    # Get current velocity in body frame
    velocity_b = asset.data.root_lin_vel_b
    
    # Normalize to get directions
    force_norm = torch.norm(external_force_b_xy, dim=1, keepdim=True).clamp(min=1e-6)
    vel_norm = torch.norm(velocity_b, dim=1, keepdim=True).clamp(min=1e-6)
    
    force_dir = external_force_b_xy / force_norm
    vel_dir = velocity_b / vel_norm
    
    # Dot product ∈ [-1, 1]
    alignment = torch.sum(force_dir * vel_dir, dim=1)
    
    # Exponential kernel: converts [-1, 1] to (0, 1]
    # alignment=1 → reward=1, alignment=-1 → reward≈0.14
    alignment_reward = torch.exp((alignment - 1) / (std**2))
    
    # Add: Speed matching component
    v_target = 0.5  # Target speed in m/s when force applied
    vel_magnitude = vel_norm.squeeze(1)
    speed_error = torch.abs(vel_magnitude - v_target)
    speed_reward = torch.exp(-speed_error / std)

    # Combine: both direction AND speed matter
    reward = alignment_reward * speed_reward


    # Only apply when force is significant
    reward = reward * (force_norm.squeeze(1) > 2.5).float()

    env.extras["force_alignment_mean"] = alignment.mean()
    env.extras["force_reward_mean"] = reward.mean()
    return reward

def action_smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the second-order finite difference of actions: |a_t - 2*a_{t-1} + a_{t-2}|.

    This penalizes jerkiness (sudden changes in acceleration) and produces much smoother
    gaits than first-order action rate alone. From Walk These Ways (Margolis & Agrawal, 2022).
    """
    am = env.action_manager
    # We need a_{t-2}. Store it on the env since the action manager only keeps one prev.
    if not hasattr(env, "_action_t_minus_2"):
        env._action_t_minus_2 = torch.zeros_like(am.action)
        env._action_t_minus_1 = torch.zeros_like(am.action)

    second_diff = am.action - 2 * env._action_t_minus_1 + env._action_t_minus_2

    # Shift: t-1 becomes t-2, current becomes t-1
    env._action_t_minus_2 = env._action_t_minus_1.clone()
    env._action_t_minus_1 = am.action.clone()

    return torch.sum(torch.square(second_diff), dim=1)


def track_lin_vel_xy_exp_staged(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Linear velocity tracking with temporal stage support.

    During RECOVERY stage, returns a frozen constant value instead of actual
    tracking reward. This allows the robot to deviate from commands without
    penalty, encouraging compliant behavior per Hartmann et al. (2024).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] -
            asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1
    )
    normal_reward = torch.exp(-lin_vel_error / std)

    if hasattr(env, '_temporal_stage_recovery_mask'):
        mask = env._temporal_stage_recovery_mask
        frozen = env._temporal_stage_frozen_value
        return (1.0 - mask) * normal_reward + mask * frozen

    return normal_reward


def track_ang_vel_z_exp_staged(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angular velocity tracking with temporal stage support.

    During RECOVERY stage, returns a frozen constant value instead of actual
    tracking reward. This allows the robot to deviate from commands without
    penalty, encouraging compliant behavior per Hartmann et al. (2024).

    Paper: "we exchange the reward rlin and rang by constant values"
    """
    asset: Articulation = env.scene[asset_cfg.name]

    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] -
        asset.data.root_ang_vel_b[:, 2]
    )
    normal_reward = torch.exp(-ang_vel_error / std)

    if hasattr(env, '_temporal_stage_recovery_mask'):
        mask = env._temporal_stage_recovery_mask
        # Use same frozen value ratio as linear (could be separate param)
        frozen = env._temporal_stage_frozen_value
        return (1.0 - mask) * normal_reward + mask * frozen

    return normal_reward