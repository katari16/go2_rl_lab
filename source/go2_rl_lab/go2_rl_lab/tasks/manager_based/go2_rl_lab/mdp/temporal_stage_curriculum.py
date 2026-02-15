"""
Temporal Stage Curriculum for Compliance Training.

This curriculum divides each episode into stages with different reward behaviors.
Based on the approach from Hartmann et al. (2024) "Deep Compliant Control".
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .events import push_by_setting_velocity_with_return

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TemporalStageCurriculum(ManagerTermBase):
    """
    Curriculum that divides episodes into temporal stages.

    Stages:
        0 (WALKING): Normal tracking rewards, accumulate performance
        1 (RECOVERY): Frozen tracking rewards, energy rewards active
        2 (POST_RECOVERY): Tracking rewards restored

    Transitions:
        WALKING -> RECOVERY: After walking_duration AND if performance > threshold
        RECOVERY -> POST_RECOVERY: After recovery_duration
        POST_RECOVERY -> WALKING: After post_recovery_duration
    """

    # Stage constants
    STAGE_WALKING = 0
    STAGE_RECOVERY = 1
    STAGE_POST_RECOVERY = 2

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Extract parameters with defaults
        self.walking_duration = cfg.params.get("walking_duration", 2.0)
        self.recovery_duration = cfg.params.get("recovery_duration", 1.0)
        self.post_recovery_duration = cfg.params.get("post_recovery_duration", 1.0)
        self.reward_threshold = cfg.params.get("reward_threshold", 0.85)
        self.debug_vis = cfg.params.get("debug_vis", False)

        push_range = cfg.params.get("push_velocity_range", {"x": (-1.0, 1.0), "y": (-1.0, 1.0)})
        self.push_vel_x = push_range.get("x", (-1.0, 1.0))
        self.push_vel_y = push_range.get("y", (-1.0, 1.0))

        # State tensors
        self._stage = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._stage_timer = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

        # Store push velocities for visualization (x, y per env)
        self._push_vel = torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)

        # Cache original reward weights
        self._track_lin_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
        self._original_lin_weight = self._track_lin_cfg.weight

        # Cache robot asset for visualization
        self._robot = env.scene["robot"]

        # Setup debug visualization
        if self.debug_vis:
            marker_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Curriculum/push_velocity")
            marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            self._push_visualizer = VisualizationMarkers(marker_cfg)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage[env_ids] = self.STAGE_WALKING
        self._stage_timer[env_ids] = 0.0
        self._push_vel[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        walking_duration: float = 2.0,
        recovery_duration: float = 1.0,
        post_recovery_duration: float = 1.0,
        reward_threshold: float = 0.85,
        push_velocity_range: dict = None,
        ) -> dict:

        dt = env.step_dt
        self._stage_timer += dt

        # Handle WALKING stage
        walking_envs = (self._stage == self.STAGE_WALKING).nonzero(as_tuple=True)[0]
        # Debug metrics (will be logged to tensorboard)
        self._debug_threshold = self._original_lin_weight * self.reward_threshold
        self._debug_reward_rate_mean = 0.0
        self._debug_reward_rate_min = 0.0
        self._debug_reward_rate_max = 0.0
        self._debug_episode_sum = 0.0
        self._debug_stage_timer = 0.0

        if len(walking_envs) > 0:
            # Same pattern as curriculums.py: episode_sum / max_episode_length_s
            episode_sum = env.reward_manager._episode_sums["track_lin_vel_xy_exp"][walking_envs]
            reward_rate = episode_sum / env.max_episode_length_s

            # Store debug values
            self._debug_reward_rate_mean = reward_rate.mean().item()
            self._debug_reward_rate_min = reward_rate.min().item()
            self._debug_reward_rate_max = reward_rate.max().item()
            self._debug_episode_sum = episode_sum.mean().item()
            self._debug_stage_timer = self._stage_timer[walking_envs].mean().item()

            ready = (
                (self._stage_timer[walking_envs] >= self.walking_duration) &
                (reward_rate > self._original_lin_weight * self.reward_threshold)
            )
            transition_envs = walking_envs[ready]
            if len(transition_envs) > 0:
                self._enter_recovery(env, transition_envs)

        # Handle RECOVERY stage
        recovery_envs = (self._stage == self.STAGE_RECOVERY).nonzero(as_tuple=True)[0]
        if len(recovery_envs) > 0:
            done = self._stage_timer[recovery_envs] >= self.recovery_duration
            transition_envs = recovery_envs[done]
            if len(transition_envs) > 0:
                self._enter_post_recovery(env, transition_envs)

        # Handle POST_RECOVERY stage
        post_envs = (self._stage == self.STAGE_POST_RECOVERY).nonzero(as_tuple=True)[0]
        if len(post_envs) > 0:
            done = self._stage_timer[post_envs] >= self.post_recovery_duration
            transition_envs = post_envs[done]
            if len(transition_envs) > 0:
                self._enter_walking(env, transition_envs)

        # Store mask for reward functions
        env._temporal_stage_recovery_mask = (self._stage == self.STAGE_RECOVERY).float()
        env._temporal_stage_frozen_value = self._original_lin_weight * self.reward_threshold

        # Update debug visualization
        if self.debug_vis:
            self._update_debug_vis(env)

        # Return metrics
        n = float(env.num_envs)
        return {
            "walking_frac": (self._stage == self.STAGE_WALKING).sum().item() / n,
            "recovery_frac": (self._stage == self.STAGE_RECOVERY).sum().item() / n,
            "post_recovery_frac": (self._stage == self.STAGE_POST_RECOVERY).sum().item() / n,
            "debug/reward_rate_mean": self._debug_reward_rate_mean,
            "debug/reward_rate_min": self._debug_reward_rate_min,
            "debug/reward_rate_max": self._debug_reward_rate_max,
            "debug/threshold": self._debug_threshold,
            "debug/episode_sum": self._debug_episode_sum,
            "debug/stage_timer": self._debug_stage_timer,
        }

    def _enter_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        self._stage[env_ids] = self.STAGE_RECOVERY
        self._stage_timer[env_ids] = 0.0

        # Apply push and store the velocity for visualization
        sampled_vel = push_by_setting_velocity_with_return(
            env, env_ids,
            velocity_range={"x": self.push_vel_x, "y": self.push_vel_y},
            asset_cfg=SceneEntityCfg("robot")
        )
        # Store x, y components for visualization
        self._push_vel[env_ids, 0] = sampled_vel[:, 0]
        self._push_vel[env_ids, 1] = sampled_vel[:, 1]

    def _enter_post_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        self._stage[env_ids] = self.STAGE_POST_RECOVERY
        self._stage_timer[env_ids] = 0.0
        # Clear push velocity for these envs
        self._push_vel[env_ids] = 0.0

    def _enter_walking(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        self._stage[env_ids] = self.STAGE_WALKING
        self._stage_timer[env_ids] = 0.0

    def _update_debug_vis(self, env: ManagerBasedRLEnv):
        """Update visualization markers for push velocity."""
        # Only visualize envs in RECOVERY stage
        recovery_mask = (self._stage == self.STAGE_RECOVERY)
        num_recovery = recovery_mask.sum().item()

        if num_recovery == 0:
            self._push_visualizer.set_visibility(False)
            return

        self._push_visualizer.set_visibility(True)

        # Get recovery env indices
        recovery_envs = recovery_mask.nonzero(as_tuple=True)[0]

        # Get robot positions for recovery envs
        base_pos_w = self._robot.data.root_pos_w[recovery_envs].clone()
        base_pos_w[:, 2] += 0.5  # Offset above robot

        # Get robot orientations for transforming arrow to world frame
        base_quat_w = self._robot.data.root_quat_w[recovery_envs]

        # Convert push velocity to arrow orientation and scale
        push_vel_xy = self._push_vel[recovery_envs]
        arrow_scale, arrow_quat = self._resolve_xy_velocity_to_arrow(push_vel_xy, base_quat_w)

        # Visualize
        self._push_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, base_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert XY velocity to arrow scale and quaternion.

        Args:
            xy_velocity: Velocity in XY plane, shape (N, 2)
            base_quat_w: Base orientation quaternion in world frame, shape (N, 4)

        Returns:
            Tuple of (arrow_scale, arrow_quat) for visualization
        """
        num_envs = xy_velocity.shape[0]
        device = xy_velocity.device

        # Default arrow scale
        default_scale = (0.5, 0.5, 0.5)
        arrow_scale = torch.tensor(default_scale, device=device).repeat(num_envs, 1)

        # Scale arrow length by velocity magnitude
        vel_magnitude = torch.linalg.norm(xy_velocity, dim=1)
        arrow_scale[:, 0] *= vel_magnitude * 3.0

        # Compute heading angle from velocity
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)

        # Create quaternion from heading angle (rotation around Z axis)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)

        # Transform from base frame to world frame
        arrow_quat = quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat